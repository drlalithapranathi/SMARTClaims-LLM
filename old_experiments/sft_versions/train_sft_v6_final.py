#!/usr/bin/env python3
"""
SFT of Qwen3-32B on MIMIC-IV radiology → CPT code prediction.
Uses v6 data: billable CPT codes only, VSAC descriptions as labels.

Starts from ep2 CPT adapter (frozen), adds fresh SFT LoRA on top.
Saves SFT adapter + full merged model.

Run:
    accelerate launch --num_processes 2 --mixed_precision bf16 train_sft_v6_final.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import torch
import wandb
import pandas as pd
import transformers
from accelerate import Accelerator
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, prepare_model_for_kbit_training, LoraConfig
from trl import SFTTrainer, SFTConfig

# ============================================================
# CONFIG
# ============================================================
BASE_MODEL      = "Qwen/Qwen3-32B"
CPT_ADAPTER     = "qwen3-32b-mimic-cpt-200k-ep2"
TRAIN_CSV       = "mimic_radiology_sft_train.csv"
OUTPUT_DIR      = "outputs_sft_qwen3_32b"
SAVE_ADAPTER    = "qwen3-32b-mimic-sft"
SAVE_FULL_MODEL = "qwen3-32b-mimic-sft-merged"

EPOCHS         = 3
BATCH_SIZE     = 2
GRAD_ACCUM     = 4
LEARNING_RATE  = 2e-5
LORA_RANK      = 16
MAX_SEQ_LENGTH = 4096
VAL_FRACTION   = 0.1
WANDB_PROJECT  = "SmartClaims-SFT"

SYSTEM_PROMPT = (
    "You are a medical coding assistant. "
    "Given a clinical note for a patient admission, identify all billable "
    "radiology procedures performed. Output only the CPT codes separated by "
    "pipe ( | ). If no billable radiology procedure can be determined, output "
    "unknown. Do not include add-on codes, explanations, or numbering."
)

# ============================================================
# ACCELERATOR
# ============================================================
accelerator = Accelerator()
local_rank  = accelerator.local_process_index
world_size  = accelerator.num_processes

if accelerator.is_main_process:
    print(f"\n{'='*60}")
    print(f"  SFT — DDP on {world_size} GPUs")
    print(f"  CPT adapter      : {CPT_ADAPTER}  [FROZEN — will not be modified]")
    print(f"  SFT adapter      : {SAVE_ADAPTER}  [fresh LoRA, trained here]")
    print(f"  Max seq length   : {MAX_SEQ_LENGTH}")
    print(f"  Effective batch  : {BATCH_SIZE} x {GRAD_ACCUM} x {world_size}"
          f" = {BATCH_SIZE * GRAD_ACCUM * world_size}")
    print(f"{'='*60}\n")

# ============================================================
# TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(CPT_ADAPTER, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============================================================
# MODEL — base + frozen CPT adapter + trainable SFT LoRA
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

if accelerator.is_main_process:
    print("Loading base model ...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": local_rank},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(
    base_model, use_gradient_checkpointing=True
)

if accelerator.is_main_process:
    print(f"Loading CPT adapter (FROZEN) from {CPT_ADAPTER} ...")

model = PeftModel.from_pretrained(base_model, CPT_ADAPTER, is_trainable=False)

if accelerator.is_main_process:
    print("Adding fresh SFT LoRA adapter ...")

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora=True,
)
model.add_adapter("sft", lora_config)
model.set_adapter("sft")

if accelerator.is_main_process:
    model.print_trainable_parameters()
    alloc = torch.cuda.memory_allocated(local_rank) / 1e9
    total = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
    print(f"\nGPU {local_rank}: {alloc:.1f} GB / {total:.1f} GB"
          f" ({total - alloc:.1f} GB free)\n")

# ============================================================
# DATASET — load CSV, re-aggregate per hadm_id, build messages
# ============================================================
# v6 CSV has one row per CPT code. We re-aggregate per hadm_id
# so each training example = one admission with all its billable
# codes as the label.
# ============================================================

if accelerator.is_main_process:
    print("Loading and preparing dataset ...")

df = pd.read_csv(TRAIN_CSV, dtype={"cpt_codes": str})

# Re-aggregate: group by hadm_id, take first report, join all labels (sorted for consistency)
agg = df.groupby("hadm_id").agg(
    reports=("reports", "first"),
    cpt_codes=("cpt_codes", lambda x: " | ".join(sorted(x))),
).reset_index()

if accelerator.is_main_process:
    print(f"  Raw CSV rows:      {len(df):,}")
    print(f"  Unique admissions: {len(agg):,}")
    codes_per_adm = df.groupby("hadm_id").size()
    print(f"  CPT codes per admission: mean={codes_per_adm.mean():.1f}  max={codes_per_adm.max()}")

# Build messages format for SFTTrainer
records = []
for _, row in agg.iterrows():
    records.append({
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": row["reports"]},
            {"role": "assistant", "content": row["cpt_codes"]},
        ]
    })

full_dataset = Dataset.from_list(records).shuffle(seed=42)

# Split: 90% train, 10% monitoring val
split = full_dataset.train_test_split(test_size=VAL_FRACTION, seed=42)
train_dataset = split["train"]
val_dataset   = split["test"]

if accelerator.is_main_process:
    print(f"  Train: {len(train_dataset)} | Val (monitoring): {len(val_dataset)}")

# ============================================================
# W&B
# ============================================================
if accelerator.is_main_process:
    wandb.login()
    wandb.init(
        project=WANDB_PROJECT,
        name=f"qwen3-32b-sft-r{LORA_RANK}-{EPOCHS}ep-DDP{world_size}gpu",
        config={
            "model":           BASE_MODEL,
            "cpt_adapter":     CPT_ADAPTER,
            "cpt_frozen":      True,
            "sft_strategy":    "frozen_cpt + fresh_sft_lora",
            "data_version":    "v6 — billable CPT codes only, VSAC descriptions",
            "lora_r":          LORA_RANK,
            "lora_dropout":    0.05,
            "learning_rate":   LEARNING_RATE,
            "epochs":          EPOCHS,
            "batch_size":      BATCH_SIZE,
            "grad_accum":      GRAD_ACCUM,
            "effective_batch": BATCH_SIZE * GRAD_ACCUM * world_size,
            "max_seq_length":  MAX_SEQ_LENGTH,
            "train_samples":   len(train_dataset),
            "val_samples":     len(val_dataset),
            "transformers":    transformers.__version__,
        },
        tags=["stage2", "sft", "mimic-iv", "qwen3-32b", "radiology", "ddp"],
    )

# ============================================================
# TRAINING ARGS
# ============================================================
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    warmup_ratio=0.1,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=1.0,
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="wandb" if accelerator.is_main_process else "none",
    seed=3407,
    max_length=MAX_SEQ_LENGTH,
    packing=False,
    remove_unused_columns=False,
    dataset_kwargs={"apply_chat_template_kwargs": {"enable_thinking": False}},
)

# ============================================================
# TRAINER
# ============================================================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
)

# ============================================================
# TRAIN
# ============================================================
if accelerator.is_main_process:
    print("\nStarting SFT ...\n")

start_time = time.time()
trainer_stats = trainer.train()
elapsed = time.time() - start_time

# ============================================================
# SAVE ADAPTER + MERGE FULL MODEL (main process only)
# ============================================================
if accelerator.is_main_process:
    print(f"\n✓ SFT complete!")
    print(f"  Duration  : {elapsed / 3600:.1f} hours")
    print(f"  Final loss: {trainer_stats.training_loss:.4f}")

    used_mem = torch.cuda.max_memory_reserved(local_rank) / 1e9
    total_mem = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
    print(f"  Peak VRAM : {used_mem:.1f} / {total_mem:.1f} GB")

    # Save SFT adapter
    os.makedirs(SAVE_ADAPTER, exist_ok=True)
    model.save_pretrained(SAVE_ADAPTER)
    tokenizer.save_pretrained(SAVE_ADAPTER)
    print(f"\n✓ SFT adapter saved → {SAVE_ADAPTER}/")

    # Merge: base + CPT + SFT → full model in bf16
    print(f"\nMerging full model (base + CPT adapter + SFT adapter) ...")
    merge_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    m = PeftModel.from_pretrained(merge_base, CPT_ADAPTER)
    m = m.merge_and_unload()
    print("  ✓ CPT adapter merged")

    m = PeftModel.from_pretrained(m, SAVE_ADAPTER)
    m = m.merge_and_unload()
    print("  ✓ SFT adapter merged")

    os.makedirs(SAVE_FULL_MODEL, exist_ok=True)
    m.save_pretrained(
        SAVE_FULL_MODEL, safe_serialization=True, max_shard_size="5GB"
    )
    tokenizer.save_pretrained(SAVE_FULL_MODEL)
    print(f"\n✓ Full merged model saved → {SAVE_FULL_MODEL}/")

    wandb.log({
        "train/final_loss":     trainer_stats.training_loss,
        "train/duration_hours": elapsed / 3600,
    })
    wandb.finish()

    print("\n" + "=" * 60)
    print("  STAGE 2 SFT COMPLETE")
    print(f"  CPT adapter  : {CPT_ADAPTER}/  [unchanged]")
    print(f"  SFT adapter  : {SAVE_ADAPTER}/")
    print(f"  Full model   : {SAVE_FULL_MODEL}/")
    print("=" * 60)
