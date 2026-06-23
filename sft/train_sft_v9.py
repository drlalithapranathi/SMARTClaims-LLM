#!/usr/bin/env python3
"""
SFT v9 — Qwen3-32B on full MIMIC-IV radiology dataset (18,805 admissions).
v9 changes vs v8:
  - 10x more training data (18,805 vs 1,815 admissions)
  - 1 epoch (sufficient with large dataset)
  - CPT codes as labels, full MIMIC radiology reports

Base model: qwen3-32b-mimic-cpt-merged (Base + CPT already merged)
Trains a single SFT LoRA on top of the merged CPT model.
Saves SFT adapter + full merged model.

Run:
    accelerate launch --num_processes 4 --mixed_precision bf16 train_sft_v9.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import torch
import wandb
import transformers
from accelerate import Accelerator
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training, LoraConfig
from trl import SFTTrainer, SFTConfig

# ============================================================
# CONFIG
# ============================================================
BASE_MODEL      = "qwen3-32b-mimic-cpt-merged"   # CPT already merged in
SFT_TRAIN_DATA  = "sft_v9_train_dataset"
SFT_VAL_DATA    = "sft_v9_val_dataset"
OUTPUT_DIR      = "sft_v9_out"
SAVE_ADAPTER    = "qwen3-32b-sft-v9"
SAVE_FULL_MODEL = "qwen3-32b-sft-v9-merged"
TOKENIZER_SRC   = "qwen3-32b-mimic-cpt-200k-ep2"  # tokenizer lives here

EPOCHS         = 1
BATCH_SIZE     = 1
GRAD_ACCUM     = 8
LEARNING_RATE  = 2e-5
LORA_RANK      = 16
MAX_SEQ_LENGTH = 4096
WANDB_PROJECT  = "SmartClaims-SFT-v9"

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
    print(f"  SFT v9 — DDP on {world_size} GPUs")
    print(f"  Base model       : {BASE_MODEL}  (CPT merged in)")
    print(f"  SFT adapter out  : {SAVE_ADAPTER}  [fresh LoRA]")
    print(f"  Max seq length   : {MAX_SEQ_LENGTH}")
    print(f"  Effective batch  : {BATCH_SIZE} x {GRAD_ACCUM} x {world_size}"
          f" = {BATCH_SIZE * GRAD_ACCUM * world_size}")
    print(f"{'='*60}\n")

# ============================================================
# TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SRC, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============================================================
# MODEL — load CPT-merged base, add trainable SFT LoRA
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

if accelerator.is_main_process:
    print(f"Loading CPT-merged base model from {BASE_MODEL} ...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": local_rank},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True
)

if accelerator.is_main_process:
    print("Model loaded. Adding SFT LoRA adapter ...")

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

if accelerator.is_main_process:
    alloc = torch.cuda.memory_allocated(local_rank) / 1e9
    total = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
    print(f"\nGPU {local_rank}: {alloc:.1f} GB / {total:.1f} GB"
          f" ({total - alloc:.1f} GB free)\n")

# ============================================================
# DATASET — load pre-built parquet datasets
# ============================================================
if accelerator.is_main_process:
    print("Loading pre-built SFT datasets ...")

train_dataset = load_from_disk(SFT_TRAIN_DATA)
val_dataset   = load_from_disk(SFT_VAL_DATA)

if accelerator.is_main_process:
    print(f"  Train: {len(train_dataset):,} admissions")
    print(f"  Val  : {len(val_dataset):,} admissions")

# ============================================================
# W&B
# ============================================================
if accelerator.is_main_process:
    wandb.login()
    wandb.init(
        project=WANDB_PROJECT,
        name=f"qwen3-32b-sft-v9-r{LORA_RANK}-{EPOCHS}ep-DDP{world_size}gpu",
        config={
            "model":           BASE_MODEL,
            "sft_strategy":    "cpt_merged_base + fresh_sft_lora",
            "data_version":    "v9 — full MIMIC radiology, 18,805 admissions",
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
        tags=["stage2", "sft-v9", "mimic-iv", "qwen3-32b", "cpt-codes", "ddp"],
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
    peft_config=lora_config,
)

# ============================================================
# TRAIN
# ============================================================
if accelerator.is_main_process:
    print("\nStarting SFT v9 ...\n")
    trainer.model.print_trainable_parameters()

start_time = time.time()
trainer_stats = trainer.train()
elapsed = time.time() - start_time

# ============================================================
# SAVE ADAPTER + MERGE FULL MODEL (main process only)
# ============================================================
if accelerator.is_main_process:
    print(f"\n✓ SFT v9 complete!")
    print(f"  Duration  : {elapsed / 3600:.1f} hours")
    print(f"  Final loss: {trainer_stats.training_loss:.4f}")

    used_mem  = torch.cuda.max_memory_reserved(local_rank) / 1e9
    total_mem = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
    print(f"  Peak VRAM : {used_mem:.1f} / {total_mem:.1f} GB")

    # Save SFT adapter
    os.makedirs(SAVE_ADAPTER, exist_ok=True)
    trainer.model.save_pretrained(SAVE_ADAPTER)
    tokenizer.save_pretrained(SAVE_ADAPTER)
    print(f"\n✓ SFT adapter saved → {SAVE_ADAPTER}/")

    # Merge: CPT-merged base + SFT adapter → full model in bf16
    print(f"\nMerging full model ({BASE_MODEL} + SFT adapter) ...")
    from peft import PeftModel
    merge_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    m = PeftModel.from_pretrained(merge_base, SAVE_ADAPTER)
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
    print("  STAGE 2 SFT v9 COMPLETE")
    print(f"  Base model   : {BASE_MODEL}/  (CPT merged in)")
    print(f"  SFT adapter  : {SAVE_ADAPTER}/")
    print(f"  Full model   : {SAVE_FULL_MODEL}/")
    print("=" * 60)
