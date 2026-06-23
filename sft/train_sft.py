#!/usr/bin/env python3
"""
Experiment: 10% unknowns, 2 epochs SFT
Base model: qwen3-32b-mimic-cpt-merged (CPT already merged in)

Run from exp_unk10_ep2/:
    accelerate launch --num_processes 3 --mixed_precision bf16 train_sft.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import torch
import wandb
import transformers
from accelerate import Accelerator
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig
from trl import SFTTrainer, SFTConfig

# ============================================================
# CONFIG
# ============================================================
BASE_DIR        = "../../.."
BASE_MODEL      = f"{BASE_DIR}/qwen3-32b-mimic-cpt-merged"
TOKENIZER_SRC   = f"{BASE_DIR}/qwen3-32b-mimic-cpt-200k-ep2"
SFT_TRAIN_DATA  = "unk10_sft_train"
SFT_VAL_DATA    = "unk10_sft_val"
OUTPUT_DIR      = "sft_out"
SAVE_ADAPTER    = "sft_adapter"
SAVE_FULL_MODEL = "sft_merged"

EPOCHS         = 2
BATCH_SIZE     = 1
GRAD_ACCUM     = 8
LEARNING_RATE  = 2e-5
LORA_RANK      = 16
MAX_SEQ_LENGTH = 4096
WANDB_PROJECT  = "SmartClaims-unk10-ep2"

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
    print(f"  SFT — 10% unknowns, 2 epochs, DDP on {world_size} GPUs")
    print(f"  Base model      : {BASE_MODEL}")
    print(f"  Train data      : {SFT_TRAIN_DATA}")
    print(f"  Adapter out     : {SAVE_ADAPTER}")
    print(f"  Merged out      : {SAVE_FULL_MODEL}")
    print(f"  Effective batch : {BATCH_SIZE} x {GRAD_ACCUM} x {world_size} = {BATCH_SIZE * GRAD_ACCUM * world_size}")
    print(f"{'='*60}\n")

# ============================================================
# TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SRC, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============================================================
# MODEL
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

if accelerator.is_main_process:
    print(f"Loading base model from {BASE_MODEL} ...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": local_rank},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

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

# ============================================================
# DATASET
# ============================================================
if accelerator.is_main_process:
    print("Loading datasets ...")

train_dataset = load_from_disk(SFT_TRAIN_DATA)
val_dataset   = load_from_disk(SFT_VAL_DATA)

if accelerator.is_main_process:
    print(f"  Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

# ============================================================
# W&B
# ============================================================
if accelerator.is_main_process:
    wandb.login()
    wandb.init(
        project=WANDB_PROJECT,
        name=f"sft-unk10-ep2-r{LORA_RANK}-DDP{world_size}gpu",
        config={
            "base_model":      BASE_MODEL,
            "unknown_fraction": 0.10,
            "epochs":          EPOCHS,
            "lora_r":          LORA_RANK,
            "learning_rate":   LEARNING_RATE,
            "batch_size":      BATCH_SIZE,
            "grad_accum":      GRAD_ACCUM,
            "max_seq_length":  MAX_SEQ_LENGTH,
            "train_samples":   len(train_dataset),
            "val_samples":     len(val_dataset),
        },
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
    print("\nStarting SFT training ...\n")
    trainer.model.print_trainable_parameters()

start_time = time.time()
trainer.train()
elapsed = time.time() - start_time

# ============================================================
# SAVE ADAPTER + MERGE
# ============================================================
if accelerator.is_main_process:
    print(f"\n✓ Training complete! Duration: {elapsed/3600:.1f} hours")

    os.makedirs(SAVE_ADAPTER, exist_ok=True)
    trainer.model.save_pretrained(SAVE_ADAPTER)
    tokenizer.save_pretrained(SAVE_ADAPTER)
    print(f"✓ Adapter saved → {SAVE_ADAPTER}/")

    print(f"\nMerging {BASE_MODEL} + {SAVE_ADAPTER} ...")
    from peft import PeftModel
    merge_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    m = PeftModel.from_pretrained(merge_base, SAVE_ADAPTER)
    m = m.merge_and_unload()

    os.makedirs(SAVE_FULL_MODEL, exist_ok=True)
    m.save_pretrained(SAVE_FULL_MODEL, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(SAVE_FULL_MODEL)
    print(f"✓ Merged model saved → {SAVE_FULL_MODEL}/")

    wandb.finish()

    print("\n" + "="*60)
    print("  SFT COMPLETE (10% unknowns, 2 epochs)")
    print(f"  Adapter : exp_unk10_ep2/{SAVE_ADAPTER}/")
    print(f"  Merged  : exp_unk10_ep2/{SAVE_FULL_MODEL}/")
    print("="*60)
