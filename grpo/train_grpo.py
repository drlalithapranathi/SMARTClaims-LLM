#!/usr/bin/env python3
"""
GRPO for unk10 ep2 experiment.
Base: unk10_sft_merged (CPT + SFT unk10 baked in)
Saves adapter to: unk10_grpo_adapter/
Output dir:       unk10_grpo_out/

Run:
    accelerate launch --num_processes 4 --mixed_precision bf16 train_grpo.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
import time
import torch
import wandb
import transformers
from accelerate import Accelerator
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig
from trl import GRPOTrainer, GRPOConfig

# ── CONFIG ────────────────────────────────────────────────────────────────────
SFT_MERGED_MODEL = "unk10_sft_merged"
TOKENIZER_SRC    = "../qwen3-32b-mimic-cpt-200k-ep2"
GRPO_TRAIN_DATA  = "unk10_grpo_train"
GRPO_VAL_DATA    = "unk10_grpo_val"
OUTPUT_DIR       = "unk10_grpo_out"
SAVE_ADAPTER     = "unk10_grpo_adapter"
WANDB_PROJECT    = "SmartClaims-unk10-GRPO"

EPOCHS           = 1
MAX_STEPS        = 300
BATCH_SIZE       = 1
GRAD_ACCUM       = 8
LEARNING_RATE    = 1e-5
LORA_RANK        = 16
MAX_SEQ_LENGTH   = 4096
MAX_COMPLETION   = 128
MAX_PROMPT       = MAX_SEQ_LENGTH - MAX_COMPLETION
NUM_GENERATIONS  = 4
TEMPERATURE      = 0.9
KL_BETA          = 0.04

# ── ACCELERATOR ───────────────────────────────────────────────────────────────
accelerator = Accelerator()
local_rank  = accelerator.local_process_index
world_size  = accelerator.num_processes

if accelerator.is_main_process:
    print(f"\n{'='*60}")
    print(f"  GRPO unk10 — DDP on {world_size} GPUs")
    print(f"  Base model      : {SFT_MERGED_MODEL}")
    print(f"  Train data      : {GRPO_TRAIN_DATA}")
    print(f"  GRPO adapter out: {SAVE_ADAPTER}")
    print(f"  Effective batch : {BATCH_SIZE} x {GRAD_ACCUM} x {world_size} = {BATCH_SIZE * GRAD_ACCUM * world_size}")
    print(f"{'='*60}\n")

# ── TOKENIZER ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SRC, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── MODEL ─────────────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

if accelerator.is_main_process:
    print(f"Loading {SFT_MERGED_MODEL} ...")

base_model = AutoModelForCausalLM.from_pretrained(
    SFT_MERGED_MODEL,
    quantization_config=bnb_config,
    device_map={"": local_rank},
    dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
for param in base_model.parameters():
    if param.dtype == torch.float32:
        param.data = param.data.to(torch.bfloat16)

if accelerator.is_main_process:
    alloc = torch.cuda.memory_allocated(local_rank) / 1e9
    total = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
    print(f"GPU {local_rank}: {alloc:.1f}/{total:.1f} GB ({total-alloc:.1f} free)\n")

# ── LORA ──────────────────────────────────────────────────────────────────────
grpo_lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora=True,
)

# ── REWARD ────────────────────────────────────────────────────────────────────
def parse_cpt_output(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text:
        return set(), False
    if text.strip().lower() == "unknown":
        return {"unknown"}, True
    parts = [p.strip() for p in text.split("|") if p.strip()]
    codes = {p for p in parts if re.match(r"^\d{5}$", p)}
    return codes, len(codes) > 0

def parse_ground_truth(gt_str):
    gt_str = gt_str.strip()
    if gt_str.lower() == "unknown":
        return {"unknown"}
    return {c.strip() for c in gt_str.split("|") if re.match(r"^\d{5}$", c.strip())}

def compute_set_f1(pred, gt):
    if not pred or not gt:
        return 0.0
    tp = len(pred & gt)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(gt)
    return 2.0 * p * r / (p + r)

def cpt_f1_reward(prompts, completions, **kwargs):
    ground_truths = kwargs["ground_truth"]
    rewards = []
    for completion, gt_str in zip(completions, ground_truths):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        pred_codes, is_valid = parse_cpt_output(text)
        gt_codes = parse_ground_truth(gt_str)
        if is_valid:
            reward = compute_set_f1(pred_codes, gt_codes)
        else:
            reward = -0.2 if not text.strip() else -0.1
        rewards.append(float(reward))
    return rewards

# ── DATASET ───────────────────────────────────────────────────────────────────
if accelerator.is_main_process:
    print("Loading GRPO datasets ...")

train_dataset = load_from_disk(GRPO_TRAIN_DATA)
val_dataset   = load_from_disk(GRPO_VAL_DATA)

if accelerator.is_main_process:
    print(f"  Train: {len(train_dataset):,} | Val: {len(val_dataset):,}\n")

# ── W&B ───────────────────────────────────────────────────────────────────────
if accelerator.is_main_process:
    wandb.login()
    wandb.init(
        project=WANDB_PROJECT,
        name=f"unk10-grpo-r{LORA_RANK}-ng{NUM_GENERATIONS}-DDP{world_size}gpu",
        config={
            "base_model":      SFT_MERGED_MODEL,
            "experiment":      "unk10_ep2",
            "reward_fn":       "set_level_f1",
            "num_generations": NUM_GENERATIONS,
            "beta":            KL_BETA,
            "temperature":     TEMPERATURE,
            "lora_r":          LORA_RANK,
            "learning_rate":   LEARNING_RATE,
            "max_steps":       MAX_STEPS,
            "batch_size":      BATCH_SIZE,
            "grad_accum":      GRAD_ACCUM,
            "train_samples":   len(train_dataset),
            "val_samples":     len(val_dataset),
        },
        tags=["unk10", "grpo", "mimic-iv", "qwen3-32b"],
    )

# ── GRPO CONFIG ───────────────────────────────────────────────────────────────
grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_generations=NUM_GENERATIONS,
    max_prompt_length=MAX_PROMPT,
    max_completion_length=MAX_COMPLETION,
    temperature=TEMPERATURE,
    beta=KL_BETA,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=1.0,
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
    logging_steps=5,
    eval_strategy="no",
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    mask_truncated_completions=True,
    log_completions=True,
    num_completions_to_print=4,
    report_to="wandb" if accelerator.is_main_process else "none",
    seed=3407,
    remove_unused_columns=False,
)

# ── TRAINER ───────────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=base_model,
    reward_funcs=cpt_f1_reward,
    args=grpo_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    peft_config=grpo_lora_config,
)

for param in trainer.model.parameters():
    if param.dtype == torch.float32:
        param.data = param.data.to(torch.bfloat16)

# ── TRAIN ─────────────────────────────────────────────────────────────────────
if accelerator.is_main_process:
    print("Starting GRPO unk10 training ...\n")

start_time = time.time()
trainer.train()
elapsed = time.time() - start_time

# ── SAVE ──────────────────────────────────────────────────────────────────────
if accelerator.is_main_process:
    print(f"\n✓ GRPO unk10 complete! Duration: {elapsed/3600:.1f} hours")
    os.makedirs(SAVE_ADAPTER, exist_ok=True)
    unwrapped = accelerator.unwrap_model(trainer.model)
    unwrapped.save_pretrained(SAVE_ADAPTER)
    tokenizer.save_pretrained(SAVE_ADAPTER)
    print(f"✓ Adapter saved → {SAVE_ADAPTER}/")
    wandb.log({"train/duration_hours": elapsed / 3600})
    wandb.finish()
    print("\n" + "="*60)
    print("  GRPO unk10 ep2 COMPLETE")
    print(f"  Base     : {SFT_MERGED_MODEL}/")
    print(f"  Adapter  : {SAVE_ADAPTER}/")
    print("="*60)
