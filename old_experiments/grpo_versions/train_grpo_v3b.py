#!/usr/bin/env python3
"""
GRPO v3 — Stage 3 training on top of SFT v9 merged model.
Uses set-level F1 as reward to directly optimize the evaluation metric.

Base model: qwen3-32b-sft-v9-merged (CPT + SFT already baked in)
Adds a fresh GRPO LoRA on top.

Run:
    accelerate launch --num_processes 2 --mixed_precision bf16 train_grpo_v3b.py
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training, LoraConfig
from trl import GRPOTrainer, GRPOConfig

# ============================================================
# CONFIG
# ============================================================
SFT_MERGED_MODEL  = "qwen3-32b-sft-v9-merged"
TOKENIZER_SRC     = "qwen3-32b-mimic-cpt-200k-ep2"
GRPO_TRAIN_DATA   = "grpo_v9_train_dataset"
GRPO_VAL_DATA     = "grpo_v9_val_dataset"
OUTPUT_DIR        = "grpo_v3b_out"
SAVE_ADAPTER      = "qwen3-32b-grpo-v3b"
WANDB_PROJECT     = "SmartClaims-GRPO-v3b"

EPOCHS            = 1
MAX_STEPS         = 300   # cap at ~15 hrs (181 sec/step x 300)
BATCH_SIZE        = 1
GRAD_ACCUM        = 8
LEARNING_RATE     = 1e-5
LORA_RANK         = 16
MAX_SEQ_LENGTH    = 4096
MAX_COMPLETION    = 128
MAX_PROMPT        = MAX_SEQ_LENGTH - MAX_COMPLETION
NUM_GENERATIONS   = 4
TEMPERATURE       = 0.9
KL_BETA           = 0.04

# ============================================================
# ACCELERATOR
# ============================================================
accelerator = Accelerator()
local_rank  = accelerator.local_process_index
world_size  = accelerator.num_processes

if accelerator.is_main_process:
    print(f"\n{'='*60}")
    print(f"  GRPO v1 — DDP on {world_size} GPUs")
    print(f"  Base model       : {SFT_MERGED_MODEL}")
    print(f"  GRPO adapter out : {SAVE_ADAPTER}")
    print(f"  Num generations  : {NUM_GENERATIONS}")
    print(f"  KL beta          : {KL_BETA}")
    print(f"  Effective batch  : {BATCH_SIZE} x {GRAD_ACCUM} x {world_size}"
          f" = {BATCH_SIZE * GRAD_ACCUM * world_size}")
    print(f"{'='*60}\n")

# ============================================================
# TOKENIZER — must come from CPT adapter for Qwen3 chat template
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SRC, trust_remote_code=True)
tokenizer.padding_side = "left"   # left-pad for generation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# MODEL — fully merged SFT model as GRPO base
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

if accelerator.is_main_process:
    print(f"Loading SFT merged model from {SFT_MERGED_MODEL} ...")

base_model = AutoModelForCausalLM.from_pretrained(
    SFT_MERGED_MODEL,
    quantization_config=bnb_config,
    device_map={"": local_rank},
    dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(
    base_model, use_gradient_checkpointing=True
)
# Cast any fp32 params (e.g. lm_head) to bf16 AFTER prepare_model_for_kbit_training
for param in base_model.parameters():
    if param.dtype == torch.float32:
        param.data = param.data.to(torch.bfloat16)

if accelerator.is_main_process:
    alloc = torch.cuda.memory_allocated(local_rank) / 1e9
    total = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
    print(f"GPU {local_rank}: {alloc:.1f} GB / {total:.1f} GB"
          f" ({total - alloc:.1f} GB free)\n")

# ============================================================
# GRPO LoRA CONFIG
# ============================================================
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

# ============================================================
# REWARD FUNCTIONS
# ============================================================
def parse_cpt_output(text: str) -> tuple:
    """Returns (set_of_codes, is_valid_format)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text:
        return set(), False
    if text.strip().lower() == "unknown":
        return {"unknown"}, True
    parts = [p.strip() for p in text.split("|") if p.strip()]
    codes = {p for p in parts if re.match(r"^\d{5}$", p)}
    return codes, len(codes) > 0


def parse_ground_truth(gt_str: str) -> set:
    gt_str = gt_str.strip()
    if gt_str.lower() == "unknown":
        return {"unknown"}
    return {c.strip() for c in gt_str.split("|")
            if re.match(r"^\d{5}$", c.strip())}


def compute_set_f1(pred: set, gt: set) -> float:
    if not pred or not gt:
        return 0.0
    tp = len(pred & gt)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall    = tp / len(gt)
    return 2.0 * precision * recall / (precision + recall)


def cpt_f1_reward(prompts, completions, **kwargs) -> list:
    """
    GRPOTrainer reward function — set-level F1 between predicted and ground truth CPT codes.

    Rewards:
      1.0  = exact set match (or unknown/unknown)
      >0   = partial F1 credit
     -0.1  = non-empty but unparseable (no 5-digit codes, not 'unknown')
     -0.2  = empty output

    W&B key: rewards/cpt_f1_reward/mean  ← training F1 in real time
    """
    ground_truths = kwargs["ground_truth"]
    rewards = []

    for completion, gt_str in zip(completions, ground_truths):
        # Conversational completions: [{"role": "assistant", "content": "..."}]
        if isinstance(completion, list):
            text = completion[0]["content"]
        else:
            text = str(completion)

        pred_codes, is_valid = parse_cpt_output(text)
        gt_codes = parse_ground_truth(gt_str)

        if is_valid:
            reward = compute_set_f1(pred_codes, gt_codes)
        else:
            reward = -0.2 if not text.strip() else -0.1

        rewards.append(float(reward))

    return rewards

# ============================================================
# DATASET
# ============================================================
if accelerator.is_main_process:
    print("Loading GRPO datasets ...")

train_dataset = load_from_disk(GRPO_TRAIN_DATA)
val_dataset   = load_from_disk(GRPO_VAL_DATA)

if accelerator.is_main_process:
    print(f"  Train: {len(train_dataset):,} | Val: {len(val_dataset):,}\n")

# ============================================================
# W&B
# ============================================================
if accelerator.is_main_process:
    wandb.login()
    wandb.init(
        project=WANDB_PROJECT,
        name=f"qwen3-32b-grpo-v2-r{LORA_RANK}-ng{NUM_GENERATIONS}-DDP{world_size}gpu",
        config={
            "base_model":      SFT_MERGED_MODEL,
            "reward_fn":       "set_level_f1",
            "num_generations": NUM_GENERATIONS,
            "beta":            KL_BETA,
            "temperature":     TEMPERATURE,
            "lora_r":          LORA_RANK,
            "learning_rate":   LEARNING_RATE,
            "epochs":          EPOCHS,
            "batch_size":      BATCH_SIZE,
            "grad_accum":      GRAD_ACCUM,
            "effective_batch": BATCH_SIZE * GRAD_ACCUM * world_size,
            "max_prompt":      MAX_PROMPT,
            "max_completion":  MAX_COMPLETION,
            "train_samples":   len(train_dataset),
            "val_samples":     len(val_dataset),
            "transformers":    transformers.__version__,
        },
        tags=["stage3", "grpo-v2", "mimic-iv", "qwen3-32b", "cpt-codes", "ddp"],
    )

# ============================================================
# GRPO CONFIG
# ============================================================
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
    remove_unused_columns=False,   # CRITICAL: keeps ground_truth in reward kwargs
)

# ============================================================
# TRAINER
# ============================================================
trainer = GRPOTrainer(
    model=base_model,
    reward_funcs=cpt_f1_reward,
    args=grpo_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    peft_config=grpo_lora_config,
)

# ============================================================
# TRAIN
# ============================================================
if accelerator.is_main_process:
    print("Starting GRPO training ...\n")
    print("Monitor: W&B → rewards/cpt_f1_reward/mean (training F1)")
    print("         W&B → eval_reward (validation F1 every 50 steps)\n")

# Cast any fp32 params (lm_head etc.) to bf16 after all PEFT/trainer setup
for param in trainer.model.parameters():
    if param.dtype == torch.float32:
        param.data = param.data.to(torch.bfloat16)

start_time = time.time()
trainer.train()
elapsed = time.time() - start_time

# ============================================================
# SAVE GRPO ADAPTER (main process only)
# ============================================================
if accelerator.is_main_process:
    print(f"\n✓ GRPO training complete!")
    print(f"  Duration: {elapsed / 3600:.1f} hours")

    os.makedirs(SAVE_ADAPTER, exist_ok=True)
    unwrapped = accelerator.unwrap_model(trainer.model)
    unwrapped.save_pretrained(SAVE_ADAPTER)
    tokenizer.save_pretrained(SAVE_ADAPTER)
    print(f"✓ GRPO adapter saved → {SAVE_ADAPTER}/")

    wandb.log({"train/duration_hours": elapsed / 3600})
    wandb.finish()

    print("\n" + "=" * 60)
    print("  STAGE 2 GRPO v1 COMPLETE")
    print(f"  Base model   : {SFT_MERGED_MODEL}/")
    print(f"  GRPO adapter : {SAVE_ADAPTER}/")
    print("=" * 60)
