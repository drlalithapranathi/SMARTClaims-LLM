#!/usr/bin/env python3
"""
Continued Pretraining of Qwen3-32B on MIMIC-IV discharge summaries — EPOCH 2.
Loads the saved LoRA adapter from epoch 1 (qwen3-32b-mimic-cpt-200k) and
trains for one more full epoch with a lower learning rate.

NO checkpoint resume — avoids the paged_adamw_8bit RNG/optimizer-state crash.
Starts cleanly from the epoch-1 adapter weights.

Run:
    accelerate launch --num_processes 4 --mixed_precision bf16 train_cpt_ep2.py
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
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import PeftModel, prepare_model_for_kbit_training
import torch.nn.functional as F

# ============================================================
# CONFIG
# ============================================================
BASE_MODEL_NAME = "Qwen/Qwen3-32B"           # original base weights
EP1_ADAPTER     = "qwen3-32b-mimic-cpt-200k" # your saved epoch-1 adapter folder
DATA_DIR        = "tokenized_chunks_200k"
OUTPUT_DIR      = "outputs_cpt_qwen3_32b_ep2" # NEW dir — keeps ep1 checkpoints safe
SAVE_MODEL_NAME = "qwen3-32b-mimic-cpt-200k-ep2"

BATCH_SIZE    = 4
GRAD_ACCUM    = 4
LEARNING_RATE = 2e-5   # lower than ep1 (5e-5) — model is already partially trained
LORA_RANK     = 16     # must match ep1
WANDB_PROJECT = "SmartClaims-CPT"

# ============================================================
# ACCELERATOR
# ============================================================
accelerator = Accelerator()
local_rank  = accelerator.local_process_index
world_size  = accelerator.num_processes

if accelerator.is_main_process:
    print(f"\n{'='*60}")
    print(f"  EPOCH 2 — DDP training on {world_size} GPUs (batch {BATCH_SIZE}/GPU)")
    print(f"  Loading adapter from : {EP1_ADAPTER}")
    print(f"  Effective batch: {BATCH_SIZE} x {GRAD_ACCUM} x {world_size} = "
          f"{BATCH_SIZE * GRAD_ACCUM * world_size}")
    print(f"{'='*60}\n")

# ============================================================
# SAFETY CHECK — refuse to overwrite ep1 adapter
# ============================================================
if accelerator.is_main_process:
    if SAVE_MODEL_NAME == EP1_ADAPTER:
        raise ValueError(
            f"SAVE_MODEL_NAME == EP1_ADAPTER ({EP1_ADAPTER}). "
            "Set a different SAVE_MODEL_NAME to avoid overwriting epoch-1 weights."
        )
    if os.path.exists(SAVE_MODEL_NAME):
        raise FileExistsError(
            f"{SAVE_MODEL_NAME} already exists. "
            "Delete it or choose a new SAVE_MODEL_NAME."
        )

# ============================================================
# TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# MODEL — load base then apply ep1 LoRA adapter
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
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": local_rank},   # each DDP process → its own GPU
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

if accelerator.is_main_process:
    print(f"Loading epoch-1 LoRA adapter from {EP1_ADAPTER} ...")

# is_trainable=True keeps all adapter params in training mode
model = PeftModel.from_pretrained(base_model, EP1_ADAPTER, is_trainable=True)

if accelerator.is_main_process:
    model.print_trainable_parameters()
    alloc = torch.cuda.memory_allocated(local_rank) / 1e9
    total = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
    print(f"\nGPU {local_rank}: {alloc:.1f} GB used / {total:.1f} GB total "
          f"({total-alloc:.1f} GB free after load)\n")

# ============================================================
# DATASET
# ============================================================
dataset = load_from_disk(DATA_DIR)
total_chunks    = len(dataset)
effective_batch = BATCH_SIZE * GRAD_ACCUM * world_size
max_steps       = total_chunks // effective_batch

if accelerator.is_main_process:
    print(f"Dataset  : {total_chunks:,} chunks")
    print(f"Eff batch: {BATCH_SIZE} x {GRAD_ACCUM} x {world_size} GPUs = {effective_batch}")
    print(f"Steps (1 epoch): {max_steps:,}\n")

# ============================================================
# DATA COLLATOR
# ============================================================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)

# ============================================================
# W&B (main process only)
# ============================================================
if accelerator.is_main_process:
    wandb.login()
    wandb.init(
        project=WANDB_PROJECT,
        name=f"qwen3-32b-cpt-r16-ep2-200k-DDP{world_size}gpu",
        config={
            "model":            BASE_MODEL_NAME,
            "epoch":            2,
            "ep1_adapter":      EP1_ADAPTER,
            "attention":        "flash_attention_2",
            "lora_r":           LORA_RANK,
            "learning_rate":    LEARNING_RATE,
            "per_device_batch": BATCH_SIZE,
            "grad_accum":       GRAD_ACCUM,
            "effective_batch":  effective_batch,
            "total_chunks":     total_chunks,
            "max_steps":        max_steps,
            "num_gpus":         world_size,
            "parallelism":      f"DDP ({world_size} GPUs)",
            "optimizer":        "paged_adamw_8bit",
            "quantization":     "NF4 + double quant",
            "gpu_type":         torch.cuda.get_device_name(local_rank),
            "transformers":     transformers.__version__,
        },
        tags=["stage1", "cpt", "mimic-iv", "qwen3-32b", "ddp", "200k", "ep2"],
    )

# ============================================================
# TRAINING ARGS
# ============================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    max_steps=max_steps,

    warmup_ratio=0.05,
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

    dataloader_num_workers=4,
    dataloader_pin_memory=False,

    logging_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,

    report_to="wandb" if accelerator.is_main_process else "none",
    seed=3407,
    remove_unused_columns=False,
)

# ============================================================
# CUSTOM LOSS — same tiled ShiftedCE to avoid OOM on backward
# (identical to ep1 — do not remove)
# ============================================================
class _ShiftedCE(torch.autograd.Function):
    """
    Shifted causal-LM cross-entropy with a tiled, fp32-stable kernel.

    Takes full 3D logits (B, L, V) and labels (B, L).
    Loss = mean CE over logits[:, :-1, :] vs labels[:, 1:].
    Gradient is returned for the full (B, L, V) shape (zeros at last
    token position), so there is never a separate slice-gradient tensor.
    """
    TILE = 256   # seq-rows per tile; ~0.16 GB fp32 at vocab=151 669

    @staticmethod
    def forward(ctx, logits, labels):
        B, L, V = logits.shape
        T = _ShiftedCE.TILE
        total   = torch.zeros(1, device=logits.device, dtype=torch.float32)
        n_valid = 0
        for b in range(B):
            for s in range(0, L - 1, T):
                end = min(s + T, L - 1)
                c   = logits[b, s:end, :].float()
                cy  = labels[b, s + 1 : end + 1]
                total.add_(F.cross_entropy(c, cy, ignore_index=-100, reduction="sum"))
                n_valid += (cy != -100).sum().item()
                del c
        n_valid = max(n_valid, 1)
        ctx.save_for_backward(logits, labels)
        ctx.n_valid = n_valid
        return (total / n_valid).squeeze()

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels = ctx.saved_tensors
        B, L, V = logits.shape
        T     = _ShiftedCE.TILE
        scale = grad_output.item() / ctx.n_valid

        g = torch.zeros_like(logits)   # (B, L, V) bf16 — single pre-allocated grad

        for b in range(B):
            for s in range(0, L - 1, T):
                end = min(s + T, L - 1)
                c   = logits[b, s:end, :].float()
                p   = torch.softmax(c, dim=-1)
                del c
                cy  = labels[b, s + 1 : end + 1]
                mask = cy != -100
                if mask.any():
                    p[mask, cy[mask]] -= 1.0
                p[~mask] = 0.0
                g[b, s:end, :] = p.mul_(scale).to(logits.dtype)
                del p

        return g, None


class BF16LossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        outputs = model(**inputs)
        if labels is not None:
            loss = _ShiftedCE.apply(outputs.logits, labels)
        else:
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


trainer = BF16LossTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# ============================================================
# TRAIN — no resume_from_checkpoint (avoids paged_adamw_8bit crash)
# ============================================================
if accelerator.is_main_process:
    print("Starting epoch 2 DDP continued pretraining ...\n")

start_time = time.time()
trainer_stats = trainer.train()   # fresh start from ep1 adapter weights
elapsed = time.time() - start_time

# ============================================================
# POST-TRAINING (main process only)
# ============================================================
if accelerator.is_main_process:
    print(f"\n✓ Training complete!")
    print(f"  Duration : {elapsed/60:.1f} min  ({elapsed/3600:.1f} hours)")
    print(f"  Final loss: {trainer_stats.training_loss:.4f}")
    print(f"  Steps    : {trainer_stats.global_step}")

    used_mem  = torch.cuda.max_memory_reserved(local_rank) / 1e9
    total_mem = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
    print(f"  Peak VRAM: {used_mem:.1f} / {total_mem:.1f} GB ({used_mem/total_mem*100:.1f}%)")

    # Save epoch-2 adapter — guard against accidental overwrite
    if os.path.exists(SAVE_MODEL_NAME):
        raise FileExistsError(
            f"{SAVE_MODEL_NAME} appeared during training. Aborting save to prevent overwrite."
        )
    os.makedirs(SAVE_MODEL_NAME, exist_ok=False)
    model.save_pretrained(SAVE_MODEL_NAME)
    tokenizer.save_pretrained(SAVE_MODEL_NAME)

    adapter_mb = sum(
        os.path.getsize(os.path.join(SAVE_MODEL_NAME, f))
        for f in os.listdir(SAVE_MODEL_NAME)
        if os.path.isfile(os.path.join(SAVE_MODEL_NAME, f))
    ) / 1e6
    print(f"\n✓ Epoch-2 adapter saved → {SAVE_MODEL_NAME}/  ({adapter_mb:.1f} MB)")
    print(f"  Epoch-1 adapter untouched → {EP1_ADAPTER}/")

    # Loss plot
    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        log_history = trainer.state.log_history
        steps  = [e["step"] for e in log_history if "loss" in e]
        losses = [e["loss"] for e in log_history if "loss" in e]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, losses, color="#00B4D8", lw=1.5, alpha=0.7)
        if len(losses) > 10:
            w = max(5, len(losses) // 20)
            ax.plot(steps, pd.Series(losses).rolling(w, center=True).mean(),
                    color="#EF476F", lw=2, label=f"Smoothed (w={w})")
            ax.legend()
        ax.set(xlabel="Step", ylabel="Loss",
               title=f"CPT Loss Ep2 — Qwen3-32B MIMIC-IV 200K [DDP {world_size}GPU]")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(f"{OUTPUT_DIR}/training_loss_ep2.png", dpi=150)

        if losses:
            print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f} "
                  f"({(losses[0]-losses[-1])/losses[0]*100:.1f}% drop)")
    except Exception as e:
        print(f"  (Plot skipped: {e})")

    wandb.log({
        "train/final_loss":      trainer_stats.training_loss,
        "train/duration_hours":  elapsed / 3600,
    })
    wandb.finish()

    print("\n" + "="*60)
    print("  STAGE 1 CPT EPOCH 2 COMPLETE")
    print("="*60)
    print(f"  Ep1 adapter : {EP1_ADAPTER}/")
    print(f"  Ep2 adapter : {SAVE_MODEL_NAME}/")
    print(f"  NEXT → Stage 2: SFT on radiology reports")
    print("="*60)
    