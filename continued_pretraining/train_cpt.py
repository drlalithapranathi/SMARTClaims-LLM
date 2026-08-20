#!/usr/bin/env python3
"""
Continued Pretraining of Qwen3-32B on MIMIC-IV discharge summaries.
TRUE DATA PARALLELISM across 4x A6000 GPUs via DDP.

batch=4 per GPU keeps peak VRAM within 47 GB.
With 4 GPUs working simultaneously, wall-clock ≈ 50-70 hours.

Run:
    accelerate launch --num_processes 4 --mixed_precision bf16 train_cpt.py
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch.nn.functional as F

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME      = "Qwen/Qwen3-32B"
DATA_DIR        = "tokenized_chunks_200k"
OUTPUT_DIR      = "outputs_cpt_qwen3_32b"
SAVE_MODEL_NAME = "qwen3-32b-mimic-cpt-200k"

BATCH_SIZE      = 4    # per GPU — logit tensor (4×2048×151k fp32) ≈ 5 GB, fits fine
GRAD_ACCUM      = 4    # eff batch = 4 × 4 × 4 GPUs = 64
LEARNING_RATE   = 5e-5
LORA_RANK       = 16
WANDB_PROJECT   = "SmartClaims-CPT"

# ============================================================
# ACCELERATOR
# ============================================================
accelerator = Accelerator()
local_rank  = accelerator.local_process_index
world_size  = accelerator.num_processes

if accelerator.is_main_process:
    print(f"\n{'='*60}")
    print(f"  DDP training on {world_size} GPUs (batch {BATCH_SIZE}/GPU)")
    print(f"  Effective batch: {BATCH_SIZE} × {GRAD_ACCUM} × {world_size} = "
          f"{BATCH_SIZE * GRAD_ACCUM * world_size}")
    print(f"{'='*60}\n")

# ============================================================
# TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# MODEL — each DDP process loads full model onto its own GPU
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": local_rank},        # each process → its own GPU
    dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
model.config.use_cache = False

# ============================================================
# LoRA
# ============================================================
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora=True,
)
model = get_peft_model(model, lora_config)

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
    print(f"Dataset : {total_chunks:,} chunks")
    print(f"Eff batch: {BATCH_SIZE} × {GRAD_ACCUM} × {world_size} GPUs = {effective_batch}")
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
        name=f"qwen3-32b-cpt-r16-ep1-200k-DDP{world_size}gpu",
        config={
            "model":            MODEL_NAME,
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
        tags=["stage1", "cpt", "mimic-iv", "qwen3-32b", "ddp", "200k"],
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
# MEMORY ANALYSIS of the OOM at batch=4:
#
# During backward, PyTorch must hold BOTH:
#   • shift_logits gradient  (4×2047×151669 bf16 = 2.48 GB)
#   • full logits gradient   (4×2048×151669 bf16 = 2.49 GB)
# simultaneously for the SliceBackward node → 4.97 GB = 4.63 GiB.
# With only 3.28 GiB free → OOM.  No "del" trick can help because
# both tensors must coexist for the slice-backward to execute.
#
# FIX: custom autograd.Function that takes the raw 3D logits (no
# intermediate slice tensor), processes CE in (B, seq_chunk) tiles,
# and writes the gradient directly into a single pre-allocated
# (B, L, V) bf16 tensor.  Peak extra memory per step:
#   forward:  0.32 GB (one fp32 tile)
#   backward: 2.49 GB (grad tensor) + 0.32 GB (one fp32 tile)
# Total peak ≈ 44.12 + 2.49 + 0.32 = 46.93 GB < 47.4 GB  ✓
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
        # logits: (B, L, V) bf16 | labels: (B, L) int64
        B, L, V = logits.shape
        T = _ShiftedCE.TILE
        total   = torch.zeros(1, device=logits.device, dtype=torch.float32)
        n_valid = 0
        for b in range(B):
            for s in range(0, L - 1, T):
                end = min(s + T, L - 1)
                c   = logits[b, s:end, :].float()       # fp32 tile, ~0.16 GB
                cy  = labels[b, s + 1 : end + 1]
                total.add_(F.cross_entropy(c, cy, ignore_index=-100,
                                           reduction="sum"))
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

        # Single pre-allocated gradient tensor; last-token column stays 0
        g = torch.zeros_like(logits)        # (B, L, V) bf16, ~2.49 GB

        for b in range(B):
            for s in range(0, L - 1, T):
                end = min(s + T, L - 1)
                c   = logits[b, s:end, :].float()       # fp32 tile
                p   = torch.softmax(c, dim=-1)          # fp32 probs
                del c
                cy  = labels[b, s + 1 : end + 1]
                mask = cy != -100
                if mask.any():
                    p[mask, cy[mask]] -= 1.0            # subtract one-hot
                p[~mask] = 0.0
                g[b, s:end, :] = p.mul_(scale).to(logits.dtype)
                del p

        return g, None   # (d_logits, d_labels)


class BF16LossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        outputs = model(**inputs)           # no labels → no fp32 cast inside model
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
# TRAIN
# ============================================================
if accelerator.is_main_process:
    print("Starting DDP continued pretraining ...\n")

start_time = time.time()
trainer_stats = trainer.train()
elapsed = time.time() - start_time

# ============================================================
# POST-TRAINING (main process only)
# ============================================================
if accelerator.is_main_process:
    print(f"\n✓ Training complete!")
    print(f"  Duration : {elapsed/60:.1f} min  ({elapsed/3600:.1f} hours)")
    print(f"  Final loss: {trainer_stats.training_loss:.4f}")
    print(f"  Steps    : {trainer_stats.global_step}")

    used_mem = torch.cuda.max_memory_reserved(local_rank) / 1e9
    total_mem = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
    print(f"  Peak VRAM: {used_mem:.1f} / {total_mem:.1f} GB ({used_mem/total_mem*100:.1f}%)")

    # Save adapter
    os.makedirs(SAVE_MODEL_NAME, exist_ok=True)
    model.save_pretrained(SAVE_MODEL_NAME)
    tokenizer.save_pretrained(SAVE_MODEL_NAME)
    adapter_mb = sum(
        os.path.getsize(os.path.join(SAVE_MODEL_NAME, f))
        for f in os.listdir(SAVE_MODEL_NAME)
        if os.path.isfile(os.path.join(SAVE_MODEL_NAME, f))
    ) / 1e6
    print(f"\n✓ Adapter saved → {SAVE_MODEL_NAME}/  ({adapter_mb:.1f} MB)")

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
               title=f"CPT Loss — Qwen3-32B MIMIC-IV 200K [DDP {world_size}GPU]")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(f"{OUTPUT_DIR}/training_loss.png", dpi=150)
        print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f} "
              f"({(losses[0]-losses[-1])/losses[0]*100:.1f}% drop)")
    except Exception as e:
        print(f"  (Plot skipped: {e})")

    wandb.log({
        "train/final_loss": trainer_stats.training_loss,
        "train/duration_hours": elapsed / 3600,
    })
    wandb.finish()

    print("\n" + "="*60)
    print("  STAGE 1 CPT COMPLETE")
    print("="*60)
    print(f"  Adapter : {SAVE_MODEL_NAME}/")
    print(f"  NEXT → Stage 2: SFT on radiology reports")
    print("="*60)
