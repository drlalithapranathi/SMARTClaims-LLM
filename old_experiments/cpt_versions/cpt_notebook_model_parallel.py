#!/usr/bin/env python
# coding: utf-8

# # Qwen3-32B Continued Pretraining — OPTIMIZED
# 
# **Key optimizations over original:**
# 1. `flash_attention_2` instead of `eager` (~2-3x speedup)
# 2. `packing=True` for CPT (eliminates padding waste)
# 3. `dataloader_num_workers=0` (fixes streaming shard warning)
# 4. `paged_adamw_8bit` optimizer (more memory-efficient than `adamw_8bit`)
# 5. Larger `per_device_train_batch_size` to improve GPU utilization
# 6. Fixed wandb config to match actual hyperparameters
# 
# **Note on multi-GPU:** `device_map="auto"` with QLoRA uses pipeline/model parallelism (layers split across GPUs). This means only 1 GPU computes at a time. For true data parallelism you'd need `accelerate launch` with FSDP/DeepSpeed — but that requires a script, not a notebook. The config below is optimized for notebook-based model-parallel training across 3 GPUs.

import os




# ========== CELL 2: VERIFY VERSIONS ==========
import torch, transformers, trl, peft
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'Transformers: {transformers.__version__}')
print(f'TRL: {trl.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f'  GPU {i}: {name} ({mem:.1f} GB)')

# Check Flash Attention 2
try:
    import flash_attn
    print(f'Flash Attention: {flash_attn.__version__} ✓')
except ImportError:
    print('⚠️  Flash Attention not installed! Run: pip install flash-attn --no-build-isolation')

from packaging import version
assert version.parse(transformers.__version__) >= version.parse("4.51.0"), \
    f'FATAL: Qwen3 needs transformers>=4.51.0, got {transformers.__version__}'
print('\n✓ All versions correct')


# ========== CELL 3: LOAD MODEL ==========
# KEY CHANGE: flash_attention_2 instead of eager (~2-3x speedup)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

MODEL_NAME = "Qwen/Qwen3-32B"
MAX_SEQ_LENGTH = 2048

# 4-bit quantization (NF4 + double quant)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",   # <<< KEY CHANGE: was "eager"
)
model.config.use_cache = False

print(f'\n✓ Qwen3-32B loaded with Flash Attention 2')
print(f'  Total params: {sum(p.numel() for p in model.parameters()):,}')
print(f'  Layers: {model.config.num_hidden_layers}')
print(f'  Vocab: {len(tokenizer):,}')
print(f'  EOS: {repr(tokenizer.eos_token)} (id={tokenizer.eos_token_id})')
print(f'  PAD: {repr(tokenizer.pad_token)} (id={tokenizer.pad_token_id})')
print(f'\n  GPU memory after load:')
for i in range(torch.cuda.device_count()):
    a = torch.cuda.memory_allocated(i) / 1e9
    if a > 0.01:
        t = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f'    GPU {i}: {a:.1f}GB / {t:.1f}GB ({t-a:.1f}GB free)')


# ========== CELL 4: ADD LoRA ==========
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora=True,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ========== CELL 5: LOAD DATA (STREAMING) ==========
from datasets import IterableDataset

DATA_PATH = "cleaned_discharge_notes_200k_fixed.csv"
TEXT_COLUMN = "cleaned_text"
EOS_TOKEN = tokenizer.eos_token
MAX_LEN = 2048

def chunk_generator():
    """Stream CSV row by row -> tokenize -> chunk -> yield text."""
    import csv
    with open(DATA_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get(TEXT_COLUMN)
            if not text or str(text).strip() == "" or text == "nan":
                continue
            text = str(text) + EOS_TOKEN
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            for j in range(0, len(tokens), MAX_LEN):
                chunk = tokens[j:j + MAX_LEN]
                if len(chunk) > 128:
                    yield {"text": tokenizer.decode(chunk)}

dataset = IterableDataset.from_generator(chunk_generator)
dataset = dataset.shuffle(buffer_size=10_000, seed=42)

# Estimate chunk count
import pandas as pd
num_notes = len(pd.read_csv(DATA_PATH, usecols=[TEXT_COLUMN]).dropna())
avg_chunks_per_note = 2
total_chunks = num_notes * avg_chunks_per_note
del pd

print(f"✓ Streaming dataset ready")
print(f"  ~{num_notes:,} notes × ~{avg_chunks_per_note} chunks ≈ {total_chunks:,} estimated chunks")


# ========== CELL 6: TRAINER (OPTIMIZED) ==========
from trl import SFTTrainer, SFTConfig

# With device_map="auto", this is MODEL parallelism (not data parallel).
# The trainer sees 1 logical device, so don't divide steps by NUM_GPUS.
BATCH_SIZE = 8           # <<< Increased from 4 (FA2 + packing saves VRAM)
GRAD_ACCUM = 4
EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM  # = 32

max_steps = total_chunks // EFFECTIVE_BATCH  # No GPU division for model parallelism
print(f"Effective batch size: {EFFECTIVE_BATCH}")
print(f"Max steps (1 epoch): {max_steps:,}")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=max_steps,
        warmup_ratio=0.05,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        logging_steps=50,
        optim="paged_adamw_8bit",       # <<< Changed: paged variant is more memory-efficient
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=2,
        seed=3407,
        report_to="wandb",
        output_dir="outputs_cpt_qwen3_32b",
        max_seq_length=MAX_SEQ_LENGTH,     # <<< Fixed: was max_length (wrong param name)
        packing=True,                      # <<< KEY CHANGE: pack sequences to eliminate padding waste
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,          # <<< Fixed: was 2, caused warning with 1 shard
        dataloader_pin_memory=True,
    ),
    processing_class=tokenizer,
)

print(f"✓ Trainer ready — {max_steps:,} steps")


# ========== CELL 7: WANDB ==========
import wandb

wandb.login()
wandb.init(
    project="SmartClaims-CPT",
    name="qwen3-32b-cpt-r16-ep1-200k-OPTIMIZED",
    config={
        "model": "Qwen/Qwen3-32B",
        "params": "32.8B",
        "layers": 64,
        "attention": "flash_attention_2",
        "lora_r": 16,
        "lora_alpha": 16,
        "learning_rate": 5e-5,
        "epochs": 1,
        "per_device_batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "effective_batch": EFFECTIVE_BATCH,
        "max_seq_length": MAX_SEQ_LENGTH,
        "packing": True,
        "dataset_size": total_chunks,
        "method": "QLoRA 4bit NF4 + rsLoRA + FA2 + packing + CPT (model parallel)",
        "num_gpus": torch.cuda.device_count(),
        "gpu_type": torch.cuda.get_device_name(0),
        "parallelism": "model_parallel (device_map=auto)",
        "optimizer": "paged_adamw_8bit",
        "quantization": "NF4 + double quant",
        "transformers_version": transformers.__version__,
    },
    tags=["stage1", "cpt", "mimic-iv", "qwen3-32b", "hf-peft", "200k", "optimized"],
)
print('✓ W&B initialized')


# ========== CELL 8: MEMORY CHECK ==========
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU 0 = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved pre-training.")


# ========== CELL 9: TRAIN ==========
import time

print('🚀 Starting continued pretraining on Qwen3-32B (OPTIMIZED)...\n')
start_time = time.time()
trainer_stats = trainer.train()
elapsed = time.time() - start_time

print(f'\n✓ Training complete!')
print(f'  Duration: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)')
print(f'  Final loss: {trainer_stats.training_loss:.4f}')
print(f'  Steps: {trainer_stats.global_step}')


# ========== CELL 10: MEMORY STATS ==========
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# ========== CELL 11: PLOT LOSS ==========
import matplotlib.pyplot as plt
import pandas as pd
import os

log_history = trainer.state.log_history
steps = [e['step'] for e in log_history if 'loss' in e]
losses = [e['loss'] for e in log_history if 'loss' in e]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(steps, losses, color='#00B4D8', lw=1.5, alpha=0.7)
if len(losses) > 10:
    w = max(5, len(losses) // 20)
    ax.plot(steps, pd.Series(losses).rolling(w, center=True).mean(),
            color='#EF476F', lw=2, label=f'Smoothed (w={w})')
    ax.legend()
ax.set(xlabel='Step', ylabel='Loss',
       title='CPT Loss — Qwen3-32B on MIMIC-IV Discharge Summaries (200K notes) [OPTIMIZED]')
ax.grid(alpha=0.3)
plt.tight_layout()
os.makedirs('outputs_cpt_qwen3_32b', exist_ok=True)
plt.savefig('outputs_cpt_qwen3_32b/training_loss.png', dpi=150)
plt.show()
print(f'First: {losses[0]:.4f} → Final: {losses[-1]:.4f} ({(losses[0]-losses[-1])/losses[0]*100:.1f}% drop)')


# ========== CELL 12: INFERENCE ==========
model.config.use_cache = True
model.eval()

test_prompts = [
    "Brief Hospital Course:\nThe patient is a 65-year-old male with a history of",
    "Discharge Medications:\n1.",
    "Discharge Disposition:",
]

print('=== AFTER CONTINUED PRETRAINING ===\n')
for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f'PROMPT: {prompt}')
    print(f'OUTPUT: {generated[:500]}')
    print('-' * 80)


# ========== CELL 13: PERPLEXITY ==========
import math
import pandas as pd

df = pd.read_csv(DATA_PATH)
eval_texts = df[TEXT_COLUMN].dropna().tail(100).tolist()
total_loss = 0
total_tokens = 0

model.eval()
model.config.use_cache = False
with torch.no_grad():
    for text in eval_texts:
        inputs = tokenizer(str(text), return_tensors='pt', truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        inputs['labels'] = inputs['input_ids'].clone()
        outputs = model(**inputs)
        total_loss += outputs.loss.item() * inputs['input_ids'].shape[1]
        total_tokens += inputs['input_ids'].shape[1]

avg_loss = total_loss / total_tokens
perplexity = math.exp(avg_loss)
print(f'Perplexity: {perplexity:.2f} (loss: {avg_loss:.4f})')
wandb.log({"eval/perplexity": perplexity, "eval/avg_loss": avg_loss})


# ========== CELL 14: SAVE ==========
import os
SAVE_MODEL_NAME = "qwen3-32b-mimic-cpt-200k"

model.save_pretrained(SAVE_MODEL_NAME)
tokenizer.save_pretrained(SAVE_MODEL_NAME)

adapter_size = sum(
    os.path.getsize(os.path.join(SAVE_MODEL_NAME, f))
    for f in os.listdir(SAVE_MODEL_NAME)
    if os.path.isfile(os.path.join(SAVE_MODEL_NAME, f))
) / 1e6
print(f'✓ Saved: {SAVE_MODEL_NAME}/ ({adapter_size:.1f} MB)')


# ========== CELL 15: SUMMARY ==========
print('=' * 60)
print('  STAGE 1 CONTINUED PRETRAINING — COMPLETE (OPTIMIZED)')
print('=' * 60)
print(f'''
  Model:          Qwen3-32B (32.8B params, 64 layers, GQA 64Q/8KV)
  Attention:       Flash Attention 2
  Context:        {MAX_SEQ_LENGTH} / 32,768 native
  Method:         QLoRA (4-bit NF4 double quant) + rsLoRA (rank 16) + packing
  Optimizer:      paged_adamw_8bit
  GPUs:           {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)} (model parallel)
  Data:           ~{total_chunks:,} chunks from {num_notes:,} MIMIC-IV discharge summaries
  Batch:          {EFFECTIVE_BATCH} effective ({BATCH_SIZE} x {GRAD_ACCUM} grad accum)
  Epochs:         1
  Training loss:  {trainer_stats.training_loss:.4f}
  Perplexity:     {perplexity:.2f}
  Duration:       {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)
  Adapter:        {SAVE_MODEL_NAME}/
  W&B:            {wandb.run.get_url()}
  
  NEXT → Stage 2: SFT on radiology reports for CPT code extraction
''')

wandb.finish()
print('✓ Done')

