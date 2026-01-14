"""
Ablation Study: Train WITHOUT RadLex context
Compare with RadLex model to prove GraphRAG helps
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
import torch
import pandas as pd
import json
from pathlib import Path
from datasets import Dataset
import wandb

# Config
DATA_DIR = Path("YOUR_NO_RADLEX_DATA_DIR")  # Data without RadLex context
OUTPUT_DIR = Path("YOUR_OUTPUT_DIR")
MODEL_NAME = "unsloth/medgemma-27b-text-it"
MAX_SEQ_LENGTH = 4096
LORA_R = 16
LORA_ALPHA = 32
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 1e-5
NUM_EPOCHS = 1
WARMUP_STEPS = 50
TRAIN_SAMPLES = 1000
VAL_SAMPLES = 250
WANDB_PROJECT = "YOUR_WANDB_PROJECT"
WANDB_RUN_NAME = "medgemma-27b-NO-RADLEX-baseline"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load datasets (WITHOUT RadLex)
train_df = pd.read_json(DATA_DIR / "train_dataset.jsonl", lines=True)
val_df = pd.read_json(DATA_DIR / "val_dataset.jsonl", lines=True)
train_dataset = Dataset.from_pandas(train_df[['text']]).select(range(TRAIN_SAMPLES))
val_dataset = Dataset.from_pandas(val_df[['text']]).select(range(VAL_SAMPLES))

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# Configure LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Training args
training_args = UnslothTrainingArguments(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    report_to="wandb",
    run_name=WANDB_RUN_NAME,
    seed=42,
)

wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)

# Train
trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
)

trainer_stats = trainer.train()

# Save
lora_dir = OUTPUT_DIR / "lora_model"
lora_dir.mkdir(exist_ok=True)
model.save_pretrained(str(lora_dir))
tokenizer.save_pretrained(str(lora_dir))

with open(OUTPUT_DIR / "training_info.json", 'w') as f:
    json.dump({"model": MODEL_NAME, "train_samples": TRAIN_SAMPLES, "loss": float(trainer_stats.training_loss), "type": "NO-RADLEX-BASELINE"}, f)

wandb.finish()
