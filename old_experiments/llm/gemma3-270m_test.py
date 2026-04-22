import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
import wandb
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(os.getenv("DATA_DIR"))
RADIOLOGY = DATA_DIR / os.getenv("RADIOLOGY_CSV")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR"))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MODEL_NAME = "google/gemma-3-270m"
HF_TOKEN = os.getenv("HF_TOKEN")

TRAIN_START_IDX = 0
TRAIN_END_IDX = 2000

BATCH_SIZE = 4
GRADIENT_ACCUM = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
MAX_LENGTH = 2048

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

wandb.init(
    project="gemma3-270m-radiology",
    name="gemma3-270m-2000-reports",
    config={"model": MODEL_NAME, "indices": f"{TRAIN_START_IDX}-{TRAIN_END_IDX}"}
)

radiology_df = pd.read_csv(RADIOLOGY, usecols=["note_id", "subject_id", "text"])
radiology_df = radiology_df[radiology_df['text'].notna()]
radiology_df = radiology_df[radiology_df['text'].str.len() > 200]

train_reports = radiology_df.iloc[TRAIN_START_IDX:TRAIN_END_IDX].copy()

def extract_impression(text):
    if "impression:" in text.lower():
        idx = text.lower().find("impression:")
        return text[idx:idx+500].strip()
    return "Report reviewed. No significant abnormalities."

examples = []
for _, row in train_reports.iterrows():
    text = row['text'][:4000]
    if len(text) < 200:
        continue

    impression = extract_impression(text)
    prompt = f"""Analyze this radiology report and provide key findings:

{text}

Key findings:"""

    examples.append({'text': f"{prompt}\n\n{impression}"})

dataset = Dataset.from_pandas(pd.DataFrame(examples))

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    trust_remote_code=True
)
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

def formatting_func(example):
    return example["text"]

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUM,
    learning_rate=LEARNING_RATE,
    warmup_steps=10,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=torch.cuda.is_bf16_supported(),
    optim="paged_adamw_8bit",
    report_to="wandb",
    save_total_limit=2,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_func,
)

result = trainer.train()

model.save_pretrained(str(OUTPUT_DIR / "lora_adapters"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "lora_adapters"))

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN
)
merged_model = PeftModel.from_pretrained(base_model, str(OUTPUT_DIR / "lora_adapters"))
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained(str(OUTPUT_DIR / "full_model"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "full_model"))

wandb.finish()