#!/usr/bin/env python3
"""
Evaluation script for SFT Qwen3-32B CPT code prediction.
Runs inference on unseen test reports and computes F1-macro.

Usage:
    python eval_sft_v6.py
"""

import os
import re
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
BASE_MODEL      = "Qwen/Qwen3-32B"
CPT_ADAPTER     = "qwen3-32b-mimic-cpt-200k-ep2"
SFT_CHECKPOINT  = "outputs_sft_qwen3_32b/checkpoint-270"

TEST_INPUTS_CSV = "mimic_radiology_sft_test_inputs.csv"   # 400 rows, 1 per admission
TEST_LABELS_CSV = "mimic_radiology_sft_test_labels.csv"   # multiple rows per admission

OUTPUT_JSON     = "eval_results.json"

MAX_NEW_TOKENS  = 256

SYSTEM_PROMPT = (
    "You are a medical coding assistant. "
    "Given a clinical note for a patient admission, identify all billable "
    "radiology procedures performed. Output only the CPT codes separated by "
    "pipe ( | ). If no billable radiology procedure can be determined, output "
    "unknown. Do not include add-on codes, explanations, or numbering."
)

# ============================================================
# LOAD MODEL
# ============================================================
print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(SFT_CHECKPOINT, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading base model ...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
base_model.config.use_cache = True

print(f"Loading CPT adapter (frozen) from {CPT_ADAPTER} ...")
model = PeftModel.from_pretrained(base_model, CPT_ADAPTER, is_trainable=False)

print(f"Loading SFT adapter from {SFT_CHECKPOINT} ...")
model.load_adapter(SFT_CHECKPOINT, adapter_name="sft")
model.set_adapter("sft")
model.eval()

# ============================================================
# LOAD DATA
# ============================================================
print("\nLoading test data ...")

# Inputs: already 1 row per admission (400 rows)
inputs_df = pd.read_csv(TEST_INPUTS_CSV, dtype={"hadm_id": str})

# Labels: one row per CPT code — aggregate per admission
labels_df = pd.read_csv(TEST_LABELS_CSV, dtype={"hadm_id": str, "cpt_codes": str})
gt_agg = (
    labels_df.groupby("hadm_id")["cpt_codes"]
    .apply(lambda x: sorted(set(x.str.strip())))
    .reset_index()
    .rename(columns={"cpt_codes": "gt_labels"})
)

# Merge
test_df = inputs_df.merge(gt_agg, on="hadm_id", how="inner")
print(f"  Test admissions: {len(test_df)}")
print(f"  Avg GT labels per admission: {gt_agg['gt_labels'].apply(len).mean():.2f}")

# ============================================================
# INFERENCE
# ============================================================
def build_prompt(report_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": report_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

def parse_output(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.strip().lower() == "unknown":
        return ["unknown"]
    codes = [c.strip() for c in text.split("|")
             if re.match(r"^\d{5}$", c.strip())]
    return sorted(set(codes))

predictions   = []
ground_truths = []
raw_outputs   = []

print("\nRunning inference ...")
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    prompt = build_prompt(row["reports"])
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    decoded   = tokenizer.decode(generated, skip_special_tokens=True)

    pred_labels = parse_output(decoded)
    predictions.append(pred_labels)
    ground_truths.append(row["gt_labels"])

    raw_outputs.append({
        "hadm_id":      row["hadm_id"],
        "predicted":    pred_labels,
        "ground_truth": row["gt_labels"],
        "raw_output":   decoded,
    })

# ============================================================
# METRICS
# ============================================================
print("\nComputing metrics ...")

all_labels = sorted(set(
    lbl for lst in ground_truths + predictions for lbl in lst
))

mlb    = MultiLabelBinarizer(classes=all_labels)
y_true = mlb.fit_transform(ground_truths)
y_pred = mlb.transform(predictions)

f1_macro  = f1_score(y_true, y_pred, average="macro",   zero_division=0)
f1_micro  = f1_score(y_true, y_pred, average="micro",   zero_division=0)
f1_sample = f1_score(y_true, y_pred, average="samples", zero_division=0)

print(f"\n{'='*50}")
print(f"  F1 Macro   : {f1_macro:.4f}")
print(f"  F1 Micro   : {f1_micro:.4f}")
print(f"  F1 Samples : {f1_sample:.4f}")
print(f"  Unique labels seen: {len(all_labels)}")
print(f"{'='*50}\n")

# Per-label report for top 20 most frequent codes
label_counts = labels_df["cpt_codes"].str.strip().value_counts()
top_labels   = label_counts.head(20).index.tolist()
top_indices  = [i for i, l in enumerate(all_labels) if l in top_labels]

if top_indices:
    print("Per-label F1 (top 20 most frequent codes):")
    report = classification_report(
        y_true[:, top_indices],
        y_pred[:, top_indices],
        target_names=[all_labels[i] for i in top_indices],
        zero_division=0,
    )
    print(report)

# ============================================================
# SAVE
# ============================================================
results = {
    "f1_macro":    round(f1_macro,  4),
    "f1_micro":    round(f1_micro,  4),
    "f1_samples":  round(f1_sample, 4),
    "n_test":      len(test_df),
    "n_labels":    len(all_labels),
    "predictions": raw_outputs,
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved → {OUTPUT_JSON}")
