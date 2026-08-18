#!/usr/bin/env python3
"""
Mixed eval: radiology test cases + unknown discharge notes → unk10 GRPO model.

Tests both:
  1. F1 on radiology cases (can the model predict correct CPT codes?)
  2. Accuracy on unknown cases (does it say 'unknown' when no radiology?)

Run:
    python eval_mixed_radiology_unknown.py
"""

import os, re, json
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

# ── CONFIG ────────────────────────────────────────────────────────────────────
N_RADIOLOGY   = 450   # radiology test samples
N_UNKNOWN     = 50    # unknown discharge notes

SFT_MERGED    = "unk10_sft_merged"
GRPO_ADAPTER  = "unk10_grpo_adapter"
TOKENIZER_SRC = "../qwen3-32b-mimic-cpt-200k-ep2"

TEST_INPUTS_CSV    = "../sft_test_inputs.csv"
TEST_LABELS_CSV    = "../sft_test_labels.csv"
HOLDOUT_NOTES_CSV  = "../cleaned_holdout_notes.csv"

MAX_NEW_TOKENS = 128
SEED           = 42

SYSTEM_PROMPT = (
    "You are a medical coding assistant. "
    "Given a clinical note for a patient admission, identify all billable "
    "radiology procedures performed. Output only the CPT codes separated by "
    "pipe ( | ). If no billable radiology procedure can be determined, output "
    "unknown. Do not include add-on codes, explanations, or numbering."
)

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SRC, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
print(f"Loading {SFT_MERGED} ...")
model = AutoModelForCausalLM.from_pretrained(
    SFT_MERGED, quantization_config=bnb_config, device_map="cuda:0",
    torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager",
)
print(f"Applying GRPO adapter: {GRPO_ADAPTER} ...")
model = PeftModel.from_pretrained(model, GRPO_ADAPTER)
model.config.use_cache = True
model.eval()
print("Model ready.\n")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading test data ...")
inputs_df = pd.read_csv(TEST_INPUTS_CSV, dtype={"hadm_id": str})
labels_df = pd.read_csv(TEST_LABELS_CSV, dtype={"hadm_id": str, "cpt_codes": str})
gt_agg = (
    labels_df.groupby("hadm_id")["cpt_codes"]
    .apply(lambda x: sorted(set(x.str.strip())))
    .reset_index().rename(columns={"cpt_codes": "gt_codes"})
)
test_df = inputs_df.merge(gt_agg, on="hadm_id", how="inner")
rad_df  = test_df.sample(n=min(N_RADIOLOGY, len(test_df)), random_state=SEED).reset_index(drop=True)
print(f"  Radiology samples: {len(rad_df)}")

print("Loading holdout discharge notes for unknown samples ...")
holdout_df = pd.read_csv(HOLDOUT_NOTES_CSV, dtype={"hadm_id": str})
unk_df = holdout_df.dropna(subset=["cleaned_text"]).sample(
    n=min(N_UNKNOWN, len(holdout_df)), random_state=SEED
).reset_index(drop=True)
print(f"  Unknown samples: {len(unk_df)}\n")


# ── HELPERS ───────────────────────────────────────────────────────────────────
def build_prompt(text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def parse_prediction(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.strip().lower() == "unknown":
        return ["unknown"]
    codes = [c.strip() for c in text.split("|") if re.match(r"^\d{5}$", c.strip())]
    return sorted(set(codes)) if codes else ["unknown"]


def run_inference(prompt):
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=4096,
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, temperature=None, top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)


# ── RADIOLOGY INFERENCE ───────────────────────────────────────────────────────
print("Running inference on radiology samples ...")
rad_predictions, rad_ground_truths, rad_raw = [], [], []
for _, row in tqdm(rad_df.iterrows(), total=len(rad_df), desc="Radiology"):
    decoded = run_inference(build_prompt(row["reports"]))
    pred    = parse_prediction(decoded)
    rad_predictions.append(pred)
    rad_ground_truths.append(row["gt_codes"])
    rad_raw.append({
        "type": "radiology", "hadm_id": row["hadm_id"],
        "predicted": pred, "ground_truth": row["gt_codes"], "raw_output": decoded,
    })

# ── UNKNOWN INFERENCE ─────────────────────────────────────────────────────────
print("\nRunning inference on unknown (discharge-only) samples ...")
unk_raw = []
n_correct_unknown = 0
for _, row in tqdm(unk_df.iterrows(), total=len(unk_df), desc="Unknown"):
    decoded = run_inference(build_prompt(row["cleaned_text"]))
    pred    = parse_prediction(decoded)
    correct = (pred == ["unknown"])
    if correct:
        n_correct_unknown += 1
    unk_raw.append({
        "type": "unknown", "hadm_id": row.get("hadm_id", ""),
        "predicted": pred, "ground_truth": ["unknown"],
        "raw_output": decoded, "correct_unknown": correct,
    })

# ── METRICS ───────────────────────────────────────────────────────────────────
print("\nComputing radiology F1 ...")
all_labels = sorted(set(l for lst in rad_ground_truths + rad_predictions for l in lst))
mlb    = MultiLabelBinarizer(classes=all_labels)
y_true = mlb.fit_transform(rad_ground_truths)
y_pred = mlb.transform(rad_predictions)

f1_samples = f1_score(y_true, y_pred, average="samples", zero_division=0)
f1_micro   = f1_score(y_true, y_pred, average="micro",   zero_division=0)
f1_macro   = f1_score(y_true, y_pred, average="macro",   zero_division=0)

unknown_accuracy = n_correct_unknown / len(unk_df) if unk_df is not None else 0.0

print(f"\n{'='*60}")
print(f"  Model          : unk10 GRPO")
print(f"  --- Radiology ({len(rad_df)} samples) ---")
print(f"  F1-samples     : {f1_samples:.4f}")
print(f"  F1-micro       : {f1_micro:.4f}")
print(f"  F1-macro       : {f1_macro:.4f}")
print(f"  --- Unknown ({len(unk_df)} samples) ---")
print(f"  Correct 'unknown' predictions : {n_correct_unknown}/{len(unk_df)} ({unknown_accuracy*100:.1f}%)")
print(f"{'='*60}\n")

# ── SAVE ──────────────────────────────────────────────────────────────────────
ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
out_json = f"eval_unk10grpo_mixed_radiology_unknown_results_{ts}.json"

results = {
    "model":             "unk10_grpo",
    "timestamp":         ts,
    "n_radiology":       len(rad_df),
    "n_unknown":         len(unk_df),
    "radiology_f1_samples": round(f1_samples, 4),
    "radiology_f1_micro":   round(f1_micro,   4),
    "radiology_f1_macro":   round(f1_macro,   4),
    "unknown_accuracy":     round(unknown_accuracy, 4),
    "unknown_correct":      n_correct_unknown,
    "predictions":          rad_raw + unk_raw,
}

with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved → {out_json}")
