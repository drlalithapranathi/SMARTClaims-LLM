#!/usr/bin/env python3
"""
Unified evaluation for SFT v9 and GRPO v3b models.
Computes F1-macro, F1-micro, F1-samples, and per-code F1.

Usage:
    # Evaluate SFT model (merged, no GRPO adapter)
    python eval_sft_grpo_v9.py --mode sft

    # Evaluate GRPO model (merged + GRPO adapter)
    python eval_sft_grpo_v9.py --mode grpo
"""

import os
import re
import json
import argparse
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report

# ============================================================
# CONFIG
# ============================================================
SFT_MERGED_MODEL = "qwen3-32b-sft-v9-merged"
GRPO_ADAPTER     = "qwen3-32b-grpo-v3b"
TOKENIZER_SRC    = "qwen3-32b-mimic-cpt-200k-ep2"

TEST_INPUTS_CSV  = "sft_test_inputs.csv"
TEST_LABELS_CSV  = "sft_test_labels.csv"

MAX_NEW_TOKENS   = 128

SYSTEM_PROMPT = (
    "You are a medical coding assistant. "
    "Given a clinical note for a patient admission, identify all billable "
    "radiology procedures performed. Output only the CPT codes separated by "
    "pipe ( | ). If no billable radiology procedure can be determined, output "
    "unknown. Do not include add-on codes, explanations, or numbering."
)

# ============================================================
# ARGS
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["sft", "grpo"], default="sft",
                    help="sft = merged SFT model only | grpo = merged SFT + GRPO adapter")
parser.add_argument("--n-eval", type=int, default=None,
                    help="Cap number of test samples (default: all). Use 300 for quick eval.")
parser.add_argument("--batch-size", type=int, default=1,
                    help="Inference batch size (default: 1)")
args = parser.parse_args()

# ============================================================
# LOAD MODEL
# ============================================================
print(f"\nMode: {args.mode}")
print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SRC, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading {SFT_MERGED_MODEL} ...")
model = AutoModelForCausalLM.from_pretrained(
    SFT_MERGED_MODEL,
    quantization_config=bnb_config,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
model.config.use_cache = True

if args.mode == "grpo":
    print(f"Loading GRPO adapter from {GRPO_ADAPTER} ...")
    model = PeftModel.from_pretrained(model, GRPO_ADAPTER)

model.eval()
print("Model loaded.\n")

# ============================================================
# LOAD TEST DATA
# ============================================================
print("Loading test data ...")
inputs_df  = pd.read_csv(TEST_INPUTS_CSV, dtype={"hadm_id": str})
labels_df  = pd.read_csv(TEST_LABELS_CSV, dtype={"hadm_id": str, "cpt_codes": str})

# Aggregate ground truth labels per admission
gt_agg = (
    labels_df.groupby("hadm_id")["cpt_codes"]
    .apply(lambda x: sorted(set(x.str.strip())))
    .reset_index()
    .rename(columns={"cpt_codes": "gt_codes"})
)

test_df = inputs_df.merge(gt_agg, on="hadm_id", how="inner")
if args.n_eval is not None:
    test_df = test_df.sample(n=min(args.n_eval, len(test_df)), random_state=42).reset_index(drop=True)
    print(f"  [--n-eval] Capped to {len(test_df)} samples")
print(f"  Test admissions : {len(test_df)}")
print(f"  Avg GT codes/admission: {gt_agg['gt_codes'].apply(len).mean():.2f}\n")

# ============================================================
# INFERENCE HELPERS
# ============================================================
def build_prompt(report_text: str) -> str:
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


def parse_prediction(text: str) -> list:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.strip().lower() == "unknown":
        return ["unknown"]
    codes = [c.strip() for c in text.split("|")
             if re.match(r"^\d{5}$", c.strip())]
    return sorted(set(codes))

# ============================================================
# INFERENCE
# ============================================================
predictions   = []
ground_truths = []
raw_outputs   = []

EARLY_STOP_CHECK = 20   # check after this many examples
EARLY_STOP_MIN   = 2    # abort if fewer than this many have any code predicted
PRINT_FIRST_N    = 5    # print raw output for first N examples
BATCH_SIZE       = args.batch_size

print(f"Running inference (batch_size={BATCH_SIZE}) ...")
rows = [row for _, row in test_df.iterrows()]

for batch_start in tqdm(range(0, len(rows), BATCH_SIZE), desc="Batches"):
    batch = rows[batch_start : batch_start + BATCH_SIZE]
    prompts = [build_prompt(row["reports"]) for row in batch]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
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

    input_len = inputs["input_ids"].shape[1]
    for i, row in enumerate(batch):
        generated = output_ids[i][input_len:]
        decoded   = tokenizer.decode(generated, skip_special_tokens=True)

        pred = parse_prediction(decoded)
        predictions.append(pred)
        ground_truths.append(row["gt_codes"])

        idx = batch_start + i
        raw_outputs.append({
            "hadm_id":      row["hadm_id"],
            "predicted":    pred,
            "ground_truth": row["gt_codes"],
            "raw_output":   decoded,
        })

        if idx < PRINT_FIRST_N:
            print(f"\n--- Sample {idx+1} ---")
            print(f"  GT       : {row['gt_codes']}")
            print(f"  Predicted: {pred}")
            print(f"  Raw      : {decoded[:200]!r}")

    # Early stop check after EARLY_STOP_CHECK examples processed
    n_done = batch_start + len(batch)
    if n_done >= EARLY_STOP_CHECK and (batch_start < EARLY_STOP_CHECK):
        n_with_codes = sum(
            1 for p in predictions[:EARLY_STOP_CHECK]
            if p and p != ["unknown"] and any(re.match(r"^\d{5}$", c) for c in p)
        )
        print(f"\n[Early-stop check] {n_with_codes}/{EARLY_STOP_CHECK} examples have predicted codes.")
        if n_with_codes < EARLY_STOP_MIN:
            print(f"[ABORT] Model is not predicting CPT codes — stopping eval early.")
            import sys; sys.exit(1)

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

f1_macro   = f1_score(y_true, y_pred, average="macro",   zero_division=0)
f1_micro   = f1_score(y_true, y_pred, average="micro",   zero_division=0)
f1_samples = f1_score(y_true, y_pred, average="samples", zero_division=0)

print(f"\n{'='*50}")
print(f"  Mode       : {args.mode.upper()}")
print(f"  F1 Macro   : {f1_macro:.4f}")
print(f"  F1 Micro   : {f1_micro:.4f}")
print(f"  F1 Samples : {f1_samples:.4f}")
print(f"  Labels seen: {len(all_labels)}")
print(f"{'='*50}\n")

# Per-code F1 report
report_dict = classification_report(
    y_true, y_pred,
    target_names=mlb.classes_,
    zero_division=0,
    output_dict=True,
)

# Print top 20 most frequent codes
label_counts = labels_df["cpt_codes"].str.strip().value_counts()
top_codes    = label_counts.head(20).index.tolist()
top_indices  = [i for i, l in enumerate(all_labels) if l in top_codes]

if top_indices:
    print("Per-code F1 (top 20 most frequent):")
    print(classification_report(
        y_true[:, top_indices],
        y_pred[:, top_indices],
        target_names=[all_labels[i] for i in top_indices],
        zero_division=0,
    ))

# ============================================================
# SAVE
# ============================================================
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
mode = args.mode

results = {
    "mode":        mode,
    "timestamp":   ts,
    "f1_macro":    round(f1_macro,   4),
    "f1_micro":    round(f1_micro,   4),
    "f1_samples":  round(f1_samples, 4),
    "n_test":      len(test_df),
    "n_labels":    len(all_labels),
    "per_code_f1": {k: v for k, v in report_dict.items()
                    if k not in ["accuracy", "macro avg", "weighted avg", "samples avg"]},
    "predictions": raw_outputs,
}

json_path = f"eval_{mode}_results_{ts}.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

# Per-code CSV
per_code_rows = []
for code in all_labels:
    if code in report_dict:
        per_code_rows.append({
            "cpt_code":  code,
            "precision": round(report_dict[code]["precision"], 4),
            "recall":    round(report_dict[code]["recall"],    4),
            "f1":        round(report_dict[code]["f1-score"],  4),
            "support":   report_dict[code]["support"],
        })
pd.DataFrame(per_code_rows).to_csv(f"eval_{mode}_per_code_{ts}.csv", index=False)

print(f"✓ Results saved → {json_path}")
print(f"✓ Per-code F1  → eval_{mode}_per_code_{ts}.csv")
