#!/usr/bin/env python3
"""
Eval script for unk10 ep2 experiment.
Evaluates unk10_sft_merged (and optionally unk10 GRPO adapter).

Run:
    python eval_test_set.py --mode sft --n-eval 500
    python eval_test_set.py --mode grpo --n-eval 500
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

# ── CONFIG ───────────────────────────────────────────────────────────────────
UNK10_SFT_MERGED = "unk10_sft_merged"
UNK10_GRPO_ADAPTER = "unk10_grpo_adapter"   # will exist after GRPO training
TOKENIZER_SRC    = "../qwen3-32b-mimic-cpt-200k-ep2"

TEST_INPUTS_CSV  = "../sft_test_inputs.csv"
TEST_LABELS_CSV  = "../sft_test_labels.csv"
MAX_NEW_TOKENS   = 128

SYSTEM_PROMPT = (
    "You are a medical coding assistant. "
    "Given a clinical note for a patient admission, identify all billable "
    "radiology procedures performed. Output only the CPT codes separated by "
    "pipe ( | ). If no billable radiology procedure can be determined, output "
    "unknown. Do not include add-on codes, explanations, or numbering."
)

# ── ARGS ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["sft", "grpo"], default="sft")
parser.add_argument("--n-eval", type=int, default=None)
parser.add_argument("--batch-size", type=int, default=1)
args = parser.parse_args()

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
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

print(f"Loading {UNK10_SFT_MERGED} ...")
model = AutoModelForCausalLM.from_pretrained(
    UNK10_SFT_MERGED,
    quantization_config=bnb_config,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
model.config.use_cache = True

if args.mode == "grpo":
    print(f"Loading GRPO adapter from {UNK10_GRPO_ADAPTER} ...")
    model = PeftModel.from_pretrained(model, UNK10_GRPO_ADAPTER)

model.eval()
print("Model loaded.\n")

# ── TEST DATA ─────────────────────────────────────────────────────────────────
print("Loading test data ...")
inputs_df = pd.read_csv(TEST_INPUTS_CSV, dtype={"hadm_id": str})
labels_df = pd.read_csv(TEST_LABELS_CSV, dtype={"hadm_id": str, "cpt_codes": str})

gt_agg = (
    labels_df.groupby("hadm_id")["cpt_codes"]
    .apply(lambda x: sorted(set(x.str.strip())))
    .reset_index()
    .rename(columns={"cpt_codes": "gt_codes"})
)

test_df = inputs_df.merge(gt_agg, on="hadm_id", how="inner")
if args.n_eval is not None:
    test_df = test_df.sample(n=min(args.n_eval, len(test_df)), random_state=42).reset_index(drop=True)
    print(f"  Capped to {len(test_df)} samples")
print(f"  Test admissions: {len(test_df)}\n")

# ── INFERENCE ─────────────────────────────────────────────────────────────────
def build_prompt(report_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": report_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

def parse_prediction(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.strip().lower() == "unknown":
        return ["unknown"]
    codes = [c.strip() for c in text.split("|") if re.match(r"^\d{5}$", c.strip())]
    return sorted(set(codes))

predictions, ground_truths, raw_outputs = [], [], []
BATCH_SIZE = args.batch_size
rows = [row for _, row in test_df.iterrows()]

print(f"Running inference (batch_size={BATCH_SIZE}) ...")
for batch_start in tqdm(range(0, len(rows), BATCH_SIZE), desc="Batches"):
    batch = rows[batch_start: batch_start + BATCH_SIZE]
    prompts = [build_prompt(row["reports"]) for row in batch]

    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=4096,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, temperature=None, top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    for i, row in enumerate(batch):
        decoded = tokenizer.decode(output_ids[i][input_len:], skip_special_tokens=True)
        pred    = parse_prediction(decoded)
        predictions.append(pred)
        ground_truths.append(row["gt_codes"])
        raw_outputs.append({
            "hadm_id": row["hadm_id"], "predicted": pred,
            "ground_truth": row["gt_codes"], "raw_output": decoded,
        })

# ── METRICS ──────────────────────────────────────────────────────────────────
print("\nComputing metrics ...")
all_labels = sorted(set(l for lst in ground_truths + predictions for l in lst))
mlb    = MultiLabelBinarizer(classes=all_labels)
y_true = mlb.fit_transform(ground_truths)
y_pred = mlb.transform(predictions)

f1_macro   = f1_score(y_true, y_pred, average="macro",   zero_division=0)
f1_micro   = f1_score(y_true, y_pred, average="micro",   zero_division=0)
f1_samples = f1_score(y_true, y_pred, average="samples", zero_division=0)

print(f"\n{'='*50}")
print(f"  Experiment : unk10 ep2 — {args.mode.upper()}")
print(f"  F1 Macro   : {f1_macro:.4f}")
print(f"  F1 Micro   : {f1_micro:.4f}")
print(f"  F1 Samples : {f1_samples:.4f}")
print(f"  N samples  : {len(test_df)}")
print(f"{'='*50}\n")

report_dict = classification_report(
    y_true, y_pred, target_names=mlb.classes_, zero_division=0, output_dict=True
)

# ── SAVE ─────────────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
results = {
    "experiment":  "unk10_ep2",
    "mode":        args.mode,
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

out_json = f"eval_unk10_{args.mode}_results_{ts}.json"
out_csv  = f"eval_unk10_{args.mode}_per_code_{ts}.csv"

with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

per_code_rows = [
    {"cpt_code": c, "precision": round(v["precision"], 4),
     "recall": round(v["recall"], 4), "f1": round(v["f1-score"], 4), "support": v["support"]}
    for c, v in report_dict.items()
    if c not in ["accuracy", "macro avg", "weighted avg", "samples avg", "micro avg"]
]
pd.DataFrame(per_code_rows).to_csv(out_csv, index=False)

print(f"✓ Results saved → {out_json}")
print(f"✓ Per-code F1  → {out_csv}")
