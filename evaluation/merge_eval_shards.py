#!/usr/bin/env python3
"""
Merges shard_{0..3}_predictions.json and computes final metrics.
Saves eval_unk10_grpo_results_<TS>.json and eval_unk10_grpo_per_code_<TS>.csv.

Run after all 4 shards finish:
    python merge_eval_shards.py
"""

import json
import glob
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report

N_SHARDS = 4

print("Loading shard predictions ...")
all_outputs = []
for i in range(N_SHARDS):
    fname = f"shard_{i}_predictions.json"
    with open(fname) as f:
        shard = json.load(f)
    all_outputs.extend(shard)
    print(f"  shard_{i}: {len(shard)} samples")

print(f"  Total: {len(all_outputs)} samples\n")

predictions   = [r["predicted"]    for r in all_outputs]
ground_truths = [r["ground_truth"] for r in all_outputs]

all_labels = sorted(set(l for lst in ground_truths + predictions for l in lst))
mlb    = MultiLabelBinarizer(classes=all_labels)
y_true = mlb.fit_transform(ground_truths)
y_pred = mlb.transform(predictions)

f1_macro   = f1_score(y_true, y_pred, average="macro",   zero_division=0)
f1_micro   = f1_score(y_true, y_pred, average="micro",   zero_division=0)
f1_samples = f1_score(y_true, y_pred, average="samples", zero_division=0)

print("=" * 55)
print("  Experiment : unk10 ep2 — GRPO (full test set)")
print(f"  F1 Samples : {f1_samples:.4f}")
print(f"  F1 Micro   : {f1_micro:.4f}")
print(f"  F1 Macro   : {f1_macro:.4f}")
print(f"  N samples  : {len(all_outputs)}")
print("=" * 55)

report_dict = classification_report(
    y_true, y_pred, target_names=mlb.classes_, zero_division=0, output_dict=True
)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_json = f"eval_unk10_grpo_results_{ts}.json"
out_csv  = f"eval_unk10_grpo_per_code_{ts}.csv"

results = {
    "experiment":  "unk10_ep2",
    "mode":        "grpo",
    "base_model":  "unk10_sft_merged",
    "adapter":     "unk10_grpo_adapter",
    "timestamp":   ts,
    "f1_samples":  round(f1_samples, 4),
    "f1_micro":    round(f1_micro,   4),
    "f1_macro":    round(f1_macro,   4),
    "n_test":      len(all_outputs),
    "n_labels":    len(all_labels),
    "per_code_f1": {k: v for k, v in report_dict.items()
                    if k not in ["accuracy", "macro avg", "weighted avg", "samples avg", "micro avg"]},
    "predictions": all_outputs,
}

with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

per_code_rows = [
    {"cpt_code":   c,
     "precision":  round(v["precision"], 4),
     "recall":     round(v["recall"],    4),
     "f1":         round(v["f1-score"],  4),
     "support":    v["support"]}
    for c, v in report_dict.items()
    if c not in ["accuracy", "macro avg", "weighted avg", "samples avg", "micro avg"]
]
pd.DataFrame(per_code_rows).to_csv(out_csv, index=False)

print(f"\n✓ Results saved -> {out_json}")
print(f"✓ Per-code F1  -> {out_csv}")
