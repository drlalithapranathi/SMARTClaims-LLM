#!/usr/bin/env python3
"""
Merge sharded nounk eval JSONs into a single result.

Usage:
    python merge_nounk_eval.py eval_nounk_sft_results_shard*of8_*.json
"""

import sys
import json
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
from datetime import datetime

shard_files = sorted(sys.argv[1:])
assert shard_files, "Pass shard JSON files as arguments"
print(f"Merging {len(shard_files)} shards: {shard_files}")

all_preds, all_gts, all_rows = [], [], []
meta = None
for f in shard_files:
    with open(f) as fh:
        d = json.load(fh)
    if meta is None:
        meta = d
    all_rows.extend(d["predictions"])

all_preds = [r["predicted"]    for r in all_rows]
all_gts   = [r["ground_truth"] for r in all_rows]

print(f"Total samples: {len(all_rows)}")

all_labels = sorted(set(l for lst in all_gts + all_preds for l in lst))
mlb    = MultiLabelBinarizer(classes=all_labels)
y_true = mlb.fit_transform(all_gts)
y_pred = mlb.transform(all_preds)

f1_macro   = f1_score(y_true, y_pred, average="macro",   zero_division=0)
f1_micro   = f1_score(y_true, y_pred, average="micro",   zero_division=0)
f1_samples = f1_score(y_true, y_pred, average="samples", zero_division=0)

print(f"\n{'='*55}")
print(f"  Experiment : nounk — SFT (merged {len(shard_files)} shards)")
print(f"  F1 Samples : {f1_samples:.4f}")
print(f"  F1 Micro   : {f1_micro:.4f}")
print(f"  F1 Macro   : {f1_macro:.4f}")
print(f"  N samples  : {len(all_rows)}")
print(f"{'='*55}\n")

report_dict = classification_report(
    y_true, y_pred, target_names=mlb.classes_, zero_division=0, output_dict=True
)

ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
out_json = f"eval_nounk_sft_results_merged_{ts}.json"
out_csv  = f"eval_nounk_sft_per_code_merged_{ts}.csv"

results = {
    "experiment":  "nounk",
    "mode":        "sft",
    "model":       meta["model"],
    "timestamp":   ts,
    "f1_samples":  round(f1_samples, 4),
    "f1_micro":    round(f1_micro,   4),
    "f1_macro":    round(f1_macro,   4),
    "n_test":      len(all_rows),
    "n_labels":    len(all_labels),
    "per_code_f1": {k: v for k, v in report_dict.items()
                    if k not in ["accuracy","macro avg","weighted avg","samples avg","micro avg"]},
    "predictions": all_rows,
}

with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

per_code_rows = [
    {"cpt_code": c, "precision": round(v["precision"],4),
     "recall": round(v["recall"],4), "f1": round(v["f1-score"],4), "support": v["support"]}
    for c, v in report_dict.items()
    if c not in ["accuracy","macro avg","weighted avg","samples avg","micro avg"]
]
pd.DataFrame(per_code_rows).to_csv(out_csv, index=False)

print(f"✓ Merged results → {out_json}")
print(f"✓ Per-code F1    → {out_csv}")
