#!/usr/bin/env python3
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_JSON = "eval_grpo_results.json"   # output of evaluation/merge_eval_shards.py or eval_test_set.py
TOP_N = 25
OUT = "confusion_matrix.png"

with open(RESULTS_JSON) as f:
    data = json.load(f)

predictions = data["predictions"]
per_code    = data["per_code_f1"]
f1_samples  = data["f1_samples"]

# Top N codes by support (exclude non-CPT keys)
top_codes = [c for c, _ in sorted(per_code.items(), key=lambda x: -x[1]["support"])
             if c.strip().isdigit()][:TOP_N]
top_set   = set(top_codes)
code_idx  = {c: i for i, c in enumerate(top_codes)}

# Build matrix: row=true, col=predicted
matrix = np.zeros((TOP_N, TOP_N), dtype=int)
for s in predictions:
    for tc in s["ground_truth"]:
        if tc in top_set:
            for pc in s["predicted"]:
                if pc in top_set:
                    matrix[code_idx[tc], code_idx[pc]] += 1

# Row-normalize (recall per true code)
matrix_norm = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)

# Plot
sz = max(14, TOP_N * 0.55)
fig, ax = plt.subplots(figsize=(sz, sz * 0.88))
fig.patch.set_facecolor("#0d0d0d")
ax.set_facecolor("#0d0d0d")

sns.heatmap(
    matrix_norm,
    xticklabels=top_codes,
    yticklabels=top_codes,
    cmap="magma",
    vmin=0, vmax=1,
    linewidths=0.4,
    linecolor="#1a1a1a",
    annot=True,
    fmt=".2f",
    annot_kws={"size": 7.5, "color": "white"},
    cbar_kws={"shrink": 0.7, "label": "Recall (row-normalised)"},
    ax=ax,
)

for i in range(TOP_N):
    ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="white", lw=1.5))

ax.set_title(f"CPT Confusion Matrix — GRPO v3b  |  F1-Samples = {f1_samples:.4f}",
             fontsize=13, fontweight="bold", pad=14, color="white")
ax.set_xlabel("Predicted Code", fontsize=11, labelpad=10, color="white")
ax.set_ylabel("True Code",      fontsize=11, labelpad=10, color="white")
ax.tick_params(axis="x", rotation=45, labelsize=9, colors="white")
ax.tick_params(axis="y", rotation=0,  labelsize=9, colors="white")
plt.setp(ax.get_xticklabels(), color="white")
plt.setp(ax.get_yticklabels(), color="white")
ax.spines[:].set_color("#333333")

cbar = ax.collections[0].colorbar
cbar.ax.yaxis.label.set_color("white")
cbar.ax.tick_params(colors="white")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {OUT}")
