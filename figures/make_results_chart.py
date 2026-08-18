#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("white")

# ── LEFT: F1 bar chart ────────────────────────────────────────────────────────
models = ["SFT", "SFT + GRPO", "Augmented\nSFT", "Augmented\nSFT + GRPO\n(Final)"]
f1s  = [0.4419, 0.4647, 0.5560, 0.5677]
f1mi = [0.3990, 0.4290, 0.5486, 0.5662]
f1ma = [0.0490, 0.0610, 0.2346, 0.1565]

x = np.arange(len(models))
w = 0.25

b1 = ax1.bar(x - w, f1s,  w, label="F1-samples", color="#4472C4")
b2 = ax1.bar(x,     f1mi, w, label="F1-micro",   color="#70AD47")
b3 = ax1.bar(x + w, f1ma, w, label="F1-macro",   color="#ED7D31")

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax1.annotate("Best Model", xy=(3 - w, 0.5677), xytext=(2.3, 0.64),
             color="red", fontsize=9, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=9)
ax1.set_ylim(0, 0.8)
ax1.set_ylabel("F1 Score", fontsize=10)
ax1.set_title("F1 Performance Across Training Stages", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.spines[["top", "right"]].set_visible(False)

# ── RIGHT: Perplexity ─────────────────────────────────────────────────────────
stages = ["Base\nQwen3-32B", "+Clinical\nPretraining", "+SFT", "+GRPO"]
perp   = [13.00, 5.36, 1.48, 1.49]
colors = ["#C00000", "#ED7D31", "#70AD47", "#4472C4"]

bars = ax2.bar(stages, perp, color=colors, alpha=0.88, edgecolor="white", linewidth=0.5)

for bar, val in zip(bars, perp):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.15,
             f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# drop annotations
ax2.annotate("−59%", xy=(1, 5.36), xytext=(1, 9.0),
             ha="center", color="#ED7D31", fontsize=9, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#ED7D31", lw=1.5))
ax2.annotate("−72%", xy=(2, 1.48), xytext=(2.6, 4.5),
             ha="center", color="#70AD47", fontsize=9, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#70AD47", lw=1.5))

ax2.set_ylim(0, 15.5)
ax2.set_ylabel("Perplexity (lower is better)", fontsize=10)
ax2.set_title("Domain Adaptation Effect (Perplexity ↓)", fontsize=11, fontweight="bold")
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout(pad=2.0)
out = "results_chart.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved → {out}")
