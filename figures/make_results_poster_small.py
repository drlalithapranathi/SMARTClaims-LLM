#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

fig = plt.figure(figsize=(22, 7))
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(1, 3, width_ratios=[2.4, 1.6, 1.0],
                       wspace=0.38, left=0.05, right=0.97,
                       top=0.88, bottom=0.13)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

C1 = "#4C72B0"
C2 = "#55A868"
C3 = "#C44E52"

# ═══════════════════════════════════════════════════════
# PANEL 1 — main F1 bar chart (4 models)
# ═══════════════════════════════════════════════════════
ax1.set_facecolor("white")

models = ["SFT\nBaseline", "SFT + GRPO\nBaseline", "SFT\n(Improved)", "SFT + GRPO\n(Final)"]
f1s  = [0.4419, 0.4647, 0.5560, 0.5677]
f1mi = [0.3990, 0.4290, 0.5486, 0.5662]
f1ma = [0.0490, 0.0610, 0.2346, 0.1565]

x = np.arange(len(models))
w = 0.26

b1 = ax1.bar(x - w, f1s,  w, label="F1-samples", color=C1, alpha=0.92)
b2 = ax1.bar(x,     f1mi, w, label="F1-micro",   color=C2, alpha=0.92)
b3 = ax1.bar(x + w, f1ma, w, label="F1-macro",   color=C3, alpha=0.92)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.006,
                 f"{h:.3f}", ha="center", va="bottom",
                 fontsize=9.5, fontweight="bold", color="#111111")

for bar in [b1[-1], b2[-1], b3[-1]]:
    bar.set_edgecolor("#e6a817")
    bar.set_linewidth(2.8)

ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=12)
ax1.set_ylim(0, 0.80)
ax1.set_ylabel("F1 Score", fontsize=13, fontweight="bold")
ax1.set_title("F1 Performance Across Training Stages",
              fontsize=14, fontweight="bold", pad=10)
ax1.legend(fontsize=11, framealpha=0.95, edgecolor="#cccccc")
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.spines[["top", "right"]].set_visible(False)
ax1.tick_params(axis="y", labelsize=11)

ax1.annotate("Best Model", xy=(3 - w, 0.5677), xytext=(2.0, 0.72),
             fontsize=11, color="#e6a817", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#e6a817", lw=2.0))

# ═══════════════════════════════════════════════════════
# PANEL 2 — Top-20 / Top-50 horizontal bar chart
# ═══════════════════════════════════════════════════════
ax2.set_facecolor("white")

labels  = ["Top-50\nMean F1", "Top-20\nMean F1"]
values  = [0.565, 0.634]
colors2 = ["#3DA8C8", "#7B68C8"]

y_pos = np.array([0, 1])
bars2 = ax2.barh(y_pos, values, height=0.45, color=colors2, alpha=0.92,
                 edgecolor="white", linewidth=1.2)

# value labels inside/outside bars
for bar, val in zip(bars2, values):
    ax2.text(val + 0.012, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", ha="left",
             fontsize=14, fontweight="bold", color="#111111")

# overall F1-samples reference line
ax2.axvline(0.568, color=C1, linestyle="--", linewidth=2.2, alpha=0.8)
ax2.text(0.568 + 0.008, 1.52, "Overall\nF1-samples\n0.568",
         fontsize=9, color=C1, fontweight="bold", va="top")

ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels, fontsize=13, fontweight="bold")
ax2.set_xlim(0, 0.82)
ax2.set_xlabel("Mean F1 Score", fontsize=12, fontweight="bold")
ax2.set_title("Per-Code F1 by Frequency Tier",
              fontsize=14, fontweight="bold", pad=10)
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(axis="x", linestyle="--", alpha=0.4)
ax2.tick_params(axis="x", labelsize=11)

# ═══════════════════════════════════════════════════════
# PANEL 3 — Unknown accuracy donut
# ═══════════════════════════════════════════════════════
ax3.set_facecolor("white")
ax3.set_aspect("equal")

wedge_kw = dict(wedgeprops=dict(width=0.42, edgecolor="white", linewidth=3))
ax3.pie([100], radius=1.0, colors=["#2CA87F"],
        startangle=90, **wedge_kw)

ax3.text(0, 0.12, "100%", ha="center", va="center",
         fontsize=26, fontweight="bold", color="#000000",
         transform=ax3.transData)
ax3.text(0, -0.22, "50 / 50\ncorrect", ha="center", va="center",
         fontsize=12, color="#333333", fontweight="bold",
         transform=ax3.transData)

ax3.set_title("Unknown\nAccuracy", fontsize=14, fontweight="bold", pad=10)

# ═══════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════
out_pdf = "results_poster_small.pdf"
plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
print(f"Saved PDF  → {out_pdf}")

out_png = "results_poster_small.png"
plt.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
print(f"Saved PNG  → {out_png}")
