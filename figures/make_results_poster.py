#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig = plt.figure(figsize=(20, 12))
fig.patch.set_facecolor("#0d1117")

# ── colour palette ────────────────────────────────────────────────────────────
C_BLUE   = "#4dabf7"
C_TEAL   = "#38d9a9"
C_PURPLE = "#da77f2"
C_ORANGE = "#ffa94d"
C_RED    = "#ff6b6b"
C_GOLD   = "#ffd43b"
C_BG     = "#0d1117"
C_PANEL  = "#161b22"
C_TEXT   = "#f0f6fc"
C_MUTED  = "#8b949e"

def panel(ax, title):
    ax.set_facecolor(C_PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.tick_params(colors=C_TEXT, labelsize=9)
    ax.set_title(title, color=C_TEXT, fontsize=11, fontweight="bold", pad=10)

# ── GRID ──────────────────────────────────────────────────────────────────────
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                      left=0.06, right=0.97, top=0.88, bottom=0.08)
ax1 = fig.add_subplot(gs[0, :2])   # wide bar chart
ax2 = fig.add_subplot(gs[0, 2])    # perplexity
ax3 = fig.add_subplot(gs[1, :2])   # top-20 codes
ax4 = fig.add_subplot(gs[1, 2])    # mixed eval donut

# ── TITLE ─────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.95, "SMARTClaims — Experimental Results",
         ha="center", va="center", fontsize=18, fontweight="bold",
         color=C_TEXT, fontfamily="DejaVu Sans")
fig.text(0.5, 0.91,
         "Automated Radiology CPT Code Prediction  •  Qwen3-32B  •  MIMIC-IV  •  n=4,702 test admissions",
         ha="center", va="center", fontsize=10, color=C_MUTED)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. MAIN F1 BAR CHART
# ═══════════════════════════════════════════════════════════════════════════════
panel(ax1, "Model Performance Across Pipeline Stages")

models   = ["SFT v9", "GRPO v3b\n(on SFT v9)", "SFT unk10\nep2", "GRPO unk10\n(Final Model)"]
f1s      = [0.4419, 0.4647, 0.5560, 0.5677]
f1mi     = [0.3990, 0.4290, 0.5486, 0.5662]
f1ma     = [0.0490, 0.0610, 0.2346, 0.1565]

x   = np.arange(len(models))
w   = 0.25
colors = [C_BLUE, C_TEAL, C_PURPLE]

b1 = ax1.bar(x - w, f1s,  w, label="F1-samples", color=C_BLUE,   alpha=0.9, zorder=3)
b2 = ax1.bar(x,     f1mi, w, label="F1-micro",   color=C_TEAL,   alpha=0.9, zorder=3)
b3 = ax1.bar(x + w, f1ma, w, label="F1-macro",   color=C_PURPLE, alpha=0.9, zorder=3)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                 f"{h:.3f}", ha="center", va="bottom",
                 fontsize=7.5, color=C_TEXT, fontweight="bold")

# highlight final model
for bar in [b1[-1], b2[-1], b3[-1]]:
    bar.set_edgecolor(C_GOLD)
    bar.set_linewidth(2)

ax1.set_xticks(x)
ax1.set_xticklabels(models, color=C_TEXT, fontsize=9)
ax1.set_ylim(0, 0.72)
ax1.set_ylabel("F1 Score", color=C_MUTED, fontsize=9)
ax1.yaxis.label.set_color(C_MUTED)
ax1.grid(axis="y", color="#30363d", linewidth=0.5, zorder=0)
ax1.legend(facecolor=C_PANEL, edgecolor="#30363d", labelcolor=C_TEXT,
           fontsize=9, loc="upper left")

# annotate best
ax1.annotate(">> Best Model", xy=(3 - w, 0.5677), xytext=(2.35, 0.63),
             color=C_GOLD, fontsize=9, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=1.5))

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PERPLEXITY
# ═══════════════════════════════════════════════════════════════════════════════
panel(ax2, "Domain Adaptation (Perplexity ↓)")

stages = ["Base\nQwen3-32B", "+Clinical\nPretraining", "+SFT", "+GRPO"]
perp   = [13.00, 5.36, 1.48, 1.49]
bar_colors = [C_RED, C_ORANGE, C_TEAL, C_BLUE]

bars = ax2.bar(stages, perp, color=bar_colors, alpha=0.9, zorder=3,
               edgecolor="#30363d", linewidth=0.5)
for bar, val in zip(bars, perp):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.2,
             f"{val:.2f}", ha="center", va="bottom",
             fontsize=9, color=C_TEXT, fontweight="bold")

ax2.set_ylim(0, 15.5)
ax2.set_ylabel("Perplexity", color=C_MUTED, fontsize=9)
ax2.yaxis.label.set_color(C_MUTED)
ax2.tick_params(axis="x", labelsize=8)
ax2.grid(axis="y", color="#30363d", linewidth=0.5, zorder=0)
ax2.annotate("−59%", xy=(1, 5.36), xytext=(1, 9),
             ha="center", color=C_ORANGE, fontsize=9, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=C_ORANGE))
ax2.annotate("−72%", xy=(2, 1.48), xytext=(2.5, 4.5),
             ha="center", color=C_TEAL, fontsize=9, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=C_TEAL))

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TOP-20 CODE F1 HORIZONTAL BAR
# ═══════════════════════════════════════════════════════════════════════════════
panel(ax3, "Per-Code F1 — Top 20 Most Frequent CPT Codes  (GRPO unk10)")

codes = ["99152","36569","99144","36558","36556","36584",
         "36247","75894","76377","75726","36245","36589",
         "49440","75898","61624","36561","75984","36217","36226","36224"]
f1s_code = [0.639,0.833,0.572,0.768,0.723,0.709,
             0.755,0.660,0.734,0.657,0.391,0.620,
             0.604,0.588,0.840,0.824,0.467,0.560,0.495,0.242]

bar_c = [C_GOLD if v >= 0.75 else C_TEAL if v >= 0.55 else C_RED for v in f1s_code]
y_pos = np.arange(len(codes))

ax3.barh(y_pos, f1s_code, color=bar_c, alpha=0.9, zorder=3,
         edgecolor="#30363d", linewidth=0.4, height=0.7)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(codes, fontsize=8, color=C_TEXT)
ax3.set_xlim(0, 1.05)
ax3.set_xlabel("F1 Score", color=C_MUTED, fontsize=9)
ax3.xaxis.label.set_color(C_MUTED)
ax3.grid(axis="x", color="#30363d", linewidth=0.5, zorder=0)
ax3.axvline(0.634, color=C_GOLD, linestyle="--", linewidth=1.2, zorder=4)
ax3.text(0.638, 19.4, "Top-20\nmean 0.634", color=C_GOLD, fontsize=7.5, va="top")

for i, (v, c) in enumerate(zip(f1s_code, codes)):
    ax3.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7, color=C_TEXT)

legend_patches = [
    mpatches.Patch(color=C_GOLD,  label="F1 ≥ 0.75"),
    mpatches.Patch(color=C_TEAL,  label="F1 ≥ 0.55"),
    mpatches.Patch(color=C_RED,   label="F1 < 0.55"),
]
ax3.legend(handles=legend_patches, facecolor=C_PANEL, edgecolor="#30363d",
           labelcolor=C_TEXT, fontsize=8, loc="lower right")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. MIXED EVAL DONUT
# ═══════════════════════════════════════════════════════════════════════════════
panel(ax4, "Mixed Eval: Radiology + Unknown (n=500)")

sizes1   = [50, 0]    # unknown: 50/50 correct
sizes2   = [265, 185] # radiology approx F1-samples 0.589 * 450

wedge_kw = dict(wedgeprops=dict(width=0.38, edgecolor=C_BG, linewidth=2))

# outer ring = unknown accuracy
w1, _ = ax4.pie([50, 0], radius=1.0, colors=[C_TEAL, "#30363d"],
                startangle=90, **wedge_kw)
# inner ring = radiology F1
w2, _ = ax4.pie([0.5888, 0.4112], radius=0.58, colors=[C_BLUE, "#30363d"],
                startangle=90, **wedge_kw)

ax4.text(0, 0.18, "100%", ha="center", va="center",
         fontsize=13, fontweight="bold", color=C_TEAL)
ax4.text(0, 0.04, "Unknown\nAccuracy", ha="center", va="center",
         fontsize=7, color=C_MUTED)

ax4.text(0, -0.18, "0.589", ha="center", va="center",
         fontsize=13, fontweight="bold", color=C_BLUE)
ax4.text(0, -0.32, "Radiology\nF1-samples", ha="center", va="center",
         fontsize=7, color=C_MUTED)

outer_patch = mpatches.Patch(color=C_TEAL, label="Unknown (50 cases)")
inner_patch = mpatches.Patch(color=C_BLUE, label="Radiology (450 cases)")
ax4.legend(handles=[outer_patch, inner_patch],
           facecolor=C_PANEL, edgecolor="#30363d",
           labelcolor=C_TEXT, fontsize=8, loc="lower center",
           bbox_to_anchor=(0.5, -0.12))

# ── SAVE ──────────────────────────────────────────────────────────────────────
out = "results_poster.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=C_BG)
print(f"Saved → {out}")
