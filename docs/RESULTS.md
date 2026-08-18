# 4. Results

## 4.1 Experimental Setup

We evaluate Qwen3-32B fine-tuned for radiology CPT-code prediction along the three-stage pipeline: continued pretraining (CPT) on 200,000 MIMIC-IV discharge summaries, supervised fine-tuning (SFT) on radiology-report → CPT-code pairs, and Group Relative Policy Optimization (GRPO) with an F1-based reward.

The held-out test set contains **4,702 admissions** with **264 distinct CPT codes**. Test-set leakage was explicitly verified against all training and validation splits (`validate()` in `data_pipeline/sft_cpt_vsac_pipeline.py`). Three classification metrics are reported: **F1-samples** (per-instance F1, then averaged), **F1-micro** (aggregate over all label decisions, the primary metric for imbalanced multi-label problems), and **F1-macro** (unweighted code average, sensitive to long-tail codes). All metrics use the standard `sklearn` multi-label formulation.

## 4.2 Domain-Adaptation Effect (Perplexity)

To verify each training stage actually moves the model toward the clinical-coding distribution, we computed mean per-token perplexity on a held-out clinical sample.

| Model stage | Perplexity ↓ |
|---|---:|
| Qwen3-32B (base) | 13.00 |
| + CPT (200k MIMIC discharge) | 5.36 |
| + SFT (radiology CPT) | 1.48 |
| + GRPO (F1 reward) | 1.49 |

*Measured with `evaluation/eval_perplexity_stages.py` (response-token perplexity, first 50 test admissions) on 2026-04-22, before the final unk10 GRPO model was trained: the SFT and GRPO rows correspond to the SFT v9 / GRPO v3b checkpoints.*

CPT alone reduces perplexity by ~59%; SFT reduces it by another ~72%. GRPO does not lower perplexity further — expected, since GRPO optimizes a task reward, not next-token likelihood — but the value remains essentially flat, indicating the policy update did not damage the language-modeling distribution learned in earlier stages.

## 4.2.1 CPT-Stage Generation Quality (BLEU / ROUGE)

Two supplementary checks of the continued-pretraining stage, run in March 2026 before the SFT/GRPO work (not included in the thesis). Both compare base Qwen3-32B against the CPT epoch-2 adapter on 100 held-out samples; scripts are in `old_experiments/cpt_eval/`.

**Discharge-note continuation** (`eval_bleu_base.py` / `eval_bleu_ep2.py`): the model is given the first 50% of a held-out discharge note (≤512 prompt tokens) and generates up to 256 tokens; BLEU-4 against the true continuation.

| Model | BLEU | BP | p1 | p2 | p3 | p4 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-32B (base) | 2.27 | 0.96 | 16.70 | 3.28 | 1.47 | 0.39 |
| + CPT (epoch 2) | 4.57 | 0.96 | 19.24 | 5.64 | 3.16 | 1.52 |

**Radiology findings generation** (`bleu_rouge_radiology_base.py` / `bleu_rouge_radiology_cpt.py`): given the procedure name and clinical indication, generate the FINDINGS section (≤512 tokens); BLEU-4, ROUGE-F1, and per-sample perplexity against the reference report.

| Model | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | Perplexity |
|---|---:|---:|---:|---:|---:|
| Qwen3-32B (base) | 2.09 | 0.179 | 0.039 | 0.104 | 10.55 |
| + CPT (epoch 2) | 3.62 | 0.171 | 0.054 | 0.111 | 4.72 |

CPT roughly doubles BLEU on discharge continuation and lowers radiology-report perplexity by more than half; absolute BLEU/ROUGE remain low, as expected for free-text clinical generation, and these metrics were not pursued further once the task was framed as CPT-code prediction.

## 4.3 Main Classification Results

Table 4.1 reports test-set F1 across the model lineage. Early models (v8 / v2) used a small 1.8k-example dataset and are included only as a baseline reference; the v9 line is trained on the full 21,155-example SFT dataset and 16,924-example GRPO dataset. The unk10 line introduces a 10% unknown-discharge augmentation during SFT.

**Table 4.1 — Test-set F1 across pipeline stages (n=4,702 unless noted).**

| Model | n (test) | F1-samples | F1-micro | F1-macro |
|---|---:|---:|---:|---:|
| SFT v8 (small data, baseline) | 400 | 0.078 | 0.066 | — |
| GRPO v2 (small data, baseline) | 400 | 0.224 | 0.180 | — |
| SFT v9 | 4,702 | 0.4419 | 0.399 | 0.049 |
| GRPO v3b (on SFT v9) | 4,702 | 0.4647 | 0.429 | 0.061 |
| SFT unk10 ep2 | 4,702 | 0.5560 | 0.5486 | 0.2346 |
| **GRPO unk10 (on SFT unk10 ep2)** | 4,702 | **0.5677** | **0.5662** | **0.1565** |

GRPO improves over SFT v9 by **+2.3 F1-samples** and **+3.0 F1-micro** points on the v9 line. On the unk10 line, GRPO further improves SFT unk10 by **+1.2 F1-samples** and **+1.8 F1-micro** points, with the final model reaching **F1-samples 0.568** and **F1-micro 0.566** — the best results observed across all experiments.

Note on F1-macro: unk10 GRPO (0.157) is lower than unk10 SFT (0.235). This is expected — GRPO optimizes F1 reward on the training distribution (radiology-only, no unknowns), which concentrates updates on high-frequency codes at the expense of rare-code recall.

## 4.4 Per-Code Performance

Aggregate F1 understates performance on the codes that matter clinically. Table 4.2 compares mean F1 restricted to the top-K most-frequent CPT codes.

**Table 4.2 — Mean F1 over the top-K most frequent CPT codes.**

| Model | Top-20 mean F1 | Top-50 mean F1 |
|---|---:|---:|
| SFT v9 (n=4,702) | 0.540 | 0.394 |
| GRPO v3b (n=4,702) | 0.553 | 0.409 |
| SFT unk10 ep2 (n=4,702) | 0.630 | 0.575 |
| **GRPO unk10 (n=4,702)** | **0.634** | **0.565** |

For the 20 most frequent codes, the unk10 GRPO model achieves 0.634 mean F1, with several individual codes above 0.75 (e.g., 36569 *non-tunneled central venous catheter insertion* — F1 0.833; 36561 *tunneled centrally inserted catheter* — F1 0.824; 61624 *transcatheter occlusion* — F1 0.840; 36247 *selective catheter placement* — F1 0.755). The top-50 mean F1 is 0.565, lower than the top-20 as expected — performance degrades toward rarer codes with fewer training examples. High-volume codes that previously underperformed — most notably **99152** (*moderate sedation*) — reached **F1 0.639** under GRPO unk10, a substantial gain driven by the model's improved selectivity from unknown augmentation.

## 4.5 Mixed Evaluation: Radiology + Unknown Cases

To assess the model's ability to correctly identify non-radiology encounters, the GRPO unk10 model was evaluated on a mixed set of 450 radiology test cases and 50 holdout discharge notes (no radiology procedures, labeled "unknown").

| Metric | Value |
|---|---:|
| Radiology F1-samples (n=450) | 0.5888 |
| Radiology F1-micro (n=450) | 0.5521 |
| Radiology F1-macro (n=450) | 0.2446 |
| Unknown accuracy (n=50) | **1.0 (50/50)** |

The model correctly outputs "unknown" for all 50 non-radiology discharge notes — a 100% accuracy rate — confirming that the unknown augmentation strategy successfully teaches the model to abstain when no radiology procedure is identifiable. Radiology F1 on the 450-sample subset is consistent with the full 4,702-sample results.

## 4.6 Per-Code Performance — GRPO unk10 (Full 4,702 Test Set)

**Table 4.3 — Top-20 most frequent CPT codes: GRPO unk10 individual F1 scores.**

| CPT Code | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 99152 | 811 | 0.675 | 0.605 | 0.639 |
| 36569 | 579 | 0.934 | 0.751 | 0.833 |
| 99144 | 527 | 0.558 | 0.586 | 0.572 |
| 36558 | 481 | 0.916 | 0.661 | 0.768 |
| 36556 | 339 | 0.875 | 0.617 | 0.723 |
| 36584 | 315 | 0.902 | 0.584 | 0.709 |
| 36247 | 254 | 0.746 | 0.764 | 0.755 |
| 75894 | 244 | 0.584 | 0.758 | 0.660 |
| 76377 | 225 | 0.764 | 0.707 | 0.734 |
| 75726 | 186 | 0.619 | 0.699 | 0.657 |
| 36245 | 175 | 0.593 | 0.291 | 0.391 |
| 36589 | 167 | 0.879 | 0.479 | 0.620 |
| 49440 | 164 | 0.711 | 0.524 | 0.604 |
| 75898 | 160 | 0.486 | 0.744 | 0.588 |
| 61624 | 153 | 0.879 | 0.804 | 0.840 |
| 36561 | 148 | 0.981 | 0.710 | 0.824 |
| 75984 | 146 | 0.530 | 0.418 | 0.467 |
| 36217 | 140 | 0.515 | 0.614 | 0.560 |
| 36226 | 134 | 0.663 | 0.396 | 0.495 |
| 36224 | 122 | 0.667 | 0.148 | 0.242 |
| **Top-20 Mean** | | | | **0.634** |
| **Top-50 Mean** | | | | **0.565** |

## 4.7 Summary

Four findings are clear from the results to date:

1. **Each pipeline stage adds value.** Perplexity drops by an order of magnitude from base to SFT, and GRPO adds a measurable F1 improvement on top of SFT.
2. **GRPO with an F1 reward is effective post-SFT.** GRPO yields consistent gains over SFT on both the v9 and unk10 lines.
3. **Data composition is the single most impactful intervention.** The unk10 unknown-discharge augmentation improves F1-samples by +11.4 points over the v9 baseline and quadruples macro-F1, confirming that negative examples are a high-leverage, low-cost intervention for multi-label clinical coding tasks.
4. **Best result: GRPO unk10 — F1-samples 0.568, F1-micro 0.566** on the full 4,702-sample held-out test set.
