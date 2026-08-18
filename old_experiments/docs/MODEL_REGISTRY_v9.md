# SmartClaims Model Registry

## Pipeline: CPT Pretraining → SFT → GRPO

---

## Active Models (Use These)

### Base + CPT
| Directory | Description |
|-----------|-------------|
| `qwen3-32b-mimic-cpt-200k-ep2` | CPT epoch 2 on 200k MIMIC discharge notes — base for SFT |

### SFT
| Directory | Description | F1-Samples |
|-----------|-------------|-----------|
| `qwen3-32b-sft-v9` | SFT v9 LoRA adapter only | — |
| `qwen3-32b-sft-v9-merged` | SFT v9 merged (CPT+SFT baked in) — **input to GRPO** | 0.44 (4702 samples) |

### GRPO
| Directory | Description | Training Reward | Status |
|-----------|-------------|----------------|--------|
| `qwen3-32b-grpo-v3b` | GRPO v3b — 300 steps, full run on grpo_v9 data | ~0.42-0.49 | **Being evaluated now** |
| `grpo_v3_out/checkpoint-150` | GRPO v3 — 150 steps (interrupted run), hit reward 0.50 at step 145 | 0.50 | **Fallback if v3b is bad** |

---

## Old Models (Do Not Use)

| Directory | Description | F1-Samples |
|-----------|-------------|-----------|
| `qwen3-32b-grpo-v2` | GRPO v2 — old small dataset (1,815 samples) | 0.224 |
| `qwen3-32b-sft-v8-merged` | SFT v8 merged — old small dataset | 0.078 |
| `qwen3-32b-mimic-sft-merged` | Very early SFT — ignore | 0.0 |

---

## Datasets (v9)

| Directory | Description |
|-----------|-------------|
| `sft_v9_train_dataset` | 21,155 examples (80% radiology CPT, 20% unknown discharge) |
| `sft_v9_val_dataset` | 2,351 examples |
| `grpo_v9_train_dataset` | 16,924 examples (radiology only, no unknowns) |
| `grpo_v9_val_dataset` | 1,881 examples |
| `sft_test_inputs.csv` | 4,702 test admissions — NO leakage confirmed |
| `sft_test_labels.csv` | Ground truth CPT codes for test set |

---

## Eval Results

| Run | Model | N samples | F1-Samples | F1-Micro | Notes |
|-----|-------|-----------|-----------|----------|-------|
| SFT v8 | `qwen3-32b-sft-v8-merged` | 400 | 0.0780 | 0.0663 | Old small dataset |
| GRPO v2 | `qwen3-32b-grpo-v2` | 400 | 0.2237 | 0.1799 | Old small dataset |
| SFT v9 (500) | `qwen3-32b-sft-v9-merged` | 500 | 0.4386 | 0.4006 | |
| SFT v9 (full) | `qwen3-32b-sft-v9-merged` | 4702 | 0.4419 | 0.3990 | **Best SFT** |
| GRPO v3b (500) | `qwen3-32b-grpo-v3b` | 500 | TBD | — | Running now |
| GRPO v3b (full) | `qwen3-32b-grpo-v3b` | 4702 | TBD | — | Running now |

---

## Eval Script Usage

```bash
# SFT eval (500 samples, GPU 7)
python eval_sft_grpo.py --mode sft --n-eval 500

# GRPO eval (full test set, GPU 7)
python eval_sft_grpo.py --mode grpo

# To eval checkpoint-150 fallback: change in eval_sft_grpo.py:
# GRPO_ADAPTER = "grpo_v3_out/checkpoint-150"
```
