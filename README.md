# SMARTClaims

**Automated radiology CPT-code prediction from clinical notes, delivered at the point of care as a SMART-on-FHIR app.**

SMARTClaims adapts a 32B-parameter large language model to read free-text radiology documentation and output the billable CPT codes for an admission — a multi-label task spanning hundreds of procedure codes. The model is exposed through a standards-based clinical app that launches inside any SMART-compatible EHR (Epic, Oracle Health/Cerner, OpenEMR), so coders get recommendations inside their existing workflow rather than a separate tool.

Manual coding is slow and error-prone (~1 in 5 claims is processed incorrectly), and off-the-shelf LLMs score below 50% on it without domain adaptation. SMARTClaims closes that gap: **Qwen3-32B**, fine-tuned on **MIMIC-IV** through a three-stage pipeline and served via serverless GPU inference.

> Master's thesis project, M.S. Health Informatics, Indiana University Indianapolis (2026).

## Architecture

At inference time, the app lives inside the clinician's EHR and never moves patient data outside the FHIR exchange:

```
EHR  (Epic · Cerner · OpenEMR)
   │
   ▼  launch + OAuth 2.0
SMART-on-FHIR app
   │
   ▼  fetch radiology reports  (FHIR DiagnosticReport)
   │
   ▼  send report text
Modal serverless GPU inference   (Qwen3-32B, weights on Hugging Face)
   │
   ▼  predicted CPT codes
SMART-on-FHIR app  →  clinician reviews: accept / edit / reject
```

- **SMART-on-FHIR app** ([`app/`](app/)) — launches from inside the EHR via the SMART App Launch / OAuth 2.0 flow, pulls the patient's radiology reports as FHIR `DiagnosticReport` resources, and renders the predicted codes for review. Launch verified against OpenEMR, the Epic on FHIR sandbox, and the Oracle Health (Cerner) sandbox.
- **Serverless inference** — weights hosted on Hugging Face, GPUs provisioned on demand on Modal: pay-per-inference, no always-on infrastructure.
- **Human-in-the-loop** — codes are surfaced as recommendations for a qualified coder to accept, edit, or reject — never applied automatically.

The model takes a radiology report and returns a pipe-delimited list of CPT codes (e.g. `36569 | 75726 | 99152`), or `unknown` when no billable radiology procedure is present.

### Model pipeline

The model behind it is built in three stages, each adapting Qwen3-32B further toward the coding task:

```
Qwen3-32B
   │
   ▼  Stage 1 — Continued Pretraining   200k MIMIC-IV discharge summaries
   │            clinical-domain adaptation (next-token LM)
   ▼  Stage 2 — Supervised Fine-Tuning  radiology report → CPT-code pairs (QLoRA)
   │            + 10% "unknown" negatives so the model learns when NOT to code
   ▼  Stage 3 — GRPO                     reinforcement learning, F1 as the reward
   │            optimizes the eval metric directly instead of a proxy loss
   ▼
SMARTClaims model
```

Training uses QLoRA (4-bit NF4 + LoRA, rank 16), FlashAttention-2, and multi-GPU DDP via Hugging Face Accelerate. The highest-impact design choice was **data composition, not model tricks**: mixing in 10% "unknown" discharge notes (no billable radiology procedure) taught the model *when not to assign a code* — as important as teaching it which code to assign.

> All data comes from **MIMIC-IV**, available to credentialed researchers via [PhysioNet](https://physionet.org/content/mimiciv/) under a Data Use Agreement. **No patient data or derived datasets are included in this repository** — MIMIC-IV cannot be redistributed. To reproduce, obtain your own credentialed access and point the pipeline at your local copy.

## Results

Held-out test set: **4,702 admissions**, **264 distinct CPT codes** (MIMIC-IV, split at the admission level, no hadm_id overlap with training).

| Model | F1-samples | F1-micro | F1-macro |
|---|---:|---:|---:|
| Small-data baseline (SFT v8, n=400) | 0.078 | 0.066 | — |
| SFT, 20% unknown (v9) | 0.442 | 0.399 | 0.049 |
| SFT + GRPO (v3b, on v9) | 0.465 | 0.429 | 0.061 |
| SFT, 10% unknown, 2 epochs | 0.556 | 0.549 | 0.235 |
| **SFT 10% unknown + GRPO (final)** | **0.568** | **0.566** | 0.157 |

Per-token perplexity by stage: base 13.00 → +CPT 5.36 → +SFT 1.48 → +GRPO 1.49 (SFT/GRPO rows measured on the v9/v3b line; see `docs/RESULTS.md`). Top-20 most-frequent-code mean F1 for the final model: 0.634. On a mixed set of 450 radiology + 50 non-radiology discharge notes, the final model outputs `unknown` for 50/50 non-radiology notes. Full tables and per-code numbers are in [`docs/RESULTS.md`](docs/RESULTS.md).

## Repository layout

```
app/                    SMART-on-FHIR app — OpenEMR version (FastAPI, EHR launch, PKCE),
                        Epic-sandbox version (static HTML + fhirclient), Modal serving scripts
data_pipeline/          MIMIC-IV preprocessing: radiology-section removal, VSAC/UMLS
                        CPT validation + train/test split, SFT/GRPO dataset builders,
                        holdout note sampler
continued_pretraining/  Stage 1 — tokenize 200k discharge notes, CPT epochs 1 & 2, merge
sft/                    Stage 2 — SFT (final 10%-unknown run and the 20%-unknown v9 run), merge
grpo/                   Stage 3 — GRPO with F1 reward, merge + upload to Hugging Face
evaluation/             Test-set F1 (single-GPU and sharded), mixed radiology/unknown eval,
                        perplexity by stage
figures/                Scripts that render the results charts / poster / confusion matrix
docs/                   RESULTS.md — full results write-up
old_experiments/        Superseded runs kept for reference: early exam-name SFT, SFT v5–v8,
                        GRPO v3b on SFT v9, no-unknown ablation, CPT-stage BLEU/ROUGE eval,
                        model-parallel CPT notebook, GraphRAG/RadLex and MedGemma
                        prototypes, early SMART app frontend, v7/v9-era planning notes
```

## Reproducing the pipeline

Requires credentialed MIMIC-IV access (`discharge.csv`, `radiology.csv`, CPT billing tables), a UMLS API key (`export UMLS_API_KEY=...`), and multi-GPU hardware (SFT used 3 GPUs, GRPO 4; 32B model in 4-bit NF4). Scripts use relative paths and expect to be run from the directory that holds the model checkpoints and data CSVs (see each script's `CONFIG` block).

1. **Clean discharge notes** — `data_pipeline/radiology_removal_pipeline.py` with `SAMPLE_SIZE = 200000` (edit the constant in the script; its default `None` processes all notes; seed 42) strips imaging sections from a 200k-note sample → `cleaned_discharge_notes_200k_fixed.csv`. Run again on the remaining notes (`data_pipeline/make_holdout.py` → `holdout_discharge_notes.csv`) to build `cleaned_holdout_notes.csv` for perplexity and unknown-note evaluation.
2. **Build the CPT dataset** — `data_pipeline/sft_cpt_vsac_pipeline.py` validates codes against VSAC/UMLS, drops add-on/invalid codes, caps at 6 codes per admission, splits 80/20 by hadm_id and asserts no leakage. Rename its outputs to `sft_train.csv`, `sft_test_inputs.csv`, `sft_test_labels.csv` for the training/eval scripts.
3. **Build training sets** — `data_pipeline/build_datasets.py` → `unk10_sft_*` / `unk10_grpo_*` (final, 10% unknown). `build_sft_grpo_datasets.py` is the earlier 20%-unknown v9 builder.
4. **Stage 1 (CPT)** — `continued_pretraining/preprocess.py` → `train_cpt.py` → `train_cpt_ep2.py` → `merge_cpt.py` → `qwen3-32b-mimic-cpt-merged`.
5. **Stage 2 (SFT)** — `sft/train_sft.py` (2 epochs, lr 2e-5, LoRA r=16) → `sft/merge_unk10.py` → `unk10_sft_merged`.
6. **Stage 3 (GRPO)** — `grpo/train_grpo.py` (300 steps, F1 reward) → `unk10_grpo_adapter`; `grpo/merge_and_upload_grpo.py` merges and pushes the standalone model to Hugging Face.
7. **Evaluate** — `evaluation/eval_test_set.py --mode sft|grpo` (or `eval_grpo_sharded.py` + `merge_eval_shards.py` across GPUs), `eval_mixed_radiology_unknown.py`, `eval_perplexity_cpt.py`, `eval_perplexity_stages.py`.
