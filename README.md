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

- **SMART-on-FHIR app** — launches from inside the EHR via the SMART App Launch / OAuth 2.0 flow, pulls the patient's radiology reports as FHIR `DiagnosticReport` resources, and renders the predicted codes for review. Launch verified against OpenEMR, the Epic on FHIR sandbox, and the Oracle Health (Cerner) sandbox.
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
