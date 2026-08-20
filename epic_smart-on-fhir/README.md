# SMARTClaims — Epic on FHIR sandbox app

The Epic sandbox version of the SMARTClaims app: static HTML + [fhirclient](https://github.com/smart-on-fhir/client-js) for the SMART App Launch flow, with predictions served by the same fine-tuned Qwen3-32B model on Modal. The primary OpenEMR version lives in [`../openemr_smart-on-fhir/`](../openemr_smart-on-fhir/).

```
epic_smart-on-fhir/
├── launch.html          EHR-launch entry point (fhirclient)
├── index.html           Redirect target: patient banner + prediction UI
└── modal_inference.py   Modal GPU endpoint with CORS headers (called from the browser)
```

## Model endpoint (Modal)

```bash
pip install modal && modal token new
modal deploy modal_inference.py
```

`modal deploy` prints the endpoint URL; paste it into the `MODAL_ENDPOINT` constant at the top of the `<script>` block in `index.html`. The container serves `lalithapranathipulavarthy/smartclaims-grpo-unk10` (Qwen3-32B after CPT → SFT → GRPO, see the root README) on one H100 and scales to zero after 5 minutes idle. First request after a cold start takes ~60–90 s; subsequent requests ~3 s. `modal app stop smartclaims-cpt-predictor-epic` tears it down.

## Setup

Serve this folder with any static web server (the thesis demo used `http://localhost:3000/app/`). Register a non-production app at [fhir.epic.com](https://fhir.epic.com) with launch URL `.../launch.html`, redirect URI `.../index.html`, Clinicians audience, and the scopes in `launch.html`; paste the client id into `EPIC_CONFIG.clientId`. Test via Epic's Launch Pad.

## Sample report

The Epic sandbox does not expose radiology report text as FHIR `Binary`, so `index.html` carries a built-in **synthetic** sample report (CT head + CTA head/neck, written for the demo, not taken from any dataset) to exercise the model end to end. The patient banner is still read live from the sandbox, and the UI scores the prediction against the sample's codes (precision / recall / F1).
