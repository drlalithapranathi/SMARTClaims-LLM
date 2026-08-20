# SMARTClaims SMART-on-FHIR app

The clinician-facing half of SMARTClaims: a SMART App Launch app that opens from a patient chart, pulls the patient's radiology report over FHIR, sends the text to the fine-tuned Qwen3-32B model running on Modal, and shows the predicted CPT codes for review. Two variants were built, one per EHR used for the thesis demos.

```
app/
├── openemr/                 Primary version (development + thesis demo)
│   ├── main.py              Single-file FastAPI app: EHR launch, OAuth2 + PKCE,
│   │                        DocumentReference/Binary fetch, Modal call, results UI
│   ├── modal_inference.py   Modal GPU endpoint serving the merged model
│   ├── seed_radiology.py    Loads demo reports into OpenEMR patients (see Data below)
│   └── requirements.txt
└── epic/                    Epic on FHIR sandbox version
    ├── launch.html          EHR-launch entry point (fhirclient)
    ├── index.html           Redirect target: patient banner + prediction UI
    └── modal_inference.py   Same endpoint with CORS headers (called from the browser)
```

Both variants call the same model: `lalithapranathipulavarthy/smartclaims-grpo-unk10` on Hugging Face (Qwen3-32B after CPT → SFT → GRPO, see the repository root README). The app posts `{"report_text": ...}` and receives `{"codes": [...], "raw": "...", "inference_ms": ...}`; the UI compares predictions against ground truth when it is known and shows precision / recall / F1.

## Model endpoint (Modal)

```bash
pip install modal && modal token new
cd app/openemr            # or app/epic
modal deploy modal_inference.py
```

`modal deploy` prints the endpoint URL; put it in `MODAL_ENDPOINT` (OpenEMR) or in the `MODAL_ENDPOINT` constant at the top of the `<script>` block in `epic/index.html`. The container runs on one H100, loads the model from a Modal volume cache, and scales to zero after 5 minutes idle. First request after a cold start takes ~60–90 s; subsequent requests ~3 s. `modal app stop <app-name>` tears it down.

## OpenEMR version

1. **Register the app** in OpenEMR (Administration → System → API Clients, or the `/oauth2/default/registration` endpoint) as a confidential client with redirect URI `https://<host>/smartclaims/callback` and the scopes listed in `main.py` (`openid fhirUser launch launch/patient api:oemr user/document.* patient/*.read`). Enable it and note the client id/secret.
2. **Configure and run**:

   ```bash
   cd app/openemr
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   export SMARTCLAIMS_CLIENT_ID=...
   export SMARTCLAIMS_CLIENT_SECRET=...
   export SMARTCLAIMS_APP_BASE=https://<host>/smartclaims
   export MODAL_ENDPOINT=https://<workspace>--smartclaims-cpt-predictor-cptpredictor-predict.modal.run
   uvicorn main:app --host 127.0.0.1 --port 8001
   ```

   Set `SMARTCLAIMS_VERIFY_TLS=true` unless the OpenEMR instance uses a self-signed certificate.
3. **Reverse proxy** under the same host as OpenEMR (nginx):

   ```nginx
   location /smartclaims/ {
       proxy_pass http://localhost:8001/;
       proxy_set_header Host $host;
       proxy_set_header X-Forwarded-Proto https;
       proxy_set_header X-Forwarded-Prefix /smartclaims;
   }
   ```
4. Launch from a patient chart: OpenEMR calls `/smartclaims/launch?iss=...&launch=...`; after authorization the app lands on `/smartclaims/app`, lists the patient's radiology `DocumentReference`s, and **Predict CPT Codes** posts the report `Binary` to Modal.

## Epic sandbox version

Static files — serve `app/epic/` with any web server (the thesis demo used `http://localhost:3000/app/`). Register a non-production app at [fhir.epic.com](https://fhir.epic.com) with launch URL `.../launch.html`, redirect URI `.../index.html`, Clinicians audience, and the scopes in `launch.html`; paste the client id into `EPIC_CONFIG.clientId`. Test via Epic's Launch Pad.

The Epic sandbox does not expose radiology report text as FHIR `Binary`, so `index.html` carries a built-in **synthetic** sample report (CT head + CTA head/neck, written for the demo) to exercise the model end to end; the patient banner is still read live from the sandbox.

## Data

`seed_radiology.py` populates an OpenEMR instance with 100 radiology reports from the held-out test set so the demo can score predictions against ground truth, and writes `mapping.json` (report → patient/document ids + codes) which `main.py` reads if present. The input `demo_reports.json` and the resulting `mapping.json` are derived from MIMIC-IV and are **not included**; build them from your own credentialed copy (`{report_id, hadm_id, report_text, ground_truth_codes}` per report). The seed script needs `OPENEMR_BASE`, `OPENEMR_ACCESS_TOKEN`, `OPENEMR_DB_ROOT_PASSWORD`, and `OPENEMR_COMPOSE_DIR` in the environment.
