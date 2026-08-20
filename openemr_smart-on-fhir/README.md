# SMARTClaims — OpenEMR SMART-on-FHIR app

The primary version of the SMARTClaims app (development and thesis demo). A SMART App Launch app that opens from an OpenEMR patient chart, pulls the patient's radiology report over FHIR, sends the text to the fine-tuned Qwen3-32B model running on Modal, and shows the predicted CPT codes for review.

```
openemr_smart-on-fhir/
├── main.py              Single-file FastAPI app: EHR launch, OAuth2 + PKCE,
│                        DocumentReference/Binary fetch, Modal call, results UI
├── modal_inference.py   Modal GPU endpoint serving the merged model
├── seed_radiology.py    Loads demo reports into OpenEMR patients (see Data below)
└── requirements.txt
```

The app posts `{"report_text": ...}` to the model endpoint and receives `{"codes": [...], "raw": "...", "inference_ms": ...}`; when ground truth is known (via `mapping.json`) the UI shows precision / recall / F1. The Epic sandbox variant lives in [`../epic_smart-on-fhir/`](../epic_smart-on-fhir/).

## Model endpoint (Modal)

```bash
pip install modal && modal token new
modal deploy modal_inference.py
```

`modal deploy` prints the endpoint URL; put it in `MODAL_ENDPOINT`. The container serves `lalithapranathipulavarthy/smartclaims-grpo-unk10` (Qwen3-32B after CPT → SFT → GRPO, see the root README) on one H100, loads weights from a Modal volume cache, and scales to zero after 5 minutes idle. First request after a cold start takes ~60–90 s; subsequent requests ~3 s. `modal app stop smartclaims-cpt-predictor` tears it down.

## Setup

1. **Register the app** in OpenEMR (Administration → System → API Clients, or the `/oauth2/default/registration` endpoint) as a confidential client with redirect URI `https://<host>/smartclaims/callback` and the scopes listed in `main.py` (`openid fhirUser launch launch/patient api:oemr user/document.* patient/*.read`). Enable it and note the client id/secret.
2. **Configure and run**:

   ```bash
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

## Data

`seed_radiology.py` populates an OpenEMR instance with 100 radiology reports from the held-out test set so the demo can score predictions against ground truth, and writes `mapping.json` (report → patient/document ids + codes) which `main.py` reads if present. The input `demo_reports.json` and the resulting `mapping.json` are derived from MIMIC-IV and are **not included**; build them from your own credentialed copy (`{report_id, hadm_id, report_text, ground_truth_codes}` per report). The seed script needs `OPENEMR_BASE`, `OPENEMR_ACCESS_TOKEN`, `OPENEMR_DB_ROOT_PASSWORD`, and `OPENEMR_COMPOSE_DIR` in the environment.
