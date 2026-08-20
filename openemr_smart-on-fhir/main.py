"""SMARTClaims - SMART on FHIR app for CPT code prediction (OpenEMR).

Single-file FastAPI app. Registered in OpenEMR as a confidential client
(EHR launch, PKCE); fetches the patient's radiology DocumentReference/Binary
and sends the report text to the Modal inference endpoint.

Configuration (environment variables):
    SMARTCLAIMS_CLIENT_ID       OpenEMR OAuth2 client id
    SMARTCLAIMS_CLIENT_SECRET   OpenEMR OAuth2 client secret
    SMARTCLAIMS_APP_BASE        public base URL, e.g. https://<host>/smartclaims
    MODAL_ENDPOINT              URL of the deployed modal_inference.py endpoint
    SMARTCLAIMS_VERIFY_TLS      "true" to verify the EHR's TLS cert (default false,
                                for self-signed OpenEMR installs)

Run:
    uvicorn main:app --host 127.0.0.1 --port 8001
(reverse-proxied at /smartclaims; see app/README.md)

Optional mapping.json (built by seed_radiology.py, not included) maps report
titles to ground-truth codes so the UI can show precision/recall/F1.
"""
import base64
import hashlib
import json
import os
import secrets
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware

CLIENT_ID = os.environ["SMARTCLAIMS_CLIENT_ID"]
CLIENT_SECRET = os.environ["SMARTCLAIMS_CLIENT_SECRET"]
APP_BASE = os.environ.get("SMARTCLAIMS_APP_BASE", "http://localhost:8001/smartclaims").rstrip("/")
REDIRECT_URI = f"{APP_BASE}/callback"
MODAL_ENDPOINT = os.environ["MODAL_ENDPOINT"]
VERIFY_TLS = os.environ.get("SMARTCLAIMS_VERIFY_TLS", "false").lower() == "true"

SCOPES = ("openid fhirUser launch launch/patient api:oemr "
          "user/document.write user/document.read "
          "patient/Binary.read patient/Patient.read patient/Encounter.read "
          "patient/DocumentReference.read patient/Condition.read "
          "patient/Procedure.read patient/Observation.read")

MAPPING_FILE = Path(__file__).parent / "mapping.json"
MAPPING_BY_TITLE = {}
if MAPPING_FILE.exists():
    try:
        with open(MAPPING_FILE) as f:
            for entry in json.load(f):
                MAPPING_BY_TITLE[f"{entry['report_id']}.txt"] = entry
    except Exception as e:
        print(f"WARNING: could not load mapping.json: {e}")

app = FastAPI(root_path="/smartclaims")
app.add_middleware(SessionMiddleware, secret_key=secrets.token_urlsafe(32))


# =========================================================================
# STYLES — clean clinical aesthetic, big readable, source on right
# =========================================================================
CSS = """<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, "SF Pro Text", "Segoe UI", system-ui, sans-serif;
  background: #f1f5f9;
  color: #0f172a;
  font-size: 14px;
  line-height: 1.5;
  min-height: 100vh;
  -webkit-font-smoothing: antialiased;
}

/* === Header strip === */
.topbar {
  background: #faf5ff;
  color: #fff;
  padding: 22px 36px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.tb-left { display: flex; align-items: center; gap: 14px; }
.tb-mark {
  width: 44px; height: 44px;
  color: #fef7ed;
  background: #4c1d95;
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-weight: 800; font-size: 13px; letter-spacing: -0.02em;
}
.tb-name { font-size: 22px; font-weight: 700; letter-spacing: -0.01em; line-height: 1.2; color: #0f172a; }
.tb-sub { font-size: 13px; color: #475569; margin-top: 2px; }

.tb-right { display: flex; gap: 28px; align-items: center; }
.tb-pt { display: flex; flex-direction: column; gap: 2px; align-items: flex-end; }
.tb-key { font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: #64748b; }
.tb-val { font-size: 15.5px; font-weight: 700; color: #0f172a; font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }

/* === Pill tabs === */
.pill-bar {
  background: #fff;
  padding: 18px 36px;
  display: flex;
  gap: 10px;
  border-bottom: 1px solid #e2e8f0;
}
.pill {
  padding: 11px 22px;
  font-size: 13.5px;
  font-weight: 600;
  color: #475569;
  background: #faf5ff;
  border: 1px solid #e2e8f0;
  border-radius: 999px;
  cursor: pointer;
  transition: all 0.15s;
  font-family: inherit;
  letter-spacing: -0.01em;
}
.pill.active { background: #1e3a8a; color: #fff; border-color: #1c4960; }
.pill:disabled { opacity: 0.5; cursor: not-allowed; }
.pill:hover:not(.active):not(:disabled) { background: #e2e8f0; }

/* === Page shell === */
.shell { padding: 20px 28px 40px; max-width: 1600px; margin: 0 auto; }

/* === Section card === */
.section {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  margin-bottom: 18px;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}
.sec-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 24px;
  border-bottom: 1px solid #e2e8f0;
  background: #fff;
}
.sec-title {
  font-size: 19px;
  font-weight: 700;
  color: #0f172a;
  letter-spacing: -0.02em;
}
.sec-meta { font-size: 12.5px; color: #64748b; margin-top: 3px; }
.bench-pill {
  display: none;
  background: #dbeafe;
  color: #0067b8;
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.06em;
  padding: 4px 10px;
  border-radius: 4px;
  margin-left: 10px;
  vertical-align: 2px;
}

.analyze-btn {
  background: #1e3a8a;
  color: #fff;
  border: none;
  padding: 11px 22px;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  transition: background 0.15s;
  font-family: inherit;
}
.analyze-btn:hover:not(:disabled) { background: #3b1377;}
.analyze-btn:disabled { opacity: 0.7; cursor: not-allowed; }

.status-pill {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 5px 12px;
  border-radius: 999px;
  background: #fef3c7;
  color: #92400e;
}
.status-pill.running { background: #dbeafe; color: #1e40af; }
.status-pill.success { background: #d1fae5; color: #065f46; }
.status-pill.error { background: #fee2e2; color: #991b1b; }

/* === Two-pane layout: analysis LEFT (compact), report RIGHT (wide) === */
.pane {
  display: grid;
  grid-template-columns: 440px 1fr;
  min-height: 540px;
}
.pane-left {
  padding: 22px;
  background: #f8fafc;
  border-right: 1px solid #e2e8f0;
  display: flex;
  flex-direction: column;
  gap: 18px;
  overflow-y: auto;
  max-height: 720px;
}
.pane-right {
  padding: 22px 28px;
  background: #fff;
  display: flex;
  flex-direction: column;
  min-height: 540px;
  max-height: 720px;
}

/* === Block in left pane === */
.block-title {
  font-size: 16px;
  font-weight: 700;
  color: #0f172a;
  margin-bottom: 14px;
  letter-spacing: -0.01em;
  display: flex;
  align-items: center;
  gap: 8px;
}
.block-dot {
  width: 9px; height: 9px;
  background: #7c3aed;
  border-radius: 50%;
}

/* === Big bold CPT chips === */
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
  font-family: "SF Mono", Menlo, monospace;
  font-size: 16px;
  font-weight: 700;
  padding: 11px 18px;
  border-radius: 7px;
  background: #1e3a8a;
  color: #fff;
  letter-spacing: 0.02em;
  border: 2px solid #1c4960;
}
.chip.pending {
  background: #fff;
  color: #cbd5e1;
  border: 2px dashed #cbd5e1;
  font-weight: 600;
}
.chip.gt {
  background: #fff;
  color: #334155;
  border: 2px solid #cbd5e1;
}
.chip.gt.correct {
  background: #d1fae5;
  color: #065f46;
  border-color: #10b981;
}
.chip.gt.missed {
  background: #fef2f2;
  color: #991b1b;
  border: 2px dashed #ef4444;
}
.chip.correct {
  background: #059669;
  color: #fff;
  border-color: #059669;
}
.chip.extra {
  background: #fef3c7;
  color: #92400e;
  border-color: #fde68a;
}

.raw-line {
  font-size: 12px;
  color: #64748b;
  margin-top: 12px;
  word-break: break-word;
}
.raw-line code {
  background: #fff;
  border: 1px solid #e2e8f0;
  padding: 4px 9px;
  border-radius: 4px;
  font-size: 11.5px;
  color: #334155;
  font-family: "SF Mono", Menlo, monospace;
}
.raw-output.empty { color: #94a3b8; font-style: italic; }

/* === Metric grid === */
.metrics {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.metric {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 9px 8px;
  text-align: center;
}
.metric-val {
  font-size: 17px;
  font-weight: 800;
  color: #0f172a;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.02em;
  line-height: 1.1;
}
.metric-val.empty { color: #cbd5e1; }
.metric-key {
  font-size: 9.5px;
  color: #64748b;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin-top: 4px;
}

/* === Legend === */
.legend {
  display: flex;
  gap: 14px;
  margin-top: 12px;
  font-size: 11.5px;
  color: #64748b;
  flex-wrap: wrap;
  align-items: center;
}
.legend .chip { font-size: 11px; padding: 4px 10px; }

/* === Right pane: source report === */
.doc-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding-bottom: 14px;
  margin-bottom: 14px;
  border-bottom: 1px solid #e2e8f0;
}
.doc-label {
  font-size: 14px;
  font-weight: 700;
  color: #0f172a;
  letter-spacing: -0.01em;
}
.doc-meta { font-size: 12px; color: #94a3b8; font-variant-numeric: tabular-nums; }
.doc-content {
  flex: 1;
  overflow-y: auto;
  font-family: "SF Mono", Menlo, monospace;
  font-size: 13px;
  line-height: 1.7;
  white-space: pre-wrap;
  color: #1e293b;
  padding: 18px 20px;
  background: #fafbfc;
  border: 1px solid #e2e8f0;
  border-radius: 7px;
}
.doc-content.empty {
  color: #94a3b8;
  font-style: italic;
  font-family: inherit;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  font-size: 13px;
}

/* === Spinner / banners === */
.spinner {
  display: inline-block;
  width: 14px; height: 14px;
  border: 2px solid rgba(255,255,255,0.4);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
  vertical-align: middle;
}
@keyframes spin { to { transform: rotate(360deg); } }
.banner {
  display: flex; align-items: center; gap: 10px;
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  color: #1e40af;
  padding: 11px 14px;
  border-radius: 7px;
  font-size: 13px;
  margin-bottom: 12px;
}
.banner .spinner {
  border-color: rgba(30,64,175,0.25);
  border-top-color: #1e40af;
}
.banner.error {
  background: #fef2f2;
  border-color: #fecaca;
  color: #991b1b;
}

/* === Empty state === */
.empty-card {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 48px;
  text-align: center;
  color: #64748b;
}

/* === Back link === */
.back-link {
  display: inline-block;
  margin-top: 14px;
  color: #7c3aed;
  text-decoration: none;
  font-size: 13px;
  font-weight: 600;
}
.back-link:hover { text-decoration: underline; }
</style>"""


# =========================================================================
# RENDER FUNCTIONS
# =========================================================================
def render_topbar(patient_name: str, sex_short: str, age: str, mrn: str) -> str:
    return f"""<div class="topbar">
  <div class="tb-left">
    <div class="tb-mark">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M14 2H6C4.9 2 4 2.9 4 4V20C4 21.1 4.9 22 6 22H18C19.1 22 20 21.1 20 20V8L14 2Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
        <path d="M14 2V8H20" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
        <path d="M12 12V18M9 15H15" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
      </svg>
    </div>
    <div>
      <div class="tb-name">SMARTClaims</div>
      <div class="tb-sub">Clinical Decision Support · Radiology</div>
    </div>
  </div>
  <div class="tb-right">
    <div class="tb-pt"><span class="tb-key">Name</span><span class="tb-val">{patient_name}</span></div>
    <div class="tb-pt"><span class="tb-key">Sex</span><span class="tb-val">{sex_short}</span></div>
    <div class="tb-pt"><span class="tb-key">Age</span><span class="tb-val">{age}</span></div>
    <div class="tb-pt"><span class="tb-key">MRN</span><span class="tb-val">{mrn}</span></div>
  </div>
</div>
<div class="pill-bar">
  <button class="pill active">Radiology Reports</button>
  <button class="pill" disabled>Encounters</button>
  <button class="pill" disabled>Conditions</button>
  <button class="pill" disabled>Procedures</button>
  <button class="pill" disabled>Lab Results</button>
</div>"""


def render_report_section(doc_id: str, doc_title: str, doc_date: str,
                          has_benchmark: bool, ground_truth: list | None) -> str:
    benchmark_html = '<span class="bench-pill">BENCHMARK</span>' if has_benchmark else ''

    gt_section = ""
    if ground_truth:
        gt_chips = "".join(
            f'<span class="chip gt" data-gt-code="{c}">{c}</span>'
            for c in ground_truth
        )
        gt_section = f"""
      <div>
        <div class="block-title"><span class="block-dot"></span>Billed Codes <span style="color:#94a3b8;font-weight:500;font-size:11.5px;margin-left:4px">ground truth</span></div>
        <div class="chips" data-gt-row>{gt_chips}</div>
        <div class="legend">
          <span class="chip gt correct">✓</span><span>matches AI</span>
          <span class="chip gt missed">✗</span><span>missed</span>
        </div>
      </div>"""

    return f"""<div class="section wrap" data-doc-id="{doc_id}">
  <div class="sec-head">
    <div>
      <div class="sec-title">Radiology Report{benchmark_html}</div>
      <div class="sec-meta">{doc_date} · Order #{doc_id[:6]}</div>
    </div>
    <div style="display:flex;gap:14px;align-items:center">
      <span class="status-pill" data-status-pill>Awaiting</span>
      <button class="analyze-btn" onclick="analyze(this, '{doc_id}')">Analyze with AI →</button>
    </div>
  </div>

  <div class="pane">
    <aside class="pane-left">
      <div class="status-area"></div>

      <div>
        <div class="block-title"><span class="block-dot"></span>Predicted CPT Codes</div>
        <div class="chips" data-codes-row="predicted">
          <span class="chip pending">— — —</span>
          <span class="chip pending">— — —</span>
          <span class="chip pending">— — —</span>
        </div>
        <div class="raw-line">Output: <code class="raw-output empty" data-raw-output>awaiting</code></div>
      </div>

      <div>
        <div class="block-title"><span class="block-dot"></span>Performance</div>
        <div class="metrics">
          <div class="metric"><div class="metric-val empty" data-metric="precision">—</div><div class="metric-key">Precision</div></div>
          <div class="metric"><div class="metric-val empty" data-metric="recall">—</div><div class="metric-key">Recall</div></div>
          <div class="metric"><div class="metric-val empty" data-metric="f1">—</div><div class="metric-key">F1 Score</div></div>
          <div class="metric"><div class="metric-val empty" data-metric="correct">—</div><div class="metric-key">Match</div></div>
        </div>
      </div>

      {gt_section}
    </aside>

    <main class="pane-right">
      <div class="doc-head">
        <span class="doc-label">Radiology Report</span>
        <span class="doc-meta">{doc_date}</span>
      </div>
      <div class="doc-content empty" data-report-content>Click <strong>Analyze with AI</strong> to load the full report and run inference.</div>
    </main>
  </div>
</div>"""


JS_SCRIPT = """<script>
async function analyze(btn, docId) {
  const card = btn.closest('.wrap');
  const statusArea = card.querySelector('.status-area');
  const statusPill = card.querySelector('[data-status-pill]');

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Analyzing...';
  statusPill.className = 'status-pill running';
  statusPill.textContent = 'Running';
  statusArea.innerHTML = `<div class="banner"><span class="spinner"></span>
    <div>Running inference on Qwen3-32B GRPO via Modal GPU...</div></div>`;

  try {
    const resp = await fetch(`./analyze/${docId}`, { method: 'POST' });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'Analysis failed');

    // Predicted codes
    const pRow = card.querySelector('[data-codes-row="predicted"]');
    pRow.innerHTML = '';
    const gtCodes = Array.from(card.querySelectorAll('[data-gt-code]')).map(e => e.dataset.gtCode);
    (data.predicted_codes || []).forEach(code => {
      const chip = document.createElement('span');
      chip.className = 'chip ' + (gtCodes.includes(code) ? 'correct' : 'extra');
      chip.textContent = code;
      pRow.appendChild(chip);
    });

    // Ground truth chips
    if (gtCodes.length) {
      card.querySelectorAll('[data-gt-code]').forEach(chip => {
        chip.classList.remove('correct', 'missed');
        chip.classList.add((data.predicted_codes || []).includes(chip.dataset.gtCode) ? 'correct' : 'missed');
      });
    }

    // Raw output
    const raw = card.querySelector('[data-raw-output]');
    raw.classList.remove('empty');
    raw.textContent = data.raw_output || '(empty)';

    // Metrics
    const m = data.metrics || {};
    const fmt = (v, d=3) => (typeof v === 'number' ? v.toFixed(d) : '—');
    const setMetric = (key, val) => {
      const el = card.querySelector(`[data-metric="${key}"]`);
      if (!el) return;
      el.textContent = val;
      el.classList.remove('empty');
    };
    setMetric('precision', fmt(m.precision));
    setMetric('recall',    fmt(m.recall));
    setMetric('f1',        fmt(m.f1));
    setMetric('correct',   typeof m.correct === 'number' ? `${m.correct}/${m.total || 0}` : '—');

    // Source doc
    const rc = card.querySelector('[data-report-content]');
    rc.classList.remove('empty');
    rc.textContent = data.report_text || '(no text)';

    // Done
    statusArea.innerHTML = '';
    statusPill.className = 'status-pill success';
    statusPill.textContent = '✓ Analyzed';
    btn.innerHTML = 'Re-analyze ↻';
    btn.disabled = false;

  } catch (err) {
    statusArea.innerHTML = `<div class="banner error">⚠️ ${err.message}</div>`;
    statusPill.className = 'status-pill error';
    statusPill.textContent = 'Error';
    btn.disabled = false;
    btn.innerHTML = 'Retry →';
  }
}
</script>"""


# =========================================================================
# OAUTH FLOW
# =========================================================================
@app.get("/")
async def root():
    return {"status": "SMARTClaims app is running", "launch_url": f"{APP_BASE}/launch"}


@app.get("/launch")
async def launch(request: Request, iss: str, launch: str | None = None):
    code_verifier = secrets.token_urlsafe(64)
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).decode().rstrip("=")
    state = secrets.token_urlsafe(16)

    request.session["code_verifier"] = code_verifier
    request.session["state"] = state
    request.session["iss"] = iss

    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "state": state,
        "aud": iss,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    if launch:
        params["launch"] = launch

    auth_url = f"{iss.rstrip('/').replace('/apis/default/fhir', '')}/oauth2/default/authorize?" + urlencode(params)
    return RedirectResponse(auth_url)


@app.get("/callback")
async def callback(request: Request, code: str, state: str):
    if state != request.session.get("state"):
        raise HTTPException(status_code=400, detail="state mismatch")
    iss = request.session["iss"]
    token_url = iss.rstrip("/").replace("/apis/default/fhir", "") + "/oauth2/default/token"
    async with httpx.AsyncClient(verify=VERIFY_TLS, timeout=120) as client:
        tok = await client.post(token_url, data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "code_verifier": request.session["code_verifier"],
        })
    tok.raise_for_status()
    token_data = tok.json()
    request.session["access_token"] = token_data["access_token"]
    request.session["patient_id"]   = token_data.get("patient")
    return RedirectResponse(f"{APP_BASE}/app")


# =========================================================================
# MAIN APP PAGE
# =========================================================================
@app.get("/app", response_class=HTMLResponse)
async def app_page(request: Request):
    token = request.session.get("access_token")
    patient_id = request.session.get("patient_id")
    iss = request.session.get("iss")
    if not token or not patient_id:
        return RedirectResponse(f"{APP_BASE}/")

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/fhir+json"}
    async with httpx.AsyncClient(verify=VERIFY_TLS, timeout=120) as client:
        p = await client.get(f"{iss}/Patient/{patient_id}", headers=headers)
        patient = p.json()
        docs = await client.get(f"{iss}/DocumentReference?patient={patient_id}", headers=headers)
        doc_bundle = docs.json()

    # Patient name
    name = "Unknown"
    if patient.get("name"):
        n0 = patient["name"][0]
        name = f"{(n0.get('given') or [''])[0]} {n0.get('family','')}".strip()

    # Sex / Age
    gender = patient.get("gender", "—")
    sex_short = "M" if gender.lower().startswith("m") else ("F" if gender.lower().startswith("f") else "—")
    dob = patient.get("birthDate", "—")
    age = "—"
    try:
        birth = datetime.strptime(dob, "%Y-%m-%d")
        age = str((datetime.now() - birth).days // 365)
    except Exception:
        pass

    mrn = patient_id[:10] if patient_id else "—"

    # Reports
    entries = doc_bundle.get("entry", []) or []
    report_blocks = []
    for e in entries:
        doc = e.get("resource", {})
        doc_id = doc.get("id", "")
        doc_title = (doc.get("content", [{}])[0].get("attachment", {}).get("title")
                     or doc.get("description") or "Untitled Report")
        doc_date = (doc.get("date", "") or "")[:10]
        mapping = MAPPING_BY_TITLE.get(doc_title)
        gt = mapping.get("ground_truth_codes") if mapping else None
        report_blocks.append(render_report_section(doc_id, doc_title, doc_date, bool(gt), gt))

    # Limit to first report only — patient typically has one current report shown
    if report_blocks:
        report_blocks = report_blocks[:1]

    body_content = ''.join(report_blocks) if report_blocks else \
        '<div class="empty-card">No radiology reports found for this patient.</div>'

    html = f"""<!DOCTYPE html><html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SMARTClaims · {name}</title>{CSS}</head>
<body>
{render_topbar(name, sex_short, age, mrn)}
<div class="shell">
{body_content}
<a href="javascript:history.back()" class="back-link">← Back to OpenEMR</a>
</div>
{JS_SCRIPT}
</body></html>"""
    return HTMLResponse(html)


# =========================================================================
# ANALYZE — runs Modal GPU inference
# =========================================================================
@app.post("/analyze/{doc_id}")
async def analyze(doc_id: str, request: Request):
    token = request.session.get("access_token")
    iss = request.session.get("iss")
    if not token or not iss:
        raise HTTPException(status_code=401, detail="Not authenticated")
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(verify=VERIFY_TLS, timeout=600) as client:
        doc_resp = await client.get(f"{iss}/DocumentReference/{doc_id}", headers=headers)
        doc = doc_resp.json()
        doc_title = (doc.get("content", [{}])[0].get("attachment", {}).get("title")
                     or "Untitled")
        bin_url = doc["content"][0]["attachment"]["url"]
        if not bin_url.startswith("http"):
            bin_url = f"{iss}/{bin_url.lstrip('/')}"
        bin_resp = await client.get(bin_url, headers={**headers, "Accept": "application/fhir+json"})
        bin_text = bin_resp.text
        try:
            bin_resp.json()
            report_text = bin_text
        except Exception:
            report_text = bin_text

        t0 = time.time()
        modal_resp = await client.post(MODAL_ENDPOINT, json={"report_text": report_text[:3000]})
        inference_ms = int((time.time() - t0) * 1000)
        modal_resp.raise_for_status()
        modal_data = modal_resp.json()

    predicted_codes = modal_data.get("codes", [])
    raw_output = modal_data.get("raw", "")

    mapping = MAPPING_BY_TITLE.get(doc_title)
    metrics = {}
    if mapping and mapping.get("ground_truth_codes"):
        gt = set(mapping["ground_truth_codes"])
        pred = set(predicted_codes)
        tp = len(gt & pred)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(gt) if gt else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
        metrics = {"precision": precision, "recall": recall, "f1": f1,
                   "correct": tp, "total": len(gt)}

    return JSONResponse({
        "predicted_codes": predicted_codes,
        "raw_output": raw_output,
        "metrics": metrics,
        "inference_ms": inference_ms,
        "report_text": report_text,
    })
