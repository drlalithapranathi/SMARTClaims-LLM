"""
Seed OpenEMR with radiology reports from the GRPO test set.
Uses docker-compose to query MariaDB (port not exposed to host).

Input demo_reports.json (not included -- derived from MIMIC-IV, build your own
from credentialed data): a list of {report_id, hadm_id, report_text, ground_truth_codes}.
Output mapping.json: report -> OpenEMR patient/document ids + ground truth, read by main.py.

Environment variables:
    OPENEMR_BASE             e.g. https://<host>/apis/default
    OPENEMR_ACCESS_TOKEN     fresh OAuth2 access token with user/document.write
    OPENEMR_DB_ROOT_PASSWORD MariaDB root password used by the docker-compose stack
    OPENEMR_COMPOSE_DIR      directory holding the OpenEMR docker-compose.yml
"""
import json
import os
import subprocess
import sys
import time
import requests

OPENEMR_BASE = os.environ.get("OPENEMR_BASE", "https://localhost/apis/default")
REPORTS_FILE = "demo_reports.json"
MAPPING_FILE = "mapping.json"
DOCKER_COMPOSE_DIR = os.environ.get("OPENEMR_COMPOSE_DIR", ".")
DB_ROOT_PASSWORD = os.environ.get("OPENEMR_DB_ROOT_PASSWORD", "")

TOKEN = os.environ.get("OPENEMR_ACCESS_TOKEN", "")


def get_patients(n=100):
    query = f"""
    SELECT DISTINCT pd.pid,
           LOWER(CONCAT_WS('-',
             SUBSTR(HEX(pd.uuid),1,8),
             SUBSTR(HEX(pd.uuid),9,4),
             SUBSTR(HEX(pd.uuid),13,4),
             SUBSTR(HEX(pd.uuid),17,4),
             SUBSTR(HEX(pd.uuid),21,12)
           )) AS uuid_formatted,
           pd.fname, pd.lname
    FROM patient_data pd
    JOIN form_encounter fe ON fe.pid = pd.pid
    ORDER BY pd.pid
    LIMIT {n};
    """
    result = subprocess.run(
        ["docker-compose", "exec", "-T", "mysql",
         "mysql", "-uroot", f"-p{DB_ROOT_PASSWORD}",
         "openemr", "-N", "-s", "-e", query],
        cwd=DOCKER_COMPOSE_DIR,
        capture_output=True, text=True, check=True,
    )
    patients = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 4:
            patients.append({
                "pid": int(parts[0]),
                "uuid_formatted": parts[1],
                "fname": parts[2],
                "lname": parts[3],
            })
    return patients


def upload_report(pid, report_id, report_text):
    url = f"{OPENEMR_BASE}/api/patient/{pid}/document?path=Radiology"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    files = {"document": (f"{report_id}.txt", report_text.encode("utf-8"), "text/plain")}
    resp = requests.post(url, headers=headers, files=files, verify=True, timeout=60)
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
    return resp.text.strip(), None


def get_latest_doc(pid):
    url = f"{OPENEMR_BASE}/api/patient/{pid}/document?path=Radiology"
    resp = requests.get(url, headers={"Authorization": f"Bearer {TOKEN}"}, verify=True, timeout=30)
    if resp.status_code != 200:
        return None
    docs = resp.json()
    if not docs:
        return None
    return max(docs, key=lambda d: d.get("id", 0))


def main():
    if not TOKEN:
        print("ERROR: set OPENEMR_ACCESS_TOKEN before running")
        sys.exit(1)

    print("Loading reports...")
    with open(REPORTS_FILE) as f:
        reports = json.load(f)
    print(f"  {len(reports)} reports to seed")

    print("\nFetching patients from DB via docker-compose...")
    patients = get_patients(n=len(reports))
    print(f"  {len(patients)} patients available")

    mapping = []
    failures = 0

    for i, report in enumerate(reports):
        patient = patients[i % len(patients)]
        pid = patient["pid"]
        uuid = patient["uuid_formatted"]
        name = f"{patient['fname']} {patient['lname']}"
        report_id = report["report_id"]
        text = report["report_text"]
        original_len = len(text)

        if len(text) > 50000:
            text = text[:50000] + f"\n\n[TRUNCATED - original was {original_len} chars]"

        print(f"[{i+1:3d}/{len(reports)}] {report_id} -> {name} (pid={pid})")

        result, err = upload_report(pid, report_id, text)
        if err:
            print(f"    FAILED: {err}")
            failures += 1
            continue

        doc_info = get_latest_doc(pid)
        doc_id = doc_info.get("id") if doc_info else None

        mapping.append({
            "report_id": report_id,
            "hadm_id": report.get("hadm_id"),
            "patient_pid": pid,
            "patient_uuid": uuid,
            "patient_name": name,
            "doc_id": doc_id,
            "ground_truth_codes": report["ground_truth_codes"],
        })

        with open(MAPPING_FILE, "w") as f:
            json.dump(mapping, f, indent=2)

        time.sleep(0.2)

    print(f"\nDONE. {len(mapping)} uploaded, {failures} failed")
    print(f"  Mapping saved to {MAPPING_FILE}")


if __name__ == "__main__":
    main()
