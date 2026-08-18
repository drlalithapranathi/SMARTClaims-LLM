#!/usr/bin/env python3
"""
Build radiology_holdout.jsonl from:
  - mimic_radiology_sft_test__1_.jsonl   (full report text per admission)
  - mimic_radiology_sft_test_labels.jsonl (pipe-separated procedure names)

Output: radiology_holdout.jsonl
  Each line = one individual report with:
    hadm_id, procedure, indication, reference (FINDINGS + IMPRESSION)
"""

import json, re

SFT_PATH   = "mimic_radiology_sft_test.jsonl"
LABEL_PATH = "mimic_radiology_sft_test_labels.jsonl"
OUT_PATH   = "radiology_holdout.jsonl"

# ── Load both files ──────────────────────────────────────────────────────────
with open(SFT_PATH) as f:
    sft_rows = [json.loads(line) for line in f]

with open(LABEL_PATH) as f:
    label_rows = {json.loads(line)["hadm_id"]: json.loads(line)["ground_truth"]
                  for line in open(LABEL_PATH)}

# ── Helper: split admission text into individual reports ─────────────────────
def split_reports(full_text: str) -> list[str]:
    """Split the SFT prompt text into individual report strings."""
    # Remove instruction prefix
    text = re.sub(
        r"^Given the following radiology reports for this admission,.*?:\n\n",
        "", full_text, flags=re.DOTALL
    )
    # Split on --- separator
    parts = re.split(r"\n---\n", text)
    reports = []
    for p in parts:
        p = re.sub(r"^Report \d+:\n", "", p.strip())
        if len(p) > 50:
            reports.append(p)
    return reports


def extract_indication(report: str) -> str:
    """Extract INDICATION / HISTORY / CLINICAL INFORMATION."""
    patterns = [
        r"(?:INDICATION|HISTORY|CLINICAL INFORMATION|CLINICAL HISTORY)\s*:\s*(.*?)(?=\n\n|\nCOMPARISON|\nTECHNIQUE|\nFINDINGS)",
    ]
    for pat in patterns:
        m = re.search(pat, report, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def extract_findings_and_impression(report: str) -> str:
    """Extract everything from FINDINGS: onward (findings + impression).
    
    Some reports don't have a FINDINGS: header — they use the view description
    as the section header (e.g., 'SINGLE FRONTAL VIEW OF THE CHEST:').
    In those cases, grab content after COMPARISON (or TECHNIQUE) section onward.
    """
    # Try to find explicit FINDINGS section
    m = re.search(r"(FINDINGS\s*:.*)", report, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Fallback: grab everything after COMPARISON: ... paragraph
    # (the next section is usually the actual findings text + IMPRESSION)
    m = re.search(
        r"COMPARISON\s*:.*?\n\n(.*)",
        report, re.DOTALL | re.IGNORECASE,
    )
    if m:
        remainder = m.group(1).strip()
        if len(remainder) > 50:
            return remainder

    # Fallback: try IMPRESSION only
    m = re.search(r"(IMPRESSION\s*:.*)", report, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


# ── Process ──────────────────────────────────────────────────────────────────
output_records = []
skipped = 0

for sft_row in sft_rows:
    hadm_id = sft_row["hadm_id"]
    full_text = sft_row["messages"][0]["content"]
    procedures_str = label_rows.get(hadm_id, "")
    procedures = [p.strip() for p in procedures_str.split("|")]

    reports = split_reports(full_text)

    # Try to pair reports with procedures 1:1 if counts match;
    # otherwise assign all procedures to each report
    for i, report in enumerate(reports):
        indication = extract_indication(report)
        reference  = extract_findings_and_impression(report)

        if len(reference) < 50:
            skipped += 1
            continue

        # Best-effort procedure assignment
        if len(reports) == len(procedures):
            procedure = procedures[i]
        else:
            # Can't align 1:1; use all procedures for this admission
            procedure = " | ".join(procedures)

        output_records.append({
            "hadm_id":    hadm_id,
            "procedure":  procedure,
            "indication": indication,
            "reference":  reference,
        })

# ── Write output ─────────────────────────────────────────────────────────────
with open(OUT_PATH, "w") as f:
    for rec in output_records:
        f.write(json.dumps(rec) + "\n")

print(f"✓ Wrote {len(output_records)} records to {OUT_PATH}")
print(f"  Skipped {skipped} reports (reference too short)")
print(f"  From {len(sft_rows)} admissions")
