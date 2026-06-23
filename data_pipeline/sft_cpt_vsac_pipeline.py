"""Build a radiology SFT dataset labelled with billable CPT codes.

Pipeline: extract CPT codes from MIMIC-IV radiology_detail, link them to
admissions, aggregate all reports per admission, fetch official CPT descriptions
from VSAC (UMLS REST fallback), drop add-on/invalid codes, then split and explode
to one row per CPT code. Requires MIMIC-IV access via PhysioNet and a UMLS API key.

Set via environment:
    UMLS_API_KEY, MIMIC_NOTES_PATH, OUTPUT_DIR
"""

import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

NOTES_PATH = os.environ.get("MIMIC_NOTES_PATH", "data/mimic-iv-note/")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output/")
UMLS_API_KEY = os.environ.get("UMLS_API_KEY", "")

MAX_CPT_CODES = 6        # drop admissions with more billable codes than this
TRAIN_FRAC = 0.8
RANDOM_SEED = 42

ADDON_PHRASES = ["list separately in addition", "each additional"]


def load_data():
    radiology = pd.read_csv(os.path.join(NOTES_PATH, "radiology.csv"))
    rad_detail = pd.read_csv(os.path.join(NOTES_PATH, "radiology_detail.csv"))
    print(f"radiology: {len(radiology):,} rows | rad_detail: {len(rad_detail):,} rows")
    return radiology, rad_detail


def extract_cpt_codes(rad_detail):
    cpt = rad_detail[rad_detail["field_name"] == "cpt_code"][["note_id", "field_value"]].copy()
    cpt.columns = ["note_id", "cpt_code"]
    cpt["cpt_code"] = cpt["cpt_code"].astype(str).str.strip()
    print(f"CPT records: {len(cpt):,} | unique codes: {cpt['cpt_code'].nunique()}")
    return cpt


def link_cpt_to_hadm(cpt_records, radiology):
    linked = cpt_records.merge(radiology[["note_id", "hadm_id"]], on="note_id", how="inner")
    linked = linked[linked["hadm_id"].notna()].copy()
    linked["hadm_id"] = linked["hadm_id"].astype(int)
    print(f"CPT records with hadm_id: {len(linked):,} | admissions: {linked['hadm_id'].nunique():,}")
    return linked


def aggregate_reports(radiology):
    """One row per admission with all its reports concatenated (no truncation)."""
    valid = radiology[radiology["hadm_id"].notna()].copy()
    valid["hadm_id"] = valid["hadm_id"].astype(int)

    def _aggregate(group):
        ordered = group.sort_values("charttime")
        text = "\n\n---\n\n".join(
            f"Report {i + 1}:\n{r.strip()}"
            for i, r in enumerate(ordered["text"].dropna().tolist())
        )
        return pd.Series({"input_text": text, "included_note_ids": ordered["note_id"].tolist()})

    aggregated = valid.groupby("hadm_id").apply(_aggregate, include_groups=False).reset_index()
    print(f"Admissions aggregated: {len(aggregated):,}")
    return aggregated


def join_reports_labels(reports_per_admission, cpt_with_hadm):
    """Attach the CPT codes for the reports actually included in each admission."""
    merged = reports_per_admission.merge(cpt_with_hadm, on="hadm_id", how="inner")
    merged = merged[merged.apply(lambda r: r["note_id"] in r["included_note_ids"], axis=1)]

    labels = (
        merged.groupby(["hadm_id", "input_text"])["cpt_code"]
        .apply(lambda x: " | ".join(sorted(set(x.str.strip()))))
        .reset_index()
        .rename(columns={"cpt_code": "all_cpt_codes"})
    )
    print(f"Admissions with reports + CPT codes: {len(labels):,}")
    return labels


def get_cpt_from_vsac(cpt_code, api_key):
    url = "https://cts.nlm.nih.gov/fhir/CodeSystem/$lookup"
    params = {"system": "http://www.ama-assn.org/go/cpt", "code": cpt_code}
    try:
        resp = requests.get(
            url, params=params, headers={"Accept": "application/fhir+json"},
            auth=HTTPBasicAuth("apikey", api_key), timeout=15,
        )
        if resp.status_code == 200:
            for param in resp.json().get("parameter", []):
                if param.get("name") == "display":
                    return param.get("valueString")
        elif resp.status_code == 401:
            print(f"  AUTH ERROR for {cpt_code} — check UMLS_API_KEY")
    except Exception as e:
        print(f"  Error {cpt_code}: {e}")
    return None


def get_cpt_from_umls_rest(cpt_code, api_key):
    url = f"https://uts-ws.nlm.nih.gov/rest/content/current/source/CPT/{cpt_code}"
    try:
        resp = requests.get(url, params={"apiKey": api_key}, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("result", {}).get("name")
    except Exception as e:
        print(f"  UMLS REST error {cpt_code}: {e}")
    return None


def build_cpt_descriptions(codes, api_key):
    """Look up CPT descriptions; cache to JSON to avoid re-querying."""
    cache_path = os.path.join(OUTPUT_DIR, "cpt_to_vsac_description.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"Loaded {len(cached)} cached CPT descriptions.")
        return cached

    descriptions, failed = {}, []
    for i, code in enumerate(codes):
        desc = get_cpt_from_vsac(code, api_key)
        if desc:
            descriptions[code] = desc
        else:
            failed.append(code)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(codes)} ({len(descriptions)} mapped, {len(failed)} failed)")
        time.sleep(0.3)

    for code in failed:  # UMLS REST fallback
        desc = get_cpt_from_umls_rest(code, api_key)
        if desc:
            descriptions[code] = desc
        time.sleep(0.3)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(descriptions, f, indent=2)
    print(f"Mapped {len(descriptions)}/{len(codes)} codes -> {cache_path}")
    return descriptions


def identify_codes_to_remove(cpt_to_description, all_codes):
    """Add-on codes (not independently billable) + codes with no description."""
    addon = {c for c, d in cpt_to_description.items()
             if any(p in d.lower() for p in ADDON_PHRASES)}
    invalid = {c for c in all_codes if c not in cpt_to_description}
    print(f"Removing {len(addon)} add-on + {len(invalid)} invalid codes.")
    return addon, invalid, addon | invalid


def filter_bad_codes(labels, codes_to_remove, cpt_to_description):
    def _keep(code_string):
        kept = [c.strip() for c in code_string.split(" | ") if c.strip() not in codes_to_remove]
        return " | ".join(kept) if kept else None

    labels = labels.copy()
    labels["all_cpt_codes"] = labels["all_cpt_codes"].apply(_keep)
    labels = labels[labels["all_cpt_codes"].notna()].copy()
    labels["all_procedures"] = labels["all_cpt_codes"].apply(
        lambda s: " | ".join(cpt_to_description.get(c.strip(), c.strip()) for c in s.split(" | "))
    )
    print(f"Admissions with billable codes: {len(labels):,}")
    return labels


def explode_cpt(df):
    rows = []
    for _, row in df.iterrows():
        codes = [c.strip() for c in row["all_cpt_codes"].split(" | ")]
        procs = [p.strip() for p in row["all_procedures"].split(" | ")]
        for code, label in zip(codes, procs):
            rows.append({
                "hadm_id": row["hadm_id"], "reports": row["input_text"],
                "cpt_codes": code, "cpt_labels": label,
            })
    return pd.DataFrame(rows)


def split_and_explode(labels):
    labels = labels.copy()
    labels["cpt_count"] = labels["all_cpt_codes"].apply(lambda x: len(x.split(" | ")))
    capped = labels[labels["cpt_count"] <= MAX_CPT_CODES].copy()
    print(f"Admissions after cap (<={MAX_CPT_CODES}): {len(capped):,}")

    hadm_ids = capped["hadm_id"].drop_duplicates().values
    shuffled = np.random.default_rng(RANDOM_SEED).permutation(hadm_ids)
    split_idx = int(TRAIN_FRAC * len(shuffled))
    train_ids, test_ids = set(shuffled[:split_idx]), set(shuffled[split_idx:])

    train = explode_cpt(capped[capped["hadm_id"].isin(train_ids)]).reset_index(drop=True)
    test = explode_cpt(capped[capped["hadm_id"].isin(test_ids)]).reset_index(drop=True)
    print(f"Train: {len(train):,} rows / {train['hadm_id'].nunique():,} hadm_ids | "
          f"Test: {len(test):,} rows / {test['hadm_id'].nunique():,} hadm_ids")
    return train, test


def export(train, test):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train[["hadm_id", "reports", "cpt_codes", "cpt_labels"]].to_csv(
        os.path.join(OUTPUT_DIR, "mimic_radiology_sft_train.csv"), index=False)
    test[["hadm_id", "reports"]].drop_duplicates(subset="hadm_id").to_csv(
        os.path.join(OUTPUT_DIR, "mimic_radiology_sft_test_inputs.csv"), index=False)
    test[["hadm_id", "cpt_codes", "cpt_labels"]].to_csv(
        os.path.join(OUTPUT_DIR, "mimic_radiology_sft_test_labels.csv"), index=False)
    print("Saved train / test_inputs / test_labels CSVs.")


def validate(train, test, addon_codes):
    train_hadm = set(train["hadm_id"])
    test_hadm = set(test["hadm_id"])
    assert not (train_hadm & test_hadm), "DATA LEAKAGE: overlapping hadm_ids"
    assert not train["cpt_codes"].str.contains("|", regex=False).any(), "multi-code rows present"
    assert not (set(train["cpt_codes"]) & addon_codes), "add-on codes leaked into train"
    print("Sanity checks passed: no leakage, one code per row, no add-on codes.")


def main():
    if not UMLS_API_KEY:
        raise SystemExit("Set UMLS_API_KEY in the environment before running.")

    radiology, rad_detail = load_data()
    cpt_records = extract_cpt_codes(rad_detail)
    cpt_with_hadm = link_cpt_to_hadm(cpt_records, radiology)
    reports_per_admission = aggregate_reports(radiology)
    labels = join_reports_labels(reports_per_admission, cpt_with_hadm)

    all_codes = {c.strip() for row in labels["all_cpt_codes"] for c in row.split(" | ")}
    cpt_to_description = build_cpt_descriptions(sorted(all_codes), UMLS_API_KEY)

    addon_codes, _, codes_to_remove = identify_codes_to_remove(cpt_to_description, all_codes)
    labels = filter_bad_codes(labels, codes_to_remove, cpt_to_description)

    train, test = split_and_explode(labels)
    export(train, test)
    validate(train, test, addon_codes)


if __name__ == "__main__":
    main()