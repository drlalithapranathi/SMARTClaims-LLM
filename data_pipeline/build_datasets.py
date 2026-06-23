#!/usr/bin/env python3
"""Build SFT and GRPO datasets for radiology CPT-code prediction.

Combines radiology admissions (from the SFT train CSV) with a sample of
discharge notes labelled 'unknown' (~10% of the final SFT set), then writes:
  - SFT  (system/user/assistant messages) train + val
  - GRPO (prompt + ground_truth, radiology only) train + val

Paths are configurable via environment variables; data files are not included
(MIMIC-IV requires PhysioNet access).
"""

import os

import pandas as pd
from datasets import Dataset

BASE_DIR = os.environ.get("DATA_DIR", "..")
TRAIN_CSV = os.path.join(BASE_DIR, "sft_train.csv")
DISCHARGE_CSV = os.path.join(BASE_DIR, "cleaned_discharge_notes_200k_fixed.csv")

UNKNOWN_FRACTION = 0.10     # share of the final SFT set labelled 'unknown'
VAL_FRACTION = 0.1
SEED = 42

SFT_TRAIN_OUT = "unk10_sft_train"
SFT_VAL_OUT = "unk10_sft_val"
GRPO_TRAIN_OUT = "unk10_grpo_train"
GRPO_VAL_OUT = "unk10_grpo_val"

SYSTEM_PROMPT = (
    "You are a medical coding assistant. "
    "Given a clinical note for a patient admission, identify all billable "
    "radiology procedures performed. Output only the CPT codes separated by "
    "pipe ( | ). If no billable radiology procedure can be determined, output "
    "unknown. Do not include add-on codes, explanations, or numbering."
)


def load_radiology(path):
    """One row per admission: reports + pipe-joined sorted CPT codes."""
    df = pd.read_csv(path, dtype={"cpt_codes": str})
    agg = df.groupby("hadm_id").agg(
        reports=("reports", "first"),
        cpt_codes=("cpt_codes", lambda x: " | ".join(sorted(x))),
    ).reset_index()
    print(f"Radiology admissions: {len(agg):,}")
    return agg


def sample_unknowns(path, n_admissions):
    """Sample discharge notes to use as 'unknown' (no billable radiology) examples."""
    n_unknown = int(n_admissions * UNKNOWN_FRACTION / (1 - UNKNOWN_FRACTION))
    discharge = pd.read_csv(
        path, usecols=["cleaned_text"], dtype={"cleaned_text": str}
    ).dropna(subset=["cleaned_text"])
    sample = discharge.sample(n=n_unknown, random_state=SEED)
    records = [
        {"hadm_id": f"unk_{i}", "reports": text, "cpt_codes": "unknown"}
        for i, text in enumerate(sample["cleaned_text"])
    ]
    print(f"Unknown examples: {len(records):,}")
    return pd.DataFrame(records)


def build_sft(combined):
    records = [
        {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["reports"]},
            {"role": "assistant", "content": row["cpt_codes"]},
        ]}
        for _, row in combined.iterrows()
    ]
    split = Dataset.from_list(records).shuffle(seed=SEED).train_test_split(
        test_size=VAL_FRACTION, seed=SEED
    )
    split["train"].save_to_disk(SFT_TRAIN_OUT)
    split["test"].save_to_disk(SFT_VAL_OUT)
    print(f"SFT  -> train {len(split['train']):,} | val {len(split['test']):,}")
    return split


def build_grpo(agg):
    """Radiology only — no 'unknown' examples for RL."""
    records = [
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["reports"]},
            ],
            "ground_truth": row["cpt_codes"],
            "hadm_id": str(row["hadm_id"]),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for _, row in agg.iterrows()
    ]
    split = Dataset.from_list(records).shuffle(seed=SEED).train_test_split(
        test_size=VAL_FRACTION, seed=SEED
    )
    split["train"].save_to_disk(GRPO_TRAIN_OUT)
    split["test"].save_to_disk(GRPO_VAL_OUT)
    print(f"GRPO -> train {len(split['train']):,} | val {len(split['test']):,}")
    return split


def main():
    agg = load_radiology(TRAIN_CSV)
    unknowns = sample_unknowns(DISCHARGE_CSV, len(agg))

    combined = pd.concat(
        [agg[["hadm_id", "reports", "cpt_codes"]], unknowns], ignore_index=True
    ).sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_unk = (combined["cpt_codes"] == "unknown").sum()
    print(f"Combined: {len(combined):,} ({n_unk / len(combined) * 100:.1f}% unknown)")

    build_sft(combined)
    build_grpo(agg)


if __name__ == "__main__":
    main()