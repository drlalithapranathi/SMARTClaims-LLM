#!/usr/bin/env python3
"""
Create a true held-out test set from MIMIC-IV discharge notes.
Filters out all note_ids that were used in training (cleaned_discharge_notes_200k_fixed.csv).

Usage:
    python make_holdout.py --discharge /path/to/discharge.csv

Output:
    holdout_discharge_notes.csv  — notes NOT in training set
"""

import argparse
import pandas as pd

TRAIN_CSV    = "cleaned_discharge_notes_200k_fixed.csv"
OUTPUT_CSV   = "holdout_discharge_notes.csv"
TEXT_COLUMN  = "text"   # column name in the original MIMIC-IV discharge.csv

parser = argparse.ArgumentParser()
parser.add_argument("--discharge", required=True, help="Path to original MIMIC-IV discharge.csv")
args = parser.parse_args()

print("Loading training note_ids ...")
train_ids = set(pd.read_csv(TRAIN_CSV, usecols=["note_id"])["note_id"].astype(str))
print(f"  Training set : {len(train_ids):,} notes")

print(f"Loading {args.discharge} ...")
discharge_df = pd.read_csv(args.discharge)
print(f"  Total notes  : {len(discharge_df):,}")

holdout_df = discharge_df[~discharge_df["note_id"].astype(str).isin(train_ids)].reset_index(drop=True)
print(f"  Held-out     : {len(holdout_df):,} notes")

holdout_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Saved → {OUTPUT_CSV}")
