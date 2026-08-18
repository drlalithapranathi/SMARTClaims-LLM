"""
Adds 15% discharge summary rows (label = "unknown") to the SFT training set.
Discharge notes are sampled from hadm_ids NOT in train or test to avoid leakage.
"""

import pandas as pd

TRAIN_CSV    = "mimic_radiology_sft_train.csv"
TEST_LABELS  = "mimic_radiology_sft_test_labels.csv"
TEST_INPUTS  = "mimic_radiology_sft_test_inputs.csv"
DISCHARGE    = "cleaned_discharge_notes_200k_fixed.csv"
OUTPUT_CSV   = "mimic_radiology_sft_train.csv"
SEED         = 42

# ── Load train and test ──────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_CSV, dtype={"cpt_codes": str})
test_labels = pd.read_csv(TEST_LABELS, dtype={"cpt_codes": str})
test_inputs = pd.read_csv(TEST_INPUTS)

excluded_hadm_ids = (
    set(train["hadm_id"].astype(str)) |
    set(test_labels["hadm_id"].astype(str)) |
    set(test_inputs["hadm_id"].astype(str))
)
print(f"Excluded hadm_ids (train + test): {len(excluded_hadm_ids)}")

# ── Load discharge notes ─────────────────────────────────────────────────────
discharge = pd.read_csv(DISCHARGE)
print(f"Total discharge notes: {len(discharge)}")

# Filter to hadm_ids not in train or test
available = discharge[~discharge["hadm_id"].astype(str).isin(excluded_hadm_ids)].copy()
print(f"Available discharge notes (unseen hadm_ids): {len(available)}")

# ── Sample 15% of current train size ────────────────────────────────────────
n_to_add = round(len(train) * 0.15)
print(f"Sampling {n_to_add} discharge notes (15% of {len(train)} train rows)")

sampled = available.sample(n=n_to_add, random_state=SEED)

# ── Format to match train columns ───────────────────────────────────────────
discharge_rows = pd.DataFrame({
    "hadm_id":   sampled["hadm_id"].values,
    "reports":   sampled["cleaned_text"].values,
    "cpt_codes": "unknown",
    "cpt_labels": "unknown",
})

# ── Append and save ──────────────────────────────────────────────────────────
train_updated = pd.concat([train, discharge_rows], ignore_index=True)

print(f"\nTrain size: {len(train)} → {len(train_updated)}")
print(f"  Radiology rows : {len(train)}")
print(f"  Discharge rows : {len(discharge_rows)}")
print(f"  Unknown %      : {len(discharge_rows)/len(train_updated)*100:.1f}%")

train_updated.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved to {OUTPUT_CSV}")
