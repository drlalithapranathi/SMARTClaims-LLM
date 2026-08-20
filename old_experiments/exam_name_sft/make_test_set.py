#!/usr/bin/env python3
"""
Build a held-out test set of N_TEST radiology SFT examples.

Uses the same pipeline as the SFT notebook but excludes the 1200
samples already used for train/val (random_state=42).

Run on the machine that holds the MIMIC-IV note files:
    python make_test_set.py

Output:
    mimic_radiology_sft_test.jsonl  (N_TEST examples)
"""

import json
import pandas as pd
from collections import Counter

NOTES_PATH = 'data/mimic-iv-note/2.2/note/'
N_TEST     = 1000
SEED       = 99   # different seed from train/val

# ── Load raw data ─────────────────────────────────────────────────────────────
print("Loading radiology.csv ...")
radiology  = pd.read_csv(NOTES_PATH + 'radiology.csv')
print("Loading radiology_detail.csv ...")
rad_detail = pd.read_csv(NOTES_PATH + 'radiology_detail.csv')
print(f'radiology:  {len(radiology):,} rows')
print(f'rad_detail: {len(rad_detail):,} rows')

# ── Extract exam_name labels ──────────────────────────────────────────────────
exam_names = rad_detail[rad_detail['field_name'] == 'exam_name'][['note_id', 'field_value']].copy()
exam_names.columns = ['note_id', 'exam_name']

exam_with_hadm = exam_names.merge(radiology[['note_id', 'hadm_id']], on='note_id', how='inner')
exam_with_hadm = exam_with_hadm[exam_with_hadm['hadm_id'].notna()].copy()
exam_with_hadm['hadm_id'] = exam_with_hadm['hadm_id'].astype(int)
print(f'\nexam_name records with hadm_id: {len(exam_with_hadm):,}')

# ── Aggregate up to 5 reports per admission ───────────────────────────────────
radiology_valid = radiology[radiology['hadm_id'].notna()].copy()
radiology_valid['hadm_id'] = radiology_valid['hadm_id'].astype(int)

def get_top5(group):
    top = group.sort_values('charttime').head(5)
    text = '\n\n---\n\n'.join(
        [f'Report {i+1}:\n{r.strip()}' for i, r in enumerate(top['text'].dropna().tolist())]
    )
    note_ids = top['note_id'].tolist()
    return pd.Series({'input_text': text, 'included_note_ids': note_ids})

print("\nAggregating reports per admission (this takes a minute) ...")
reports_per_admission = (
    radiology_valid
    .groupby('hadm_id')[['charttime', 'text', 'note_id']]
    .apply(get_top5)
    .reset_index()
)
print(f'Admissions with reports: {len(reports_per_admission):,}')

# ── Join + filter labels to only included reports ─────────────────────────────
dataset = reports_per_admission.merge(exam_with_hadm, on='hadm_id', how='inner')
dataset = dataset[dataset.apply(lambda r: r['note_id'] in r['included_note_ids'], axis=1)]

labels_filtered = (
    dataset.groupby(['hadm_id', 'input_text'])['exam_name']
    .apply(lambda x: ' | '.join(sorted(set(x.str.strip()))))
    .reset_index()
    .rename(columns={'exam_name': 'all_procedures'})
)
print(f'Admissions with matched labels: {len(labels_filtered):,}')

# ── Exclude the 1200 already used for train/val ───────────────────────────────
used = labels_filtered.sample(1200, random_state=42)
used_hadm_ids = set(used['hadm_id'].tolist())

remaining = labels_filtered[~labels_filtered['hadm_id'].isin(used_hadm_ids)].reset_index(drop=True)
print(f'Remaining (unseen) admissions: {len(remaining):,}')

# ── Sample N_TEST test examples ───────────────────────────────────────────────
test = remaining.sample(n=min(N_TEST, len(remaining)), random_state=SEED).reset_index(drop=True)
print(f'Test set size: {len(test)}')

# ── Label distribution ────────────────────────────────────────────────────────
all_labels = Counter()
for row in test['all_procedures']:
    for p in row.split(' | '):
        all_labels[p.strip()] += 1
print(f'\nUnique procedures in test: {len(all_labels)}')
print('Top 10:')
for proc, cnt in all_labels.most_common(10):
    print(f'  {cnt:>4}  {proc}')

# ── Save test JSONL (no labels) ───────────────────────────────────────────────
out_path     = 'mimic_radiology_sft_test.jsonl'
labels_path  = 'mimic_radiology_sft_test_labels.jsonl'

records = []
label_records = []

for _, row in test.iterrows():
    records.append({
        'hadm_id': row['hadm_id'],
        'messages': [
            {
                'role': 'user',
                'content': f'Given the following radiology reports for this admission, identify all procedures performed:\n\n{row["input_text"]}'
            }
        ]
    })
    label_records.append({
        'hadm_id': row['hadm_id'],
        'ground_truth': row['all_procedures']
    })

with open(out_path, 'w') as f:
    for r in records:
        f.write(json.dumps(r) + '\n')

with open(labels_path, 'w') as f:
    for r in label_records:
        f.write(json.dumps(r) + '\n')

print(f'\n✓ Test inputs saved  → {out_path}  ({len(records)} records)')
print(f'✓ Labels saved       → {labels_path}  (keep separate for eval)')

# Spot check
print('\n--- Sample ---')
print(f'hadm_id: {records[0]["hadm_id"]}')
print(f'GT:      {label_records[0]["ground_truth"]}')
print(f'Report:  {records[0]["messages"][0]["content"][:300]}...')
