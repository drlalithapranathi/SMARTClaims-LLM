#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import json

NOTES_PATH = 'data/mimic-iv-note/2.2/note/'

radiology  = pd.read_csv(NOTES_PATH + 'radiology.csv')
rad_detail = pd.read_csv(NOTES_PATH + 'radiology_detail.csv')

print(f'radiology:   {len(radiology):,} rows')
print(f'rad_detail:  {len(rad_detail):,} rows')


# Extract exam_name labels, join back to get hadm_id via note_id
exam_names = rad_detail[rad_detail['field_name'] == 'exam_name'][['note_id', 'field_value']].copy()
exam_names.columns = ['note_id', 'exam_name']

# Join to radiology to get hadm_id — keep note_id for label filtering later
exam_with_hadm = exam_names.merge(radiology[['note_id', 'hadm_id']], on='note_id', how='inner')
exam_with_hadm = exam_with_hadm[exam_with_hadm['hadm_id'].notna()].copy()
exam_with_hadm['hadm_id'] = exam_with_hadm['hadm_id'].astype(int)

print(f'exam_name records with hadm_id: {len(exam_with_hadm):,}')
print(f'Unique admissions: {exam_with_hadm["hadm_id"].nunique():,}')


# Aggregate reports per hadm_id (cap at 5, sorted by charttime)
# Also track which note_ids were included so labels match exactly
radiology_valid = radiology[radiology['hadm_id'].notna()].copy()
radiology_valid['hadm_id'] = radiology_valid['hadm_id'].astype(int)

def get_top5(group, max_reports=5):
    top = group.sort_values('charttime').head(max_reports)
    text = '\n\n---\n\n'.join([f'Report {i+1}:\n{r.strip()}' for i, r in enumerate(top['text'].dropna().tolist())])
    note_ids = top['note_id'].tolist()
    return pd.Series({'input_text': text, 'included_note_ids': note_ids})

reports_per_admission = (
    radiology_valid
    .groupby('hadm_id')[['charttime', 'text', 'note_id']]
    .apply(get_top5)
    .reset_index()
)

print(f'Admissions with reports: {len(reports_per_admission):,}')


# Join reports + labels, but only keep labels from the 5 included reports
dataset = reports_per_admission.merge(exam_with_hadm, on='hadm_id', how='inner')

# Filter: only exam_names whose note_id was in the top 5
dataset = dataset[dataset.apply(lambda r: r['note_id'] in r['included_note_ids'], axis=1)]

# Aggregate filtered labels per hadm_id — drop included_note_ids from groupby (unhashable list)
labels_filtered = (
    dataset.groupby(['hadm_id', 'input_text'])['exam_name']
    .apply(lambda x: ' | '.join(sorted(set(x.str.strip()))))
    .reset_index()
    .rename(columns={'exam_name': 'all_procedures'})
)

print(f'Admissions with matched reports and labels: {len(labels_filtered):,}')
print(f'\nSample:')
print(labels_filtered[['hadm_id', 'all_procedures']].head(5).to_string(index=False))


# Sample 1200, split 1000 train / 200 val
from collections import Counter

sft = labels_filtered.sample(1200, random_state=42).reset_index(drop=True)
train = sft.iloc[:1000]
val   = sft.iloc[1000:]

print(f'Train: {len(train)} | Val: {len(val)}')

all_labels = Counter()
for row in train['all_procedures']:
    for p in row.split(' | '):
        all_labels[p.strip()] += 1
print(f'\nUnique procedures in train: {len(all_labels)}')
print(f'\nTop 20:')
for proc, cnt in all_labels.most_common(20):
    print(f'  {cnt:>5}  {proc}')

sft.to_csv('mimic_radiology_sft_1200.csv', index=False)


# ---
# ## Validation

# 1. Shape + nulls
print(f'Rows: {len(sft)}  (expected 1200)')
print(f'Duplicate hadm_ids: {sft["hadm_id"].duplicated().sum()}  (expected 0)')
print(f'\nNulls:')
print(sft[['input_text', 'all_procedures']].isnull().sum())


# 2. Spot check — read 3 admissions manually
for _, row in sft.sample(3, random_state=0).iterrows():
    print('='*60)
    print(f'hadm_id:    {row["hadm_id"]}')
    print(f'procedures: {row["all_procedures"]}')
    print(f'reports:\n{row["input_text"][:600]}')
    print()


# ---
# ## Convert to JSONL for SFT

def to_jsonl(df, path):
    records = []
    for _, row in df.iterrows():
        records.append({
            'messages': [
                {
                    'role': 'user',
                    'content': f'Given the following radiology reports for this admission, identify all procedures performed:\n\n{row["input_text"]}'
                },
                {
                    'role': 'assistant',
                    'content': row['all_procedures']
                }
            ]
        })
    with open(path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    print(f'Saved: {path}  ({len(records)} records)')

to_jsonl(train, 'mimic_radiology_sft_train.jsonl')
to_jsonl(val,   'mimic_radiology_sft_val.jsonl')

# Preview one training example
with open('mimic_radiology_sft_train.jsonl') as f:
    print(f'\nSample training record:')
    print(json.dumps(json.loads(f.readline()), indent=2))










