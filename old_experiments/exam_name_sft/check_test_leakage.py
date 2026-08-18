#!/usr/bin/env python3
"""
Verify no hadm_id overlap between test set and train/val sets.
"""

import json

def load_hadm_ids(path):
    ids = set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            ids.add(r['hadm_id'])
    return ids

train_ids = load_hadm_ids('mimic_radiology_sft_train_cleaned.jsonl')
val_ids   = load_hadm_ids('mimic_radiology_sft_val_cleaned.jsonl')
test_ids  = load_hadm_ids('mimic_radiology_sft_test.jsonl')

train_overlap = test_ids & train_ids
val_overlap   = test_ids & val_ids

print(f'Train size : {len(train_ids)}')
print(f'Val size   : {len(val_ids)}')
print(f'Test size  : {len(test_ids)}')
print()
print(f'Test ∩ Train : {len(train_overlap)} overlap')
print(f'Test ∩ Val   : {len(val_overlap)} overlap')

if train_overlap or val_overlap:
    print('\n❌ LEAKAGE DETECTED')
    if train_overlap:
        print(f'   Overlapping train hadm_ids: {sorted(train_overlap)[:10]}')
    if val_overlap:
        print(f'   Overlapping val hadm_ids:   {sorted(val_overlap)[:10]}')
else:
    print('\n✓ No leakage — test set is clean')
