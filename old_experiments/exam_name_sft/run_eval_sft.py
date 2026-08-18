#!/usr/bin/env python3
"""
Full evaluation of SFT model on 1000 held-out test examples.
Saves predictions + cosine similarity scores.

Run:
    python run_eval_sft.py
"""

import os, json, re, csv
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# ── CONFIG ───────────────────────────────────────────────────────────────────
TEST_JSONL    = "mimic_radiology_sft_test.jsonl"
LABELS_JSONL  = "mimic_radiology_sft_test_labels.jsonl"
OUTPUT_CSV    = f"eval_sft_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
SUMMARY_TXT   = f"eval_sft_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
GPU           = 0   # logical GPU index
MAX_SEQ_LEN   = 4096
MAX_NEW_TOKS  = 128

SYSTEM_PROMPT = (
    'You are a medical coding assistant. '
    'Given radiology reports for a patient admission, identify all procedures performed. '
    'Output only the procedure names. '
    'If multiple procedures were performed, separate each procedure name with a pipe character ( | ). '
    'Do not include any explanation, numbering, or extra text.'
)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading test data ...")
with open(TEST_JSONL) as f:
    test_examples = [json.loads(l) for l in f]

with open(LABELS_JSONL) as f:
    labels = {r['hadm_id']: r['ground_truth'] for r in (json.loads(l) for l in f)}

print(f"Test examples: {len(test_examples)}")

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("\nLoading model ...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained("qwen3-32b-mimic-sft")

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-32B",
    quantization_config=bnb,
    device_map={"": GPU},
    dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
base.config.use_cache = True

model = PeftModel.from_pretrained(base, "qwen3-32b-mimic-cpt-200k-ep2", is_trainable=False)
model.load_adapter("qwen3-32b-mimic-sft", adapter_name="sft")
model.set_adapter("sft")
model.eval()
print("Model loaded.\n")

# ── LOAD SENTENCE TRANSFORMER ─────────────────────────────────────────────────
print("Loading sentence transformer ...")
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Sentence transformer loaded.\n")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def split_procedures(text):
    procs = re.split(r'\||\n', text)
    return [p.strip() for p in procs if p.strip()]

def procedure_cosine_sim(gt, pred):
    gt_procs   = split_procedures(gt)
    pred_procs = split_procedures(pred)
    if not gt_procs or not pred_procs:
        return 0.0, 0.0, 0.0
    gt_embs   = st_model.encode(gt_procs,   convert_to_numpy=True)
    pred_embs = st_model.encode(pred_procs, convert_to_numpy=True)
    sim_matrix = cosine_similarity(gt_embs, pred_embs)
    recall_sim  = sim_matrix.max(axis=1).mean()
    prec_sim    = sim_matrix.max(axis=0).mean()
    f1_sim      = (recall_sim + prec_sim) / 2
    return recall_sim, prec_sim, f1_sim

def predict(example):
    user_msg = example['messages'][0]['content']
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user',   'content': user_msg},
    ]
    text   = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LEN).to(f"cuda:{GPU}")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKS, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    pred = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    gt   = labels.get(example['hadm_id'], '')
    return gt, pred

# ── RUN INFERENCE ─────────────────────────────────────────────────────────────
print(f"Running inference on {len(test_examples)} examples ...\n")

records = []
for ex in tqdm(test_examples, desc="Evaluating"):
    gt, pred = predict(ex)
    rec_sim, prec_sim, f1_sim = procedure_cosine_sim(gt, pred)
    records.append({
        'hadm_id':      ex['hadm_id'],
        'ground_truth': gt,
        'prediction':   pred,
        'cos_recall':   round(rec_sim,  4),
        'cos_precision':round(prec_sim, 4),
        'cos_f1':       round(f1_sim,   4),
    })

# ── SAVE RESULTS ──────────────────────────────────────────────────────────────
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
print(f"\n✓ Results saved → {OUTPUT_CSV}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
avg_rec  = np.mean([r['cos_recall']    for r in records])
avg_prec = np.mean([r['cos_precision'] for r in records])
avg_f1   = np.mean([r['cos_f1']        for r in records])

summary = f"""
{'='*50}
  SFT EVAL SUMMARY — {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*50}
  Test examples  : {len(records)}
  Avg Cos Recall : {avg_rec:.4f}
  Avg Cos Prec   : {avg_prec:.4f}
  Avg Cos F1     : {avg_f1:.4f}
{'='*50}
"""
print(summary)

with open(SUMMARY_TXT, 'w') as f:
    f.write(summary)
print(f"✓ Summary saved → {SUMMARY_TXT}")
