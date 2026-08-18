#!/usr/bin/env python3
"""
Perplexity comparison on the test set across training stages
(base → CPT → SFT → GRPO). Computes perplexity on response tokens only
(the CPT-code string), on the first N_SAMPLES test admissions.

This is the protocol behind the perplexity table in docs/RESULTS.md
(13.00 / 5.36 / 1.48 / 1.49). Those reported values were measured on the
SFT v9 / GRPO v3b checkpoints; MODELS below points at the final unk10
pipeline — swap paths to reproduce the reported run.

Run:
    python eval_perplexity_stages.py
"""

import os
import json
import math
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# ── CONFIG ───────────────────────────────────────────────────────────────────
TOKENIZER_SRC    = "../qwen3-32b-mimic-cpt-200k-ep2"
TEST_INPUTS_CSV  = "../sft_test_inputs.csv"
TEST_LABELS_CSV  = "../sft_test_labels.csv"
N_SAMPLES        = 50           # cap for speed; set to None for all 4702
OUT_JSON         = "perplexity_test_results.json"

# (label, base model, optional adapter)
MODELS = [
    ("Base", "Qwen/Qwen3-32B",                None),
    ("CPT",  "../qwen3-32b-mimic-cpt-merged", None),
    ("SFT",  "unk10_sft_merged",              None),
    ("GRPO", "unk10_sft_merged",              "unk10_grpo_adapter"),
]

SYSTEM_PROMPT = (
    "You are a medical coding assistant. "
    "Given a clinical note for a patient admission, identify all billable "
    "radiology procedures performed. Output only the CPT codes separated by "
    "pipe ( | ). If no billable radiology procedure can be determined, output "
    "unknown. Do not include add-on codes, explanations, or numbering."
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ── LOAD TEST DATA ───────────────────────────────────────────────────────────
print("Loading test data ...")
inputs_df = pd.read_csv(TEST_INPUTS_CSV, dtype=str)
labels_df = pd.read_csv(TEST_LABELS_CSV, dtype=str)
labels_agg = labels_df.groupby("hadm_id")["cpt_codes"].apply(
    lambda x: " | ".join(sorted(x))
).reset_index()
df = inputs_df.merge(labels_agg, on="hadm_id")
if N_SAMPLES:
    df = df.head(N_SAMPLES)
print(f"  {len(df)} samples")

# ── TOKENIZER ────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SRC, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def compute_perplexity(model, tokenizer, df):
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Computing PPL"):
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": str(row["reports"])},
            {"role": "assistant", "content": str(row["cpt_codes"])},
        ]

        # Full sequence
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False
        )
        # Prefix without assistant response (to find where response starts)
        prefix_messages = messages[:2]
        prefix_text = tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )

        full_ids   = tokenizer(full_text,   return_tensors="pt").input_ids
        prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids

        prefix_len = prefix_ids.shape[1]
        full_len   = full_ids.shape[1]

        if full_len <= prefix_len:
            continue

        input_ids = full_ids.to(model.device if hasattr(model, 'device') else next(model.parameters()).device)

        # Labels: -100 for prefix, real tokens for response
        labels = input_ids.clone()
        labels[:, :prefix_len] = -100

        with torch.no_grad():
            out = model(input_ids=input_ids, labels=labels)
            nll = out.loss.item()

        n_response_tokens = full_len - prefix_len
        total_nll    += nll * n_response_tokens
        total_tokens += n_response_tokens

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    return ppl

# ── MAIN LOOP ────────────────────────────────────────────────────────────────
results = {}

for label, base_path, adapter_path in MODELS:
    print(f"\n{'='*50}")
    print(f"  Loading {label}: {base_path}")
    if adapter_path:
        print(f"  + adapter: {adapter_path}")

    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.config.use_cache = False

    ppl = compute_perplexity(model, tokenizer, df)
    results[label] = round(ppl, 4)
    print(f"\n  {label} Perplexity: {ppl:.4f}")

    del model
    torch.cuda.empty_cache()

# ── SAVE + PRINT ─────────────────────────────────────────────────────────────
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*50)
print("  PERPLEXITY RESULTS (test set, response tokens)")
print("="*50)
for label, ppl in results.items():
    print(f"  {label:6s}: {ppl:.4f}")
print(f"\nSaved → {OUT_JSON}")
