#!/usr/bin/env python3
"""
Sharded GRPO eval for unk10 ep2.
Loads unk10_sft_merged + unk10_grpo_adapter, runs inference on one shard,
saves raw predictions to shard_{idx}_predictions.json.

Run one per GPU:
    python eval_grpo_sharded.py --shard-idx 0 --n-shards 4
    python eval_grpo_sharded.py --shard-idx 1 --n-shards 4
    python eval_grpo_sharded.py --shard-idx 2 --n-shards 4
    python eval_grpo_sharded.py --shard-idx 3 --n-shards 4

Then run merge_eval_shards.py to compute final metrics.
"""

import os
import re
import json
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

UNK10_SFT_MERGED   = "unk10_sft_merged"
UNK10_GRPO_ADAPTER = "unk10_grpo_adapter"
TOKENIZER_SRC      = "../qwen3-32b-mimic-cpt-200k-ep2"
TEST_INPUTS_CSV    = "../sft_test_inputs.csv"
TEST_LABELS_CSV    = "../sft_test_labels.csv"
MAX_NEW_TOKENS     = 128

SYSTEM_PROMPT = (
    "You are a medical coding assistant. "
    "Given a clinical note for a patient admission, identify all billable "
    "radiology procedures performed. Output only the CPT codes separated by "
    "pipe ( | ). If no billable radiology procedure can be determined, output "
    "unknown. Do not include add-on codes, explanations, or numbering."
)

parser = argparse.ArgumentParser()
parser.add_argument("--shard-idx", type=int, required=True)
parser.add_argument("--n-shards",  type=int, required=True)
args = parser.parse_args()
assert 0 <= args.shard_idx < args.n_shards

print(f"\n[Shard {args.shard_idx}/{args.n_shards}] Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SRC, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"[Shard {args.shard_idx}] Loading {UNK10_SFT_MERGED} ...")
model = AutoModelForCausalLM.from_pretrained(
    UNK10_SFT_MERGED,
    quantization_config=bnb_config,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
model.config.use_cache = True

print(f"[Shard {args.shard_idx}] Loading GRPO adapter from {UNK10_GRPO_ADAPTER} ...")
model = PeftModel.from_pretrained(model, UNK10_GRPO_ADAPTER)
model.eval()
print(f"[Shard {args.shard_idx}] Model ready.\n")

print(f"[Shard {args.shard_idx}] Loading test data ...")
inputs_df = pd.read_csv(TEST_INPUTS_CSV, dtype={"hadm_id": str})
labels_df = pd.read_csv(TEST_LABELS_CSV, dtype={"hadm_id": str, "cpt_codes": str})

gt_agg = (
    labels_df.groupby("hadm_id")["cpt_codes"]
    .apply(lambda x: sorted(set(x.str.strip())))
    .reset_index()
    .rename(columns={"cpt_codes": "gt_codes"})
)
test_df = inputs_df.merge(gt_agg, on="hadm_id", how="inner").reset_index(drop=True)

total      = len(test_df)
shard_size = (total + args.n_shards - 1) // args.n_shards
start      = args.shard_idx * shard_size
end        = min(start + shard_size, total)
shard_df   = test_df.iloc[start:end].reset_index(drop=True)

print(f"[Shard {args.shard_idx}] Rows {start}-{end-1} ({len(shard_df)} of {total} total)\n")


def build_prompt(report_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": report_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def parse_prediction(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.strip().lower() == "unknown":
        return ["unknown"]
    codes = [c.strip() for c in text.split("|") if re.match(r"^\d{5}$", c.strip())]
    return sorted(set(codes))


raw_outputs = []
print(f"[Shard {args.shard_idx}] Running inference ...")
for _, row in tqdm(shard_df.iterrows(), total=len(shard_df), desc=f"Shard {args.shard_idx}"):
    prompt = build_prompt(row["reports"])
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=4096,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, temperature=None, top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    decoded   = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    pred      = parse_prediction(decoded)

    raw_outputs.append({
        "hadm_id":      row["hadm_id"],
        "predicted":    pred,
        "ground_truth": row["gt_codes"],
        "raw_output":   decoded,
    })

out_file = f"shard_{args.shard_idx}_predictions.json"
with open(out_file, "w") as f:
    json.dump(raw_outputs, f, indent=2)

print(f"\n[Shard {args.shard_idx}] Done. Saved -> {out_file}")
