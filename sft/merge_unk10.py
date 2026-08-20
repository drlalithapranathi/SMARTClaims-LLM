#!/usr/bin/env python3
"""
Merge unk10 SFT adapter into full model.
Output: unk10_sft_merged/

Run:
    python merge_unk10.py
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL   = "../qwen3-32b-mimic-cpt-merged"
ADAPTER_PATH = "sft_adapter"
MERGED_OUT   = "unk10_sft_merged"
TOKENIZER    = "../qwen3-32b-mimic-cpt-200k-ep2"

print(f"Loading base model from {BASE_MODEL} ...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print(f"Loading adapter from {ADAPTER_PATH} ...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("Merging ...")
model = model.merge_and_unload()

print(f"Saving merged model to {MERGED_OUT}/ ...")
os.makedirs(MERGED_OUT, exist_ok=True)
model.save_pretrained(MERGED_OUT, safe_serialization=True, max_shard_size="5GB")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
tokenizer.save_pretrained(MERGED_OUT)

print(f"✓ Done → {MERGED_OUT}/")
