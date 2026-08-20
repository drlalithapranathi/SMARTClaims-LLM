#!/usr/bin/env python3
"""
Step 0: Merge Qwen3-32B base + CPT adapter → qwen3-32b-mimic-cpt-merged/

This creates the merged base for SFT. Run this ONCE before sft/train_sft.py.

Run:
    python merge_cpt.py
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL  = "Qwen/Qwen3-32B"
CPT_ADAPTER = "qwen3-32b-mimic-cpt-200k-ep2"
SAVE_TO     = "qwen3-32b-mimic-cpt-merged"

if os.path.exists(SAVE_TO) and os.listdir(SAVE_TO):
    print(f"✓ {SAVE_TO}/ already exists — skipping merge.")
    exit(0)

print(f"Loading base model {BASE_MODEL} ...")
m = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print(f"Merging CPT adapter from {CPT_ADAPTER} ...")
m = PeftModel.from_pretrained(m, CPT_ADAPTER)
m = m.merge_and_unload()
print("  ✓ CPT adapter merged")

print(f"Saving → {SAVE_TO}/ ...")
os.makedirs(SAVE_TO, exist_ok=True)
m.save_pretrained(SAVE_TO, safe_serialization=True, max_shard_size="5GB")

tokenizer = AutoTokenizer.from_pretrained(CPT_ADAPTER, trust_remote_code=True)
tokenizer.save_pretrained(SAVE_TO)

print(f"\n✓ CPT-merged model saved → {SAVE_TO}/")
print("Next: accelerate launch --num_processes 3 --mixed_precision bf16 train_sft.py  (from sft/)")
