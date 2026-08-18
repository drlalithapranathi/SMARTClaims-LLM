#!/usr/bin/env python3
"""
Recovery: training completed but merge step OOM'd.
1. Copy adapter from checkpoint-342 → qwen3-32b-mimic-sft-v7/
2. Merge base + CPT adapter + SFT adapter → qwen3-32b-mimic-sft-v7-merged/

Run on a single process (no accelerate):
    python recover_sft_v7.py
"""
import os
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

CHECKPOINT   = "outputs_sft_v7_qwen3_32b/checkpoint-342/sft"   # named adapter subdir
TOKENIZER_CHECKPOINT = "outputs_sft_v7_qwen3_32b/checkpoint-342"  # tokenizer files in root
SAVE_ADAPTER = "qwen3-32b-mimic-sft-v7"
SAVE_MERGED  = "qwen3-32b-mimic-sft-v7-merged"
BASE_MODEL   = "Qwen/Qwen3-32B"
CPT_ADAPTER  = "qwen3-32b-mimic-cpt-200k-ep2"

ADAPTER_FILES  = ["adapter_config.json", "adapter_model.safetensors"]
TOKENIZER_FILES = ["added_tokens.json", "chat_template.jinja", "merges.txt",
                   "special_tokens_map.json", "tokenizer_config.json",
                   "tokenizer.json", "vocab.json"]

# ── Step 1: copy adapter ────────────────────────────────────────────────────
print(f"Copying SFT adapter from {CHECKPOINT}/ → {SAVE_ADAPTER}/")
os.makedirs(SAVE_ADAPTER, exist_ok=True)
for fname in ADAPTER_FILES:
    src = os.path.join(CHECKPOINT, fname)
    shutil.copy2(src, os.path.join(SAVE_ADAPTER, fname))
    print(f"  copied {fname}")
for fname in TOKENIZER_FILES:
    src = os.path.join(TOKENIZER_CHECKPOINT, fname)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(SAVE_ADAPTER, fname))
        print(f"  copied {fname}")
print(f"✓ SFT adapter saved → {SAVE_ADAPTER}/")

# ── Step 2: merge ───────────────────────────────────────────────────────────
if os.path.exists(SAVE_MERGED):
    print(f"✓ {SAVE_MERGED}/ already exists, skipping merge")
else:
    print(f"\nLoading base model {BASE_MODEL} ...")
    m = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Merging CPT adapter ({CPT_ADAPTER}) ...")
    m = PeftModel.from_pretrained(m, CPT_ADAPTER)
    m = m.merge_and_unload()
    print("  ✓ CPT adapter merged")

    print(f"Merging SFT adapter ({SAVE_ADAPTER}) ...")
    m = PeftModel.from_pretrained(m, SAVE_ADAPTER)
    m = m.merge_and_unload()
    print("  ✓ SFT adapter merged")

    print(f"Saving merged model → {SAVE_MERGED}/ ...")
    os.makedirs(SAVE_MERGED, exist_ok=True)
    m.save_pretrained(SAVE_MERGED, safe_serialization=True, max_shard_size="5GB")

    tokenizer = AutoTokenizer.from_pretrained(CPT_ADAPTER, trust_remote_code=True)
    tokenizer.save_pretrained(SAVE_MERGED)
    print(f"✓ Full merged model saved → {SAVE_MERGED}/")

print("\n" + "=" * 50)
print("  RECOVERY COMPLETE")
print(f"  SFT adapter  : {SAVE_ADAPTER}/")
print(f"  Merged model : {SAVE_MERGED}/")
print("=" * 50)
