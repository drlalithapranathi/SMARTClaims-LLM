#!/usr/bin/env python3
"""
Merges unk10_grpo_adapter into unk10_sft_merged to produce a single
standalone model, then uploads it to Hugging Face.

Result on HF: lalithapranathipulavarthy/smartclaims-grpo-unk10
  — fully self-contained, no adapter needed at inference

Run:
    huggingface-cli login          # once, to store token
    python merge_and_upload_grpo.py
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, create_repo

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = "."
SFT_MERGED   = f"{BASE_DIR}/unk10_sft_merged"      # full SFT model (CPT+SFT merged)
GRPO_ADAPTER = f"{BASE_DIR}/unk10_grpo_adapter"    # GRPO LoRA adapter
OUT_DIR      = f"{BASE_DIR}/unk10_grpo_merged"     # where merged model is saved
HF_REPO      = "lalithapranathipulavarthy/smartclaims-grpo-unk10"

# ── step 1: merge ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  Step 1: Loading SFT merged model ...")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(
    SFT_MERGED,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(SFT_MERGED, trust_remote_code=True)
print(f"  ✓ Base loaded from {SFT_MERGED}")

print("\n  Applying GRPO LoRA adapter ...")
model = PeftModel.from_pretrained(model, GRPO_ADAPTER)
print(f"  ✓ Adapter loaded from {GRPO_ADAPTER}")

print("\n  Merging and unloading adapter into base weights ...")
model = model.merge_and_unload()
print("  ✓ Merge complete — single standalone model ready")

print(f"\n  Saving merged model to {OUT_DIR} ...")
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, safe_serialization=True, max_shard_size="5GB")
tokenizer.save_pretrained(OUT_DIR)
print(f"  ✓ Saved to {OUT_DIR}/")

# ── step 2: upload ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  Step 2: Uploading to HF → {HF_REPO}")
print("=" * 60)

api = HfApi()
create_repo(HF_REPO, repo_type="model", exist_ok=True, private=True)
print("  ✓ Repo ready")

print("  Uploading (large model — this will take a while) ...")
api.upload_folder(
    folder_path=OUT_DIR,
    repo_id=HF_REPO,
    repo_type="model",
    commit_message="SMARTClaims final model: Qwen3-32B CPT+SFT+GRPO merged (unk10 ep2)",
)
print(f"\n  ✓ Done → https://huggingface.co/{HF_REPO}")
print("\n" + "=" * 60)
print("  UPLOAD COMPLETE")
print(f"  Model : {HF_REPO}")
print(f"  Usage : AutoModelForCausalLM.from_pretrained('{HF_REPO}')")
print("=" * 60)
