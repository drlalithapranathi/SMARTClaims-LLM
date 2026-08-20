#!/usr/bin/env python3
"""
Upload SmartClaims models to HuggingFace Hub.

  SFT base  : qwen3-32b-sft-v9-merged  → lalithapranathipulavarthy/smartclaims-sft-v9
  GRPO LoRA : qwen3-32b-grpo-v3b       → lalithapranathipulavarthy/smartclaims-grpo-v3b-adapter

Usage:
    huggingface-cli login   # once, to store your token
    python upload_sft_v9_grpo_v3b_to_hf.py
"""

import os
from huggingface_hub import HfApi, create_repo

BASE_DIR = "."

UPLOADS = [
    {
        "local_dir": f"{BASE_DIR}/qwen3-32b-sft-v9-merged",
        "repo_id":   "lalithapranathipulavarthy/smartclaims-sft-v9",
        "repo_type": "model",
        "commit_msg": "Upload SmartClaims SFT v9 merged model (CPT + SFT baked in)",
    },
    {
        "local_dir": f"{BASE_DIR}/qwen3-32b-grpo-v3b",
        "repo_id":   "lalithapranathipulavarthy/smartclaims-grpo-v3b-adapter",
        "repo_type": "model",
        "commit_msg": "Upload SmartClaims GRPO v3b LoRA adapter (base: smartclaims-sft-v9)",
    },
]

api = HfApi()

for u in UPLOADS:
    print(f"\n{'='*60}")
    print(f"  Repo  : {u['repo_id']}")
    print(f"  Local : {u['local_dir']}")
    print(f"{'='*60}")

    # Create repo if it doesn't exist (private by default — change to private=False if you want public)
    create_repo(u["repo_id"], repo_type=u["repo_type"], exist_ok=True, private=True)
    print(f"  ✓ Repo ready")

    print(f"  Uploading (this will take a while for the 62GB base)...")
    api.upload_folder(
        folder_path=u["local_dir"],
        repo_id=u["repo_id"],
        repo_type=u["repo_type"],
        commit_message=u["commit_msg"],
    )
    print(f"  ✓ Done → https://huggingface.co/{u['repo_id']}")

print("\n✓ All uploads complete.")
