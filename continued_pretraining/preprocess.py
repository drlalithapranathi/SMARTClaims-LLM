#!/usr/bin/env python3
"""
Pre-tokenize discharge notes CSV into a HuggingFace Arrow dataset.

Run ONCE before training:
    python preprocess.py

Output: tokenized_chunks_200k/  (~400K chunks, ready for DDP training)
"""

import os
import csv
import time
from transformers import AutoTokenizer
from datasets import Dataset

# ========== CONFIG ==========
MODEL_NAME   = "Qwen/Qwen3-32B"
DATA_PATH    = "cleaned_discharge_notes_200k_fixed.csv"
TEXT_COLUMN  = "cleaned_text"
MAX_LEN      = 2048
MIN_LEN      = 128
OUTPUT_DIR   = "tokenized_chunks_200k"

# ========== TOKENIZER ==========
print(f"Loading tokenizer from {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

EOS_ID = tokenizer.eos_token_id
print(f"  EOS id : {EOS_ID}")
print(f"  PAD id : {tokenizer.pad_token_id}")
print(f"  Vocab  : {len(tokenizer):,}\n")

# ========== GENERATOR (memory-efficient) ==========
def chunk_generator():
    """Read CSV row-by-row, tokenize, chunk, yield dicts."""
    skipped = 0
    total_chunks = 0

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i % 10_000 == 0:
                print(f"  row {i:,}  chunks so far: {total_chunks:,}", flush=True)

            text = row.get(TEXT_COLUMN)
            if not text or str(text).strip() in ("", "nan"):
                skipped += 1
                continue

            tokens = tokenizer(str(text), add_special_tokens=False)["input_ids"]
            tokens = tokens + [EOS_ID]   # append EOS at end of each note

            for j in range(0, len(tokens), MAX_LEN):
                chunk = tokens[j : j + MAX_LEN]
                if len(chunk) >= MIN_LEN:
                    yield {
                        "input_ids":      chunk,
                        "attention_mask": [1] * len(chunk),
                    }
                    total_chunks += 1

    print(f"\n  Skipped {skipped:,} empty rows")
    print(f"  Total chunks yielded: {total_chunks:,}")

# ========== BUILD & SAVE DATASET ==========
if os.path.exists(OUTPUT_DIR):
    print(f"⚠️  {OUTPUT_DIR}/ already exists — delete it first if you want to rerun.")
else:
    print(f"Tokenizing {DATA_PATH} ...")
    t0 = time.time()

    # writer_batch_size keeps RAM usage low (~1k chunks buffered at a time)
    dataset = Dataset.from_generator(chunk_generator, writer_batch_size=1_000)
    dataset = dataset.shuffle(seed=42)
    dataset.save_to_disk(OUTPUT_DIR)

    elapsed = time.time() - t0
    print(f"\n✓ Saved {len(dataset):,} chunks → {OUTPUT_DIR}/")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Estimated training steps @ eff-batch 96 (8×4×3 GPUs): {len(dataset)//96:,}")
    print(f"\nNext step:")
    print(f"  accelerate launch --num_processes 3 --mixed_precision bf16 train_cpt.py")
