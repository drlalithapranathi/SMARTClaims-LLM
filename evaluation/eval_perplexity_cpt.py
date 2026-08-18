#!/usr/bin/env python3
"""
Perplexity evaluation of the base model and continued-pretraining (CPT)
adapters on held-out discharge notes. Lower perplexity = better language modeling.

Run:
    python eval_perplexity_cpt.py
"""

import os, math, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from datetime import datetime

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen3-32B"
ADAPTERS     = [
    None,                              # base Qwen3-32B, no adapter
    "qwen3-32b-mimic-cpt-200k-ep2",   # CPT epoch 2 (used for SFT)
    "qwen3-32b-mimic-cpt-200k",        # CPT epoch 1 — for comparison
]
CSV_PATH     = "cleaned_holdout_notes.csv"
N_EVAL       = 100
MAX_TOK      = 2048    # tokens per note (truncate if longer)
GPU          = 0
SEED         = 42
OUTPUT_DIR   = "perplexity_results"
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)

# ── LOAD DATA ────────────────────────────────────────────────────────────────
print("Loading holdout CSV ...")
df = pd.read_csv(CSV_PATH)
eval_df = df.sample(n=min(N_EVAL, len(df)), random_state=SEED).reset_index(drop=True)
print(f"Evaluating on {len(eval_df)} notes")

# ── LOAD TOKENIZER ───────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

results = []

for adapter in ADAPTERS:
    print(f"\n{'='*50}")
    print(f"Evaluating: {adapter or 'base (no adapter)'}")
    print(f"{'='*50}")

    # Load base model
    print("Loading base model ...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map={"": GPU},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    base.config.use_cache = True

    if adapter:
        print(f"Loading adapter ...")
        model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    else:
        model = base
    model.eval()

    total_loss   = 0.0
    total_tokens = 0

    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Computing perplexity"):
        text = row["cleaned_text"]
        if not isinstance(text, str) or len(text.strip()) < 50:
            continue

        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOK,
        )
        input_ids = enc["input_ids"].to(f"cuda:{GPU}")
        n_tokens  = input_ids.shape[1]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss    = outputs.loss.item()

        total_loss   += loss * n_tokens
        total_tokens += n_tokens

    avg_loss   = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    print(f"\n  Adapter    : {adapter or 'base (no adapter)'}")
    print(f"  Avg loss   : {avg_loss:.4f}")
    print(f"  Perplexity : {perplexity:.2f}")

    results.append({
        "adapter":     adapter or "base",
        "n_eval":      len(eval_df),
        "max_tok":     MAX_TOK,
        "avg_loss":    round(avg_loss, 4),
        "perplexity":  round(perplexity, 2),
        "timestamp":   datetime.now().strftime("%Y%m%d_%H%M%S"),
    })

    # Free VRAM before next adapter
    del model, base
    torch.cuda.empty_cache()

# ── SAVE RESULTS ─────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"{OUTPUT_DIR}/perplexity_results_{ts}.csv"
pd.DataFrame(results).to_csv(out_path, index=False)
print(f"\n✓ Results saved → {out_path}")

print("\n" + "="*50)
print("  PERPLEXITY SUMMARY")
print("="*50)
for r in results:
    print(f"  {r['adapter']}")
    print(f"    Perplexity : {r['perplexity']:.2f}  |  Loss : {r['avg_loss']:.4f}")
print("="*50)
