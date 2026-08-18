#!/usr/bin/env python3
"""
BLEU evaluation for base Qwen3-32B (no adapter) — comparison baseline.

Run:
    python eval_bleu_base.py
"""

import os, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from datetime import datetime

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_MODEL     = "Qwen/Qwen3-32B"
CSV_PATH       = "cleaned_holdout_notes.csv"
N_EVAL         = 100
SPLIT_FRAC     = 0.50
MAX_NEW_TOK    = 256
PROMPT_MAX_TOK = 512
GPU            = 0
SEED           = 42
OUTPUT_DIR     = "bleu_results"
ADAPTER_TAG    = "base_qwen3-32b"   # used in output filenames
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)

# ── LOAD TOKENIZER ───────────────────────────────────────────────────────────
print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# ── LOAD MODEL ───────────────────────────────────────────────────────────────
print("Loading base model (4-bit) ...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": GPU},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
model.config.use_cache = True
model.eval()

# ── LOAD DATA ────────────────────────────────────────────────────────────────
print("Loading holdout CSV ...")
df = pd.read_csv(CSV_PATH)
eval_df = df.sample(n=min(N_EVAL, len(df)), random_state=SEED).reset_index(drop=True)
print(f"Evaluating on {len(eval_df)} notes")

# ── GENERATION ───────────────────────────────────────────────────────────────
records = []

for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Generating"):
    text = row["cleaned_text"]
    if not isinstance(text, str) or len(text.strip()) < 100:
        continue

    split_idx   = int(len(text) * SPLIT_FRAC)
    prompt_text = text[:split_idx]
    ref_text    = text[split_idx:]

    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=PROMPT_MAX_TOK,
    ).to(f"cuda:{GPU}")

    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOK,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids  = out_ids[0][enc["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    gen_words = gen_text.split()
    ref_words = ref_text.strip().split()
    ref_truncated = " ".join(ref_words[:len(gen_words)])

    records.append({
        "note_id":        row.get("note_id", ""),
        "prompt":         prompt_text,
        "reference_full": ref_text.strip(),
        "reference":      ref_truncated,
        "hypothesis":     gen_text,
    })

# ── SAVE PER-SAMPLE CSV ───────────────────────────────────────────────────────
results_df = pd.DataFrame(records)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
samples_path = f"{OUTPUT_DIR}/bleu_samples_{ADAPTER_TAG}_{ts}.csv"
results_df.to_csv(samples_path, index=False)
print(f"\n✓ Per-sample results saved → {samples_path}")

# ── BLEU ─────────────────────────────────────────────────────────────────────
hypotheses = results_df["hypothesis"].tolist()
references  = results_df["reference"].tolist()

print(f"Computing BLEU over {len(hypotheses)} samples ...")
bleu   = BLEU(tokenize="13a")
result = bleu.corpus_score(hypotheses, [references])

summary = {
    "adapter":        ADAPTER_TAG,
    "n_eval":         len(hypotheses),
    "split_frac":     SPLIT_FRAC,
    "max_new_tok":    MAX_NEW_TOK,
    "prompt_max_tok": PROMPT_MAX_TOK,
    "bleu":           round(result.score, 4),
    "brevity_penalty":round(result.bp, 4),
    "p1": round(result.precisions[0], 4),
    "p2": round(result.precisions[1], 4),
    "p3": round(result.precisions[2], 4),
    "p4": round(result.precisions[3], 4),
    "timestamp":      ts,
}

summary_path = f"{OUTPUT_DIR}/bleu_summary_{ADAPTER_TAG}_{ts}.csv"
pd.DataFrame([summary]).to_csv(summary_path, index=False)
print(f"✓ Summary saved → {summary_path}")

print("\n" + "="*50)
print(f"  Model       : {BASE_MODEL} (no adapter)")
print(f"  BLEU score  : {result.score:.2f}")
print(f"  Brevity Pen : {result.bp:.4f}")
print(f"  n-gram prec : {[round(p,2) for p in result.precisions]}")
print(f"  Num samples : {len(hypotheses)}")
print("="*50)
