#!/usr/bin/env python3
"""
BLEU / ROUGE / Perplexity evaluation — BASE Qwen3-32B on radiology reports.

Expects: radiology_holdout.jsonl  (built by build_rad_holdout.py)
  Each line: {"hadm_id": ..., "procedure": "CHEST (PA AND LAT)",
              "indication": "Shortness of breath.", "reference": "FINDINGS: ..."}

Run:
    python bleu_rouge_radiology_base.py
"""

import os, json, math, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from tqdm import tqdm
from datetime import datetime

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_MODEL     = "Qwen/Qwen3-32B"
DATA_PATH      = "radiology_holdout.jsonl"
N_EVAL         = 100
MAX_NEW_TOK    = 512
PROMPT_MAX_TOK = 512
GPU            = 0
SEED           = 42
OUTPUT_DIR     = "bleu_results_rad"
TAG            = "base_qwen3-32b"
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)


# ── PROMPT BUILDER (no CPT code in prompt) ────────────────────
def build_prompt(example: dict) -> str:
    procedure  = example.get("procedure", "")
    indication = example.get("indication", "")

    prompt = "Generate a radiology report for the following procedure.\n\n"
    prompt += f"Procedure: {procedure}\n"
    if indication.strip():
        prompt += f"Indication: {indication}\n"
    prompt += "\nFINDINGS:"
    return prompt


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
print(f"Loading data from {DATA_PATH} ...")
with open(DATA_PATH) as f:
    data = [json.loads(line) for line in f]

if N_EVAL is not None:
    import random
    random.seed(SEED)
    data = random.sample(data, min(N_EVAL, len(data)))

print(f"Evaluating on {len(data)} examples")

# ── ROUGE SCORER ─────────────────────────────────────────────────────────────
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

# ── GENERATION + PERPLEXITY ──────────────────────────────────────────────────
records = []

for ex in tqdm(data, desc="Generating"):
    ref_text = ex.get("reference", "").strip()
    if len(ref_text) < 50:
        continue

    prompt = build_prompt(ex)

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=PROMPT_MAX_TOK,
    ).to(f"cuda:{GPU}")

    # ── Generate ──
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

    # Truncate reference to same word count for fair BLEU
    gen_words = gen_text.split()
    ref_words = ref_text.split()
    ref_truncated = " ".join(ref_words[:len(gen_words)])

    # ── Perplexity on reference text (prompt + reference) ──
    full_text = prompt + " " + ref_text
    full_enc  = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=PROMPT_MAX_TOK + MAX_NEW_TOK,
    ).to(f"cuda:{GPU}")

    prompt_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(
            input_ids=full_enc["input_ids"],
            attention_mask=full_enc["attention_mask"],
        )
        logits = outputs.logits  # (1, seq_len, vocab)

    # Compute loss only on the reference portion (after prompt tokens)
    shift_logits = logits[:, prompt_len - 1 : -1, :].contiguous()
    shift_labels = full_enc["input_ids"][:, prompt_len:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss()
    if shift_labels.shape[1] > 0:
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        ppl = math.exp(min(loss.item(), 100))  # cap to avoid overflow
    else:
        ppl = float("nan")

    # ── ROUGE ──
    rouge_scores = scorer.score(ref_truncated, gen_text)

    records.append({
        "hadm_id":        ex.get("hadm_id", ""),
        "procedure":      ex.get("procedure", ""),
        "prompt":         prompt,
        "reference_full": ref_text,
        "reference":      ref_truncated,
        "hypothesis":     gen_text,
        "perplexity":     round(ppl, 4),
        "rouge1_f":       round(rouge_scores["rouge1"].fmeasure, 4),
        "rouge2_f":       round(rouge_scores["rouge2"].fmeasure, 4),
        "rougeL_f":       round(rouge_scores["rougeL"].fmeasure, 4),
    })

# ── SAVE PER-SAMPLE CSV ───────────────────────────────────────────────────────
results_df = pd.DataFrame(records)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
samples_path = f"{OUTPUT_DIR}/rad_samples_{TAG}_{ts}.csv"
results_df.to_csv(samples_path, index=False)
print(f"\n✓ Per-sample results → {samples_path}")

# ── CORPUS METRICS ───────────────────────────────────────────────────────────
hypotheses = results_df["hypothesis"].tolist()
references = results_df["reference"].tolist()

print(f"Computing BLEU over {len(hypotheses)} samples ...")
bleu_metric = BLEU(tokenize="13a")
bleu_result = bleu_metric.corpus_score(hypotheses, [references])

avg_ppl     = round(results_df["perplexity"].mean(), 4)
avg_rouge1  = round(results_df["rouge1_f"].mean(), 4)
avg_rouge2  = round(results_df["rouge2_f"].mean(), 4)
avg_rougeL  = round(results_df["rougeL_f"].mean(), 4)

summary = {
    "model":           TAG,
    "n_eval":          len(hypotheses),
    "max_new_tok":     MAX_NEW_TOK,
    "prompt_max_tok":  PROMPT_MAX_TOK,
    "bleu":            round(bleu_result.score, 4),
    "brevity_penalty": round(bleu_result.bp, 4),
    "p1":              round(bleu_result.precisions[0], 4),
    "p2":              round(bleu_result.precisions[1], 4),
    "p3":              round(bleu_result.precisions[2], 4),
    "p4":              round(bleu_result.precisions[3], 4),
    "avg_perplexity":  avg_ppl,
    "avg_rouge1_f":    avg_rouge1,
    "avg_rouge2_f":    avg_rouge2,
    "avg_rougeL_f":    avg_rougeL,
    "timestamp":       ts,
}

summary_path = f"{OUTPUT_DIR}/rad_summary_{TAG}_{ts}.csv"
pd.DataFrame([summary]).to_csv(summary_path, index=False)
print(f"✓ Summary → {summary_path}")

print("\n" + "=" * 60)
print(f"  Model         : {BASE_MODEL} (no adapter)")
print(f"  BLEU score    : {bleu_result.score:.2f}")
print(f"  Brevity Pen   : {bleu_result.bp:.4f}")
print(f"  n-gram prec   : {[round(p, 2) for p in bleu_result.precisions]}")
print(f"  Avg Perplexity: {avg_ppl:.2f}")
print(f"  ROUGE-1 (F)   : {avg_rouge1:.4f}")
print(f"  ROUGE-2 (F)   : {avg_rouge2:.4f}")
print(f"  ROUGE-L (F)   : {avg_rougeL:.4f}")
print(f"  Num samples   : {len(hypotheses)}")
print("=" * 60)
