#!/usr/bin/env python3
"""
vLLM-based eval for exp_nounk SFT — fast inference.

Run:
    python eval_nounk_vllm.py
"""

import os
import re
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from vllm import LLM, SamplingParams
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report

NOUNK_SFT_MERGED = "nounk_sft_merged"
TOKENIZER_SRC    = "../qwen3-32b-mimic-cpt-200k-ep2"
TEST_INPUTS_CSV  = "../sft_test_inputs.csv"
TEST_LABELS_CSV  = "../sft_test_labels.csv"
MAX_NEW_TOKENS   = 128

SYSTEM_PROMPT = (
    "You are a medical coding assistant. "
    "Given a clinical note for a patient admission, identify all billable "
    "radiology procedures performed. Output only the CPT codes separated by "
    "pipe ( | ). If no billable radiology procedure can be determined, output "
    "unknown. Do not include add-on codes, explanations, or numbering."
)

print("Loading test data ...")
inputs_df = pd.read_csv(TEST_INPUTS_CSV, dtype={"hadm_id": str})
labels_df = pd.read_csv(TEST_LABELS_CSV, dtype={"hadm_id": str, "cpt_codes": str})

gt_agg = (
    labels_df.groupby("hadm_id")["cpt_codes"]
    .apply(lambda x: sorted(set(x.str.strip())))
    .reset_index()
    .rename(columns={"cpt_codes": "gt_codes"})
)
test_df = inputs_df.merge(gt_agg, on="hadm_id", how="inner").reset_index(drop=True)
print(f"  Test admissions: {len(test_df)}")

print(f"\nLoading {NOUNK_SFT_MERGED} with vLLM ...")
llm = LLM(
    model=NOUNK_SFT_MERGED,
    tokenizer=TOKENIZER_SRC,
    dtype="bfloat16",
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    max_model_len=4096,
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
    enforce_eager=False,
)
print("Model loaded.\n")

sampling_params = SamplingParams(
    max_tokens=MAX_NEW_TOKENS,
    temperature=0.0,
    stop_token_ids=llm.get_tokenizer().eos_token_id
        if hasattr(llm.get_tokenizer().eos_token_id, '__iter__') else
        [llm.get_tokenizer().eos_token_id],
)

tokenizer = llm.get_tokenizer()


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


print("Building prompts ...")
prompts  = [build_prompt(row["reports"]) for _, row in test_df.iterrows()]
hadm_ids = test_df["hadm_id"].tolist()
gt_list  = test_df["gt_codes"].tolist()

print(f"Running vLLM inference on {len(prompts)} samples ...")
outputs = llm.generate(prompts, sampling_params)

predictions, ground_truths, raw_outputs = [], [], []
for i, out in enumerate(tqdm(outputs, desc="Parsing")):
    decoded = out.outputs[0].text
    pred    = parse_prediction(decoded)
    predictions.append(pred)
    ground_truths.append(gt_list[i])
    raw_outputs.append({
        "hadm_id":      hadm_ids[i],
        "predicted":    pred,
        "ground_truth": gt_list[i],
        "raw_output":   decoded,
    })

print("\nComputing metrics ...")
all_labels = sorted(set(l for lst in ground_truths + predictions for l in lst))
mlb    = MultiLabelBinarizer(classes=all_labels)
y_true = mlb.fit_transform(ground_truths)
y_pred = mlb.transform(predictions)

f1_macro   = f1_score(y_true, y_pred, average="macro",   zero_division=0)
f1_micro   = f1_score(y_true, y_pred, average="micro",   zero_division=0)
f1_samples = f1_score(y_true, y_pred, average="samples", zero_division=0)

print(f"\n{'='*55}")
print(f"  Experiment : nounk — SFT (vLLM)")
print(f"  F1 Samples : {f1_samples:.4f}")
print(f"  F1 Micro   : {f1_micro:.4f}")
print(f"  F1 Macro   : {f1_macro:.4f}")
print(f"  N samples  : {len(test_df)}")
print(f"{'='*55}\n")

report_dict = classification_report(
    y_true, y_pred, target_names=mlb.classes_, zero_division=0, output_dict=True
)

ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
out_json = f"eval_nounk_sft_vllm_results_{ts}.json"
out_csv  = f"eval_nounk_sft_vllm_per_code_{ts}.csv"

results = {
    "experiment":  "nounk",
    "mode":        "sft",
    "engine":      "vllm",
    "model":       NOUNK_SFT_MERGED,
    "timestamp":   ts,
    "f1_samples":  round(f1_samples, 4),
    "f1_micro":    round(f1_micro,   4),
    "f1_macro":    round(f1_macro,   4),
    "n_test":      len(test_df),
    "n_labels":    len(all_labels),
    "per_code_f1": {k: v for k, v in report_dict.items()
                    if k not in ["accuracy","macro avg","weighted avg","samples avg","micro avg"]},
    "predictions": raw_outputs,
}

with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

per_code_rows = [
    {"cpt_code": c, "precision": round(v["precision"],4),
     "recall": round(v["recall"],4), "f1": round(v["f1-score"],4), "support": v["support"]}
    for c, v in report_dict.items()
    if c not in ["accuracy","macro avg","weighted avg","samples avg","micro avg"]
]
pd.DataFrame(per_code_rows).to_csv(out_csv, index=False)

print(f"✓ Results saved → {out_json}")
print(f"✓ Per-code F1  → {out_csv}")
