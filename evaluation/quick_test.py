#!/usr/bin/env python3
import csv, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_DIR = "unk10_grpo_merged"

SYSTEM_PROMPT = (
    "You are a medical coding assistant. "
    "Given a clinical note for a patient admission, identify all billable "
    "radiology procedures performed. Output only the CPT codes separated by "
    "pipe ( | ). If no billable radiology procedure can be determined, output "
    "unknown. Do not include add-on codes, explanations, or numbering."
)

# load 3 test samples
samples = []
with open("../sft_test_inputs.csv") as f:
    for i, row in enumerate(csv.DictReader(f)):
        samples.append(row)
        if i == 2:
            break

labels = {}
with open("../sft_test_labels.csv") as f:
    for row in csv.DictReader(f):
        labels[row["hadm_id"]] = row["cpt_codes"]

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading model (4-bit)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb_config,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.eval()
print("Model loaded.\n")

for s in samples:
    hadm_id = s["hadm_id"]
    report  = s["reports"]
    gold    = labels.get(hadm_id, "?")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": report},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda:0")

    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=128, do_sample=False)

    pred = tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()

    print(f"hadm_id : {hadm_id}")
    print(f"gold    : {gold}")
    print(f"pred    : {pred}")
    print("-" * 60)
