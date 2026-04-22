"""
Compare WITH RadLex vs WITHOUT RadLex models
Run after training both models
"""
from unsloth import FastLanguageModel
import pandas as pd

# Paths - UPDATE THESE
RADLEX_MODEL = "YOUR_RADLEX_MODEL_PATH/lora_model"
NO_RADLEX_MODEL = "YOUR_NO_RADLEX_MODEL_PATH/lora_model"
TEST_DATA = "YOUR_TEST_DATA_PATH/train_dataset.jsonl"

# Load both models
print("Loading WITH-RadLex model...")
model_radlex, tokenizer_radlex = FastLanguageModel.from_pretrained(
    model_name=RADLEX_MODEL, max_seq_length=4096, dtype=None, load_in_4bit=True,
)
FastLanguageModel.for_inference(model_radlex)

print("Loading NO-RadLex model...")
model_no_radlex, tokenizer_no_radlex = FastLanguageModel.from_pretrained(
    model_name=NO_RADLEX_MODEL, max_seq_length=4096, dtype=None, load_in_4bit=True,
)
FastLanguageModel.for_inference(model_no_radlex)

def extract_procedures(report_text, model, tokenizer):
    prompt = f"""Radiology Report:
{report_text}

Based on the above radiology report, list all the billable procedures performed.
Format as a comma-separated list of procedure names.

Billable Procedures:"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.3, top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Billable Procedures:" in generated:
        text = generated.split("Billable Procedures:")[-1].strip()
    else:
        text = generated[len(prompt):].strip()
    return [p.strip() for p in text.split(",") if p.strip()]

# Test on unseen reports (indices 1000-1020)
test_df = pd.read_json(TEST_DATA, lines=True).iloc[1000:1020]
print(f"Testing on {len(test_df)} unseen reports...\n")

results = []
for idx, row in test_df.iterrows():
    report_with_radlex = row['text']
    report_no_radlex = report_with_radlex.split("Radiology Report:")[-1].strip() if "Radiology Report:" in report_with_radlex else report_with_radlex

    proc_radlex = extract_procedures(report_with_radlex, model_radlex, tokenizer_radlex)
    proc_no_radlex = extract_procedures(report_no_radlex, model_no_radlex, tokenizer_no_radlex)

    results.append({
        'report_id': idx,
        'with_radlex': ', '.join(proc_radlex),
        'without_radlex': ', '.join(proc_no_radlex),
        'radlex_count': len(proc_radlex),
        'no_radlex_count': len(proc_no_radlex),
    })
    print(f"Report {idx}: RadLex={len(proc_radlex)}, NoRadLex={len(proc_no_radlex)}")

results_df = pd.DataFrame(results)
results_df.to_csv('ablation_results.csv', index=False)

print(f"\nResults saved to ablation_results.csv")
print(f"\nAverage procedures extracted:")
print(f"  WITH RadLex:    {results_df['radlex_count'].mean():.2f}")
print(f"  WITHOUT RadLex: {results_df['no_radlex_count'].mean():.2f}")
