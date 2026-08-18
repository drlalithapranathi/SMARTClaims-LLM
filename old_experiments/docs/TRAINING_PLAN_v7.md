# CPT Code Prediction: Two-Stage SFT + GRPO Training Plan (v7)

## Background
The original SFT model was trained with procedure names (cpt_labels) as targets → F1 = 0 due to free-text mismatch.

**Changes in v7:**
- Labels switched to CPT codes (5-digit, fixed tokens)
- 15% discharge summaries added to training set with "unknown" label
- Stage 1 SFT retrained from scratch with new labels
- Stage 2 GRPO added using set-level F1 as reward

**Existing directories (DO NOT OVERWRITE):**
- `qwen3-32b-mimic-cpt-200k-ep2/` — frozen CPT adapter
- `qwen3-32b-mimic-sft/` — old SFT adapter (procedure names)
- `qwen3-32b-mimic-sft-merged/` — old merged model (procedure names)
- `outputs_sft_qwen3_32b/` — old SFT checkpoints

---

## New Output Directories (created during execution)

| Directory | Created By | Contents |
|---|---|---|
| `sft_v7_train_dataset/` | build_sft_grpo_datasets.py | SFT train parquet |
| `sft_v7_val_dataset/` | build_sft_grpo_datasets.py | SFT val parquet |
| `grpo_v7_train_dataset/` | build_sft_grpo_datasets.py | GRPO train parquet |
| `grpo_v7_val_dataset/` | build_sft_grpo_datasets.py | GRPO val parquet |
| `outputs_sft_v7_qwen3_32b/` | train_sft_v7.py | SFT checkpoints |
| `qwen3-32b-mimic-sft-v7/` | train_sft_v7.py | New SFT adapter |
| `qwen3-32b-mimic-sft-v7-merged/` | train_sft_v7.py | New merged model |
| `outputs_grpo_v7_qwen3_32b/` | train_grpo_v1.py | GRPO checkpoints |
| `qwen3-32b-mimic-sft-v7-grpo/` | train_grpo_v1.py | GRPO LoRA adapter |

---

## Files to Create / Modify

| File | Action |
|---|---|
| `build_sft_grpo_datasets.py` | CREATE |
| `train_sft_v7.py` | CREATE (copy of train_sft_v6_final.py with parquet loading) |
| `train_grpo_v1.py` | CREATE |
| `eval_sft_grpo.py` | CREATE (unified evaluation) |
| `eval_sft_v6.py` | MODIFY — fix system prompt + parse_output only |

---

## Script 1: `build_sft_grpo_datasets.py`

**Purpose:** Single source of truth for dataset preparation. Run once before either training stage.

**Input:** `mimic_radiology_sft_train.csv` (3200 rows: 2783 radiology + 417 discharge/unknown)

**System prompt (used in ALL scripts — must be identical):**
```
You are a medical coding assistant. Given a clinical note for a patient admission,
identify all billable radiology procedures performed. Output only the CPT codes
separated by pipe ( | ). If no billable radiology procedure can be determined,
output unknown. Do not include add-on codes, explanations, or numbering.
```

**Aggregation logic:**
```python
import pandas as pd
from datasets import Dataset

df = pd.read_csv("mimic_radiology_sft_train.csv", dtype={"cpt_codes": str})

agg = df.groupby("hadm_id").agg(
    reports=("reports", "first"),
    cpt_codes=("cpt_codes", lambda x: " | ".join(sorted(x))),
).reset_index()
# Result: ~2017 rows (1600 radiology admissions + 417 discharge admissions)
# "unknown" admissions aggregate to single "unknown" label
```

**SFT dataset format:**
```python
records = []
for _, row in agg.iterrows():
    records.append({
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": row["reports"]},
            {"role": "assistant", "content": row["cpt_codes"]},
        ]
    })

dataset = Dataset.from_list(records).shuffle(seed=42)
splits = dataset.train_test_split(test_size=0.1, seed=42)
splits["train"].save_to_disk("sft_v7_train_dataset")
splits["test"].save_to_disk("sft_v7_val_dataset")
```

**GRPO dataset format (no assistant turn, adds ground_truth column):**
```python
records = []
for _, row in agg.iterrows():
    records.append({
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": row["reports"]},
        ],
        "ground_truth":         row["cpt_codes"],
        "hadm_id":              str(row["hadm_id"]),
        "chat_template_kwargs": {"enable_thinking": False},
    })

dataset = Dataset.from_list(records).shuffle(seed=42)
splits = dataset.train_test_split(test_size=0.1, seed=42)
splits["train"].save_to_disk("grpo_v7_train_dataset")
splits["test"].save_to_disk("grpo_v7_val_dataset")
```

**Important:** Both datasets use the same `shuffle(seed=42)` + `train_test_split(seed=42)` so SFT and GRPO train on the same admissions.

**Run:**
```bash
cd ~/Capstone
python build_sft_grpo_datasets.py
```

**Expected output:**
```
Loaded 3200 rows → 2017 unique admissions after aggregation
  Radiology admissions: 1600
  Unknown/discharge:    417
  Multi-code admissions: ~729 (have 2+ CPT codes)
SFT  train: 1815 | val: 202
GRPO train: 1815 | val: 202
Saved: sft_v7_train_dataset/, sft_v7_val_dataset/
Saved: grpo_v7_train_dataset/, grpo_v7_val_dataset/
```

---

## Script 2: `train_sft_v7.py`

Copy of `train_sft_v6_final.py` with these changes:

### Constants to update
```python
SAVE_ADAPTER = "qwen3-32b-mimic-sft-v7"
OUTPUT_DIR   = "outputs_sft_v7_qwen3_32b"
WANDB_PROJECT = "SmartClaims-SFT-v7"
```

### Replace CSV loading + aggregation with parquet load
Remove lines 148-173 (pd.read_csv + groupby + records loop + full_dataset).

Replace with:
```python
from datasets import load_from_disk

if accelerator.is_main_process:
    print("Loading pre-built SFT datasets ...")

full_dataset = load_from_disk("sft_v7_train_dataset")
val_dataset_raw = load_from_disk("sft_v7_val_dataset")

# SFTTrainer expects DatasetDict or separate train/eval
# Split is already done — pass directly
train_dataset = full_dataset
val_dataset   = val_dataset_raw

if accelerator.is_main_process:
    print(f"  Train: {len(train_dataset):,} admissions")
    print(f"  Val:   {len(val_dataset):,} admissions")
```

Remove `VAL_FRACTION` constant and `train_test_split` call.

### Update SFTTrainer call
```python
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,       # use pre-split val
    processing_class=tokenizer,
)
```

### Update model save block (end of script)
Change saved adapter name from `qwen3-32b-mimic-sft` to `qwen3-32b-mimic-sft-v7` and merged name to `qwen3-32b-mimic-sft-v7-merged`.

**Run:**
```bash
accelerate launch --num_processes 2 \
    --mixed_precision bf16 \
    train_sft_v7.py
```

---

## Script 3: `train_grpo_v1.py`

### Constants
```python
SFT_MERGED_MODEL  = "qwen3-32b-mimic-sft-v7-merged"   # output of Stage 1
TOKENIZER_SRC     = "qwen3-32b-mimic-cpt-200k-ep2"     # has Qwen3 chat template
GRPO_TRAIN        = "grpo_v7_train_dataset"
GRPO_VAL          = "grpo_v7_val_dataset"
SAVE_ADAPTER      = "qwen3-32b-mimic-sft-v7-grpo"
OUTPUT_DIR        = "outputs_grpo_v7_qwen3_32b"
WANDB_PROJECT     = "SmartClaims-GRPO-v7"

EPOCHS            = 1
BATCH_SIZE        = 1        # per device
GRAD_ACCUM        = 8        # effective batch = 1 * 8 * 2 GPUs = 16
LEARNING_RATE     = 1e-5
LORA_RANK         = 16
MAX_SEQ_LENGTH    = 4096
MAX_COMPLETION    = 128      # CPT codes are short
MAX_PROMPT        = MAX_SEQ_LENGTH - MAX_COMPLETION   # 3968
NUM_GENERATIONS   = 4        # completions per prompt for GRPO
TEMPERATURE       = 0.9
KL_BETA           = 0.04     # KL penalty to prevent reward hacking
```

### Model loading
```python
from accelerate import Accelerator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig

accelerator = Accelerator()
local_rank = accelerator.local_process_index

# Tokenizer — must come from CPT adapter for Qwen3 chat template
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SRC, trust_remote_code=True)
tokenizer.padding_side = "left"   # left-pad for generation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# BnB config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load fully merged SFT model as GRPO base
base_model = AutoModelForCausalLM.from_pretrained(
    SFT_MERGED_MODEL,
    quantization_config=bnb_config,
    device_map={"": local_rank},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

# GRPO LoRA config (same as SFT)
grpo_lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora=True,
)
```

### Reward functions
```python
import re

def parse_cpt_output(text: str) -> tuple:
    """Returns (set_of_codes, is_valid_format)"""
    # Strip Qwen3 thinking tags (defensive)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text:
        return set(), False
    if text.strip().lower() == "unknown":
        return {"unknown"}, True
    parts = [p.strip() for p in text.split("|") if p.strip()]
    codes = {p for p in parts if re.match(r"^\d{5}$", p)}
    return codes, len(codes) > 0

def parse_ground_truth(gt_str: str) -> set:
    gt_str = gt_str.strip()
    if gt_str.lower() == "unknown":
        return {"unknown"}
    return {c.strip() for c in gt_str.split("|")
            if re.match(r"^\d{5}$", c.strip())}

def compute_set_f1(pred: set, gt: set) -> float:
    if not pred or not gt:
        return 0.0
    tp = len(pred & gt)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall    = tp / len(gt)
    return 2.0 * precision * recall / (precision + recall)

def cpt_f1_reward(prompts, completions, **kwargs) -> list:
    """
    GRPOTrainer reward function.
    kwargs["ground_truth"] — list of pipe-sep CPT codes or "unknown"
    Returns rewards in range [-0.2, 1.0]:
      1.0  = exact set match
      >0   = partial F1 credit
     -0.1  = non-empty but unparseable format
     -0.2  = empty output
    """
    ground_truths = kwargs["ground_truth"]
    rewards = []
    for completion, gt_str in zip(completions, ground_truths):
        # GRPOTrainer passes conversational completions as list of dicts
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        pred_codes, is_valid = parse_cpt_output(text)
        gt_codes = parse_ground_truth(gt_str)
        if is_valid:
            reward = compute_set_f1(pred_codes, gt_codes)
        else:
            reward = -0.2 if not text.strip() else -0.1
        rewards.append(float(reward))
    return rewards
```

### GRPOConfig
```python
grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_generations=NUM_GENERATIONS,
    max_prompt_length=MAX_PROMPT,
    max_completion_length=MAX_COMPLETION,
    temperature=TEMPERATURE,
    beta=KL_BETA,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=1.0,
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    mask_truncated_completions=True,
    log_completions=True,
    num_completions_to_print=4,
    report_to="wandb",
    seed=3407,
    remove_unused_columns=False,   # CRITICAL — keeps ground_truth column in kwargs
)
```

### GRPOTrainer + dataset + save
```python
from datasets import load_from_disk
import wandb

train_dataset = load_from_disk(GRPO_TRAIN)
val_dataset   = load_from_disk(GRPO_VAL)

if accelerator.is_main_process:
    wandb.init(project=WANDB_PROJECT,
               name=f"qwen3-32b-grpo-r{LORA_RANK}-ng{NUM_GENERATIONS}")

trainer = GRPOTrainer(
    model=base_model,
    reward_funcs=cpt_f1_reward,
    args=grpo_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    peft_config=grpo_lora_config,
)

trainer.train()

# Save GRPO adapter
if accelerator.is_main_process:
    import os
    os.makedirs(SAVE_ADAPTER, exist_ok=True)
    unwrapped = accelerator.unwrap_model(trainer.model)
    unwrapped.save_pretrained(SAVE_ADAPTER)
    tokenizer.save_pretrained(SAVE_ADAPTER)
    print(f"GRPO adapter saved to {SAVE_ADAPTER}")
```

### W&B metrics to monitor
| Metric | Meaning | Warning threshold |
|---|---|---|
| `rewards/cpt_f1_reward/mean` | **Training F1 (primary signal)** | Should increase over steps |
| `reward_std` | Variance across 4 completions | If ≈0 from step 1, model collapsed |
| `frac_reward_zero_std` | Prompts where all 4 completions identical | > 0.8 → raise temperature |
| `eval_reward` | Validation F1 every 50 steps | Should track train reward |
| `train/loss` | GRPO policy gradient loss | — |

**Run:**
```bash
accelerate launch --num_processes 2 \
    --mixed_precision bf16 \
    train_grpo_v1.py
```

---

## Script 4: `eval_sft_grpo.py`

### CLI usage
```bash
# Evaluate SFT model (merged, no GRPO)
python eval_sft_grpo.py --mode sft

# Evaluate GRPO model (merged + GRPO adapter)
python eval_sft_grpo.py --mode grpo
```

### Constants
```python
import argparse, re, json
from datetime import datetime

SFT_MERGED_MODEL = "qwen3-32b-mimic-sft-v7-merged"
GRPO_ADAPTER     = "qwen3-32b-mimic-sft-v7-grpo"
TOKENIZER_SRC    = "qwen3-32b-mimic-cpt-200k-ep2"
TEST_INPUTS_CSV  = "mimic_radiology_sft_test_inputs.csv"
TEST_LABELS_CSV  = "mimic_radiology_sft_test_labels.csv"
MAX_NEW_TOKENS   = 128

SYSTEM_PROMPT = (
    "You are a medical coding assistant. "
    "Given a clinical note for a patient admission, identify all billable "
    "radiology procedures performed. Output only the CPT codes separated by "
    "pipe ( | ). If no billable radiology procedure can be determined, output "
    "unknown. Do not include add-on codes, explanations, or numbering."
)
```

### Model loading
```python
def load_model(mode):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SRC, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        SFT_MERGED_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    if mode == "grpo":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, GRPO_ADAPTER)
    model.eval()
    return model, tokenizer
```

### Output parsing
```python
def parse_prediction(text: str) -> list:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.strip().lower() == "unknown":
        return ["unknown"]
    codes = [c.strip() for c in text.split("|")
             if re.match(r"^\d{5}$", c.strip())]
    return sorted(set(codes))
```

### Evaluation
```python
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report

# Load test inputs (400 admissions) and aggregate labels per hadm_id
test_inputs = pd.read_csv(TEST_INPUTS_CSV)
test_labels = pd.read_csv(TEST_LABELS_CSV, dtype={"cpt_codes": str})

gt_per_hadm = test_labels.groupby("hadm_id")["cpt_codes"] \
    .apply(list).reset_index()
gt_per_hadm.columns = ["hadm_id", "gt_codes"]

# Run inference, parse predictions
# ...

# Compute metrics
mlb = MultiLabelBinarizer()
all_labels = gt_per_hadm["gt_codes"].tolist() + predictions
mlb.fit(all_labels)

Y_true = mlb.transform(gt_per_hadm["gt_codes"])
Y_pred = mlb.transform(predictions)

f1_macro   = f1_score(Y_true, Y_pred, average="macro",   zero_division=0)
f1_micro   = f1_score(Y_true, Y_pred, average="micro",   zero_division=0)
f1_samples = f1_score(Y_true, Y_pred, average="samples", zero_division=0)

# Per-code F1
report = classification_report(Y_true, Y_pred,
    target_names=mlb.classes_, zero_division=0, output_dict=True)

# Save results
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"eval_{mode}_results_{ts}.json", "w") as f:
    json.dump({"f1_macro": f1_macro, "f1_micro": f1_micro,
               "f1_samples": f1_samples, "per_code": report}, f, indent=2)
```

---

## Script 5: Fix `eval_sft_v6.py` (minimal changes only)

**Change 1 — System prompt (lines ~35-40):**
```python
# OLD
SYSTEM_PROMPT = (
    "You are a medical coding assistant. "
    "Given radiology reports for a patient admission, identify all billable "
    "procedures performed. Output only the procedure names separated by "
    "pipe ( | ). Do not include add-on codes, explanations, or numbering."
)

# NEW
SYSTEM_PROMPT = (
    "You are a medical coding assistant. "
    "Given a clinical note for a patient admission, identify all billable "
    "radiology procedures performed. Output only the CPT codes separated by "
    "pipe ( | ). If no billable radiology procedure can be determined, output "
    "unknown. Do not include add-on codes, explanations, or numbering."
)
```

**Change 2 — parse_output function:**
```python
# OLD
def parse_output(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return [s.strip() for s in text.split("|") if s.strip()]

# NEW
def parse_output(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.strip().lower() == "unknown":
        return ["unknown"]
    codes = [c.strip() for c in text.split("|")
             if re.match(r"^\d{5}$", c.strip())]
    return sorted(set(codes))
```

---

## End-to-End Run Sequence

```bash
cd ~/Capstone

# Step 1: Build datasets (run once)
python build_sft_grpo_datasets.py

# Step 2: Stage 1 — SFT (retrain with CPT code labels)
accelerate launch --num_processes 2 --mixed_precision bf16 \
    train_sft_v7.py
# → produces: qwen3-32b-mimic-sft-v7-merged/

# Step 3: Stage 2 — GRPO (after SFT merge completes)
accelerate launch --num_processes 2 --mixed_precision bf16 \
    train_grpo_v1.py
# → produces: qwen3-32b-mimic-sft-v7-grpo/
# → monitor: W&B project SmartClaims-GRPO-v7, watch rewards/cpt_f1_reward/mean

# Step 4: Evaluate SFT baseline
python eval_sft_grpo.py --mode sft

# Step 5: Evaluate GRPO model
python eval_sft_grpo.py --mode grpo
```

---

## Critical Pitfalls

1. **`remove_unused_columns=False`** in GRPOConfig — without this, `ground_truth` and `hadm_id` columns are silently dropped before the reward function sees them

2. **Tokenizer must come from `qwen3-32b-mimic-cpt-200k-ep2/`** — the merged model directory does not have the Qwen3 chat template with `enable_thinking` support

3. **`chat_template_kwargs: {"enable_thinking": False}`** in every GRPO dataset record — TRL's `maybe_apply_chat_template()` reads this to suppress Qwen3 chain-of-thought during generation

4. **Padding side** — `padding_side = "right"` for SFT training; `padding_side = "left"` for GRPO generation and evaluation inference

5. **SFT must fully complete and merge before GRPO** — GRPO loads `qwen3-32b-mimic-sft-v7-merged/` as its base model

6. **GRPO reward function signature** — `cpt_f1_reward(prompts, completions, **kwargs)` — the function name becomes the W&B metric key `rewards/cpt_f1_reward/mean`

7. **Completion format** — when prompt is conversational (list of dicts), GRPOTrainer passes completions as `[{"role": "assistant", "content": "..."}]` — extract with `completion[0]["content"]`
