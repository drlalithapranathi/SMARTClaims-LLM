"""
Modal inference endpoint for SMARTClaims CPT prediction.
Loads fully merged Qwen3-32B (CPT + SFT + GRPO) from HuggingFace.
"""
import modal, re

app = modal.App("smartclaims-cpt-predictor")

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.51.0",
        "accelerate>=1.2.0",
        "bitsandbytes>=0.45.0",
        "huggingface_hub>=0.26.0",
        "fastapi[standard]",
    )
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL = "lalithapranathipulavarthy/smartclaims-grpo-unk10"

SYSTEM_PROMPT = (
    "You are a medical coding assistant. Given a clinical note for a patient admission, "
    "identify all billable radiology procedures performed. Output only the CPT codes "
    "separated by pipe ( | ). If no billable radiology procedure can be determined, "
    "output unknown. Do not include add-on codes, explanations, or numbering."
)


@app.cls(
    image=inference_image,
    gpu="H100",
    volumes={"/cache": hf_cache},
    timeout=1800,
    scaledown_window=300,
    min_containers=0,
)
class CPTPredictor:
    @modal.enter()
    def load_model(self):
        import os, torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ["HF_HOME"] = "/cache/hf"
        os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"

        print(f"Loading merged model: {MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        print("Model ready")

    @modal.fastapi_endpoint(method="POST")
    def predict(self, item: dict) -> dict:
        import time, torch

        report_text = item.get("report_text", "").strip()
        if not report_text:
            return {"error": "report_text is required"}

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": report_text},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed_ms = int((time.time() - t0) * 1000)

        generated = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        generated = re.sub(r"<think>\s*</think>", "", generated).strip()

        if generated.lower() == "unknown":
            codes = []
        else:
            codes = [c.strip() for c in generated.split("|") if c.strip()]

        return {
            "codes": codes,
            "raw": generated,
            "inference_ms": elapsed_ms,
        }
