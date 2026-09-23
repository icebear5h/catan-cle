"""
VLM Catan Board Benchmark - Modal Edition

Runs GLM-4.1V-9B-Thinking (and other VLMs) on Modal GPUs to test
whether they can read Catan board state from screenshots.

Usage:
    # Run benchmark with local screenshots
    modal run scripts/probes/vlm_benchmark_modal.py --screenshots playground/screenshots

    # Run with a specific model
    modal run scripts/probes/vlm_benchmark_modal.py --screenshots playground/screenshots --model "zai-org/GLM-4.1V-9B-Thinking"

    # Run OpenRouter API comparison (no GPU needed)
    modal run scripts/probes/vlm_benchmark_modal.py --screenshots playground/screenshots --openrouter

Prerequisites:
    pip install modal
    modal setup  # one-time auth
    modal secret create huggingface HF_TOKEN=hf_xxx  # optional, for gated models
    modal secret create openrouter OPENROUTER_API_KEY=sk-xxx  # for API comparison
"""

import base64
import io
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import modal

from scripts.probes.modal_typing import (
    ContainerImports,
    ProcessorLoader,
    SpawnedCall,
    VLMResult,
)
from scripts.probes.vlm_prompts import (
    BENCHMARK_PROMPTS,
    OPENROUTER_MODELS,
    SYSTEM_PROMPT,
)

# ---- Modal app & image ----
app = modal.App("catan-vlm-benchmark")

vlm_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch", "transformers>=4.46.0", "accelerate", "Pillow", "sentencepiece", "protobuf"
)

api_image = modal.Image.debian_slim(python_version="3.12").pip_install("httpx", "Pillow")

# Persistent volume for HF model weights (survives across runs)
model_volume = modal.Volume.from_name("vlm-model-cache", create_if_missing=True)

_vlm_imports: ContainerImports = vlm_image.imports
_api_imports: ContainerImports = api_image.imports

with _vlm_imports():
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

with _api_imports():
    import httpx

# ---- Local VLM inference on GPU ----
@app.cls(
    image=vlm_image,
    gpu="A100-80GB",
    volumes={"/models": model_volume},
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface")],
)
class VLMBenchmark:
    model_id: str = modal.parameter(default="zai-org/GLM-4.1V-9B-Thinking")

    @modal.enter()
    def load_model(self) -> None:
        os.environ.setdefault("HF_HOME", "/models/huggingface")

        print(f"Loading {self.model_id} in bf16...")
        load_processor: ProcessorLoader = AutoProcessor.from_pretrained
        self.processor = load_processor(
            self.model_id, trust_remote_code=True, cache_dir="/models/huggingface"
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            cache_dir="/models/huggingface",
        )
        set_eval_mode: Callable[[], object] = self.model.eval
        set_eval_mode()
        self.device = "cuda"

        vram = torch.cuda.memory_allocated() / 1e9
        print(f"Model loaded. VRAM used: {vram:.1f} GB")

    @modal.method()
    def run_prompt(self, image_bytes: bytes, prompt: str, system_prompt: str = "",
                   max_new_tokens: int = 4096, temperature: float = 0.2) -> VLMResult:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        messages: list[dict[str, object]] = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append(
            {"role": "user", "content": [
                {"type": "image", "image": img}, {"type": "text", "text": prompt},
            ]}
        )

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(self.device)

        start = time.time()
        # transformers 5's generate() self type rejects PreTrainedModel under mypy.
        generate = getattr(self.model, "generate")
        with torch.no_grad():
            outputs = generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
        latency_ms = int((time.time() - start) * 1000)

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.processor.decode(generated, skip_special_tokens=True)

        return {"response": response, "latency_ms": latency_ms, "model": self.model_id}


# ---- OpenRouter API comparison (no GPU needed) ----
@app.function(
    image=api_image,
    timeout=300,
    secrets=[modal.Secret.from_name("openrouter")],
)
def query_openrouter(image_bytes: bytes, prompt: str, system_prompt: str,
                     model_id: str, temperature: float = 0.2) -> VLMResult:
    api_key = os.environ["OPENROUTER_API_KEY"]
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    messages: list[dict[str, object]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
            {"type": "text", "text": prompt},
        ]}
    )

    payload = {
        "model": model_id, "messages": messages,
        "temperature": temperature, "max_tokens": 4096,
    }
    headers = {
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/catan-learning", "X-Title": "Catan VLM Benchmark",
    }

    start = time.time()
    with httpx.Client(timeout=120.0) as client:
        resp = client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
    latency_ms = int((time.time() - start) * 1000)

    data = resp.json()
    content = data["choices"][0]["message"].get("content", "")
    reasoning = data["choices"][0]["message"].get("reasoning_content", "")
    if reasoning and content:
        content = f"<thinking>\n{reasoning}\n</thinking>\n\n{content}"
    elif reasoning:
        content = reasoning

    return {
        "response": content, "latency_ms": latency_ms,
        "model": model_id, "usage": data.get("usage", {}),
    }


# ---- Entrypoint ----
@app.local_entrypoint()
def main(
    screenshots: str = "playground/screenshots",
    model: str = "zai-org/GLM-4.1V-9B-Thinking",
    openrouter: bool = False,
    prompt: str = "tile_reading",
    api_models: str = "gemma3-12b-free,qwen3-vl-8b,nemotron-12b-free",
) -> None:
    """Run the VLM benchmark.

    Args:
        screenshots: Path to directory containing PNG screenshots
        model: HuggingFace model ID for local GPU inference
        openrouter: If True, run OpenRouter API comparison instead of local GPU
        prompt: Which benchmark prompt to use (tile_reading, building_reading, count_hexes)
        api_models: Comma-separated list of OpenRouter model keys to test
    """
    screenshots_dir = Path(screenshots)
    png_files = sorted(screenshots_dir.glob("*.png"))
    if not png_files:
        print(f"No PNG files found in {screenshots_dir}")
        return

    print(f"Found {len(png_files)} screenshots")

    # Load prompt
    if prompt in BENCHMARK_PROMPTS:
        test_prompt = BENCHMARK_PROMPTS[prompt]
        print(f"Using benchmark prompt: {prompt}")
    else:
        test_prompt = prompt
        print(f"Using custom prompt: {prompt[:80]}...")

    # Load all images as bytes
    images: dict[str, bytes] = {}
    for p in png_files:
        images[p.name] = p.read_bytes()
        print(f"  Loaded: {p.name} ({len(images[p.name]) / 1024:.0f} KB)")

    results: dict[str, VLMResult | dict[str, VLMResult]] = {}

    if openrouter:
        # --- OpenRouter API comparison ---
        model_keys = [k.strip() for k in api_models.split(",")]
        print(f"\nRunning OpenRouter comparison with {len(model_keys)} models...")

        for img_name, img_bytes in images.items():
            per_model: dict[str, VLMResult] = {}
            results[img_name] = per_model
            print(f"\n{'='*60}")
            print(f"Screenshot: {img_name}")
            print(f"{'='*60}")

            # Fan out all models in parallel for this image
            calls: list[tuple[str, str]] = []
            for key in model_keys:
                if key not in OPENROUTER_MODELS:
                    print(f"  Unknown model key: {key}, skipping")
                    continue
                model_id = OPENROUTER_MODELS[key]
                calls.append((key, model_id))

            # Run in parallel via Modal
            futures: list[tuple[str, SpawnedCall]] = []
            for key, model_id in calls:
                futures.append((key, query_openrouter.spawn(
                    img_bytes, test_prompt, SYSTEM_PROMPT, model_id
                )))

            for key, future in futures:
                try:
                    result: VLMResult = future.get()
                    per_model[key] = result
                    print(f"\n--- {key} ({result['model']}) ---")
                    print(f"Latency: {result['latency_ms']}ms")
                    print(result["response"][:1000])
                except Exception as e:
                    print(f"\n--- {key} FAILED ---")
                    print(f"Error: {e}")
                    per_model[key] = {"error": str(e)}

    else:
        # --- Local GPU inference ---
        print(f"\nRunning local inference with {model}...")
        bench = VLMBenchmark(model_id=model)

        for img_name, img_bytes in images.items():
            print(f"\n{'='*60}")
            print(f"Screenshot: {img_name}")
            print(f"{'='*60}")

            local_result: VLMResult = bench.run_prompt.remote(
                img_bytes, test_prompt, SYSTEM_PROMPT
            )
            results[img_name] = local_result
            print(f"Latency: {local_result['latency_ms']}ms")
            print(local_result["response"][:2000])

    # Save results
    output_path = Path("artifacts/generated/pretraining/legacy_corpus/vlm_benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
