"""
VLM Catan Board Benchmark - Modal Edition

Runs GLM-4.1V-9B-Thinking (and other VLMs) on Modal GPUs to test
whether they can read Catan board state from screenshots.

Usage:
    # Run benchmark with local screenshots
    modal run scripts/vlm_benchmark_modal.py --screenshots playground/screenshots

    # Run with a specific model
    modal run scripts/vlm_benchmark_modal.py --screenshots playground/screenshots --model "zai-org/GLM-4.1V-9B-Thinking"

    # Run OpenRouter API comparison (no GPU needed)
    modal run scripts/vlm_benchmark_modal.py --screenshots playground/screenshots --openrouter

Prerequisites:
    pip install modal
    modal setup  # one-time auth
    modal secret create huggingface HF_TOKEN=hf_xxx  # optional, for gated models
    modal secret create openrouter OPENROUTER_API_KEY=sk-xxx  # for API comparison
"""

import modal
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Modal app & image
# ---------------------------------------------------------------------------

app = modal.App("catan-vlm-benchmark")

vlm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers>=4.46.0",
        "accelerate",
        "Pillow",
        "sentencepiece",
        "protobuf",
    )
)

api_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "httpx",
        "Pillow",
    )
)

# Persistent volume for HF model weights (survives across runs)
model_volume = modal.Volume.from_name("vlm-model-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Local VLM inference on GPU
# ---------------------------------------------------------------------------

@app.cls(
    image=vlm_image,
    gpu="A100-80GB",
    volumes={"/models": model_volume},
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface", required_hint="optional, for gated models")],
)
class VLMBenchmark:
    model_id: str = modal.parameter(default="zai-org/GLM-4.1V-9B-Thinking")

    @modal.enter()
    def load_model(self):
        import torch
        import os
        from transformers import AutoModelForImageTextToText, AutoProcessor

        os.environ.setdefault("HF_HOME", "/models/huggingface")

        print(f"Loading {self.model_id} in bf16...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            cache_dir="/models/huggingface",
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            cache_dir="/models/huggingface",
        )
        self.model.eval()
        self.device = "cuda"

        vram = torch.cuda.memory_allocated() / 1e9
        print(f"Model loaded. VRAM used: {vram:.1f} GB")

    @modal.method()
    def run_prompt(self, image_bytes: bytes, prompt: str, system_prompt: str = "",
                   max_new_tokens: int = 4096, temperature: float = 0.2) -> dict:
        import torch
        import time
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        })

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(self.device)

        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
        latency_ms = int((time.time() - start) * 1000)

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.processor.decode(generated, skip_special_tokens=True)

        return {
            "response": response,
            "latency_ms": latency_ms,
            "model": self.model_id,
        }


# ---------------------------------------------------------------------------
# OpenRouter API comparison (no GPU needed)
# ---------------------------------------------------------------------------

OPENROUTER_MODELS = {
    "nemotron-12b-free": "nvidia/nemotron-nano-12b-v2-vl:free",
    "gemma3-27b-free": "google/gemma-3-27b-it:free",
    "gemma3-12b-free": "google/gemma-3-12b-it:free",
    "gemma3-4b-free": "google/gemma-3-4b-it:free",
    "qwen3-vl-8b": "qwen/qwen3-vl-8b-instruct",
    "qwen3-vl-8b-thinking": "qwen/qwen3-vl-8b-thinking",
    "qwen3-vl-30b-a3b": "qwen/qwen3-vl-30b-a3b-instruct",
    "qwen3-vl-32b": "qwen/qwen3-vl-32b-instruct",
    "ui-tars-7b": "bytedance/ui-tars-1.5-7b",
    "llama-3.2-11b-vision": "meta-llama/llama-3.2-11b-vision-instruct",
    "mistral-small-3.1-24b": "mistralai/mistral-small-3.1-24b-instruct",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-5.4-nano": "openai/gpt-5.4-nano",
}


@app.function(
    image=api_image,
    timeout=300,
    secrets=[modal.Secret.from_name("openrouter")],
)
def query_openrouter(image_bytes: bytes, prompt: str, system_prompt: str,
                     model_id: str, temperature: float = 0.2) -> dict:
    import httpx
    import base64
    import os
    import time

    api_key = os.environ["OPENROUTER_API_KEY"]
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
            {"type": "text", "text": prompt},
        ],
    })

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4096,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/catan-learning",
        "X-Title": "Catan VLM Benchmark",
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
        "response": content,
        "latency_ms": latency_ms,
        "model": model_id,
        "usage": data.get("usage", {}),
    }


# ---------------------------------------------------------------------------
# Benchmark prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a visual analyzer for a Settlers of Catan board game screenshot.

Rules:
- ONLY describe what you can literally see in the image.
- If you cannot read a number or color clearly, say "unclear" instead of guessing.
- Do not use any prior knowledge about Catan boards to fill in gaps.
- Do not describe what a Catan board "typically" looks like.
- Be specific: use exact numbers, exact colors, exact positions.
- If you are unsure about something, skip it entirely."""

BENCHMARK_PROMPTS = {
    "tile_reading": """For each hex tile on this board, state the resource type and the number on it.
Use these resource names only: wood, brick, sheep, wheat, ore, desert.
Format as one per line: RESOURCE NUMBER
Example: wheat 6
List all tiles you can see.""",

    "building_reading": """List every settlement and city on the board.
For each one, state the color and whether it's a settlement (small) or city (large).
Format: COLOR TYPE
Example: red settlement
List all buildings, nothing else.""",

    "count_hexes": "How many hexagonal land tiles are on this board? Just give the number.",
}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    screenshots: str = "playground/screenshots",
    model: str = "zai-org/GLM-4.1V-9B-Thinking",
    openrouter: bool = False,
    prompt: str = "tile_reading",
    api_models: str = "gemma3-12b-free,qwen3-vl-8b,nemotron-12b-free",
):
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
    images = {}
    for p in png_files:
        images[p.name] = p.read_bytes()
        print(f"  Loaded: {p.name} ({len(images[p.name]) / 1024:.0f} KB)")

    results = {}

    if openrouter:
        # --- OpenRouter API comparison ---
        model_keys = [k.strip() for k in api_models.split(",")]
        print(f"\nRunning OpenRouter comparison with {len(model_keys)} models...")

        for img_name, img_bytes in images.items():
            results[img_name] = {}
            print(f"\n{'='*60}")
            print(f"Screenshot: {img_name}")
            print(f"{'='*60}")

            # Fan out all models in parallel for this image
            calls = []
            for key in model_keys:
                if key not in OPENROUTER_MODELS:
                    print(f"  Unknown model key: {key}, skipping")
                    continue
                model_id = OPENROUTER_MODELS[key]
                calls.append((key, model_id))

            # Run in parallel via Modal
            futures = []
            for key, model_id in calls:
                futures.append((key, query_openrouter.spawn(
                    img_bytes, test_prompt, SYSTEM_PROMPT, model_id
                )))

            for key, future in futures:
                try:
                    result = future.get()
                    results[img_name][key] = result
                    print(f"\n--- {key} ({result['model']}) ---")
                    print(f"Latency: {result['latency_ms']}ms")
                    print(result["response"][:1000])
                except Exception as e:
                    print(f"\n--- {key} FAILED ---")
                    print(f"Error: {e}")
                    results[img_name][key] = {"error": str(e)}

    else:
        # --- Local GPU inference ---
        print(f"\nRunning local inference with {model}...")
        bench = VLMBenchmark(model_id=model)

        for img_name, img_bytes in images.items():
            print(f"\n{'='*60}")
            print(f"Screenshot: {img_name}")
            print(f"{'='*60}")

            result = bench.run_prompt.remote(img_bytes, test_prompt, SYSTEM_PROMPT)
            results[img_name] = result
            print(f"Latency: {result['latency_ms']}ms")
            print(result["response"][:2000])

    # Save results
    output_path = Path("data_pipeline/training/pretraining/output/vlm_benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
