"""
VLM Catan Board Benchmark - Local Edition

Calls OpenRouter APIs + self-hosted GLM-4.1V endpoint in parallel.
No GPU needed locally.

Usage:
    # Run all free models
    python3 scripts/vlm_benchmark.py

    # Run specific models
    python3 scripts/vlm_benchmark.py --models gemma3-12b-free,qwen3-vl-8b,glm-4.1v

    # Custom prompt
    python3 scripts/vlm_benchmark.py --prompt count_hexes

    # Use specific screenshot
    python3 scripts/vlm_benchmark.py --screenshot playground/screenshots/turn_0_RED.png

Prerequisites:
    export OPENROUTER_API_KEY=sk-xxx
    export GLM_API_URL=https://xxx.modal.run/completions  # from modal deploy
"""

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import httpx
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    # Free tier
    "nemotron-12b-free": ("openrouter", "nvidia/nemotron-nano-12b-v2-vl:free"),
    "gemma3-27b-free": ("openrouter", "google/gemma-3-27b-it:free"),
    "gemma3-12b-free": ("openrouter", "google/gemma-3-12b-it:free"),
    "gemma3-4b-free": ("openrouter", "google/gemma-3-4b-it:free"),

    # Cheap
    "llama-3.2-11b": ("openrouter", "meta-llama/llama-3.2-11b-vision-instruct"),
    "qwen3-vl-8b": ("openrouter", "qwen/qwen3-vl-8b-instruct"),
    "qwen3-vl-8b-thinking": ("openrouter", "qwen/qwen3-vl-8b-thinking"),
    "qwen3-vl-30b-a3b": ("openrouter", "qwen/qwen3-vl-30b-a3b-instruct"),
    "qwen3-vl-32b": ("openrouter", "qwen/qwen3-vl-32b-instruct"),
    "ui-tars-7b": ("openrouter", "bytedance/ui-tars-1.5-7b"),
    "mistral-small-24b": ("openrouter", "mistralai/mistral-small-3.1-24b-instruct"),

    # Ceiling tests
    "gemini-2.5-flash": ("openrouter", "google/gemini-2.5-flash"),
    "gpt-4o-mini": ("openrouter", "openai/gpt-4o-mini"),
    "gpt-5.4-nano": ("openrouter", "openai/gpt-5.4-nano"),

    # Self-hosted
    "glm-4.1v": ("modal", None),
}

DEFAULT_MODELS = [
    "gemma3-12b-free",
    "gemma3-27b-free",
    "nemotron-12b-free",
    "qwen3-vl-8b",
]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a visual analyzer for a Settlers of Catan board game screenshot.

Rules:
- ONLY describe what you can literally see in the image.
- If you cannot read a number or color clearly, say "unclear" instead of guessing.
- Do not use any prior knowledge about Catan boards to fill in gaps.
- Do not describe what a Catan board "typically" looks like.
- Be specific: use exact numbers, exact colors, exact positions.
- If you are unsure about something, skip it entirely."""

PROMPTS = {
    "count_hexes": "How many hexagonal land tiles are on this board? Just give the number.",

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

    "numbers_only": """List every number token you can see on the hex tiles.
Format: one number per line, nothing else.""",
}


# ---------------------------------------------------------------------------
# API callers
# ---------------------------------------------------------------------------

def call_openrouter(model_id: str, image_bytes: bytes, prompt: str,
                    system_prompt: str, temperature: float = 0.2) -> Dict:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return {"error": "OPENROUTER_API_KEY not set"}

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Some free-tier models (Gemma) don't support system prompts -- fold into user msg
    no_system = "free" in model_id or "gemma" in model_id.lower()

    messages = []
    if system_prompt and not no_system:
        messages.append({"role": "system", "content": system_prompt})

    user_text = prompt
    if system_prompt and no_system:
        user_text = f"{system_prompt}\n\n---\n\n{prompt}"

    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": user_text},
        ],
    })

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/catan-learning",
        "X-Title": "Catan VLM Benchmark",
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4096,
    }

    start = time.time()
    with httpx.Client(timeout=120.0) as client:
        resp = client.post("https://openrouter.ai/api/v1/chat/completions",
                           headers=headers, json=payload)
        if resp.status_code >= 400:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:500]}"}
        resp.raise_for_status()
    latency_ms = int((time.time() - start) * 1000)

    data = resp.json()
    content = data["choices"][0]["message"].get("content") or ""
    reasoning = data["choices"][0]["message"].get("reasoning_content") or ""
    if reasoning and content:
        content = f"<thinking>\n{reasoning}\n</thinking>\n\n{content}"
    elif reasoning:
        content = reasoning

    return {
        "response": content,
        "latency_ms": latency_ms,
        "model": data.get("model", model_id),
        "usage": data.get("usage", {}),
    }


def call_modal_glm(image_bytes: bytes, prompt: str, system_prompt: str,
                    temperature: float = 0.2) -> Dict:
    api_url = os.environ.get("GLM_API_URL")
    if not api_url:
        return {"error": "GLM_API_URL not set (deploy with: modal deploy scripts/deploy_glm_api.py)"}

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ],
    })

    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4096,
    }

    start = time.time()
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(api_url, json=payload)
            if resp.status_code >= 400:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:500]}"}
            if not resp.text.strip():
                return {"error": "Empty response (container may be cold starting)"}
    except httpx.ReadTimeout:
        return {"error": "Timeout (300s) -- container may be cold starting"}
    latency_ms = int((time.time() - start) * 1000)

    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    return {
        "response": content,
        "latency_ms": latency_ms,
        "model": data.get("model", "glm-4.1v-9b"),
        "usage": data.get("usage", {}),
    }


def call_model(key: str, image_bytes: bytes, prompt: str, system_prompt: str) -> Dict:
    provider, model_id = MODELS[key]
    if provider == "openrouter":
        return call_openrouter(model_id, image_bytes, prompt, system_prompt)
    elif provider == "modal":
        return call_modal_glm(image_bytes, prompt, system_prompt)
    else:
        return {"error": f"Unknown provider: {provider}"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VLM Catan Board Benchmark")
    parser.add_argument("--screenshots", default="playground/screenshots",
                        help="Directory with PNG screenshots")
    parser.add_argument("--screenshot", default=None,
                        help="Single screenshot file to test")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model keys (default: free models)")
    parser.add_argument("--prompt", default="tile_reading",
                        help="Prompt key or custom text")
    parser.add_argument("--all-models", action="store_true",
                        help="Run all models")
    parser.add_argument("--output", default="data_pipeline/training/pretraining/output/vlm_benchmark_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    # Load screenshots
    if args.screenshot:
        png_files = [Path(args.screenshot)]
    else:
        png_files = sorted(Path(args.screenshots).glob("*.png"))

    if not png_files:
        print(f"No PNG files found")
        sys.exit(1)

    # Pick first screenshot
    img_path = png_files[0]
    img_bytes = img_path.read_bytes()
    print(f"Screenshot: {img_path.name} ({len(img_bytes) / 1024:.0f} KB)")

    # Pick prompt
    if args.prompt in PROMPTS:
        prompt = PROMPTS[args.prompt]
        prompt_key = args.prompt
    else:
        prompt = args.prompt
        prompt_key = "custom"
    print(f"Prompt: {prompt_key}")

    # Pick models
    if args.all_models:
        model_keys = list(MODELS.keys())
    elif args.models:
        model_keys = [k.strip() for k in args.models.split(",")]
    else:
        model_keys = DEFAULT_MODELS

    # Validate
    for k in model_keys:
        if k not in MODELS:
            print(f"Unknown model: {k}")
            print(f"Available: {', '.join(MODELS.keys())}")
            sys.exit(1)

    print(f"Models: {', '.join(model_keys)}")
    print(f"{'='*70}\n")

    # Run all models in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=len(model_keys)) as pool:
        futures = {}
        for key in model_keys:
            futures[key] = pool.submit(call_model, key, img_bytes, prompt, SYSTEM_PROMPT)
            print(f"  Spawned: {key}")

        print()

        for key, future in futures.items():
            try:
                result = future.result(timeout=180)
                results[key] = result

                if "error" in result:
                    print(f"--- {key}: ERROR ---")
                    print(f"  {result['error']}\n")
                    continue

                # Count resource lines as quality heuristic
                response = result["response"]
                lines = [l.strip() for l in response.split("\n") if l.strip()]
                resource_words = {"wood", "brick", "sheep", "wheat", "ore", "desert"}
                resource_lines = [l for l in lines
                                  if any(r in l.lower() for r in resource_words)]

                print(f"--- {key} ({result.get('model', '?')}) ---")
                print(f"  Latency: {result['latency_ms']}ms | Resource lines: {len(resource_lines)}")
                # Print first 25 lines
                for line in lines[:25]:
                    print(f"  {line}")
                if len(lines) > 25:
                    print(f"  ... ({len(lines) - 25} more lines)")
                print()

            except Exception as e:
                print(f"--- {key}: EXCEPTION ---")
                print(f"  {e}\n")
                results[key] = {"error": str(e)}

    # Save results
    output = {
        "screenshot": img_path.name,
        "prompt": prompt_key,
        "system_prompt": SYSTEM_PROMPT,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
