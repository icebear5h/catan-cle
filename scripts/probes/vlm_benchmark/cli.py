"""Command line for the local VLM board benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from scripts.probes.vlm_benchmark.catalog import (
    DEFAULT_MODELS,
    MODELS,
    PROMPTS,
    SYSTEM_PROMPT,
    ModelResult,
)
from scripts.probes.vlm_benchmark.providers import call_model

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ``python -m`` would otherwise name the program __main__.py.
    parser = argparse.ArgumentParser(
        prog="vlm_benchmark.py", description="VLM Catan Board Benchmark"
    )
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
    parser.add_argument("--output", default="artifacts/generated/pretraining/legacy_corpus/vlm_benchmark_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    # Load screenshots
    if args.screenshot:
        png_files = [Path(args.screenshot)]
    else:
        png_files = sorted(Path(args.screenshots).glob("*.png"))

    if not png_files:
        print("No PNG files found")
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
    results: dict[str, ModelResult] = {}
    with ThreadPoolExecutor(max_workers=len(model_keys)) as pool:
        futures: dict[str, Future[ModelResult]] = {}
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
                lines = [entry.strip() for entry in response.split("\n") if entry.strip()]
                resource_words = {"wood", "brick", "sheep", "wheat", "ore", "desert"}
                resource_lines = [entry for entry in lines
                                  if any(r in entry.lower() for r in resource_words)]

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
