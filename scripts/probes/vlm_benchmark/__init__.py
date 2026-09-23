"""
VLM Catan Board Benchmark - Local Edition

Calls OpenRouter APIs + self-hosted GLM-4.1V endpoint in parallel.
No GPU needed locally.

Usage:
    # Run all free models
    python3 -m scripts.probes.vlm_benchmark

    # Run specific models
    python3 -m scripts.probes.vlm_benchmark --models gemma3-12b-free,qwen3-vl-8b,glm-4.1v

    # Custom prompt
    python3 -m scripts.probes.vlm_benchmark --prompt count_hexes

    # Use specific screenshot
    python3 -m scripts.probes.vlm_benchmark --screenshot playground/screenshots/turn_0_RED.png

Prerequisites:
    export OPENROUTER_API_KEY=sk-xxx
    export GLM_API_URL=https://xxx.modal.run/completions  # from modal deploy
"""

from dotenv import load_dotenv

from scripts.probes.vlm_benchmark.catalog import (
    DEFAULT_MODELS,
    MODELS,
    PROMPTS,
    SYSTEM_PROMPT,
    ModelResult,
)
from scripts.probes.vlm_benchmark.cli import main
from scripts.probes.vlm_benchmark.providers import (
    call_modal_glm,
    call_model,
    call_openrouter,
)

load_dotenv()

__all__ = [
    "DEFAULT_MODELS",
    "MODELS",
    "PROMPTS",
    "SYSTEM_PROMPT",
    "ModelResult",
    "call_modal_glm",
    "call_model",
    "call_openrouter",
    "main",
]
