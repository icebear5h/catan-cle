"""Paths, the system prompt, and small prompt/model helpers."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from evals.catan_board_bench.scoring import sentinel_hint
from scripts.board_bench.shapes import JsonDict

load_dotenv()

BENCH_DIR = Path("evals/catan_board_bench/datasets/catan_board_bench_100")
QUESTION_DIR = BENCH_DIR / "questions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_PROMPT = """You are answering engine-scored questions about an authoritative public Catan board state encoded as text.

Rules:
- Use only the supplied board state.
- Literal tokens such as <T07>, <N18>, and <E03_17> are canonical identifiers.
- In sparse formats, declared EMPTY defaults apply to omitted nodes and edges.
- Derive requested counts from the listed buildings and roads; no hidden state is present.
- Return only the final answer string, with no explanation.
- Preserve exact tokens such as <RED>, <SETTLEMENT>, <WOOD>, and <ORE>."""

__all__ = [
    "BENCH_DIR",
    "OPENROUTER_URL",
    "QUESTION_DIR",
    "SYSTEM_PROMPT",
    "build_prompt",
    "model_supports_reasoning_control",
    "split_csv",
]


def build_prompt(representation: str, board_text: str, qa: JsonDict) -> str:
    hint = sentinel_hint(qa)
    hint_text = f"{hint}\n\n" if hint else ""
    return (
        f"Representation: {representation}\n"
        "Authoritative public board state:\n"
        f"{board_text}\n\n"
        f"{hint_text}"
        f"Question: {qa['question']}\n\n"
        "Return only the answer."
    )


def model_supports_reasoning_control(model_id: str) -> bool:
    model_id_lower = model_id.lower()
    return any(family in model_id_lower for family in ("qwen3.5", "qwen3.6", "qwen3.7", "qwen3.8"))


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
