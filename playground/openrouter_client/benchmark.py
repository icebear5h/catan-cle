"""Side-by-side VLM comparison and the Gemini judge that scores it."""

import json

from .completions import query_text, query_vlm
from .config import MODELS, VlmResult

__all__ = ["judge_responses", "query_both_vlms"]


def query_both_vlms(
    image_bytes: bytes,
    prompt: str,
    system_prompt: str = "",
    model_a_key: str = "glm_4_6v",
    model_b_key: str = "glm_4_6v_novita",
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> dict[str, VlmResult]:
    """Query two VLMs with the same image+prompt, return side-by-side.

    Args:
        model_a_key: Key into MODELS dict (e.g. "glm_4_6v")
        model_b_key: Key into MODELS dict (e.g. "glm_4_6v_novita")

    Returns:
        {
            "model_a": {"content": ..., "model": ..., "latency_ms": ...},
            "model_b": {"content": ..., "model": ..., "latency_ms": ...},
        }
    """
    provider_a, model_a = MODELS[model_a_key]
    provider_b, model_b = MODELS[model_b_key]

    result_a = query_vlm(
        model_a, image_bytes, prompt, system_prompt, temperature, max_tokens,
        provider=provider_a,
    )
    result_b = query_vlm(
        model_b, image_bytes, prompt, system_prompt, temperature, max_tokens,
        provider=provider_b,
    )

    return {"model_a": result_a, "model_b": result_b}


def judge_responses(
    task_description: str,
    response_a: str,
    response_b: str,
    ground_truth: str = "",
    model_a_name: str = "GLM-4.6V (OpenRouter)",
    model_b_name: str = "GLM-4.6V (Novita)",
    judge_key: str = "gemini_judge",
) -> dict[str, object]:
    """Use Gemini to judge two VLM responses.

    Returns:
        {
            "scores_a": {"accuracy": N, "completeness": N, "strategic_depth": N, "specificity": N},
            "scores_b": {"accuracy": N, "completeness": N, "strategic_depth": N, "specificity": N},
            "winner": "A" | "B" | "tie",
            "explanation": str,
            "raw_response": str,
        }
    """
    ground_truth_section = ""
    if ground_truth:
        ground_truth_section = f"\nGROUND TRUTH (verified from game engine):\n{ground_truth}\n"

    judge_prompt = f"""You are evaluating two AI vision models on their understanding of a Catan board game position.

TASK: {task_description}

MODEL A ({model_a_name}) Response:
{response_a}

MODEL B ({model_b_name}) Response:
{response_b}
{ground_truth_section}
Score each response on these dimensions (1-5 scale, 5 is best):
1. ACCURACY: Are the factual claims about the board correct?
2. COMPLETENESS: Does it address all aspects of the question?
3. STRATEGIC_DEPTH: Does it show genuine understanding of Catan strategy?
4. SPECIFICITY: Does it reference specific board elements (tiles, numbers, positions)?

Respond with ONLY valid JSON (no markdown code fences):
{{
    "scores_a": {{"accuracy": N, "completeness": N, "strategic_depth": N, "specificity": N}},
    "scores_b": {{"accuracy": N, "completeness": N, "strategic_depth": N, "specificity": N}},
    "winner": "A" or "B" or "tie",
    "explanation": "brief explanation of your scoring"
}}"""

    judge_provider, judge_model = MODELS[judge_key]
    result = query_text(
        judge_model,
        judge_prompt,
        system_prompt="You are an expert Catan player and fair judge. Respond with JSON only.",
        temperature=0.1,
        max_tokens=1024,
        provider=judge_provider,
    )

    raw = result["content"]

    # Parse JSON from response (handle possible markdown fences)
    json_str = raw.strip()
    if json_str.startswith("```"):
        json_str = json_str.split("\n", 1)[1]
        json_str = json_str.rsplit("```", 1)[0]

    try:
        parsed: dict[str, object] = json.loads(json_str)
    except json.JSONDecodeError:
        parsed = {
            "scores_a": {"accuracy": 0, "completeness": 0, "strategic_depth": 0, "specificity": 0},
            "scores_b": {"accuracy": 0, "completeness": 0, "strategic_depth": 0, "specificity": 0},
            "winner": "error",
            "explanation": f"Failed to parse judge response: {raw[:200]}",
        }

    parsed["raw_response"] = raw
    parsed["judge_latency_ms"] = result["latency_ms"]

    return parsed
