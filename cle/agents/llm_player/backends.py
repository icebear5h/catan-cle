"""Vision and text completion backends plus the loose action-index fallback."""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from playground.openrouter_client import MODELS as VLM_MODELS
from playground.openrouter_client import query_vlm

if TYPE_CHECKING:
    from cle.agents.llm_player import LLMPlayer

__all__: list[str] = []


def _call_vlm(
    self: LLMPlayer, system_prompt: str, user_prompt: str, image_bytes: bytes
) -> tuple[str, float]:
    """Call VLM via OpenRouter/Novita. Returns (response_text, api_time_sec)."""
    provider, model_id = VLM_MODELS[self.vision_model]
    print(f"Calling VLM: {model_id} via {provider}...")

    result = query_vlm(
        model_id,
        image_bytes,
        user_prompt,
        system_prompt=system_prompt,
        provider=provider,
        temperature=self.temperature,
        max_tokens=8192,
    )

    api_time: float = result['latency_ms'] / 1000
    usage = result.get('usage', {})
    print(f"VLM call completed: {api_time:.3f}s")
    print(f"Tokens: {usage}")

    content: str = result['content']
    return content, api_time


def _call_groq(self: LLMPlayer, system_prompt: str, user_prompt: str) -> tuple[str, float]:
    """Call Groq text-only API. Returns (response_text, api_time_sec)."""
    print(f"Calling Groq API with model: {self.groq_model}...")

    assert self.groq_client is not None
    api_start = time.time()
    response = self.groq_client.chat.completions.create(
        model=self.groq_model,
        max_tokens=8192,
        temperature=self.temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    api_time = time.time() - api_start
    print(f"Groq call completed: {api_time:.3f}s")
    assert response.usage is not None
    print(f"Input tokens: {response.usage.prompt_tokens}")
    print(f"Output tokens: {response.usage.completion_tokens}")

    message_content = response.choices[0].message.content
    assert message_content is not None
    return message_content.strip(), api_time


def _parse_action_choice(self: LLMPlayer, text: str, num_actions: int) -> int:
    """Fallback: extract first number from LLM response as action index."""
    numbers = re.findall(r'\b\d+\b', text)

    if not numbers:
        raise ValueError(f"No number found in response: {text}")

    idx = int(numbers[0])

    if idx < 0 or idx >= num_actions:
        raise ValueError(f"Index {idx} out of range [0, {num_actions-1}]")

    return idx
