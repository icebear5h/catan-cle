"""Novita text request with thinking disabled."""

from __future__ import annotations

import argparse
import time

import httpx

from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats import FormatJob
from scripts.board_bench.run.eval_catan_board_bench_openrouter import (
    NOVITA_URL,
    extract_message_text,
)
from scripts.board_bench.run.eval_catan_strict_text_probe_novita.constants import SYSTEM_PROMPT
from scripts.board_bench.shapes import JsonDict, obj, values

__all__ = ["call_job", "call_novita_text"]


def call_job(api_key: str, args: argparse.Namespace, job: FormatJob) -> JsonDict:
    try:
        return call_novita_text(
            api_key,
            args.model,
            prompt=job["prompt"],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    finally:
        if args.request_interval:
            time.sleep(args.request_interval)


def call_novita_text(
    api_key: str,
    model_id: str,
    *,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> JsonDict:
    payload: JsonDict = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_thinking": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    start = time.time()
    with httpx.Client(timeout=timeout) as client:
        response = client.post(NOVITA_URL, headers=headers, json=payload)
    latency_ms = int((time.time() - start) * 1000)
    if response.status_code >= 400:
        return {
            "error": f"HTTP {response.status_code}: {response.text[:800]}",
            "latency_ms": latency_ms,
        }
    data = obj(response.json(), "novita response")
    message = obj(obj(values(data["choices"], "choices")[0], "choice")["message"], "message")
    return {
        "response": extract_message_text(message).strip(),
        "latency_ms": latency_ms,
        "usage": data.get("usage", {}),
        "served_model": data.get("model", model_id),
        "provider": "novita",
    }
