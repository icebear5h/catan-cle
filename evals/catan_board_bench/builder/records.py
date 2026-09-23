"""Question/answer projections and artifact writers."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Iterator, TextIO

from PIL import Image, ImageOps

from evals.json_types import JsonDict


def _question_view(qa: JsonDict) -> JsonDict:
    return {
        "id": qa["id"],
        "sample_id": qa["sample_id"],
        "image_path": qa["image_path"],
        "contract_path": qa["contract_path"],
        "category": qa["category"],
        "question": qa["question"],
    }


def _answer_view(qa: JsonDict) -> JsonDict:
    return {
        "id": qa["id"],
        "sample_id": qa["sample_id"],
        "category": qa["category"],
        "answer": qa["answer"],
        "target": qa["target"],
        "scoring": qa["scoring"],
    }


def _write_square_png(png: bytes, path: Path, *, size: int) -> None:
    image = Image.open(io.BytesIO(png)).convert("RGB")
    image = ImageOps.contain(image, (size, size), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), color=(13, 111, 165))
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")


def _write_json(path: Path, value: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(handle: TextIO, value: JsonDict) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")


@contextlib.contextmanager
def _quiet_context(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield
