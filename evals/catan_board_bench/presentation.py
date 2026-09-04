"""Typed board-presentation adapters for existing image and text evaluations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from cle.harness.board_surface import (
    BoardPresentationProvenance,
    ImageBoardPresentation,
    TextBoardPresentation,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_edge_token(value: str) -> str:
    first, second = sorted(int(part) for part in value.split(","))
    return f"E{first:02d}_{second:02d}"


def canonical_id_map(aliases: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Invert one frozen opaque alias map into alias-to-engine-token pairs."""

    pairs: list[tuple[str, str]] = []
    tokenizers = {
        "tiles": lambda value: f"T{int(value):02d}",
        "nodes": lambda value: f"N{int(value):02d}",
        "edges": _canonical_edge_token,
        "ports": lambda value: f"P{int(value):02d}",
    }
    for group, tokenizer in tokenizers.items():
        mapping = aliases.get(group)
        if not isinstance(mapping, dict):
            raise ValueError(f"Alias map is missing {group}")
        for canonical, alias in mapping.items():
            if not isinstance(alias, str):
                raise ValueError(f"Alias map {group} contains a non-string alias")
            pairs.append((alias, tokenizer(str(canonical))))
    return tuple(sorted(pairs))


def load_text_board_presentation(
    path: Path,
    *,
    source_id: str,
    board_sha256: str,
    aliases_path: Path,
    format: str,
    renderer_version: str,
    aliases_sha256: str | None = None,
) -> TextBoardPresentation:
    """Wrap one frozen text-eval payload without changing its prompt content."""

    content_bytes = path.read_bytes()
    content = content_bytes.decode("utf-8").rstrip("\n")
    aliases_bytes = aliases_path.read_bytes()
    aliases = json.loads(aliases_bytes)
    actual_aliases_sha256 = _sha256_bytes(aliases_bytes)
    if aliases_sha256 is not None and aliases_sha256 != actual_aliases_sha256:
        raise ValueError("Alias map digest does not match the frozen artifact")
    provenance = BoardPresentationProvenance(
        source_id=source_id,
        perspective=None,
        board_schema="catan_full_public_graph/v1",
        board_sha256=board_sha256,
        identity_space="opaque_board_local_ids",
        renderer_id=f"catan_board_bench.{format}",
        renderer_version=renderer_version,
        canonical_id_map=canonical_id_map(aliases),
        canonical_id_map_sha256=actual_aliases_sha256,
    )
    return TextBoardPresentation.create(
        provenance=provenance,
        format=f"{format}/v{renderer_version}",
        content=content,
    )


def load_indexed_tile_rows_presentation(
    path: Path,
    *,
    source_id: str,
    board_sha256: str,
    aliases_path: Path,
    aliases_sha256: str | None = None,
) -> TextBoardPresentation:
    """Wrap the frozen accuracy-first text projection."""

    return load_text_board_presentation(
        path,
        source_id=source_id,
        board_sha256=board_sha256,
        aliases_path=aliases_path,
        format="indexed_tile_rows",
        renderer_version="3",
        aliases_sha256=aliases_sha256,
    )


def load_raw_board_image_presentation(
    path: Path,
    *,
    source_id: str,
    board_sha256: str,
    expected_sha256: str | None = None,
    canonical_id_map_sha256: str | None = None,
) -> ImageBoardPresentation:
    """Wrap one ordinary board PNG while preserving exact existing bytes."""

    data = path.read_bytes()
    actual_sha256 = _sha256_bytes(data)
    if expected_sha256 is not None and expected_sha256 != actual_sha256:
        raise ValueError("Board image digest does not match the frozen artifact")
    with Image.open(path) as image:
        width, height = image.size
        image_format = image.format
    if image_format != "PNG":
        raise ValueError("Strict board image presentation must be PNG")
    provenance = BoardPresentationProvenance(
        source_id=source_id,
        perspective=None,
        board_schema="catan_full_public_graph/v1",
        board_sha256=board_sha256,
        identity_space="canonical_engine_ids",
        renderer_id="catan_board_bench.raw_full_board_png",
        renderer_version="1",
        canonical_id_map_sha256=canonical_id_map_sha256,
    )
    return ImageBoardPresentation.create(
        provenance=provenance,
        format="raw_full_board_png/v1",
        media_type="image/png",
        data=data,
        width=width,
        height=height,
        contains_entity_labels=False,
    )
