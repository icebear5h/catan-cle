"""Catan board presenters backed by the evaluated text and image renderers."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal, cast

from cle.game_engine.public_board import snapshot_public_board
from cle.harness.board_surface import (
    BoardPresentation,
    BoardPresentationProvenance,
    BoardPresenter,
    ImageBoardPresentation,
    TextBoardPresentation,
)
from cle.players.contracts import PlayerContext
from evals.catan_board_bench.ascii_variations.facts import full_facts_from_json
from evals.catan_board_bench.render import render_contract_image
from evals.catan_board_bench.text_format_optimization import render_text_format

BoardSurfaceKind = Literal[
    "legacy_semantic",
    "indexed_tile_rows",
    "image",
]
_OPAQUE_HEADER = "ENTITY IDS ARE OPAQUE, PERMUTED, AND BOARD-LOCAL."
_CANONICAL_HEADER = (
    "ENTITY IDS USE CANONICAL ENGINE TILE, NODE, EDGE, AND PORT LABELS."
)


@dataclass(frozen=True, slots=True)
class NoBoardPresenter:
    """Preserve historical requests that predate explicit board surfaces."""

    def present(self, context: PlayerContext) -> None:
        del context
        return None


@dataclass(frozen=True, slots=True)
class IndexedTileRowsBoardPresenter:
    """Surface the complete board with the current accuracy-first text format."""

    def present(self, context: PlayerContext) -> TextBoardPresentation:
        snapshot = snapshot_public_board(context.observation)
        facts = snapshot.facts()
        content = render_text_format(
            "indexed_tile_rows",
            full_facts_from_json(facts),
            sample_id=context.context_id,
        ).replace(_OPAQUE_HEADER, _CANONICAL_HEADER, 1)
        provenance = BoardPresentationProvenance(
            source_id=context.context_id,
            perspective=context.actor,
            board_schema=cast("str", facts["schema"]),
            board_sha256=snapshot.facts_sha256,
            identity_space="canonical_engine_ids",
            renderer_id="catan_board_bench.indexed_tile_rows",
            renderer_version="3",
        )
        return TextBoardPresentation.create(
            provenance=provenance,
            format="indexed_tile_rows/v3",
            content=content,
        )


@dataclass(frozen=True, slots=True)
class ImageBoardPresenter:
    """Surface an ordinary unannotated PNG rendered from public board facts."""

    image_size: int = 1024

    def __post_init__(self) -> None:
        if self.image_size < 64 or self.image_size > 2048:
            raise ValueError("Board image size must be within 64..2048 pixels")

    def present(self, context: PlayerContext) -> ImageBoardPresentation:
        snapshot = snapshot_public_board(context.observation)
        facts = snapshot.facts()
        image = render_contract_image(
            snapshot.contract(),
            image_size=self.image_size,
        )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        provenance = BoardPresentationProvenance(
            source_id=context.context_id,
            perspective=context.actor,
            board_schema=cast("str", facts["schema"]),
            board_sha256=snapshot.facts_sha256,
            identity_space="canonical_engine_ids",
            renderer_id="catan_board_bench.raw_full_board_png",
            renderer_version="1",
        )
        return ImageBoardPresentation.create(
            provenance=provenance,
            format="raw_full_board_png/v1",
            media_type="image/png",
            data=buffer.getvalue(),
            width=image.width,
            height=image.height,
            contains_entity_labels=False,
        )


def create_board_presenter(kind: BoardSurfaceKind) -> BoardPresenter:
    """Create the configured runtime public-board presenter."""

    presenters: dict[str, BoardPresenter] = {
        "legacy_semantic": NoBoardPresenter(),
        "indexed_tile_rows": IndexedTileRowsBoardPresenter(),
        "image": ImageBoardPresenter(),
    }
    try:
        return presenters[kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported board surface: {kind}") from exc


def present_board(
    context: PlayerContext,
    *,
    kind: BoardSurfaceKind = "indexed_tile_rows",
) -> BoardPresentation:
    """Convenience adapter shared by runtime callers and focused evaluations."""

    presenter = create_board_presenter(kind)
    presentation = presenter.present(context)
    if presentation is None:
        raise ValueError("The legacy semantic mode has no separate presentation")
    return presentation
