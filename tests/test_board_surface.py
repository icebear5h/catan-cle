from dataclasses import replace

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.game_engine.public_board import snapshot_public_board
from cle.harness import ModelMessage
from cle.harness.board_surface import (
    board_presentation_payload,
    openai_messages_with_board,
    sanitize_provider_payload,
)
from cle.harness.catan_board_surface import (
    ImageBoardPresenter,
    IndexedTileRowsBoardPresenter,
)
from cle.sandbox.decision import build_decision_context
from evals.catan_board_bench.text_format_optimization import parse_text_format


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def _engine() -> GameEngine:
    return GameEngine(COLORS, seed=7, shuffle_players=False)


def test_public_board_snapshot_is_private_state_independent_and_immutable():
    engine = _engine()
    red_snapshot = snapshot_public_board(engine.observe(Color.RED))
    blue_snapshot = snapshot_public_board(engine.observe(Color.BLUE))

    assert red_snapshot.facts_sha256 == blue_snapshot.facts_sha256
    original_facts = red_snapshot.facts_json

    engine.state.player_state["P0_WOOD_IN_HAND"] += 3
    engine.state.player_state["P0_KNIGHT_IN_HAND"] += 1
    hidden_state_changed = snapshot_public_board(engine.observe(Color.RED))

    assert hidden_state_changed.facts_sha256 == red_snapshot.facts_sha256
    assert red_snapshot.facts_json == original_facts

    engine.step(engine.state.playable_actions[0])
    public_state_changed = snapshot_public_board(engine.observe(Color.RED))

    assert public_state_changed.facts_sha256 != red_snapshot.facts_sha256
    assert red_snapshot.facts_json == original_facts


def test_indexed_tile_rows_surface_is_complete_canonical_and_parseable():
    context = build_decision_context(_engine())
    presentation = IndexedTileRowsBoardPresenter().present(context)

    assert presentation.kind == "text"
    assert presentation.format == "indexed_tile_rows/v3"
    assert presentation.provenance.source_id == context.context_id
    assert presentation.provenance.perspective == Color.RED
    assert presentation.provenance.identity_space == "canonical_engine_ids"
    assert "ENTITY IDS USE CANONICAL ENGINE" in presentation.content
    assert "ENTITY IDS ARE OPAQUE" not in presentation.content
    parsed = parse_text_format("indexed_tile_rows", presentation.content)
    assert len(parsed["tiles"]) == 19
    assert len(parsed["nodes"]) == 54
    assert len(parsed["edges"]) == 72
    assert len(parsed["ports"]) == 9


def test_image_surface_is_bounded_unannotated_and_state_paired():
    engine = _engine()
    context = build_decision_context(engine)
    blue_context = build_decision_context(
        engine,
        actor=Color.BLUE,
        advertised_actions=tuple(engine.state.playable_actions),
    )
    text = IndexedTileRowsBoardPresenter().present(context)
    image = ImageBoardPresenter(image_size=512).present(context)
    blue_image = ImageBoardPresenter(image_size=512).present(blue_context)

    assert image.kind == "image"
    assert image.media_type == "image/png"
    assert image.data.startswith(b"\x89PNG\r\n\x1a\n")
    assert (image.width, image.height) == (512, 512)
    assert image.contains_entity_labels is False
    assert image.provenance.board_sha256 == text.provenance.board_sha256
    assert blue_image.provenance.perspective == Color.BLUE
    assert blue_image.provenance.board_sha256 == image.provenance.board_sha256
    assert blue_image.content_sha256 == image.content_sha256
    metadata = board_presentation_payload(image, include_text_content=True)
    assert metadata["data"] is None
    assert image.data.hex() not in str(metadata)


def test_board_presentation_contract_rejects_tampering():
    context = build_decision_context(_engine())
    text = IndexedTileRowsBoardPresenter().present(context)
    image = ImageBoardPresenter(image_size=512).present(context)

    with pytest.raises(ValueError, match="inconsistent"):
        replace(text, content=text.content + "\nTAMPERED")
    with pytest.raises(ValueError, match="inconsistent"):
        replace(image, data=image.data + b"tampered")
    with pytest.raises(ValueError, match="dimensions"):
        replace(image, width=image.width - 1)
    with pytest.raises(ValueError, match="Unsupported"):
        replace(image, media_type="image/gif")


def test_openai_board_encoding_attaches_only_to_latest_user_turn():
    messages = (
        ModelMessage("system", "identity"),
        ModelMessage("user", "old state"),
        ModelMessage("assistant", "old answer"),
        ModelMessage("user", "current state"),
    )
    context = build_decision_context(_engine())
    text = IndexedTileRowsBoardPresenter().present(context)
    image = ImageBoardPresenter(image_size=512).present(context)

    text_payload = openai_messages_with_board(
        messages,
        text,
        allow_image_input=False,
    )
    assert text_payload[1]["content"] == "old state"
    assert text_payload[-1]["content"].startswith("PUBLIC BOARD:\n")
    assert text_payload[-1]["content"].endswith("current state")
    assert text.content in text_payload[-1]["content"]

    with pytest.raises(ValueError, match="explicitly enabled"):
        openai_messages_with_board(
            messages,
            image,
            allow_image_input=False,
        )

    image_payload = openai_messages_with_board(
        messages,
        image,
        allow_image_input=True,
    )
    content = image_payload[-1]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert content[1] == {"type": "text", "text": "current state"}

    sanitized = sanitize_provider_payload(image_payload, image)
    assert "data:image" not in str(sanitized)
    assert image.content_sha256 in str(sanitized)
