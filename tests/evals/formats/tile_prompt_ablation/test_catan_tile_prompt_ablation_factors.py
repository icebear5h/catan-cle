"""Prompt factor coverage and strict condition scoring."""

from __future__ import annotations

from typing import Any

import pytest

import scripts.board_bench.run.eval_catan_tile_prompt_ablation as tile_ablation
from scripts.board_bench.run.eval_catan_tile_prompt_ablation import (
    build_prompt,
    expected_response,
    extract_message_text,
    score_response,
    scorer_sha256,
)


def test_prompts_cross_wrapper_and_description_factors() -> None:
    angle_labels = build_prompt("angle_labels")
    plain_labels = build_prompt("plain_labels")
    angle_described = build_prompt("angle_described")
    plain_described = build_prompt("plain_described")

    for resource in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE", "DESERT"):
        assert f"<{resource}>" in angle_labels
        assert f"<{resource}>" in angle_described
        assert f"<{resource}>" not in plain_labels
        assert f"<{resource}>" not in plain_described
        assert resource in plain_labels
        assert resource in plain_described

    assert "Visual guide" not in angle_labels
    assert "Visual guide" not in plain_labels
    assert "evergreen tree" in angle_described
    assert "stacked brick blocks" in plain_described
    assert "NO_NUMBER" not in angle_described
    assert "FOREST" not in plain_described

    assert angle_labels.replace("<", "").replace(">", "") == plain_labels
    assert (
        angle_described.replace("<", "").replace(">", "")
        == plain_described
    )


def test_strict_condition_scoring_rejects_aliases_extras_and_wrong_wrappers() -> None:
    wood: Any = {"resource": "WOOD", "number": 2}
    desert: Any = {"resource": "DESERT", "number": None}

    exact_angle = score_response(wood, "<WOOD> 2", "angle_labels")
    assert exact_angle["pair_correct"] is True
    assert exact_angle["protocol_exact"] is True
    assert exact_angle["wrapper_rescued_content_correct"] is False

    wrong_wrapper = score_response(wood, "WOOD 2", "angle_labels")
    assert wrong_wrapper["protocol_valid"] is False
    assert wrong_wrapper["pair_correct"] is False
    assert wrong_wrapper["wrapper_rescued_content_correct"] is True

    for invalid in (
        "<FOREST> 2",
        "<WOOD> 2.",
        "Answer: <WOOD> 2",
        "```<WOOD> 2```",
        " <WOOD> 2",
        "<WOOD> 2\n",
    ):
        assert score_response(wood, invalid, "angle_labels")[
            "protocol_valid"
        ] is False

    assert extract_message_text({"content": " <WOOD> 2\n"}) == " <WOOD> 2\n"

    wrong_resource = score_response(wood, "<ORE> 2", "angle_labels")
    assert wrong_resource["protocol_valid"] is True
    assert wrong_resource["number_correct"] is True
    assert wrong_resource["resource_correct"] is False
    assert wrong_resource["pair_correct"] is False

    assert score_response(desert, "<DESERT>", "angle_labels")["pair_correct"]
    assert score_response(desert, "DESERT", "plain_labels")["pair_correct"]
    assert not score_response(desert, "<DESERT> 2", "angle_labels")[
        "protocol_valid"
    ]
    assert expected_response(wood, "plain_described") == "WOOD 2"
    assert expected_response(desert, "angle_described") == "<DESERT>"


def test_scorer_hash_covers_condition_sets_and_regex_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = scorer_sha256()
    monkeypatch.setattr(
        tile_ablation,
        "ANGLE_CONDITIONS",
        frozenset({"angle_labels"}),
    )
    assert scorer_sha256() != baseline

    monkeypatch.undo()
    monkeypatch.setattr(
        tile_ablation,
        "ANGLE_NUMBER_RE",
        tile_ablation.re.compile(
            tile_ablation.ANGLE_NUMBER_RE.pattern,
            tile_ablation.re.IGNORECASE,
        ),
    )
    assert scorer_sha256() != baseline
