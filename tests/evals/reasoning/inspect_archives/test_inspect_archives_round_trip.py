"""Archive round trips and sample identity preservation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ContentImage, ContentText

from evals.inspect_archives import (
    DEFAULT_STRICT_VISION_RUNS,
    _policy_available_actions,
    build_policy_archive,
    build_strict_vision_archive,
    is_valid_policy_selection,
    verify_inspect_archive_log,
    write_inspect_archive_log,
)


def test_strict_vision_archive_preserves_prompt_targets_and_scores() -> None:
    bundle: Any = build_strict_vision_archive(DEFAULT_STRICT_VISION_RUNS["gemma4_31b"])

    assert bundle.model_id == "google/gemma-4-31b-it"
    assert str(bundle.model) == "archive/google/gemma-4-31b-it"
    assert bundle.source_records == 60
    assert bundle.imported_records == 60
    assert bundle.task.metadata["model_selection_report"].endswith(
        "local-27b-80b-models-for-catan.md"
    )
    assert bundle.expected_metrics == {
        "exact": 28,
        "json_valid": 60,
        "protocol_exact": 24,
        "requests": 60,
    }

    sample: Any = bundle.task.dataset[0]
    assert sample.id == "ascii_q000_direction_to_tile"
    assert sample.target == '{"tile":"T04"}'
    assert isinstance(sample.input[0], ChatMessageSystem)
    assert isinstance(sample.input[1], ChatMessageUser)
    user_content = sample.input[1].content
    assert isinstance(user_content, list)
    assert isinstance(user_content[0], ContentImage)
    assert Path(user_content[0].image).is_file()
    assert isinstance(user_content[1], ContentText)
    assert "Question: Which tile is directly LEFT of T00?" in user_content[1].text
    assert sample.metadata["archive"]["score"]["correct"] is True
    assert sample.metadata["provenance"]["scorer_version"] == "strict_typed_json/v2"


def test_policy_archive_uses_original_provider_run_without_repair_overlays() -> None:
    bundle: Any = build_policy_archive()

    assert bundle.model_id == "qwen/qwen3.8-27b"
    assert str(bundle.model) == "archive/qwen/qwen3.8-27b"
    assert bundle.source_records == 164
    assert bundle.imported_records == 111
    assert bundle.task.metadata["model_selection_report"].endswith(
        "local-27b-80b-models-for-catan.md"
    )
    assert bundle.expected_metrics == {
        "decision_count": 164,
        "response_count": 111,
        "exact_responses": 111,
        "exact_human_matches": 50,
        "nontrivial_responses": 89,
        "nontrivial_human_matches": 28,
        "parse_warnings": 35,
        "valid_selections": 111,
    }

    sample: Any = bundle.task.dataset[0]
    assert sample.id == "242781000:0"
    assert "not a quality oracle" in sample.target
    assert sample.metadata["policy"]["model_action_index"] == 0
    assert sample.metadata["policy"]["human_action_index"] == 7
    assert sample.metadata["policy"]["agreement"] is False
    assert sample.metadata["available_actions"][7]["index"] == 7
    assert "three distinct resources: Sheep, Wood, and Wood" in sample.metadata[
        "visible_rationale"
    ]


def test_provider_free_inspect_log_round_trip(tmp_path: Path) -> None:
    bundle = build_policy_archive()
    log_path = write_inspect_archive_log(bundle, tmp_path, embed_images=False)
    verification = verify_inspect_archive_log(bundle, log_path)

    assert verification == {
        "status": "success",
        "model": "archive/qwen/qwen3.8-27b",
        "samples": 111,
        "scores": {
            "policy_selection_valid": 1.0,
            "policy_parse_clean": pytest.approx(76 / 111),
            "recorded_human_action_match": pytest.approx(50 / 111),
            "nontrivial_recorded_human_action_match": pytest.approx(28 / 89),
        },
        "source_scores_verified": True,
    }

    log: Any = read_eval_log(str(log_path))
    assert log.eval.metadata["archive_variant"] == "original_provider_run"
    assert log.eval.metadata["interpretation"].endswith(
        "is not a policy-quality oracle."
    )
    first: Any = log.samples[0]
    assert first.messages[-1].content.startswith("<goals>")
    assert first.scores["recorded_human_action_match"].value == "I"
    assert first.metadata["archive"]["provider"] is None
    assert first.metadata["archive"]["adapter_source"] == "action_diff"


def test_embedded_image_log_preserves_exact_sample_identity(tmp_path: Path) -> None:
    bundle = build_strict_vision_archive(
        DEFAULT_STRICT_VISION_RUNS["gemma4_31b"],
        limit=1,
    )
    log_path = write_inspect_archive_log(bundle, tmp_path, embed_images=True)
    verification = verify_inspect_archive_log(bundle, log_path)

    assert verification["samples"] == 1
    log: Any = read_eval_log(str(log_path), resolve_attachments=True)
    user_content = log.samples[0].messages[1].content
    assert isinstance(user_content, list)
    assert isinstance(user_content[0], ContentImage)
    assert user_content[0].image.startswith("data:image/png;base64,")


def test_policy_selection_validates_exact_menu_membership() -> None:
    bundle = build_policy_archive(limit=1)
    sample: Any = bundle.task.dataset[0]
    policy = dict(sample.metadata["policy"])
    actions = sample.metadata["available_actions"]

    assert is_valid_policy_selection(policy, actions)
    policy["model_action_index"] = 999
    assert not is_valid_policy_selection(policy, actions)
    policy["model_action_index"] = False
    assert not is_valid_policy_selection(policy, actions)
    policy["model_action_index"] = 0
    policy["model_action"] = "different action"
    assert not is_valid_policy_selection(policy, actions)


def test_policy_archive_rejects_missing_provider_legal_menu() -> None:
    with pytest.raises(ValueError, match="missing its exact legal menu"):
        _policy_available_actions({"decision_id": "game:1"}, {"result": {}})


def test_partial_strict_log_round_trip_skips_full_aggregate_assertion(
    tmp_path: Path,
) -> None:
    bundle = build_strict_vision_archive(
        DEFAULT_STRICT_VISION_RUNS["gemma4_31b"],
        limit=2,
    )
    log_path = write_inspect_archive_log(bundle, tmp_path, embed_images=False)
    verification = verify_inspect_archive_log(bundle, log_path)

    assert verification["samples"] == 2
    assert verification["source_scores_verified"] is False
    assert verification["scores"] == {
        "strict_exact": 1.0,
        "strict_json_valid": 1.0,
        "strict_protocol_exact": 1.0,
    }
