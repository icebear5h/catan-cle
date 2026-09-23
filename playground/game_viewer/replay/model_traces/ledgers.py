"""The optional side ledgers, and the identity the plan must declare."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .readers import _read_json, _read_jsonl, _require_equal
from .schema import ModelTraceArtifactError

__all__ = ["SideLedgers", "read_side_ledgers", "validate_plan_identity"]


@dataclass(frozen=True)
class SideLedgers:
    """Setup overrides, rationale repairs, and the ledgers that authorize them."""

    response_overrides: list[dict[str, object]]
    overrides_partial: bool
    rationale_repairs: list[dict[str, object]]
    repairs_partial: bool
    rationale_repair_attempts: list[dict[str, object]]
    repair_attempts_partial: bool
    decision_quality_warnings: dict[str, list[str]]
    response_attempts: list[dict[str, object]]
    attempts_partial: bool


def read_side_ledgers(
    *,
    response_overrides_path: Path | None,
    response_attempts_path: Path | None,
    quality_path: Path | None,
    rationale_repairs_path: Path | None,
    rationale_repair_attempts_path: Path | None,
) -> SideLedgers:
    """Read every optional ledger, enforcing the pairings each one requires."""
    response_overrides: list[dict[str, object]] = []
    overrides_partial = False
    rationale_repairs: list[dict[str, object]] = []
    repairs_partial = False
    rationale_repair_attempts: list[dict[str, object]] = []
    repair_attempts_partial = False
    decision_quality_warnings: dict[str, list[str]] = {}
    response_attempts: list[dict[str, object]] = []
    attempts_partial = False
    if response_overrides_path is not None:
        response_overrides, overrides_partial = _read_jsonl(
            Path(response_overrides_path),
            tolerate_incomplete_final_line=False,
        )
        if response_attempts_path is None:
            raise ModelTraceArtifactError(
                "Setup overrides require an immutable attempts ledger"
            )
        response_attempts, attempts_partial = _read_jsonl(
            Path(response_attempts_path),
            tolerate_incomplete_final_line=False,
        )
        if quality_path is not None:
            quality = _read_json(Path(quality_path))
            _require_equal(
                "setup quality schema",
                quality.get("schema"),
                "setup-strategy-quality-v1",
            )
            warnings = quality.get("decision_warnings")
            if not isinstance(warnings, dict) or not all(
                isinstance(decision_id, str)
                and isinstance(items, list)
                and items
                and all(isinstance(item, str) and item for item in items)
                for decision_id, items in warnings.items()
            ):
                raise ModelTraceArtifactError(
                    "Setup strategy quality warnings are malformed"
                )
            decision_quality_warnings = warnings
        if rationale_repairs_path is not None:
            if rationale_repair_attempts_path is None:
                raise ModelTraceArtifactError(
                    "Selected rationale repairs require an attempts ledger"
                )
            rationale_repairs, repairs_partial = _read_jsonl(
                Path(rationale_repairs_path),
                tolerate_incomplete_final_line=False,
            )
            rationale_repair_attempts, repair_attempts_partial = _read_jsonl(
                Path(rationale_repair_attempts_path),
                tolerate_incomplete_final_line=False,
            )
    return SideLedgers(
        response_overrides=response_overrides,
        overrides_partial=overrides_partial,
        rationale_repairs=rationale_repairs,
        repairs_partial=repairs_partial,
        rationale_repair_attempts=rationale_repair_attempts,
        repair_attempts_partial=repair_attempts_partial,
        decision_quality_warnings=decision_quality_warnings,
        response_attempts=response_attempts,
        attempts_partial=attempts_partial,
    )


def validate_plan_identity(
    plan: Mapping[str, object],
    *,
    expected_game_id: str,
    expected_model_id: str,
    expected_player_id: int,
    expected_engine_color: str,
    narrator: Mapping[str, object] | None,
) -> Mapping[str, object]:
    """Check the plan names this run, and return the settings it declares."""
    _require_equal("game", str(plan.get("game_id")), str(expected_game_id))
    _require_equal("models", plan.get("models"), [expected_model_id])
    _require_equal("target player", plan.get("target_player_id"), expected_player_id)
    _require_equal(
        "target engine color",
        plan.get("target_engine_color"),
        expected_engine_color,
    )
    if narrator:
        _require_equal(
            "narrator player",
            narrator.get("colonist_color"),
            expected_player_id,
        )

    settings = plan.get("settings")
    if not isinstance(settings, dict):
        raise ModelTraceArtifactError("Model-trace plan is missing settings")
    _require_equal("stateless goals policy", settings.get("stateless_goals"), True)
    _require_equal("lookahead policy", settings.get("allow_lookahead"), False)
    _require_equal(
        "model-action execution policy",
        settings.get("execute_model_actions"),
        False,
    )
    return settings
