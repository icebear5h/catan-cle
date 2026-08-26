"""Versioned scenario buckets for qualitative Catan decision evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


BUCKET_SCHEMA = "decision-bucket-suite-v1"
STATE_FEATURE_SCHEMA = "decision-state-features-v1"
STAGE_IDS = ("setup", "early", "mid", "late")
LATE_PUBLIC_VP = 7
CRITICAL_VP = 9
EARLY_ROUNDS = 3

BUILD_PRIORITY_ACTIONS = {
    "BUILD_ROAD",
    "BUILD_SETTLEMENT",
    "BUILD_CITY",
    "BUY_DEVELOPMENT_CARD",
}
ROBBER_ACTIONS = {"MOVE_ROBBER", "STEAL", "PLAY_KNIGHT_CARD"}
TRADE_OFFER_ACTIONS = {"OFFER_TRADE", "COUNTER_OFFER"}
TRADE_RESPONSE_ACTIONS = {
    "ACCEPT_TRADE",
    "REJECT_TRADE",
    "COUNTER_OFFER",
    "CONFIRM_TRADE",
}
TRADE_ACTIONS = TRADE_OFFER_ACTIONS | TRADE_RESPONSE_ACTIONS | {"MARITIME_TRADE"}
DEVELOPMENT_CARD_ACTIONS = {
    "PLAY_KNIGHT_CARD",
    "PLAY_MONOPOLY",
    "MONOPOLY_RESOURCE",
    "PLAY_YEAR_OF_PLENTY",
    "YEAR_OF_PLENTY_RESOURCES",
    "PLAY_ROAD_BUILDING",
}
CONSEQUENTIAL_ACTIONS = BUILD_PRIORITY_ACTIONS | TRADE_ACTIONS | DEVELOPMENT_CARD_ACTIONS
VP_ACTIONS = {"BUILD_SETTLEMENT", "BUILD_CITY", "BUY_DEVELOPMENT_CARD"}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class StageRule(_StrictModel):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    detection: str = Field(min_length=1)


class CriticalRule(_StrictModel):
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    detection: str = Field(min_length=1)


class BucketDefinition(_StrictModel):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    group: Literal[
        "setup",
        "early",
        "economy",
        "sequencing",
        "development",
        "midgame",
        "late",
    ]
    description: str = Field(min_length=1)
    review_unit: Literal["single_decision", "whole_turn", "trade_episode", "robber_episode", "setup_pair"]
    target_samples: int = Field(ge=1)
    detection: str = Field(min_length=1)
    rubric: tuple[str, ...] = Field(min_length=1)


class ReviewLabel(_StrictModel):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)


class VerdictDefinition(_StrictModel):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SamplingPolicy(_StrictModel):
    default_target_per_bucket: int = Field(ge=1)
    random_fraction: float = Field(ge=0, le=1)
    reasoning_disagreement_fraction: float = Field(ge=0, le=1)
    high_stakes_fraction: float = Field(ge=0, le=1)
    include_model_agreements: bool

    @model_validator(mode="after")
    def validate_fractions(self) -> "SamplingPolicy":
        total = (
            self.random_fraction
            + self.reasoning_disagreement_fraction
            + self.high_stakes_fraction
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError("sampling fractions must sum to 1")
        return self


class DecisionBucketSuite(_StrictModel):
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    schema_id: Literal[BUCKET_SCHEMA] = Field(alias="schema")
    stage_rules: tuple[StageRule, ...]
    critical_rule: CriticalRule
    buckets: tuple[BucketDefinition, ...]
    review_labels: tuple[ReviewLabel, ...]
    verdicts: tuple[VerdictDefinition, ...]
    sampling: SamplingPolicy

    @model_validator(mode="after")
    def validate_catalog(self) -> "DecisionBucketSuite":
        stage_ids = tuple(rule.id for rule in self.stage_rules)
        if stage_ids != STAGE_IDS:
            raise ValueError(f"stage rule order must be {STAGE_IDS!r}")
        _require_unique("bucket", [bucket.id for bucket in self.buckets])
        _require_unique("review label", [label.id for label in self.review_labels])
        _require_unique("verdict", [verdict.id for verdict in self.verdicts])
        return self

    @property
    def bucket_ids(self) -> tuple[str, ...]:
        return tuple(bucket.id for bucket in self.buckets)


class DecisionStateFeatures(_StrictModel):
    schema_id: Literal[STATE_FEATURE_SCHEMA] = Field(
        default=STATE_FEATURE_SCHEMA,
        alias="schema",
    )
    completed_turns: int = Field(ge=0)
    table_round: int = Field(ge=1)
    player_count: int = Field(ge=2)
    actor_color: str = Field(min_length=1)
    turn_owner_color: str = Field(min_length=1)
    actor_is_turn_owner: bool
    actor_public_vp: int = Field(ge=0)
    actor_actual_vp: int = Field(ge=0)
    opponent_max_public_vp: int = Field(ge=0)
    max_public_vp: int = Field(ge=0)
    public_vp: dict[str, int]
    actor_longest_road_length: int = Field(ge=0)
    actor_played_knights: int = Field(ge=0)
    actor_has_longest_road: bool
    actor_has_largest_army: bool
    available_action_types: tuple[str, ...]


class BucketAssignment(_StrictModel):
    suite_id: str
    suite_version: str
    stage: Literal["setup", "early", "mid", "late", "unknown"]
    critical: bool
    bucket_ids: tuple[str, ...]
    episode_ids: dict[str, str]
    evidence: dict[str, Any]


def _require_unique(label: str, values: Sequence[str]) -> None:
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {label} ids: {duplicates}")


def default_bucket_suite_path() -> Path:
    return Path(__file__).resolve().parent / "suites" / "decision_spot_checks_v1.yaml"


def load_decision_bucket_suite(path: str | Path | None = None) -> DecisionBucketSuite:
    suite_path = Path(path) if path is not None else default_bucket_suite_path()
    with suite_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Decision bucket suite {suite_path} must contain a YAML mapping")
    return DecisionBucketSuite.model_validate(data)


def decision_state_features(
    game_state: Any,
    actor_color: Any,
    legal_actions: Iterable[Any],
) -> dict[str, Any]:
    """Capture only state evidence needed for deterministic scenario tagging."""
    colors = tuple(game_state.colors)
    actor_index = game_state.color_to_index[actor_color]
    actor_key = f"P{actor_index}"
    public_vp = {
        _color_name(color): int(
            game_state.player_state.get(
                f"P{game_state.color_to_index[color]}_VICTORY_POINTS",
                0,
            )
        )
        for color in colors
    }
    opponent_vp = [
        score for color, score in public_vp.items() if color != _color_name(actor_color)
    ]
    action_types = tuple(sorted({_action_type_name(action) for action in legal_actions}))
    completed_turns = int(game_state.num_turns)
    payload = DecisionStateFeatures(
        completed_turns=completed_turns,
        table_round=(completed_turns // len(colors)) + 1,
        player_count=len(colors),
        actor_color=_color_name(actor_color),
        turn_owner_color=_color_name(colors[game_state.current_turn_index]),
        actor_is_turn_owner=game_state.current_turn_index == actor_index,
        actor_public_vp=int(game_state.player_state.get(f"{actor_key}_VICTORY_POINTS", 0)),
        actor_actual_vp=int(
            game_state.player_state.get(f"{actor_key}_ACTUAL_VICTORY_POINTS", 0)
        ),
        opponent_max_public_vp=max(opponent_vp, default=0),
        max_public_vp=max(public_vp.values(), default=0),
        public_vp=public_vp,
        actor_longest_road_length=int(
            game_state.player_state.get(f"{actor_key}_LONGEST_ROAD_LENGTH", 0)
        ),
        actor_played_knights=int(
            game_state.player_state.get(f"{actor_key}_PLAYED_KNIGHT", 0)
        ),
        actor_has_longest_road=bool(
            game_state.player_state.get(f"{actor_key}_HAS_ROAD", False)
        ),
        actor_has_largest_army=bool(
            game_state.player_state.get(f"{actor_key}_HAS_ARMY", False)
        ),
        available_action_types=action_types,
    )
    return payload.model_dump(mode="json", by_alias=True)


def classify_decision_records(
    records: Sequence[Mapping[str, Any]],
    suite: DecisionBucketSuite | None = None,
) -> list[dict[str, Any]]:
    """Attach stable multi-label assignments to an ordered decision manifest."""
    bucket_suite = suite or load_decision_bucket_suite()
    known_bucket_ids = set(bucket_suite.bucket_ids)
    bucket_definitions = {bucket.id: bucket for bucket in bucket_suite.buckets}
    setup_ordinals: defaultdict[str, int] = defaultdict(int)
    turn_consequential_counts = _consequential_counts_by_turn(records)
    classified: list[dict[str, Any]] = []

    for record in records:
        action_type = _record_action_type(record)
        phase = str(record.get("phase") or "")
        features = _coerce_features(record.get("state_features"))
        stage = _decision_stage(phase, features)
        critical = bool(
            features
            and (
                features.actor_actual_vp >= CRITICAL_VP
                or features.opponent_max_public_vp >= CRITICAL_VP
            )
        )
        available_types = set(features.available_action_types if features else ())
        bucket_ids: list[str] = []

        if stage == "setup":
            ordinal = setup_ordinals[action_type]
            setup_ordinals[action_type] += 1
            if action_type == "BUILD_SETTLEMENT":
                bucket_ids.append("first_settlement" if ordinal == 0 else "second_settlement")
            elif action_type == "BUILD_ROAD":
                bucket_ids.append("initial_road")

        if stage == "early":
            if features and features.table_round == 1 and not record.get("forced", False):
                bucket_ids.append("early_opening_turn")
            if action_type in {"BUILD_ROAD", "BUILD_SETTLEMENT"} or available_types.intersection(
                {"BUILD_ROAD", "BUILD_SETTLEMENT"}
            ):
                bucket_ids.append("early_expansion")
            if len(available_types.intersection(BUILD_PRIORITY_ACTIONS)) >= 2:
                bucket_ids.append("early_build_priority")
            if action_type in ROBBER_ACTIONS:
                bucket_ids.append("early_robber")
            if action_type == "DISCARD":
                bucket_ids.append("early_discard")
            if action_type in TRADE_OFFER_ACTIONS:
                bucket_ids.append("early_trade_offer")
            if action_type in TRADE_RESPONSE_ACTIONS:
                bucket_ids.append("early_trade_response")

        if action_type == "MARITIME_TRADE":
            bucket_ids.append("maritime_trade")
        if action_type in DEVELOPMENT_CARD_ACTIONS:
            bucket_ids.append("development_card_timing")

        turn_key = _turn_key(record, features)
        if (
            stage != "setup"
            and turn_key is not None
            and turn_consequential_counts.get(turn_key, 0) >= 2
        ):
            bucket_ids.append("multi_action_turn")

        award_race = _is_award_race(action_type, available_types, features)
        if award_race:
            bucket_ids.append("award_race")

        if stage == "mid":
            if len(available_types.intersection(BUILD_PRIORITY_ACTIONS)) >= 2:
                bucket_ids.append("midgame_build_priority")
            if action_type in ROBBER_ACTIONS | {"BUILD_ROAD", "BUILD_SETTLEMENT"}:
                bucket_ids.append("midgame_blocking")

        if stage == "late":
            if not record.get("forced", False):
                bucket_ids.append("late_turn")
            if features and features.actor_actual_vp >= CRITICAL_VP and available_types.intersection(
                VP_ACTIONS
            ):
                bucket_ids.append("late_win_conversion")
            if features and LATE_PUBLIC_VP <= features.actor_actual_vp < CRITICAL_VP:
                bucket_ids.append("late_win_setup")
            if features and features.opponent_max_public_vp >= CRITICAL_VP and not record.get(
                "forced", False
            ):
                bucket_ids.append("late_opponent_denial")
            if action_type in TRADE_ACTIONS:
                bucket_ids.append("late_trade")
            if action_type in ROBBER_ACTIONS:
                bucket_ids.append("late_robber")
            if award_race:
                bucket_ids.append("late_award_swing")

        if action_type == "END_TURN" and not record.get("forced", False):
            bucket_ids.append("end_turn_discipline")

        unique_ids = tuple(dict.fromkeys(bucket_ids))
        unknown_ids = set(unique_ids) - known_bucket_ids
        if unknown_ids:
            raise ValueError(f"Classifier emitted unknown bucket ids: {sorted(unknown_ids)}")

        evidence = {
            "action_type": action_type,
            "phase": phase or None,
            "completed_turns": features.completed_turns if features else None,
            "table_round": features.table_round if features else None,
            "actor_is_turn_owner": features.actor_is_turn_owner if features else None,
            "actor_public_vp": features.actor_public_vp if features else None,
            "actor_actual_vp": features.actor_actual_vp if features else None,
            "opponent_max_public_vp": (
                features.opponent_max_public_vp if features else None
            ),
            "available_action_types": sorted(available_types),
        }
        episode_ids = {
            bucket_id: _episode_id(
                record,
                bucket_definitions[bucket_id].review_unit,
                features,
            )
            for bucket_id in unique_ids
        }
        assignment = BucketAssignment(
            suite_id=bucket_suite.id,
            suite_version=bucket_suite.version,
            stage=stage,
            critical=critical,
            bucket_ids=unique_ids,
            episode_ids=episode_ids,
            evidence=evidence,
        )
        classified.append(assignment.model_dump(mode="json"))

    return classified


def _episode_id(
    record: Mapping[str, Any],
    review_unit: str,
    features: DecisionStateFeatures | None,
) -> str:
    game_id = str(record.get("game_id") or "unknown-game")
    decision_id = str(record.get("decision_id") or "unknown-decision")
    actor = record.get("actor")
    actor_color = (
        str(actor.get("engine_color") or "unknown-actor")
        if isinstance(actor, Mapping)
        else "unknown-actor"
    )
    completed_turns = features.completed_turns if features else record.get("replay_index", 0)
    if review_unit == "setup_pair":
        return f"{game_id}:setup:{actor_color}"
    if review_unit == "whole_turn":
        return f"{game_id}:turn:{completed_turns}:{actor_color}"
    if review_unit == "trade_episode":
        return f"{game_id}:turn:{completed_turns}:trade"
    if review_unit == "robber_episode":
        return f"{game_id}:turn:{completed_turns}:robber"
    return decision_id


def _coerce_features(raw: Any) -> DecisionStateFeatures | None:
    if not isinstance(raw, Mapping):
        return None
    return DecisionStateFeatures.model_validate(dict(raw))


def _decision_stage(
    phase: str,
    features: DecisionStateFeatures | None,
) -> Literal["setup", "early", "mid", "late", "unknown"]:
    if phase in {"initial_placement", "initial_settlement_1", "initial_settlement_2", "initial_road"}:
        return "setup"
    if features is None:
        return "unknown"
    if features.max_public_vp >= LATE_PUBLIC_VP:
        return "late"
    if features.completed_turns < EARLY_ROUNDS * features.player_count:
        return "early"
    return "mid"


def _consequential_counts_by_turn(
    records: Sequence[Mapping[str, Any]],
) -> Counter[tuple[str, int]]:
    counts: Counter[tuple[str, int]] = Counter()
    seen: set[tuple[tuple[str, int], int]] = set()
    for record in records:
        features = _coerce_features(record.get("state_features"))
        turn_key = _turn_key(record, features)
        replay_index = record.get("replay_index")
        action_type = _record_action_type(record)
        if turn_key is None or not isinstance(replay_index, int):
            continue
        if action_type not in CONSEQUENTIAL_ACTIONS:
            continue
        identity = (turn_key, replay_index)
        if identity in seen:
            continue
        seen.add(identity)
        counts[turn_key] += 1
    return counts


def _turn_key(
    record: Mapping[str, Any],
    features: DecisionStateFeatures | None,
) -> tuple[str, int] | None:
    actor = record.get("actor")
    actor_color = actor.get("engine_color") if isinstance(actor, Mapping) else None
    if (
        not isinstance(actor_color, str)
        or features is None
        or not features.actor_is_turn_owner
    ):
        return None
    return actor_color, features.completed_turns


def _is_award_race(
    action_type: str,
    available_types: set[str],
    features: DecisionStateFeatures | None,
) -> bool:
    if features is None:
        return False
    road_relevant = action_type in {"BUILD_ROAD", "PLAY_ROAD_BUILDING"} or bool(
        available_types.intersection({"BUILD_ROAD", "PLAY_ROAD_BUILDING"})
    )
    army_relevant = action_type == "PLAY_KNIGHT_CARD" or "PLAY_KNIGHT_CARD" in available_types
    return (
        road_relevant
        and (features.actor_longest_road_length >= 4 or features.actor_has_longest_road)
    ) or (
        army_relevant
        and (features.actor_played_knights >= 2 or features.actor_has_largest_army)
    )


def _record_action_type(record: Mapping[str, Any]) -> str:
    return str(record.get("effective_action_type") or record.get("source_action_type") or "")


def _action_type_name(action: Any) -> str:
    action_type = getattr(action, "action_type", None)
    value = getattr(action_type, "value", None)
    if isinstance(value, str):
        return value
    return str(action_type).replace("ActionType.", "").replace("AT.", "")


def _color_name(color: Any) -> str:
    name = getattr(color, "name", None)
    if isinstance(name, str):
        return name
    value = getattr(color, "value", None)
    return str(value if value is not None else color)
