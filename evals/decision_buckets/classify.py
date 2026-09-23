"""Multi-label bucket assignment over an ordered decision manifest."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

from evals.decision_buckets.features import (
    _coerce_features,
    _consequential_counts_by_turn,
    _decision_stage,
    _is_award_race,
    _record_action_type,
    _turn_key,
)
from evals.decision_buckets.loading import load_decision_bucket_suite
from evals.decision_buckets.models import (
    BucketAssignment,
    DecisionBucketSuite,
    DecisionStateFeatures,
)
from evals.decision_buckets.taxonomy import (
    BUILD_PRIORITY_ACTIONS,
    CRITICAL_VP,
    DEVELOPMENT_CARD_ACTIONS,
    LATE_PUBLIC_VP,
    ROBBER_ACTIONS,
    TRADE_ACTIONS,
    TRADE_OFFER_ACTIONS,
    TRADE_RESPONSE_ACTIONS,
    VP_ACTIONS,
)
from evals.json_types import JsonValue


def classify_decision_records(
    records: Sequence[Mapping[str, object]],
    suite: DecisionBucketSuite | None = None,
) -> list[dict[str, JsonValue]]:
    """Attach stable multi-label assignments to an ordered decision manifest."""
    bucket_suite = suite or load_decision_bucket_suite()
    known_bucket_ids = set(bucket_suite.bucket_ids)
    bucket_definitions = {bucket.id: bucket for bucket in bucket_suite.buckets}
    setup_ordinals: defaultdict[str, int] = defaultdict(int)
    turn_consequential_counts = _consequential_counts_by_turn(records)
    classified: list[dict[str, JsonValue]] = []

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
    record: Mapping[str, object],
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

