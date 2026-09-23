"""Full-replay scan that collects every comparable decision."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from cle.replay.contracts import mapping_field
from evals.decision_buckets import classify_decision_records
from evals.json_types import JsonDict
from evals.replay_action_diff.actors import infer_actor
from evals.replay_action_diff.contracts import (
    SCHEMA_VERSION,
    replay_parsed_actions,
    require_archive,
    require_engine,
    to_json,
)
from evals.replay_action_diff.decisions import (
    _load_replay,
    _resolve_target_player,
    _step_replay,
    build_decision_point,
)
from evals.replay_action_diff.identity import _color_name
from playground.game_viewer.state import server_state


def scan_replay(game_id: str, target_player: str = "captured") -> JsonDict:
    """Replay the entire game locally and classify the target human's actions."""
    state = _load_replay(game_id)
    try:
        target_player_id = _resolve_target_player(state, target_player)
        archive = require_archive(state)
        mapping = mapping_field(archive, "colonist_color_to_engine_idx")
        target_index = mapping.get(str(target_player_id))
        if not isinstance(target_index, int):
            raise ValueError(
                f"Target Colonist player {target_player_id} is not in replay play order"
            )
        target_color = require_engine(state).state.colors[target_index]
        parsed_actions = replay_parsed_actions(state)
        records: list[JsonDict] = []
        step_statuses: Counter[str] = Counter()
        compound_followups: set[int] = set()

        while state.replay_index < len(parsed_actions):
            replay_index = state.replay_index
            hint: Mapping[str, object] = parsed_actions[replay_index]
            actor_index, actor_source = infer_actor(state, hint)

            if replay_index in compound_followups and actor_index == target_index:
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "decision_id": f"{game_id}:{replay_index}",
                        "game_id": game_id,
                        "replay_index": replay_index,
                        "source_replay_index": to_json(
                            hint.get("_source_replay_index", replay_index),
                            "_source_replay_index",
                        ),
                        "source_event_index": to_json(hint.get("index"), "index"),
                        "source_action_type": to_json(hint.get("type"), "type"),
                        "effective_action_type": to_json(hint.get("type"), "type"),
                        "classification": "lifecycle",
                        "reason": "compound decision follow-up scored at the prior announcement",
                        "available_actions": [],
                        "actor": {
                            "colonist_player": to_json(hint.get("player"), "player"),
                            "engine_index": actor_index,
                            "engine_color": _color_name(target_color),
                            "source": actor_source,
                        },
                    }
                )
            elif actor_index is not None and actor_index == target_index:
                record, _, followup_index = build_decision_point(
                    state, replay_index, actor_index, actor_source
                )
                records.append(record)
                if followup_index is not None:
                    compound_followups.add(followup_index)

            result = _step_replay(state)
            # JSON object keys are strings; the summary is always written as JSON.
            step_statuses[str(result.get("status", "unknown"))] += 1

        if state.replay_index != len(parsed_actions):
            raise RuntimeError(
                f"Replay stopped at {state.replay_index}/{len(parsed_actions)}"
            )
        bucket_assignments = classify_decision_records(records)
        for record, assignment in zip(records, bucket_assignments, strict=True):
            record["bucket_assignment"] = assignment

        semantic_errors = [
            issue
            for issue in state.replay_semantic_issues
            if issue.get("severity") == "error"
        ]
        return {
            "schema_version": SCHEMA_VERSION,
            "game_id": game_id,
            "target_player_id": target_player_id,
            "target_engine_index": target_index,
            "target_engine_color": _color_name(target_color),
            "replay_file": to_json(archive.get("file"), "file"),
            "parsed_action_count": len(parsed_actions),
            "canonicalizations": to_json(
                archive.get("evaluation_canonicalizations", []),
                "evaluation_canonicalizations",
            ),
            "records": [*records],
            "step_statuses": dict(step_statuses),
            "semantic_errors": to_json(semantic_errors, "semantic_errors"),
        }
    finally:
        server_state.reset()

