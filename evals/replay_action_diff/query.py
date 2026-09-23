"""Query driver across a replay's decisions and models."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Sequence

from cle.replay.contracts import as_mapping
from evals.json_types import JsonDict, as_dict, as_str
from evals.replay_action_diff.actors import infer_actor
from evals.replay_action_diff.contracts import (
    SCHEMA_VERSION,
    replay_parsed_actions,
    require_archive,
)
from evals.replay_action_diff.decisions import (
    _load_replay,
    _resolve_target_player,
    _step_replay,
    build_decision_point,
)
from evals.replay_action_diff.identity import _state_fingerprint, utc_now
from evals.replay_action_diff.model_query import _query_model
from evals.replay_action_diff.responses import (
    _append_jsonl,
    _latest_responses,
    _read_jsonl,
    _response_cost_usd,
    normalize_response_selection,
)
from playground.game_viewer.state import server_state


def query_replay(
    game_id: str,
    target_player: str,
    manifest: Sequence[JsonDict],
    models: Sequence[str],
    responses_path: Path,
    reasoning_effort: str = "xhigh",
    max_tokens: int = 8_192,
    max_requests_per_model: Optional[int] = None,
    max_cost_usd: Optional[float] = None,
    retry_errors: bool = False,
) -> Dict[str, int]:
    """Query each model at every exact point, then advance only the human replay."""
    exact_by_index = {
        row["replay_index"]: row
        for row in manifest
        if row.get("classification") == "exact"
    }
    existing_rows = _read_jsonl(responses_path)
    latest = _latest_responses(existing_rows)
    new_requests: Counter[str] = Counter()
    manifest_ids = {record["decision_id"] for record in manifest}
    spent_usd = sum(
        _response_cost_usd(row)
        for (decision_id, model_id), row in latest.items()
        if decision_id in manifest_ids and model_id in models
    )
    state = _load_replay(game_id)

    try:
        target_player_id = _resolve_target_player(state, target_player)
        archive = require_archive(state)
        target_index = as_mapping(
            archive["colonist_color_to_engine_idx"], "colonist_color_to_engine_idx"
        ).get(str(target_player_id))
        parsed_actions = replay_parsed_actions(state)

        while state.replay_index < len(parsed_actions):
            replay_index = state.replay_index
            expected = exact_by_index.get(replay_index)
            if expected is not None:
                hint: Mapping[str, object] = parsed_actions[replay_index]
                actor_index, actor_source = infer_actor(state, hint)
                if actor_index is None or actor_index != target_index:
                    raise RuntimeError(
                        f"Decision actor changed at replay row {replay_index}: "
                        f"expected seat {target_index}, got {actor_index}"
                    )
                record, point, _ = build_decision_point(
                    state, replay_index, actor_index, actor_source
                )
                if point is None or record.get("classification") != "exact":
                    raise RuntimeError(
                        f"Exact decision no longer maps at replay row {replay_index}: {record}"
                    )
                if record["signature"] != expected["signature"]:
                    raise RuntimeError(
                        f"Decision menu changed at replay row {replay_index}"
                    )

                decision_id = as_str(record["decision_id"], "decision_id")
                models_to_query = []
                for model_id in models:
                    prior = latest.get((decision_id, model_id))
                    if prior is not None and not (retry_errors and prior.get("error")):
                        continue
                    if (
                        max_requests_per_model is not None
                        and new_requests[model_id] >= max_requests_per_model
                    ):
                        continue
                    if max_cost_usd is not None and spent_usd >= max_cost_usd:
                        continue
                    models_to_query.append(model_id)

                before = _state_fingerprint(state)
                response_rows: list[JsonDict] = []
                if models_to_query:
                    with ThreadPoolExecutor(max_workers=len(models_to_query)) as executor:
                        futures = {
                            executor.submit(
                                _query_model,
                                model_id,
                                point.context,
                                reasoning_effort,
                                max_tokens,
                            ): model_id
                            for model_id in models_to_query
                        }
                        for future in as_completed(futures):
                            model_id = futures[future]
                            new_requests[model_id] += 1
                            row: JsonDict
                            try:
                                result = future.result()
                                model_action_index = result.get("action_index")
                                row = normalize_response_selection({
                                    "schema_version": SCHEMA_VERSION,
                                    "recorded_at": utc_now(),
                                    "decision_id": record["decision_id"],
                                    "game_id": game_id,
                                    "replay_index": replay_index,
                                    "model_id": model_id,
                                    "human_action_index": as_dict(record["human"], "human")["action_index"],
                                    "model_action_index": model_action_index,
                                    "agreement": model_action_index
                                    == as_dict(record["human"], "human")["action_index"],
                                    "error": None,
                                    "result": result,
                                })
                            except Exception as exc:
                                row = {
                                    "schema_version": SCHEMA_VERSION,
                                    "recorded_at": utc_now(),
                                    "decision_id": record["decision_id"],
                                    "game_id": game_id,
                                    "replay_index": replay_index,
                                    "model_id": model_id,
                                    "human_action_index": as_dict(record["human"], "human")["action_index"],
                                    "model_action_index": None,
                                    "agreement": False,
                                    "error": {
                                        "type": type(exc).__name__,
                                        "message": str(exc),
                                    },
                                    "result": None,
                                }
                            response_rows.append(row)
                            latest[(decision_id, model_id)] = row
                            spent_usd += _response_cost_usd(row)
                    _append_jsonl(responses_path, response_rows)

                after = _state_fingerprint(state)
                if after != before:
                    raise RuntimeError(
                        f"Model query mutated replay state at row {replay_index}"
                    )

            _step_replay(state)

        if state.replay_index != len(parsed_actions):
            raise RuntimeError(
                f"Query replay stopped at {state.replay_index}/{len(parsed_actions)}"
            )
        semantic_errors = [
            issue
            for issue in state.replay_semantic_issues
            if issue.get("severity") == "error"
        ]
        if semantic_errors:
            raise RuntimeError(
                f"Causal query replay produced {len(semantic_errors)} semantic errors"
            )
        return dict(new_requests)
    finally:
        server_state.reset()

