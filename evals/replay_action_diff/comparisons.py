"""Per-decision comparison rows and percentile helpers."""

from __future__ import annotations

import math
from typing import Optional, Sequence

from evals.json_types import JsonDict, JsonList, as_dict, as_list, as_str
from evals.replay_action_diff.contracts import object_or_empty
from evals.replay_action_diff.responses import _latest_responses, normalize_response_selection
from evals.replay_action_diff.shapes import Comparison, ModelComparison


def _percent(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def build_comparisons(
    manifest: Sequence[JsonDict],
    response_rows: Sequence[JsonDict],
    models: Sequence[str],
) -> list[Comparison]:
    latest = _latest_responses(
        [normalize_response_selection(row) for row in response_rows]
    )
    comparisons: list[Comparison] = []
    for decision in manifest:
        if decision.get("classification") != "exact":
            continue
        decision_id = as_str(decision["decision_id"], "decision_id")
        model_results: dict[str, ModelComparison] = {}
        human_candidates: list[tuple[int, JsonDict]] = []
        menu_orders: list[JsonList] = []
        for model_id in models:
            response = latest.get((decision_id, model_id))
            result = object_or_empty(response.get("result"), "result") if response else {}
            menu = (
                as_list(result.get("available_actions", []), "available_actions")
                if result
                else []
            )
            model_index = response.get("model_action_index") if response else None
            human_index = response.get("human_action_index") if response else None
            selected = (
                as_dict(menu[model_index], "selected action")
                if isinstance(model_index, int) and 0 <= model_index < len(menu)
                else None
            )
            stored_human = (
                as_dict(menu[human_index], "stored human action")
                if isinstance(human_index, int) and 0 <= human_index < len(menu)
                else None
            )
            if stored_human is not None and isinstance(human_index, int):
                human_candidates.append((human_index, stored_human))
            if menu:
                menu_orders.append(
                    [as_dict(action, "menu action").get("action") for action in menu]
                )
            agreement = (
                selected.get("action") == stored_human.get("action")
                if selected and stored_human
                else bool(response and response.get("agreement"))
            )
            model_results[model_id] = {
                "response_present": response is not None,
                "error": response.get("error") if response else None,
                "parse_error": result.get("parse_error") if result else None,
                "action_index": model_index,
                "action": selected.get("action") if selected else None,
                "description": selected.get("description") if selected else None,
                "agreement": agreement,
                "agreement_scope": "primary_action",
                "agreement_is_coarse": bool(response and response.get("agreement_is_coarse")),
                "followup_scoring": response.get("followup_scoring") if response else None,
                "followup_agreement": None,
                "knight_destination": result.get("knight_destination") if result else None,
                "requested_action_sequence": result.get("requested_action_sequence") if result else None,
            }

        if human_candidates:
            human_actions = {
                candidate.get("action") for _, candidate in human_candidates
            }
            if len(human_actions) != 1:
                raise RuntimeError(
                    f"Stored models disagree on the human label for {decision['decision_id']}"
                )
            first_index, human_selected = human_candidates[0]
            human: JsonDict = {
                "action_index": first_index,
                "action": human_selected.get("action"),
                "description": human_selected.get("description"),
                "normalization": as_dict(decision["human"], "human")["normalization"],
            }
            menu_size = len(menu_orders[0])
        else:
            human = as_dict(decision["human"], "human")
            menu_size = len(as_list(decision["available_actions"], "available_actions"))

        comparisons.append(
            {
                "decision_id": decision_id,
                "replay_index": decision["replay_index"],
                "source_replay_index": decision["source_replay_index"],
                "source_event_index": decision["source_event_index"],
                "source_action_type": decision["source_action_type"],
                "effective_action_type": as_str(
                    decision["effective_action_type"], "effective_action_type"
                ),
                "phase": decision.get("phase"),
                "forced": decision.get("forced"),
                "bucket_assignment": decision.get("bucket_assignment"),
                "menu_size": menu_size,
                "menu_order_consistent_across_models": all(
                    menu == menu_orders[0] for menu in menu_orders[1:]
                ),
                "human": human,
                "models": model_results,
            }
        )
    return comparisons

