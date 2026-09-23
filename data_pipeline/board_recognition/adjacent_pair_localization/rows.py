"""Piece metadata and forward/inverse supervision for adjacent pairs."""

from __future__ import annotations

from typing import Sequence, TypedDict

from data_pipeline.board_recognition import adjacent_pair_localization as api
from data_pipeline.board_recognition.replay_dataset import JsonDict
from data_pipeline.json_coerce import as_str


class _RowContext(TypedDict):
    image_name: str
    metadata: JsonDict
    spatial_target: JsonDict
    schema: str
    grounding_stage: str
    task_family: str


def _piece_metadata(state: JsonDict, piece: api.Placement, partner: api.Placement, pair_kind: str) -> JsonDict:
    token, kind, color = piece
    partner_token, partner_kind, partner_color = partner
    return {
        "split": state["split"],
        "state_id": state["sample_id"],
        "entity_type": api.entity_of(token),
        "target_token": token,
        "piece": kind,
        "color": api.color_words(color),
        "color_heldout": api.is_novel_color(color),
        "pair_kind": pair_kind,
        "partner_token": partner_token,
        "partner_piece": partner_kind,
        "partner_color": api.color_words(partner_color),
        "partner_distance": "far" if api.is_far_kind(pair_kind) else 1,
        "same_color": color == partner_color,
    }


def rows_for_pair(
    *,
    state: JsonDict,
    regions: dict[str, JsonDict],
    controls: dict[str, JsonDict],
    first: api.Placement,
    second: api.Placement,
    pair_kind: str,
    image_name: str,
    empties: Sequence[JsonDict],
) -> list[JsonDict]:
    stem = f"{state['sample_id']}_{pair_kind}_{first[0][1:-1]}_{first[1]}_{first[2]}_{second[0][1:-1]}_{second[1]}_{second[2]}"
    rows: list[JsonDict] = []
    for piece, partner in ((first, second), (second, first)):
        token, kind, color = piece
        entity_type = api.entity_of(token)
        noun = api.LOCATION_NOUN[entity_type]
        category = "node.occupancy" if entity_type == "node" else "edge.owner"
        metadata = api._piece_metadata(state, piece, partner, pair_kind)
        target = api._spatial_target(regions[token], controls[token])
        piece_stem = f"{stem}_{token[1:-1]}"
        common: _RowContext = {
            "image_name": image_name,
            "metadata": metadata,
            "spatial_target": target,
            "schema": api.ROW_SCHEMA,
            "grounding_stage": api.GROUNDING_STAGE,
            "task_family": api.TASK_FAMILY,
        }
        if not api.is_novel_color(color):
            rows.append(
                api._row(
                    row_id=f"{piece_stem}_occupancy_positive",
                    prompt=f"{token} {api.FORWARD_QUERY[entity_type]}",
                    answer=api.forward_answer(color, kind),
                    task_type="occupancy_positive", category=category, polarity="positive", **common,
                )
            )
            rows.append(
                api._row(
                    row_id=f"{piece_stem}_colored_piece_to_token",
                    prompt=f"Which {noun} has the {api.forward_answer(color, kind)}?",
                    answer=token,
                    task_type="colored_piece_to_token", category="localization", polarity="token_return", **common,
                )
            )
        if kind != partner[1]:
            rows.append(
                api._row(
                    row_id=f"{piece_stem}_piece_to_token",
                    prompt=f"Which {noun} has the {api.piece_words(kind)}?",
                    answer=token,
                    task_type="piece_to_token", category="localization", polarity="token_return", **common,
                )
            )
    for index, empty in enumerate(empties):
        token = as_str(empty["token"])
        entity_type = api.entity_of(token)
        anchor = first if empty["anchor"] == first[0] else second
        partner = second if anchor is first else first
        rows.append(
            api._row(
                row_id=f"{stem}_occupancy_negative_{empty['kind']}_{index}",
                image_name=image_name,
                prompt=f"{token} {api.FORWARD_QUERY[entity_type]}",
                answer="empty",
                task_type=f"occupancy_negative_{empty['kind']}",
                category="node.occupancy" if entity_type == "node" else "edge.owner",
                polarity="hard_negative",
                metadata={
                    **api._piece_metadata(state, anchor, partner, pair_kind),
                    "queried_token": token,
                    "negative_kind": empty["kind"],
                    "negative_distance": empty["negative_distance"],
                },
                spatial_target=None,
                schema=api.ROW_SCHEMA, grounding_stage=api.GROUNDING_STAGE, task_family=api.TASK_FAMILY,
            )
        )
    return rows
