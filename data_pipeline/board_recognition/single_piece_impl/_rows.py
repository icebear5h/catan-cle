"""The row record and every question asked about one placed piece."""

from __future__ import annotations

from data_pipeline.board_recognition.single_piece_impl._colors import (
    color_words,
    is_novel_color,
)
from data_pipeline.board_recognition.single_piece_impl._config import (
    FORWARD_QUERY,
    LOCATION_NOUN,
    NEGATIVE_KINDS,
    ROW_SCHEMA,
)
from data_pipeline.board_recognition.single_piece_impl._placements import (
    forward_answer,
    piece_words,
)
from data_pipeline.board_recognition.spatial_localization import _spatial_target
from data_pipeline.json_coerce import as_str
from data_pipeline.json_types import JsonDict


def _row(
    *,
    row_id: str,
    image_name: str,
    prompt: str,
    answer: str,
    task_type: str,
    category: str,
    polarity: str,
    metadata: JsonDict,
    spatial_target: JsonDict | None,
    schema: str = ROW_SCHEMA,
    grounding_stage: str = "single_piece",
    task_family: str = "single_piece_localization",
) -> JsonDict:
    row: JsonDict = {
        "schema": schema,
        "row_id": row_id,
        "curriculum_stage": "spatial_grounding",
        "grounding_stage": grounding_stage,
        "task_family": task_family,
        "task_type": task_type,
        "category": category,
        "polarity": polarity,
        "images": [image_name],
        "messages": [
            {"role": "user", "content": f"<image>\n{prompt}"},
            {"role": "assistant", "content": answer},
        ],
        **metadata,
    }
    if spatial_target is not None:
        row["spatial_targets"] = [spatial_target]
    return row


def rows_for_placement(
    *,
    state: JsonDict,
    regions: dict[str, JsonDict],
    controls: dict[str, JsonDict],
    token: str,
    piece: str,
    color: str,
    image_name: str,
    empty_tokens: dict[str, list[str]],
    distances: dict[str, int] | None = None,
) -> list[JsonDict]:
    entity_type = as_str(regions[token]["entity_type"])
    distances = distances or {}
    noun = LOCATION_NOUN[entity_type]
    novel = is_novel_color(color)
    metadata = {
        "split": state["split"],
        "state_id": state["sample_id"],
        "entity_type": entity_type,
        "target_token": token,
        "piece": piece,
        "color": color_words(color),
        "color_heldout": novel,
    }
    target = _spatial_target(regions[token], controls[token])
    stem = f"{state['sample_id']}_{token[1:-1]}_{piece}_{color}"
    category = "node.occupancy" if entity_type == "node" else "edge.owner"
    named_rows = [] if novel else [
        _row(
            row_id=f"{stem}_colored_piece_to_token",
            image_name=image_name,
            prompt=f"Which {noun} has the {forward_answer(color, piece)}?",
            answer=token,
            task_type="colored_piece_to_token",
            category="localization",
            polarity="token_return",
            metadata=metadata,
            spatial_target=target,
        ),
        _row(
            row_id=f"{stem}_occupancy_positive",
            image_name=image_name,
            prompt=f"{token} {FORWARD_QUERY[entity_type]}",
            answer=forward_answer(color, piece),
            task_type="occupancy_positive",
            category=category,
            polarity="positive",
            metadata=metadata,
            spatial_target=target,
        ),
    ]
    return [
        _row(
            row_id=f"{stem}_piece_to_token",
            image_name=image_name,
            prompt=f"Which {noun} has the {piece_words(piece)}?",
            answer=token,
            task_type="piece_to_token",
            category="localization",
            polarity="token_return",
            metadata=metadata,
            spatial_target=target,
        ),
        *named_rows,
        *(
            _row(
                row_id=f"{stem}_occupancy_negative_{kind}_{index}",
                image_name=image_name,
                prompt=f"{empty_token} {FORWARD_QUERY[entity_type]}",
                answer="empty",
                task_type=f"occupancy_negative_{kind}",
                category=category,
                polarity="hard_negative",
                metadata={
                    **metadata,
                    "queried_token": empty_token,
                    "negative_kind": kind,
                    "negative_distance": distances.get(empty_token, "far"),
                },
                spatial_target=None,
            )
            for kind in NEGATIVE_KINDS
            for index, empty_token in enumerate(empty_tokens.get(kind, ()))
        ),
    ]
