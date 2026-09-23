"""Deterministic witnesses over the existing board-recognition fixtures."""
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from data_pipeline.board_recognition import adjacent_pair_localization as pairs
from data_pipeline.board_recognition import (
    dataset,
    density_curriculum,
    query_schedule,
    replay_ms_swift,
)
from data_pipeline.board_recognition import inverse_grounding as inverse
from data_pipeline.board_recognition import production_curriculum as production
from data_pipeline.board_recognition import spatial_localization as spatial
from data_pipeline.board_recognition.replay_dataset import JsonDict, read_jsonl

FIXTURE = Path("artifacts/fixtures/board_recognition/curriculum_smoke")
REPLAY = Path("artifacts/generated/board_recognition/replay_v1")


def digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def witnesses(output: Path) -> dict[str, str]:
    states: Any = read_jsonl(REPLAY / "manifest.jsonl")
    result: dict[str, str] = {}
    inverse_rows: list[tuple[JsonDict, JsonDict]] = []
    pair_rows: Any = []
    for index, state in enumerate(states):
        contract = json.loads((REPLAY / state["contract_path"]).read_text())
        queries: Any = inverse.inverse_queries_for_state(state, contract, state_index=index)
        inverse_rows.extend(
            (inverse.inverse_ms_swift_row(query, image_name=Path(state["image_path"]).name),
             inverse.inverse_audit_row(state, query, state_index=index))
            for query in queries
        )
        if state["density_bin"] != "empty":
            continue
        for kind in pairs.TOUCHING_KINDS + pairs.FAR_KINDS:
            locations: Any = pairs.location_pairs(contract, kind)
            pair_rows.extend(pairs.sample_pairs(
                sample_id=state["sample_id"], pair_kind=kind, pairs=locations,
                colors=pairs.COLORS, count=3,
            ))
    result["inverse_rows"] = digest(inverse_rows)
    result["pair_placements"] = digest(pair_rows)
    plans: Any = query_schedule.build_split_query_plan(
        [state for state in states if state["split"] == "validation"],
        dataset_dir=REPLAY, split="validation", seed=381427,
    )
    result["query_schedule"] = digest(plans)
    states_by_id: Any = {state["sample_id"]: state for state in states}
    semantic_rows: Any = []
    for plan in plans:
        state = states_by_id[plan["state_id"]]
        for query in plan["queries"]:
            semantic_rows.append((
                replay_ms_swift.ms_swift_row(query, image_name=Path(state["image_path"]).name),
                replay_ms_swift.semantic_audit_row(plan, query, state=state),
            ))
    result["semantic_rows"] = digest(semantic_rows)
    semantic_root = REPLAY / "ms_swift_semantic_v1"
    records = density_curriculum.ordered_curriculum_records(
        read_jsonl(semantic_root / "train.jsonl"),
        read_jsonl(semantic_root / "audit/train.jsonl"),
    )
    result["density_records"] = digest(records)
    base: Any = production._base_records(REPLAY)
    supplement: Any = production._supplement_records(REPLAY / "spatial_robber_v1")
    by_stage: Any = {production.CURRICULUM_STAGES[0]: [
        record for record in supplement
        if record["audit"]["curriculum_stage"] == production.CURRICULUM_STAGES[0]
    ]}
    for stage in production.CURRICULUM_STAGES[1:]:
        by_stage[stage] = production.interleave_four_to_one(
            [record for record in base if record["audit"]["density_bin"] in production.BASE_DENSITIES_BY_STAGE[stage]],
            [record for record in supplement if record["audit"]["curriculum_stage"] == stage],
            stage=stage,
        )
    result["production_rows"] = digest(production._training_and_audit_rows(by_stage))
    fixture_states = read_jsonl(FIXTURE / "manifest.jsonl")
    fixture_state: Any = next(row for row in fixture_states if row["sample_id"] == "empty_setup_node_p000_base")
    contract = json.loads((FIXTURE / fixture_state["contract_path"]).read_text())
    regions = spatial.atlas_regions(contract, image_size=256, view_padding_factor=1.2)
    result["regions"] = digest(regions)
    result["relations"] = digest(spatial._deterministic_shuffle(
        spatial._relation_rows(fixture_states, repetitions=8, balance_polarity=True), "stage2"
    ))
    state = {**fixture_state, "image_size": [256, 256]}
    with Image.open(FIXTURE / fixture_state["image_path"]) as image:
        base_image = image.convert("RGB").resize((256, 256))
    marker_rows = spatial.marker_rows_for_board(
        state=state, contract=contract, base_image=base_image, output_images=output,
        board_index=0, view_padding_factor=1.2, entity_shaped=True,
    )
    probe_rows = spatial.neutral_probe_rows(
        state=state, contract=contract, base_image=base_image, output_images=output,
        view_padding_factor=1.2,
    )
    result["marker_rows"] = digest(marker_rows)
    result["probe_rows"] = digest(probe_rows)
    result["png_bytes"] = digest({
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.glob("*.png"))
    })
    loaded = dataset.BoardRecognitionStateDataset(FIXTURE, split="train")
    result["dataset_queries"] = digest([loaded[index]["queries"] for index in range(len(loaded))])
    return result
