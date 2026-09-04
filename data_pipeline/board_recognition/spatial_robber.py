"""Deterministic spatial-grounding and robber-localization SFT supplement."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Sequence

from data_pipeline.board_recognition.replay_dataset import (
    ALL_SPLITS,
    file_sha256,
    read_jsonl,
    validate_replay_v1_dataset,
)
from evals.catan_board_bench.tokens import atlas_metadata, atlas_tokens


JsonDict = dict[str, Any]
EXPORT_SCHEMA = "catan_spatial_robber_supplement/v1"
AUDIT_SCHEMA = "catan_spatial_robber_audit/v1"
DEFAULT_OUTPUT_NAME = "spatial_robber_v1"
SPATIAL_ROWS_PER_EMPTY_STATE = 24
ROBBER_ROWS_PER_STATE = 3
SMOKE_ROWS_PER_STAGE = 8
CURRICULUM_STAGES = (
    "spatial_grounding",
    "clean_board_grounding",
    "pieces_and_colors",
    "real_game_distribution",
)
STAGE_BY_DENSITY = {
    "empty": "clean_board_grounding",
    "setup": "clean_board_grounding",
    "sparse": "pieces_and_colors",
    "dense": "real_game_distribution",
}
ATLAS_TOKENS = frozenset(atlas_tokens())


class SpatialRobberError(RuntimeError):
    """Raised when the supplement violates its deterministic contract."""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _stable_rank(*parts: Any) -> int:
    text = ":".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def _training_row(prompt: str, answer: str, image_name: str, stage: str) -> JsonDict:
    return {
        "curriculum_stage": stage,
        "images": [image_name],
        "messages": [
            {"role": "user", "content": f"<image>\n{prompt}"},
            {"role": "assistant", "content": answer},
        ],
    }


def _pixel(coord: Sequence[int]) -> tuple[float, float]:
    q, _cube_y, r = coord
    return math.sqrt(3) * (q + r / 2), 1.5 * r


def _atlas_geometry() -> tuple[
    dict[str, tuple[float, float]],
    dict[str, set[str]],
    dict[str, tuple[float, float]],
    dict[str, set[str]],
]:
    atlas = atlas_metadata()
    tile_positions = {row["token"]: _pixel(row["coord"]) for row in atlas["tiles"]}
    node_positions: dict[str, tuple[float, float]] = {}
    offsets = {
        "NORTH": (0.0, -1.0),
        "NORTHEAST": (math.sqrt(3) / 2, -0.5),
        "SOUTHEAST": (math.sqrt(3) / 2, 0.5),
        "SOUTH": (0.0, 1.0),
        "SOUTHWEST": (-math.sqrt(3) / 2, 0.5),
        "NORTHWEST": (-math.sqrt(3) / 2, -0.5),
    }
    for tile in atlas["tiles"]:
        center = tile_positions[tile["token"]]
        for direction, node_id in tile["nodes"].items():
            dx, dy = offsets[direction]
            node_positions.setdefault(f"<N{node_id:02d}>", (center[0] + dx, center[1] + dy))

    node_graph = {token: set() for token in node_positions}
    for edge in atlas["edges"]:
        left, right = (f"<N{node_id:02d}>" for node_id in edge["id"])
        node_graph[left].add(right)
        node_graph[right].add(left)

    tile_graph = {token: set() for token in tile_positions}
    tile_nodes = {row["token"]: set(row["nodes"].values()) for row in atlas["tiles"]}
    tokens = sorted(tile_positions)
    for index, left in enumerate(tokens):
        for right in tokens[index + 1 :]:
            if len(tile_nodes[left] & tile_nodes[right]) == 2:
                tile_graph[left].add(right)
                tile_graph[right].add(left)
    return node_positions, node_graph, tile_positions, tile_graph


def _graph_distance(graph: dict[str, set[str]], source: str, target: str) -> int:
    queue = deque([(source, 0)])
    seen = {source}
    while queue:
        current, distance = queue.popleft()
        if current == target:
            return distance
        for neighbor in sorted(graph[current]):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
    raise SpatialRobberError(f"disconnected atlas graph: {source}, {target}")


def _direction(
    left: str,
    right: str,
    positions: dict[str, tuple[float, float]],
) -> str:
    dx = positions[left][0] - positions[right][0]
    dy = positions[left][1] - positions[right][1]
    if abs(dx) > abs(dy):
        return "right of" if dx > 0 else "left of"
    return "below" if dy > 0 else "above"


def _opposite(relation: str) -> str:
    return {
        "above": "below",
        "below": "above",
        "left of": "right of",
        "right of": "left of",
    }[relation]


def _relation_bank(
    entity: str,
    positions: dict[str, tuple[float, float]],
    graph: dict[str, set[str]],
) -> dict[str, list[JsonDict]]:
    noun = "node" if entity == "node" else "tile"
    bank: dict[str, list[JsonDict]] = defaultdict(list)
    positive_pairs = sorted(
        (left, right) for left, neighbors in graph.items() for right in neighbors if left < right
    )
    tokens = sorted(graph)
    hard_negative_pairs = [
        (left, right)
        for index, left in enumerate(tokens)
        for right in tokens[index + 1 :]
        if _graph_distance(graph, left, right) == 2
    ]

    for left, right in positive_pairs:
        for first, second in ((left, right), (right, left)):
            relation = _direction(first, second, positions)
            bank[f"{entity}_direction_yes"].append(
                {
                    "prompt": f"Is {first} {relation} {second}?",
                    "answer": "yes",
                    "relationship": relation.replace(" ", "_"),
                    "polarity": "positive",
                    "tokens": [first, second],
                }
            )
            bank[f"{entity}_direction_no"].append(
                {
                    "prompt": f"Is {first} {_opposite(relation)} {second}?",
                    "answer": "no",
                    "relationship": _opposite(relation).replace(" ", "_"),
                    "polarity": "hard_negative",
                    "tokens": [first, second],
                }
            )
            bank[f"{entity}_direction_token"].append(
                {
                    "prompt": (f"Which {noun} is {relation} the other: {first} or {second}?"),
                    "answer": first,
                    "relationship": relation.replace(" ", "_"),
                    "polarity": "token_return",
                    "tokens": [first, second],
                }
            )

        bank[f"{entity}_adjacent_yes"].append(
            {
                "prompt": f"Are {left} and {right} adjacent {noun}s?",
                "answer": "yes",
                "relationship": "adjacent",
                "polarity": "positive",
                "tokens": [left, right],
            }
        )
        if entity == "node":
            bank["node_connected_yes"].append(
                {
                    "prompt": f"Are {left} and {right} connected by an edge?",
                    "answer": "yes",
                    "relationship": "connected",
                    "polarity": "positive",
                    "tokens": [left, right],
                }
            )

    for left, right in hard_negative_pairs:
        bank[f"{entity}_adjacent_no"].append(
            {
                "prompt": f"Are {left} and {right} adjacent {noun}s?",
                "answer": "no",
                "relationship": "adjacent",
                "polarity": "hard_negative",
                "tokens": [left, right],
            }
        )
        if entity == "node":
            bank["node_connected_no"].append(
                {
                    "prompt": f"Are {left} and {right} connected by an edge?",
                    "answer": "no",
                    "relationship": "connected",
                    "polarity": "hard_negative",
                    "tokens": [left, right],
                }
            )
    return dict(bank)


def spatial_query_bank() -> dict[str, list[JsonDict]]:
    """Return fixed local-direction and topology pools for the canonical board."""

    node_positions, node_graph, tile_positions, tile_graph = _atlas_geometry()
    bank = _relation_bank("node", node_positions, node_graph)
    bank.update(_relation_bank("tile", tile_positions, tile_graph))
    expected = {
        "node_direction_yes",
        "node_direction_no",
        "node_direction_token",
        "node_adjacent_yes",
        "node_adjacent_no",
        "node_connected_yes",
        "node_connected_no",
        "tile_direction_yes",
        "tile_direction_no",
        "tile_direction_token",
        "tile_adjacent_yes",
        "tile_adjacent_no",
    }
    if set(bank) != expected or any(not rows for rows in bank.values()):
        raise SpatialRobberError("spatial query bank is incomplete")
    return bank


def spatial_queries_for_state(state: JsonDict, *, state_index: int) -> list[JsonDict]:
    """Select two examples from each of 12 balanced spatial pools."""

    if state["density_bin"] != "empty":
        raise SpatialRobberError("pure spatial queries require an empty board state")
    queries = []
    for family, pool in sorted(spatial_query_bank().items()):
        start = _stable_rank(state["sample_id"], state_index, family) % len(pool)
        for offset in range(2):
            source = pool[(start + offset) % len(pool)]
            queries.append(
                {
                    **source,
                    "query_id": f"{state['sample_id']}_spatial_{family}_{offset}",
                    "task_family": "spatial_grounding",
                    "task_type": family,
                    "curriculum_stage": "spatial_grounding",
                }
            )
    if len(queries) != SPATIAL_ROWS_PER_EMPTY_STATE:
        raise SpatialRobberError("spatial selection has the wrong row count")
    return queries


def robber_queries_for_state(
    state: JsonDict,
    contract: JsonDict,
    *,
    state_index: int,
) -> list[JsonDict]:
    """Return positive, hard-negative, and token-return robber supervision."""

    robber_tiles = [tile["token"] for tile in contract["tiles"] if tile["has_robber"]]
    if len(robber_tiles) != 1:
        raise SpatialRobberError(f"state must have exactly one robber: {state['sample_id']}")
    robber = robber_tiles[0]
    non_robber = sorted(tile["token"] for tile in contract["tiles"] if not tile["has_robber"])
    negative = non_robber[_stable_rank(state["sample_id"], state_index, "robber") % len(non_robber)]
    stage = STAGE_BY_DENSITY[state["density_bin"]]
    return [
        {
            "query_id": f"{state['sample_id']}_robber_positive",
            "task_family": "robber",
            "task_type": "robber_presence_positive",
            "prompt": f"Is the robber on {robber}?",
            "answer": "yes",
            "relationship": "presence",
            "polarity": "positive",
            "tokens": [robber],
            "target_token": robber,
            "curriculum_stage": stage,
        },
        {
            "query_id": f"{state['sample_id']}_robber_negative",
            "task_family": "robber",
            "task_type": "robber_presence_negative",
            "prompt": f"Is the robber on {negative}?",
            "answer": "no",
            "relationship": "presence",
            "polarity": "hard_negative",
            "tokens": [negative],
            "target_token": robber,
            "curriculum_stage": stage,
        },
        {
            "query_id": f"{state['sample_id']}_robber_localize",
            "task_family": "robber",
            "task_type": "robber_token_return",
            "prompt": "Where is the robber? Answer with one tile token.",
            "answer": robber,
            "relationship": "localization",
            "polarity": "token_return",
            "tokens": [robber],
            "target_token": robber,
            "curriculum_stage": stage,
        },
    ]


def _audit_row(state: JsonDict, query: JsonDict, *, image_name: str) -> JsonDict:
    return {
        "schema": AUDIT_SCHEMA,
        "query_id": query["query_id"],
        "state_id": state["sample_id"],
        "split": state["split"],
        "density_bin": state["density_bin"],
        "image_name": image_name,
        "task_family": query["task_family"],
        "task_type": query["task_type"],
        "curriculum_stage": query["curriculum_stage"],
        "relationship": query["relationship"],
        "polarity": query["polarity"],
        "tokens": query["tokens"],
        "target_token": query.get("target_token"),
        "prompt": query["prompt"],
        "answer": query["answer"],
    }


def _diverse_indices(
    audits: Sequence[JsonDict],
    *,
    count: int,
    group_fields: Sequence[str],
) -> list[int]:
    buckets: dict[tuple[str, ...], deque[int]] = defaultdict(deque)
    for index, audit in enumerate(audits):
        key = tuple(str(audit.get(field, "")) for field in group_fields)
        buckets[key].append(index)
    selected: list[int] = []
    ordered_keys = sorted(buckets)
    while len(selected) < count:
        progressed = False
        for key in ordered_keys:
            if buckets[key]:
                selected.append(buckets[key].popleft())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise SpatialRobberError(f"cannot select {count} diverse smoke rows")
    return selected


def build_curriculum_smoke_rows(
    dataset_root: Path,
    supplement_rows: Sequence[JsonDict],
    supplement_audits: Sequence[JsonDict],
) -> list[JsonDict]:
    """Compose 32 rows so each optimizer step sees one ordered curriculum stage."""

    if len(supplement_rows) != len(supplement_audits):
        raise SpatialRobberError("supplement rows and audits are not aligned")
    base_root = dataset_root / "ms_swift_bidirectional_v1"
    base_rows = read_jsonl(base_root / "mixed" / "train.jsonl")
    base_index = read_jsonl(base_root / "mixed_index" / "train.jsonl")
    if len(base_rows) != len(base_index):
        raise SpatialRobberError("base mixed rows and index are not aligned")
    density_by_state = {
        state["sample_id"]: state["density_bin"]
        for state in read_jsonl(dataset_root / "manifest.jsonl")
        if state["split"] == "train"
    }
    base_audits = [
        {
            **index,
            "density_bin": density_by_state[index["state_id"]],
        }
        for index in base_index
    ]

    result: list[JsonDict] = []
    for stage in CURRICULUM_STAGES:
        supplement_candidates = [
            (row, audit)
            for row, audit in zip(supplement_rows, supplement_audits, strict=True)
            if audit["curriculum_stage"] == stage
        ]
        if stage == "spatial_grounding":
            chosen = _diverse_indices(
                [audit for _, audit in supplement_candidates],
                count=SMOKE_ROWS_PER_STAGE,
                group_fields=("task_type",),
            )
            result.extend(supplement_candidates[index][0] for index in chosen)
            continue

        allowed_density = {
            "clean_board_grounding": {"empty", "setup"},
            "pieces_and_colors": {"sparse"},
            "real_game_distribution": {"dense"},
        }[stage]
        base_candidates = [
            (row, audit)
            for row, audit in zip(base_rows, base_audits, strict=True)
            if audit["density_bin"] in allowed_density
        ]
        base_chosen = _diverse_indices(
            [audit for _, audit in base_candidates],
            count=SMOKE_ROWS_PER_STAGE // 2,
            group_fields=("density_bin", "row_kind"),
        )
        supplement_chosen = _diverse_indices(
            [audit for _, audit in supplement_candidates],
            count=SMOKE_ROWS_PER_STAGE // 2,
            group_fields=("task_type",),
        )
        selected_base = []
        for index in base_chosen:
            row = dict(base_candidates[index][0])
            row["curriculum_stage"] = stage
            selected_base.append(row)
        selected_supplement = [supplement_candidates[index][0] for index in supplement_chosen]
        for base_row, supplement_row in zip(selected_base, selected_supplement, strict=True):
            result.extend((base_row, supplement_row))

    expected_rows = len(CURRICULUM_STAGES) * SMOKE_ROWS_PER_STAGE
    if len(result) != expected_rows:
        raise SpatialRobberError(
            f"curriculum smoke has {len(result)} rows; expected {expected_rows}"
        )
    return result


def export_spatial_robber_supplement(
    dataset_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> JsonDict:
    dataset_root = Path(dataset_dir).resolve()
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (dataset_root / DEFAULT_OUTPUT_NAME).resolve()
    )
    validate_replay_v1_dataset(dataset_root, rerender=False)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        if output.parent != dataset_root:
            raise SpatialRobberError("refusing to overwrite output outside the dataset root")
        shutil.rmtree(output)
    (output / "audit").mkdir(parents=True)

    manifest_path = dataset_root / "manifest.jsonl"
    states = read_jsonl(manifest_path)
    files: dict[str, JsonDict] = {}
    split_counts: dict[str, int] = {}
    split_task_counts: dict[str, dict[str, int]] = {}
    split_stage_counts: dict[str, dict[str, int]] = {}
    split_relationship_counts: dict[str, dict[str, int]] = {}

    for split in ALL_SPLITS:
        split_states = [state for state in states if state["split"] == split]
        pairs_by_stage: dict[str, list[tuple[JsonDict, JsonDict]]] = {
            stage: [] for stage in CURRICULUM_STAGES
        }
        empty_index = 0
        for state_index, state in enumerate(split_states):
            contract_path = dataset_root / state["contract_path"]
            if file_sha256(contract_path) != state["sha256"]["contract"]:
                raise SpatialRobberError(f"contract hash changed: {state['sample_id']}")
            contract = json.loads(contract_path.read_text())
            image_name = Path(state["image_path"]).name
            queries = robber_queries_for_state(state, contract, state_index=state_index)
            if state["density_bin"] == "empty":
                queries.extend(spatial_queries_for_state(state, state_index=empty_index))
                empty_index += 1
            for query in queries:
                row = _training_row(
                    query["prompt"], query["answer"], image_name, query["curriculum_stage"]
                )
                audit = _audit_row(state, query, image_name=image_name)
                pairs_by_stage[query["curriculum_stage"]].append((row, audit))

        rows = [pair[0] for stage in CURRICULUM_STAGES for pair in pairs_by_stage[stage]]
        audits = [pair[1] for stage in CURRICULUM_STAGES for pair in pairs_by_stage[stage]]
        annotation_path = output / f"{split}.jsonl"
        audit_path = output / "audit" / f"{split}.jsonl"
        _write_jsonl(annotation_path, rows)
        _write_jsonl(audit_path, audits)
        split_counts[split] = len(rows)
        split_task_counts[split] = dict(
            sorted(Counter(row["task_family"] for row in audits).items())
        )
        split_stage_counts[split] = dict(
            sorted(Counter(row["curriculum_stage"] for row in audits).items())
        )
        split_relationship_counts[split] = dict(
            sorted(Counter(row["relationship"] for row in audits).items())
        )
        files[split] = {
            "annotations": annotation_path.name,
            "annotations_sha256": file_sha256(annotation_path),
            "audit": str(audit_path.relative_to(output)),
            "audit_sha256": file_sha256(audit_path),
        }

    train_rows = read_jsonl(output / files["train"]["annotations"])
    train_audits = read_jsonl(output / files["train"]["audit"])
    smoke_rows = build_curriculum_smoke_rows(dataset_root, train_rows, train_audits)
    smoke_path = output / "curriculum_smoke_32.jsonl"
    _write_jsonl(smoke_path, smoke_rows)

    metadata = {
        "schema": EXPORT_SCHEMA,
        "source_dataset": str(dataset_root),
        "source_manifest_sha256": file_sha256(manifest_path),
        "image_root": str((dataset_root / "images").resolve()),
        "spatial_rows_per_empty_state": SPATIAL_ROWS_PER_EMPTY_STATE,
        "robber_rows_per_state": ROBBER_ROWS_PER_STATE,
        "split_counts": split_counts,
        "task_counts": split_task_counts,
        "stage_counts": split_stage_counts,
        "relationship_counts": split_relationship_counts,
        "curriculum_smoke": {
            "annotations": smoke_path.name,
            "annotations_sha256": file_sha256(smoke_path),
            "rows": len(smoke_rows),
            "rows_per_stage": SMOKE_ROWS_PER_STAGE,
            "optimizer_steps_at_gradient_accumulation_8": len(smoke_rows) // 8,
        },
        "files": files,
    }
    _write_json(output / "metadata.json", metadata)
    return validate_spatial_robber_supplement(dataset_root, output_dir=output)


def validate_spatial_robber_supplement(
    dataset_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> JsonDict:
    dataset_root = Path(dataset_dir).resolve()
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (dataset_root / DEFAULT_OUTPUT_NAME).resolve()
    )
    metadata = json.loads((output / "metadata.json").read_text())
    if metadata.get("schema") != EXPORT_SCHEMA:
        raise SpatialRobberError("supplement metadata schema mismatch")
    manifest_path = dataset_root / "manifest.jsonl"
    if metadata["source_manifest_sha256"] != file_sha256(manifest_path):
        raise SpatialRobberError("source manifest changed")
    states = read_jsonl(manifest_path)
    observed_total = 0
    train_binary = Counter()
    train_robber_types = Counter()
    train_relationships = Counter()
    train_spatial_tokens: set[str] = set()
    smoke_metadata = metadata.get("curriculum_smoke", {})
    smoke_path = output / str(smoke_metadata.get("annotations", ""))
    if not smoke_path.is_file() or file_sha256(smoke_path) != smoke_metadata.get(
        "annotations_sha256"
    ):
        raise SpatialRobberError("curriculum smoke file is missing or changed")
    smoke_rows = read_jsonl(smoke_path)
    smoke_stage_counts = Counter(row.get("curriculum_stage") for row in smoke_rows)
    if list(smoke_stage_counts) != list(CURRICULUM_STAGES) or set(smoke_stage_counts.values()) != {
        SMOKE_ROWS_PER_STAGE
    }:
        raise SpatialRobberError("curriculum smoke does not contain eight ordered rows per stage")

    for split in ALL_SPLITS:
        file_metadata = metadata["files"][split]
        annotation_path = output / file_metadata["annotations"]
        audit_path = output / file_metadata["audit"]
        if file_sha256(annotation_path) != file_metadata["annotations_sha256"]:
            raise SpatialRobberError(f"{split} annotation hash changed")
        if file_sha256(audit_path) != file_metadata["audit_sha256"]:
            raise SpatialRobberError(f"{split} audit hash changed")
        rows = read_jsonl(annotation_path)
        audits = read_jsonl(audit_path)
        if len(rows) != len(audits) or len(rows) != metadata["split_counts"][split]:
            raise SpatialRobberError(f"{split} rows and audits are misaligned")
        split_states = [state for state in states if state["split"] == split]
        empty_states = sum(state["density_bin"] == "empty" for state in split_states)
        expected = (
            len(split_states) * ROBBER_ROWS_PER_STATE + empty_states * SPATIAL_ROWS_PER_EMPTY_STATE
        )
        if len(rows) != expected:
            raise SpatialRobberError(f"{split} has {len(rows)} rows; expected {expected}")
        previous_stage = -1
        query_ids = set()
        for row, audit in zip(rows, audits, strict=True):
            stage = row.get("curriculum_stage")
            if stage != audit["curriculum_stage"] or stage not in CURRICULUM_STAGES:
                raise SpatialRobberError(f"{split} has invalid curriculum stage")
            stage_index = CURRICULUM_STAGES.index(stage)
            if stage_index < previous_stage:
                raise SpatialRobberError(f"{split} curriculum order regressed")
            previous_stage = stage_index
            if audit["query_id"] in query_ids:
                raise SpatialRobberError(f"{split} has duplicate query IDs")
            query_ids.add(audit["query_id"])
            messages = row.get("messages")
            images = row.get("images")
            if not isinstance(messages, list) or len(messages) != 2:
                raise SpatialRobberError(f"{split} row has invalid messages")
            if not isinstance(images, list) or len(images) != 1:
                raise SpatialRobberError(f"{split} row has invalid images")
            if not (dataset_root / "images" / images[0]).is_file():
                raise SpatialRobberError(f"{split} row image is missing")
            prompt = messages[0].get("content")
            answer = messages[1].get("content")
            if prompt != f"<image>\n{audit['prompt']}" or answer != audit["answer"]:
                raise SpatialRobberError(f"{split} training and audit text disagree")
            if answer.startswith("<") and answer not in ATLAS_TOKENS:
                raise SpatialRobberError(f"{split} answer has an unknown atlas token")
            if split == "train" and audit["task_family"] == "spatial_grounding":
                train_relationships[audit["relationship"]] += 1
                train_spatial_tokens.update(audit["tokens"])
                if answer in {"yes", "no"}:
                    train_binary[answer] += 1
            if split == "train" and audit["task_family"] == "robber":
                train_robber_types[audit["task_type"]] += 1
        observed_total += len(rows)

    if train_binary["yes"] != train_binary["no"] or not train_binary["yes"]:
        raise SpatialRobberError("train spatial yes/no rows are not balanced")
    train_state_count = sum(state["split"] == "train" for state in states)
    expected_robber_types = {
        "robber_presence_positive": train_state_count,
        "robber_presence_negative": train_state_count,
        "robber_token_return": train_state_count,
    }
    if dict(train_robber_types) != expected_robber_types:
        raise SpatialRobberError("train robber task types are not exhaustive")
    required_relationships = {"above", "below", "left_of", "right_of", "adjacent", "connected"}
    if not required_relationships.issubset(train_relationships):
        raise SpatialRobberError("train spatial rows omit required relationships")
    expected_spatial_tokens = {
        *(f"<N{index:02d}>" for index in range(54)),
        *(f"<T{index:02d}>" for index in range(19)),
    }
    if train_spatial_tokens != expected_spatial_tokens:
        raise SpatialRobberError("train spatial rows do not cover every node and tile token")
    return {
        "valid": True,
        "output_dir": str(output),
        "rows": observed_total,
        "train_rows": metadata["split_counts"]["train"],
        "train_task_counts": metadata["task_counts"]["train"],
        "train_stage_counts": metadata["stage_counts"]["train"],
        "train_relationship_counts": metadata["relationship_counts"]["train"],
        "train_spatial_binary_counts": dict(train_binary),
        "train_spatial_token_coverage": {
            "nodes": sum(token.startswith("<N") for token in train_spatial_tokens),
            "tiles": sum(token.startswith("<T") for token in train_spatial_tokens),
        },
        "train_robber_type_counts": dict(train_robber_types),
        "curriculum_smoke": {
            "rows": len(smoke_rows),
            "stage_counts": dict(smoke_stage_counts),
            "optimizer_steps_at_gradient_accumulation_8": len(smoke_rows) // 8,
        },
    }
