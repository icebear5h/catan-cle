"""Inverse visual grounding and 2:1 forward/inverse ms-swift projection."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from jsonschema import Draft202012Validator

from data_pipeline.board_recognition.location_descriptions import (
    compact_node_signatures,
    inverse_location_descriptions,
)
from data_pipeline.board_recognition.replay_dataset import (
    ALL_SPLITS,
    PRIMARY_SPLITS,
    PROJECT_ROOT,
    file_sha256,
    read_jsonl,
    validate_replay_v1_dataset,
)
from data_pipeline.board_recognition.replay_ms_swift import (
    DEFAULT_OUTPUT_NAME as DEFAULT_FORWARD_PROJECTION,
    validate_ms_swift_row as validate_forward_row,
    validate_replay_v1_ms_swift_semantic,
)
from evals.catan_board_bench.tokens import atlas_tokens, semantic_recognition_token_inventory


JsonDict = dict[str, Any]
EXPORT_SCHEMA = "catan_board_recognition_ms_swift_bidirectional/v1"
AUDIT_SCHEMA = "catan_board_recognition_ms_swift_inverse_audit/v1"
CONTRACT_SCHEMA = "catan_board_recognition_inverse_grounding_contract/v1"
DEFAULT_OUTPUT_NAME = "ms_swift_bidirectional_v1"
INVERSE_ROWS_PER_STATE = 4
FORWARD_ROWS_PER_STATE = 8
MIXED_ROWS_PER_STATE = INVERSE_ROWS_PER_STATE + FORWARD_ROWS_PER_STATE
ENTITY_TYPES = ("tile", "node", "edge", "port")
ROW_KEYS = {"messages", "images"}
ATLAS_TOKENS = frozenset(atlas_tokens())
ATLAS_TOKEN_RE = re.compile(r"^<(?:T\d{2}|N\d{2}|E\d{2}_\d{2}|P\d{2})>$")


class InverseGroundingError(RuntimeError):
    """Raised when the inverse or mixed projection violates its contract."""


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _prepare_output_dir(path: Path, *, dataset_root: Path, overwrite: bool) -> None:
    protected = {
        dataset_root,
        dataset_root / "images",
        dataset_root / "contracts",
        dataset_root / "dense_labels",
        dataset_root / "qwen_sft",
        dataset_root / DEFAULT_FORWARD_PROJECTION,
    }
    if path in {candidate.resolve() for candidate in protected}:
        raise InverseGroundingError(f"refusing to replace protected source path: {path}")
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def inverse_grounding_contract() -> JsonDict:
    return {
        "schema": CONTRACT_SCHEMA,
        "task": "image plus board-local description to one canonical atlas token",
        "prompt_format": "<image>\\nWhere is <unambiguous board-local description>?",
        "answer_format": "exactly one of the 154 atlas tokens",
        "rows_per_state": INVERSE_ROWS_PER_STATE,
        "entity_types": list(ENTITY_TYPES),
        "resource_codes": {
            "WO": "wood",
            "B": "brick",
            "S": "sheep",
            "WH": "wheat",
            "O": "ore",
            "D": "desert",
        },
        "node_surface_forms": [
            "O10/WH5/S6",
            "ore 10/wheat 5/sheep 6",
            "O/WH/S when unique",
            "ore/wheat/sheep when unique",
        ],
        "ambiguity_policy": (
            "resource-only and resource-number descriptions are used only when unique "
            "within the current board; otherwise use a unique tile-anchored direction"
        ),
        "piece_policy": (
            "alternating complete node and edge atlas cycles prefer occupied targets "
            "and include visible color plus settlement, city, or road"
        ),
        "mixed_projection": {
            "forward_rows_per_state": FORWARD_ROWS_PER_STATE,
            "inverse_rows_per_state": INVERSE_ROWS_PER_STATE,
            "ratio": "2:1 forward:inverse",
            "order": "two forward rows followed by one inverse row, repeated four times",
        },
    }


def _words(value: str) -> str:
    return value.lower().replace("_", " ")


def _cyclic_eligible(rows: Sequence[JsonDict], eligible: set[str], index: int) -> JsonDict:
    if not eligible:
        raise InverseGroundingError("inverse description has no eligible targets")
    for offset in range(len(rows)):
        row = rows[(index + offset) % len(rows)]
        if row["token"] in eligible:
            return row
    raise InverseGroundingError("eligible inverse target is absent from canonical entity rows")


def _select_entity(
    rows: Sequence[JsonDict],
    eligible: set[str],
    *,
    state_index: int,
    occupied_field: str | None = None,
) -> JsonDict:
    prefer_occupied = (state_index // len(rows)) % 2 == 1
    if occupied_field is not None and prefer_occupied:
        occupied = [row for row in rows if row["token"] in eligible and row.get(occupied_field)]
        if occupied:
            return occupied[state_index % len(occupied)]
    return _cyclic_eligible(rows, eligible, state_index)


def _node_description(
    node: JsonDict,
    signatures: dict[str, JsonDict],
    natural: dict[str, str],
    *,
    state_index: int,
) -> tuple[str, str, str | None]:
    token = node["token"]
    signature = signatures[token]
    choices: list[tuple[str, str]] = []
    if signature["full_unique"]:
        choices.extend(
            [
                ("node_compact_resource_number", signature["full"]),
                ("node_words_resource_number", signature["full_words"]),
            ]
        )
    if signature["resources_unique"]:
        choices.extend(
            [
                ("node_compact_resources", signature["resources"]),
                ("node_words_resources", signature["resource_words"]),
            ]
        )
    if token in natural:
        choices.append(("node_directional_anchor", natural[token]))
    if not choices:
        raise InverseGroundingError(f"node has no unambiguous description: {token}")
    style, base = choices[state_index % len(choices)]
    building = node.get("building")
    color = node.get("color")
    qualifier = None
    if building is not None or color is not None:
        if not isinstance(building, str) or not isinstance(color, str):
            raise InverseGroundingError(f"node has partial building state: {token}")
        qualifier = f"{_words(color)} {_words(building)}"
        base = f"the {qualifier} at {base}"
        style = f"piece_{style}"
    return base, style, qualifier


def _edge_description(
    edge: JsonDict,
    signatures: dict[str, JsonDict],
    natural: dict[str, str],
    *,
    state_index: int,
) -> tuple[str, str, str | None]:
    token = edge["token"]
    endpoint_tokens = edge.get("node_tokens")
    if not isinstance(endpoint_tokens, list) or len(endpoint_tokens) != 2:
        raise InverseGroundingError(f"edge has invalid node tokens: {token}")
    left, right = (signatures[endpoint] for endpoint in endpoint_tokens)
    choices: list[tuple[str, str]] = []
    if left["full_unique"] and right["full_unique"]:
        choices.extend(
            [
                ("edge_compact_endpoints", f"the edge between {left['full']} and {right['full']}"),
                (
                    "edge_words_endpoints",
                    f"the edge between {left['full_words']} and {right['full_words']}",
                ),
            ]
        )
    if token in natural:
        choices.append(("edge_directional_anchor", natural[token]))
    if not choices:
        raise InverseGroundingError(f"edge has no unambiguous description: {token}")
    style, base = choices[state_index % len(choices)]
    color = edge.get("road_color")
    qualifier = None
    if color is not None:
        if not isinstance(color, str):
            raise InverseGroundingError(f"edge has invalid road color: {token}")
        qualifier = f"{_words(color)} road"
        if " between " in base:
            base = base.replace("the edge between ", f"the {qualifier} between ", 1)
        else:
            base = f"the {qualifier} on {base.removeprefix('the ')}"
        style = f"piece_{style}"
    return base, style, qualifier


def inverse_queries_for_state(
    state: JsonDict,
    contract: JsonDict,
    *,
    state_index: int,
) -> list[JsonDict]:
    """Create one deterministic inverse query for each atlas entity type."""

    natural = inverse_location_descriptions(contract)
    signatures = compact_node_signatures(contract)
    compact_nodes = {
        token for token, row in signatures.items() if row["full_unique"]
    }
    node_eligible = compact_nodes | {token for token in natural if token.startswith("<N")}
    edge_endpoint_eligible = {
        edge["token"]
        for edge in contract["edges"]
        if all(signatures[token]["full_unique"] for token in edge["node_tokens"])
    }
    edge_eligible = edge_endpoint_eligible | {
        token for token in natural if token.startswith("<E")
    }
    eligible_by_entity = {
        "tile": {token for token in natural if token.startswith("<T")},
        "node": node_eligible,
        "edge": edge_eligible,
        "port": {token for token in natural if token.startswith("<P")},
    }

    selected = {
        "tile": _select_entity(
            contract["tiles"], eligible_by_entity["tile"], state_index=state_index
        ),
        "node": _select_entity(
            contract["nodes"],
            eligible_by_entity["node"],
            state_index=state_index,
            occupied_field="building",
        ),
        "edge": _select_entity(
            contract["edges"],
            eligible_by_entity["edge"],
            state_index=state_index,
            occupied_field="road_color",
        ),
        "port": _select_entity(
            contract["ports"], eligible_by_entity["port"], state_index=state_index
        ),
    }

    queries = []
    for entity_type in ENTITY_TYPES:
        target = selected[entity_type]
        token = target["token"]
        if entity_type == "node":
            description, style, qualifier = _node_description(
                target, signatures, natural, state_index=state_index
            )
        elif entity_type == "edge":
            description, style, qualifier = _edge_description(
                target, signatures, natural, state_index=state_index
            )
        else:
            description = natural[token]
            style = f"{entity_type}_fact" if entity_type == "tile" else "port_directional_anchor"
            qualifier = None
        prompt = f"<image>\nWhere is {description}?"
        queries.append(
            {
                "query_id": f"{state['sample_id']}_inverse_{entity_type}",
                "entity_type": entity_type,
                "target_token": token,
                "description": description,
                "description_style": style,
                "visual_qualifier": qualifier,
                "prompt": prompt,
                "answer": token,
            }
        )
    return queries


def inverse_ms_swift_row(query: JsonDict, *, image_name: str) -> JsonDict:
    return {
        "messages": [
            {"role": "user", "content": query["prompt"]},
            {"role": "assistant", "content": query["answer"]},
        ],
        "images": [image_name],
    }


def inverse_audit_row(state: JsonDict, query: JsonDict, *, state_index: int) -> JsonDict:
    return {
        "schema": AUDIT_SCHEMA,
        "query_id": query["query_id"],
        "state_id": state["sample_id"],
        "state_index": state_index,
        "split": state["split"],
        "image_name": Path(state["image_path"]).name,
        "image_sha256": state["sha256"]["image"],
        "contract_path": state["contract_path"],
        "contract_sha256": state["sha256"]["contract"],
        "source_kind": state["source"]["kind"],
        "game_id": state["source"].get("game_id"),
        "trajectory_id": state["source"]["trajectory_id"],
        "density_bin": state["density_bin"],
        "building_count": state["building_count"],
        "road_count": state["road_count"],
        "piece_count": state["building_count"] + state["road_count"],
        "entity_type": query["entity_type"],
        "target_token": query["target_token"],
        "description": query["description"],
        "description_style": query["description_style"],
        "visual_qualifier": query["visual_qualifier"],
        "semantic_prompt": query["prompt"],
        "semantic_answer": query["answer"],
        "prompt_sha256": hashlib.sha256(query["prompt"].encode()).hexdigest(),
        "answer_sha256": hashlib.sha256(query["answer"].encode()).hexdigest(),
    }


def validate_inverse_ms_swift_row(row: JsonDict) -> None:
    if set(row) != ROW_KEYS:
        raise InverseGroundingError(f"inverse row keys must be {sorted(ROW_KEYS)}")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise InverseGroundingError("inverse row must have one user and one assistant turn")
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        raise InverseGroundingError("inverse message roles are invalid")
    prompt = messages[0].get("content")
    answer = messages[1].get("content")
    if not isinstance(prompt, str) or not prompt.startswith("<image>\nWhere is ") or not prompt.endswith("?"):
        raise InverseGroundingError("inverse prompt has invalid format")
    description = prompt.removeprefix("<image>\nWhere is ").removesuffix("?")
    if not description or "<" in description or ">" in description:
        raise InverseGroundingError("inverse prompt leaks an atlas token")
    if not isinstance(answer, str) or answer not in ATLAS_TOKENS or not ATLAS_TOKEN_RE.fullmatch(answer):
        raise InverseGroundingError("inverse answer must be exactly one canonical atlas token")
    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise InverseGroundingError("inverse row must reference exactly one image")
    image_name = images[0]
    if not isinstance(image_name, str) or Path(image_name).name != image_name:
        raise InverseGroundingError("inverse image must be a filename relative to image_root")


def _validate_schema_rows(schema_path: Path, rows: Sequence[JsonDict], *, label: str) -> int:
    validator = Draft202012Validator(json.loads(schema_path.read_text()))
    for index, row in enumerate(rows):
        errors = sorted(validator.iter_errors(row), key=lambda error: list(error.path))
        if errors:
            error = errors[0]
            path = ".".join(str(part) for part in error.path) or "<root>"
            raise InverseGroundingError(
                f"{label}[{index}] failed {schema_path.name} at {path}: {error.message}"
            )
    return len(rows)


def _forward_rows_by_state(
    annotations: Sequence[JsonDict], audits: Sequence[JsonDict]
) -> dict[str, list[tuple[JsonDict, JsonDict]]]:
    if len(annotations) != len(audits):
        raise InverseGroundingError("forward annotations and audits are not aligned")
    grouped: dict[str, list[tuple[JsonDict, JsonDict]]] = defaultdict(list)
    for annotation, audit in zip(annotations, audits, strict=True):
        grouped[audit["state_id"]].append((annotation, audit))
    if any(len(rows) != FORWARD_ROWS_PER_STATE for rows in grouped.values()):
        raise InverseGroundingError("forward projection is not eight rows per state")
    return dict(grouped)


def _mixed_rows_for_state(
    forward: Sequence[tuple[JsonDict, JsonDict]],
    inverse_rows: Sequence[JsonDict],
    inverse_audits: Sequence[JsonDict],
) -> tuple[list[JsonDict], list[JsonDict]]:
    if len(forward) != FORWARD_ROWS_PER_STATE or len(inverse_rows) != INVERSE_ROWS_PER_STATE:
        raise InverseGroundingError("cannot construct the fixed 2:1 mixed block")
    rows = []
    index = []
    for group in range(INVERSE_ROWS_PER_STATE):
        for annotation, audit in forward[group * 2 : group * 2 + 2]:
            rows.append(annotation)
            index.append(
                {
                    "row_kind": "forward",
                    "state_id": audit["state_id"],
                    "query_id": audit["query_id"],
                }
            )
        rows.append(inverse_rows[group])
        index.append(
            {
                "row_kind": "inverse",
                "state_id": inverse_audits[group]["state_id"],
                "query_id": inverse_audits[group]["query_id"],
            }
        )
    return rows, index


def export_replay_v1_ms_swift_bidirectional(
    dataset_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    forward_projection_dir: str | Path | None = None,
    overwrite: bool = False,
) -> JsonDict:
    dataset_root = Path(dataset_dir).resolve()
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (dataset_root / DEFAULT_OUTPUT_NAME).resolve()
    )
    forward_root = (
        Path(forward_projection_dir).resolve()
        if forward_projection_dir is not None
        else (dataset_root / DEFAULT_FORWARD_PROJECTION).resolve()
    )
    validate_replay_v1_dataset(dataset_root, rerender=False)
    forward_report = validate_replay_v1_ms_swift_semantic(
        dataset_root, export_dir=forward_root
    )
    _prepare_output_dir(output, dataset_root=dataset_root, overwrite=overwrite)
    (output / "audit").mkdir()
    (output / "mixed").mkdir()
    (output / "mixed_index").mkdir()

    manifest_path = dataset_root / "manifest.jsonl"
    manifest = read_jsonl(manifest_path)
    states_by_split = {
        split: [state for state in manifest if state["split"] == split]
        for split in ALL_SPLITS
    }
    forward_metadata_path = forward_root / "metadata.json"
    forward_metadata = json.loads(forward_metadata_path.read_text())
    files: dict[str, JsonDict] = {}
    row_counts: dict[str, int] = {}
    mixed_row_counts: dict[str, int] = {}
    style_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    qualified_counts: Counter[str] = Counter()

    for split in ALL_SPLITS:
        states = states_by_split[split]
        forward_annotations = read_jsonl(
            forward_root / forward_metadata["files"][split]["annotations"]
        )
        forward_audits = read_jsonl(
            forward_root / forward_metadata["files"][split]["audit"]
        )
        forward_by_state = _forward_rows_by_state(forward_annotations, forward_audits)
        inverse_rows: list[JsonDict] = []
        inverse_audits: list[JsonDict] = []
        mixed_rows: list[JsonDict] = []
        mixed_index: list[JsonDict] = []
        for state_index, state in enumerate(states):
            contract_path = dataset_root / state["contract_path"]
            if file_sha256(contract_path) != state["sha256"]["contract"]:
                raise InverseGroundingError(f"contract hash changed: {state['sample_id']}")
            contract = json.loads(contract_path.read_text())
            queries = inverse_queries_for_state(
                state, contract, state_index=state_index
            )
            state_rows = [
                inverse_ms_swift_row(query, image_name=Path(state["image_path"]).name)
                for query in queries
            ]
            state_audits = [
                inverse_audit_row(state, query, state_index=state_index)
                for query in queries
            ]
            forward = forward_by_state.get(state["sample_id"])
            if forward is None:
                raise InverseGroundingError(
                    f"forward projection omitted state: {state['sample_id']}"
                )
            state_mixed, state_index_rows = _mixed_rows_for_state(
                forward, state_rows, state_audits
            )
            inverse_rows.extend(state_rows)
            inverse_audits.extend(state_audits)
            mixed_rows.extend(state_mixed)
            mixed_index.extend(state_index_rows)
            for query in queries:
                style_counts[query["description_style"]] += 1
                target_counts[query["target_token"]] += 1
                if query["visual_qualifier"] is not None:
                    qualified_counts[query["visual_qualifier"]] += 1

        for index_value, row in enumerate(mixed_index):
            row["mixed_index"] = index_value
        annotation_path = output / f"{split}.jsonl"
        audit_path = output / "audit" / f"{split}.jsonl"
        mixed_path = output / "mixed" / f"{split}.jsonl"
        mixed_index_path = output / "mixed_index" / f"{split}.jsonl"
        _write_jsonl(annotation_path, inverse_rows)
        _write_jsonl(audit_path, inverse_audits)
        _write_jsonl(mixed_path, mixed_rows)
        _write_jsonl(mixed_index_path, mixed_index)
        row_counts[split] = len(inverse_rows)
        mixed_row_counts[split] = len(mixed_rows)
        files[split] = {
            "annotations": annotation_path.name,
            "annotations_sha256": file_sha256(annotation_path),
            "audit": str(audit_path.relative_to(output)),
            "audit_sha256": file_sha256(audit_path),
            "mixed_annotations": str(mixed_path.relative_to(output)),
            "mixed_annotations_sha256": file_sha256(mixed_path),
            "mixed_index": str(mixed_index_path.relative_to(output)),
            "mixed_index_sha256": file_sha256(mixed_index_path),
        }

    contract_payload = inverse_grounding_contract()
    inventory = semantic_recognition_token_inventory()
    _write_json(output / "inverse_grounding_contract.json", contract_payload)
    _write_json(output / "trainable_tokens.json", inventory)
    metadata = {
        "schema": EXPORT_SCHEMA,
        "source_dataset": str(dataset_root),
        "source_manifest_sha256": file_sha256(manifest_path),
        "source_metadata_sha256": file_sha256(dataset_root / "metadata.json"),
        "forward_projection": str(forward_root),
        "forward_projection_metadata_sha256": file_sha256(forward_metadata_path),
        "forward_projection_validation": forward_report,
        "image_root": str((dataset_root / "images").resolve()),
        "state_counts": {split: len(states_by_split[split]) for split in ALL_SPLITS},
        "inverse_rows_per_state": INVERSE_ROWS_PER_STATE,
        "mixed_rows_per_state": MIXED_ROWS_PER_STATE,
        "row_counts": row_counts,
        "mixed_row_counts": mixed_row_counts,
        "primary_inverse_row_count": sum(row_counts[split] for split in PRIMARY_SPLITS),
        "primary_mixed_row_count": sum(mixed_row_counts[split] for split in PRIMARY_SPLITS),
        "description_style_counts": dict(sorted(style_counts.items())),
        "target_token_counts": dict(sorted(target_counts.items())),
        "piece_qualifier_counts": dict(sorted(qualified_counts.items())),
        "inverse_grounding_contract": "inverse_grounding_contract.json",
        "inverse_grounding_contract_sha256": file_sha256(
            output / "inverse_grounding_contract.json"
        ),
        "trainable_token_inventory": "trainable_tokens.json",
        "trainable_token_inventory_sha256": file_sha256(output / "trainable_tokens.json"),
        "files": files,
    }
    _write_json(output / "metadata.json", metadata)
    report = validate_replay_v1_ms_swift_bidirectional(dataset_root, export_dir=output)
    report["output_dir"] = str(output)
    return report


def validate_replay_v1_ms_swift_bidirectional(
    dataset_dir: str | Path,
    *,
    export_dir: str | Path | None = None,
) -> JsonDict:
    dataset_root = Path(dataset_dir).resolve()
    output = (
        Path(export_dir).resolve()
        if export_dir is not None
        else (dataset_root / DEFAULT_OUTPUT_NAME).resolve()
    )
    metadata = json.loads((output / "metadata.json").read_text())
    if metadata.get("schema") != EXPORT_SCHEMA:
        raise InverseGroundingError("bidirectional export schema mismatch")
    manifest_path = dataset_root / "manifest.jsonl"
    if metadata["source_manifest_sha256"] != file_sha256(manifest_path):
        raise InverseGroundingError("source manifest changed after bidirectional export")
    if metadata["source_metadata_sha256"] != file_sha256(dataset_root / "metadata.json"):
        raise InverseGroundingError("source metadata changed after bidirectional export")
    if Path(metadata["image_root"]).resolve() != (dataset_root / "images").resolve():
        raise InverseGroundingError("bidirectional image_root is invalid")
    contract_path = output / metadata["inverse_grounding_contract"]
    if metadata["inverse_grounding_contract_sha256"] != file_sha256(contract_path):
        raise InverseGroundingError("inverse grounding contract hash changed")
    if json.loads(contract_path.read_text()) != inverse_grounding_contract():
        raise InverseGroundingError("inverse grounding contract content changed")
    inventory_path = output / metadata["trainable_token_inventory"]
    if metadata["trainable_token_inventory_sha256"] != file_sha256(inventory_path):
        raise InverseGroundingError("trainable token inventory hash changed")
    inventory = json.loads(inventory_path.read_text())
    if inventory != semantic_recognition_token_inventory():
        raise InverseGroundingError("bidirectional inventory is not the exact atlas inventory")

    forward_root = Path(metadata["forward_projection"]).resolve()
    forward_metadata_path = forward_root / "metadata.json"
    if metadata["forward_projection_metadata_sha256"] != file_sha256(
        forward_metadata_path
    ):
        raise InverseGroundingError("forward projection metadata changed")
    validate_replay_v1_ms_swift_semantic(dataset_root, export_dir=forward_root)
    forward_metadata = json.loads(forward_metadata_path.read_text())

    manifest = read_jsonl(manifest_path)
    states_by_split = {
        split: [state for state in manifest if state["split"] == split]
        for split in ALL_SPLITS
    }
    schema_root = PROJECT_ROOT / "data" / "curriculum" / "board_recognition" / "schemas"
    schema_counts: dict[str, int] = {}
    observed_counts: dict[str, int] = {}
    observed_mixed_counts: dict[str, int] = {}
    observed_styles: Counter[str] = Counter()
    observed_targets: Counter[str] = Counter()
    observed_qualifiers: Counter[str] = Counter()

    for split in ALL_SPLITS:
        file_metadata = metadata["files"][split]
        paths = {
            key: output / file_metadata[key]
            for key in ("annotations", "audit", "mixed_annotations", "mixed_index")
        }
        for key, path in paths.items():
            if file_sha256(path) != file_metadata[f"{key}_sha256"]:
                raise InverseGroundingError(f"{split} {key} hash changed")
        inverse_rows = read_jsonl(paths["annotations"])
        inverse_audits = read_jsonl(paths["audit"])
        mixed_rows = read_jsonl(paths["mixed_annotations"])
        mixed_index = read_jsonl(paths["mixed_index"])
        schema_counts[f"annotations.{split}"] = _validate_schema_rows(
            schema_root / "ms_swift_inverse_v1.schema.json",
            inverse_rows,
            label=f"annotations.{split}",
        )
        schema_counts[f"audit.{split}"] = _validate_schema_rows(
            schema_root / "ms_swift_inverse_audit_v1.schema.json",
            inverse_audits,
            label=f"audit.{split}",
        )
        expected_inverse_rows: list[JsonDict] = []
        expected_inverse_audits: list[JsonDict] = []
        expected_mixed_rows: list[JsonDict] = []
        expected_mixed_index: list[JsonDict] = []
        forward_annotations = read_jsonl(
            forward_root / forward_metadata["files"][split]["annotations"]
        )
        forward_audits = read_jsonl(
            forward_root / forward_metadata["files"][split]["audit"]
        )
        forward_by_state = _forward_rows_by_state(forward_annotations, forward_audits)
        for state_index, state in enumerate(states_by_split[split]):
            contract_file = dataset_root / state["contract_path"]
            if file_sha256(contract_file) != state["sha256"]["contract"]:
                raise InverseGroundingError(f"contract hash changed: {state['sample_id']}")
            contract = json.loads(contract_file.read_text())
            queries = inverse_queries_for_state(state, contract, state_index=state_index)
            state_rows = [
                inverse_ms_swift_row(query, image_name=Path(state["image_path"]).name)
                for query in queries
            ]
            state_audits = [
                inverse_audit_row(state, query, state_index=state_index)
                for query in queries
            ]
            state_mixed, state_mixed_index = _mixed_rows_for_state(
                forward_by_state[state["sample_id"]], state_rows, state_audits
            )
            expected_inverse_rows.extend(state_rows)
            expected_inverse_audits.extend(state_audits)
            expected_mixed_rows.extend(state_mixed)
            expected_mixed_index.extend(state_mixed_index)
        for index_value, row in enumerate(expected_mixed_index):
            row["mixed_index"] = index_value
        if inverse_rows != expected_inverse_rows or inverse_audits != expected_inverse_audits:
            raise InverseGroundingError(f"{split} inverse rows are not reproducible")
        if mixed_rows != expected_mixed_rows or mixed_index != expected_mixed_index:
            raise InverseGroundingError(f"{split} mixed rows are not reproducible")
        for row in inverse_rows:
            validate_inverse_ms_swift_row(row)
        for row, index_row in zip(mixed_rows, mixed_index, strict=True):
            if index_row["row_kind"] == "inverse":
                validate_inverse_ms_swift_row(row)
            else:
                validate_forward_row(row)
        for audit in inverse_audits:
            observed_styles[audit["description_style"]] += 1
            observed_targets[audit["target_token"]] += 1
            if audit["visual_qualifier"] is not None:
                observed_qualifiers[audit["visual_qualifier"]] += 1
        observed_counts[split] = len(inverse_rows)
        observed_mixed_counts[split] = len(mixed_rows)

    if observed_counts != metadata["row_counts"]:
        raise InverseGroundingError("inverse row counts are stale")
    if observed_mixed_counts != metadata["mixed_row_counts"]:
        raise InverseGroundingError("mixed row counts are stale")
    if dict(sorted(observed_styles.items())) != metadata["description_style_counts"]:
        raise InverseGroundingError("description style counts are stale")
    if dict(sorted(observed_targets.items())) != metadata["target_token_counts"]:
        raise InverseGroundingError("target token counts are stale")
    if dict(sorted(observed_qualifiers.items())) != metadata["piece_qualifier_counts"]:
        raise InverseGroundingError("piece qualifier counts are stale")
    expected_counts = {
        split: len(states_by_split[split]) * INVERSE_ROWS_PER_STATE
        for split in ALL_SPLITS
    }
    expected_mixed = {
        split: len(states_by_split[split]) * MIXED_ROWS_PER_STATE
        for split in ALL_SPLITS
    }
    if observed_counts != expected_counts or observed_mixed_counts != expected_mixed:
        raise InverseGroundingError("bidirectional split counts are not canonical")
    missing_train_targets = ATLAS_TOKENS - {
        audit["target_token"]
        for audit in read_jsonl(output / metadata["files"]["train"]["audit"])
    }
    if missing_train_targets:
        raise InverseGroundingError(
            f"train inverse rows omit atlas targets: {sorted(missing_train_targets)}"
        )
    return {
        "valid": True,
        "states": sum(len(states_by_split[split]) for split in PRIMARY_SPLITS),
        "diagnostic_states": len(states_by_split["color_diagnostic"]),
        "train_inverse_rows": observed_counts["train"],
        "train_mixed_rows": observed_mixed_counts["train"],
        "inverse_rows_per_state": INVERSE_ROWS_PER_STATE,
        "mixed_rows_per_state": MIXED_ROWS_PER_STATE,
        "forward_inverse_ratio": "2:1",
        "train_target_coverage": len(ATLAS_TOKENS),
        "piece_qualified_rows": sum(observed_qualifiers.values()),
        "schema_counts": schema_counts,
        "identity_sha256": _canonical_sha256(
            {
                "metadata": file_sha256(output / "metadata.json"),
                "contract": metadata["inverse_grounding_contract_sha256"],
                "inventory": metadata["trainable_token_inventory_sha256"],
            }
        ),
    }
