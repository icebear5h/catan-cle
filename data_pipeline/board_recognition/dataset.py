"""PyTorch state batches for direct Catan board-slot classification."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data_pipeline.board_recognition.query_schedule import build_split_query_plan
from data_pipeline.board_recognition.replay_dataset import DATASET_SCHEMA


JsonDict = dict[str, Any]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENTITY_TYPES = ("tile", "node", "edge", "port")
ENTITY_TYPE_IDS = {name: index for index, name in enumerate(ENTITY_TYPES)}
COLLECTION_BY_ENTITY = {
    "tile": "tiles",
    "node": "nodes",
    "edge": "edges",
    "port": "ports",
}
CLASS_VOCAB_KEY = {
    ("tile", "resource"): "tile_resource",
    ("tile", "number"): "tile_number",
    ("tile", "robber"): "tile_robber",
    ("node", "occupancy"): "node_occupancy",
    ("edge", "owner"): "edge_owner",
    ("port", "port_type"): "port_type",
}


def read_jsonl(path: Path) -> list[JsonDict]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def normalize_class_name(attribute: str, value: Any) -> str:
    if attribute == "number":
        return "NONE" if value is None else str(value)
    if attribute == "robber":
        return "PRESENT" if value else "ABSENT"
    return str(value)


def class_vocabulary(spec: JsonDict, entity_type: str, attribute: str) -> tuple[str, ...]:
    key = CLASS_VOCAB_KEY[(entity_type, attribute)]
    return tuple(
        normalize_class_name(attribute, value) for value in spec["class_vocabularies"][key]
    )


def attribute_ids(spec: JsonDict) -> dict[tuple[str, str], int]:
    identifiers = {}
    for entity_type in ENTITY_TYPES:
        for attribute in spec["sampling"]["attributes_by_entity_type"][entity_type]:
            identifiers[(entity_type, attribute)] = len(identifiers)
    return identifiers


def sample_state_queries(
    dense_labels: JsonDict,
    *,
    spec: JsonDict,
    queries_per_state: int,
    seed: int,
    epoch: int = 0,
    forced_query: JsonDict | None = None,
    sampling_key: str | None = None,
    sampling_dense_labels: JsonDict | None = None,
) -> list[JsonDict]:
    """Sample an exactly entity-balanced query set from one dense state."""

    if queries_per_state <= 0 or queries_per_state % len(ENTITY_TYPES):
        raise ValueError("queries_per_state must be positive and divisible by four")
    state_id = dense_labels["sample_id"]
    query_sampling_key = sampling_key or state_id
    entities = dense_labels["entities"]
    sampling_entities = (sampling_dense_labels or dense_labels)["entities"]
    attr_ids = attribute_ids(spec)
    state_offset = int(hashlib.sha256(query_sampling_key.encode()).hexdigest()[:16], 16)
    occurrences = Counter()
    forced_entity_type = (forced_query or {}).get("entity_type")

    queries = []
    for query_index in range(queries_per_state):
        entity_type = ENTITY_TYPES[query_index % len(ENTITY_TYPES)]
        occurrence = occurrences[entity_type]
        occurrences[entity_type] += 1
        attributes = spec["sampling"]["attributes_by_entity_type"][entity_type]
        rows = entities[COLLECTION_BY_ENTITY[entity_type]]
        sampling_rows = sampling_entities[COLLECTION_BY_ENTITY[entity_type]]
        if entity_type == forced_entity_type and occurrence == 0:
            attribute = forced_query["attribute"]
            slot_index = next(
                index for index, row in enumerate(rows) if row["id"] == forced_query["entity_id"]
            )
        else:
            attribute = attributes[(state_offset + epoch + occurrence) % len(attributes)]
            sampling_slot_index = class_balanced_slot_index(
                sampling_rows,
                attribute=attribute,
                vocabulary=class_vocabulary(spec, entity_type, attribute),
                seed=stable_seed(
                    seed,
                    epoch,
                    query_sampling_key,
                    entity_type,
                    attribute,
                    occurrence,
                ),
                excluded_slot=(
                    (forced_query or {}).get("entity_id")
                    if entity_type == forced_entity_type
                    else None
                ),
            )
            sampled_slot = sampling_rows[sampling_slot_index]["id"]
            slot_index = next(index for index, row in enumerate(rows) if row["id"] == sampled_slot)
        row = rows[slot_index]
        class_name = normalize_class_name(attribute, row[attribute])
        vocabulary = class_vocabulary(spec, entity_type, attribute)
        if class_name not in vocabulary:
            raise ValueError(
                f"unknown class {class_name!r} for {entity_type}.{attribute} in {state_id}"
            )
        queries.append(
            {
                "query_id": f"{state_id}_e{epoch:04d}_q{query_index:03d}",
                "entity_type": entity_type,
                "entity_type_id": ENTITY_TYPE_IDS[entity_type],
                "attribute": attribute,
                "attribute_id": attr_ids[(entity_type, attribute)],
                "head": f"{entity_type}.{attribute}",
                "slot": row["id"],
                "slot_index": slot_index,
                "class_name": class_name,
                "class_index": vocabulary.index(class_name),
                "class_vocabulary": list(vocabulary),
            }
        )
    if Counter(query["entity_type"] for query in queries) != {
        entity_type: queries_per_state // len(ENTITY_TYPES) for entity_type in ENTITY_TYPES
    }:
        raise AssertionError("query sampler lost entity-type balance")
    return queries


def class_balanced_slot_index(
    rows: Sequence[JsonDict],
    *,
    attribute: str,
    vocabulary: Sequence[str],
    seed: int,
    excluded_slot: str | None = None,
) -> int:
    indices_by_class: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if row["id"] == excluded_slot:
            continue
        class_name = normalize_class_name(attribute, row[attribute])
        indices_by_class.setdefault(class_name, []).append(index)
    available_classes = [name for name in vocabulary if name in indices_by_class]
    if not available_classes:
        raise ValueError(f"no available classes for attribute {attribute}")
    rng = random.Random(seed)
    class_name = available_classes[rng.randrange(len(available_classes))]
    candidates = indices_by_class[class_name]
    return candidates[rng.randrange(len(candidates))]


def stable_seed(*parts: Any) -> int:
    payload = ":".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)


@lru_cache(maxsize=512)
def load_json_cached(path: str) -> JsonDict:
    return json.loads(Path(path).read_text())


class BoardRecognitionStateDataset(Dataset):
    """One raw board image plus multiple balanced symbolic slot queries."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        split: str | None = "train",
        queries_per_state: int | None = None,
        seed: int = 381_427,
        image_transform: Callable[[Image.Image], Any] | None = None,
        include_dense_labels: bool = False,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.seed = seed
        self.image_transform = image_transform
        self.include_dense_labels = include_dense_labels
        self.epoch = 0
        metadata_path = self.dataset_dir / "metadata.json"
        manifest_path = self.dataset_dir / "manifest.jsonl"
        if not metadata_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"board-recognition dataset is incomplete: {self.dataset_dir}")
        self.metadata = json.loads(metadata_path.read_text())
        self.is_replay_v1 = self.metadata.get("schema") == DATASET_SCHEMA
        self.queries_per_state = queries_per_state or (8 if self.is_replay_v1 else 16)
        if self.queries_per_state <= 0 or self.queries_per_state % len(ENTITY_TYPES):
            raise ValueError("queries_per_state must be positive and divisible by four")
        rows = read_jsonl(manifest_path)
        valid_splits = (
            {"train", "validation", "test", "color_diagnostic"}
            if self.is_replay_v1
            else {"train", "validation", "test"}
        )
        if split is not None:
            if split not in valid_splits:
                raise ValueError(f"unknown split: {split}")
            rows = [row for row in rows if row["split"] == split]
        if not rows:
            raise ValueError(f"no board-recognition states selected for split={split!r}")
        self.split = split
        self.rows = sorted(rows, key=lambda row: row["sample_id"])
        if self.is_replay_v1:
            self.spec = None
            self.base_rows_by_group = {}
            self.query_plans_by_state: dict[str, JsonDict] = {}
            self._build_replay_query_plans()
        else:
            spec_path = resolve_project_path(self.metadata["curriculum_path"])
            self.spec = json.loads(spec_path.read_text())
            if self.spec.get("schema") != "catan_board_recognition_curriculum/v1":
                raise ValueError("unsupported board-recognition curriculum schema")
            self.base_rows_by_group = {
                row["counterfactual_group_id"]: row
                for row in self.rows
                if row["counterfactual_role"] == "base"
            }
            if len(self.base_rows_by_group) * 2 != len(self.rows):
                raise ValueError("selected split does not contain complete counterfactual pairs")

    def __len__(self) -> int:
        return len(self.rows)

    def _build_replay_query_plans(self) -> None:
        self.query_plans_by_state = {}
        split_names = sorted({row["split"] for row in self.rows})
        for split_name in split_names:
            plans = build_split_query_plan(
                [row for row in self.rows if row["split"] == split_name],
                dataset_dir=self.dataset_dir,
                split=split_name,
                seed=self.seed,
                queries_per_state=self.queries_per_state,
                epoch_index=self.epoch,
            )
            self.query_plans_by_state.update({plan["state_id"]: plan for plan in plans})

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch
        if self.is_replay_v1:
            self._build_replay_query_plans()

    def __getitem__(self, index: int) -> JsonDict:
        row = self.rows[index]
        image_path = resolve_dataset_path(self.dataset_dir, row["image_path"])
        label_path = resolve_dataset_path(self.dataset_dir, row["label_path"])
        labels = load_json_cached(str(label_path.resolve()))
        with Image.open(image_path) as image_file:
            image: Any = image_file.convert("RGB").copy()
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.is_replay_v1:
            queries = self.query_plans_by_state[row["sample_id"]]["queries"]
            group_id = row["source"]["trajectory_id"]
            role = row["source"]["kind"]
            target = None
            stage = row["density_bin"]
        else:
            base_row = self.base_rows_by_group[row["counterfactual_group_id"]]
            sampling_labels = load_json_cached(
                str(resolve_dataset_path(self.dataset_dir, base_row["label_path"]).resolve())
            )
            queries = sample_state_queries(
                labels,
                spec=self.spec,
                queries_per_state=self.queries_per_state,
                seed=self.seed,
                epoch=self.epoch,
                forced_query=row["counterfactual"]["target"],
                sampling_key=row["counterfactual_group_id"],
                sampling_dense_labels=sampling_labels,
            )
            group_id = row["counterfactual_group_id"]
            role = row["counterfactual_role"]
            target = row["counterfactual"]["target"]
            stage = row["stage"]
        item = {
            "state_id": row["sample_id"],
            "counterfactual_group_id": group_id,
            "counterfactual_role": role,
            "counterfactual_target": target,
            "stage": stage,
            "split": row["split"],
            "image": image,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "queries": queries,
        }
        if self.include_dense_labels:
            item["dense_labels"] = labels
        return item


def collate_board_recognition_states(examples: Sequence[JsonDict]) -> JsonDict:
    if not examples:
        raise ValueError("cannot collate an empty board-recognition batch")
    query_counts = {len(example["queries"]) for example in examples}
    if len(query_counts) != 1:
        raise ValueError("all states in a batch must have the same query count")
    images = [example["image"] for example in examples]
    if all(isinstance(image, torch.Tensor) for image in images):
        image_batch: Any = torch.stack(images)
    else:
        image_batch = images
    return {
        "images": image_batch,
        "state_ids": [example["state_id"] for example in examples],
        "counterfactual_group_ids": [example["counterfactual_group_id"] for example in examples],
        "counterfactual_roles": [example["counterfactual_role"] for example in examples],
        "counterfactual_targets": [example["counterfactual_target"] for example in examples],
        "stages": [example["stage"] for example in examples],
        "splits": [example["split"] for example in examples],
        "entity_type_ids": query_tensor(examples, "entity_type_id"),
        "attribute_ids": query_tensor(examples, "attribute_id"),
        "slot_indices": query_tensor(examples, "slot_index"),
        "class_indices": query_tensor(examples, "class_index"),
        "queries": [example["queries"] for example in examples],
    }


def query_tensor(examples: Sequence[JsonDict], key: str) -> torch.Tensor:
    return torch.tensor(
        [[int(query[key]) for query in example["queries"]] for example in examples],
        dtype=torch.long,
    )


def make_board_recognition_dataloader(
    dataset: BoardRecognitionStateDataset,
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    seed: int = 381_427,
) -> DataLoader:
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        # Respawn workers for each iterator so dataset.set_epoch() propagates.
        persistent_workers=False,
        collate_fn=collate_board_recognition_states,
        generator=generator,
    )


def resolve_dataset_path(dataset_dir: Path, relative: str) -> Path:
    path = (dataset_dir / relative).resolve()
    root = dataset_dir.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"dataset-relative path escapes root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def resolve_project_path(relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path
