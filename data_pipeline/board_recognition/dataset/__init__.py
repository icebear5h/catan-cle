"""PyTorch state batches for direct Catan board-slot classification."""

from __future__ import annotations

import hashlib as hashlib
import json
import random as random
from collections import Counter as Counter
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence, TypeAlias, cast

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data_pipeline.board_recognition.query_schedule import build_split_query_plan
from data_pipeline.board_recognition.replay_dataset import DATASET_SCHEMA
from data_pipeline.json_coerce import as_dict, as_int, as_list, as_str
from data_pipeline.json_types import JsonDict

from .sampling import CLASS_VOCAB_KEY as CLASS_VOCAB_KEY
from .sampling import COLLECTION_BY_ENTITY as COLLECTION_BY_ENTITY
from .sampling import ENTITY_TYPE_IDS as ENTITY_TYPE_IDS
from .sampling import ENTITY_TYPES as ENTITY_TYPES
from .sampling import attribute_ids as attribute_ids
from .sampling import class_balanced_slot_index as class_balanced_slot_index
from .sampling import class_vocabulary as class_vocabulary
from .sampling import normalize_class_name as normalize_class_name
from .sampling import sample_state_queries as sample_state_queries
from .sampling import stable_seed as stable_seed

# Dataset items and collated batches carry tensors and images, not JSON.
Example: TypeAlias = dict[str, object]
Batch: TypeAlias = dict[str, object]

PROJECT_ROOT = Path(__file__).resolve().parents[3]


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


@lru_cache(maxsize=512)
def load_json_cached(path: str) -> JsonDict:
    return cast(JsonDict, json.loads(Path(path).read_text()))


class BoardRecognitionStateDataset(Dataset[Example]):
    """One raw board image plus multiple balanced symbolic slot queries."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        split: str | None = "train",
        queries_per_state: int | None = None,
        seed: int = 381_427,
        image_transform: Callable[[Image.Image], object] | None = None,
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
        self.rows = sorted(rows, key=lambda row: as_str(row["sample_id"]))
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
        split_names = sorted({as_str(row["split"]) for row in self.rows})
        for split_name in split_names:
            plans = build_split_query_plan(
                [row for row in self.rows if row["split"] == split_name],
                dataset_dir=self.dataset_dir,
                split=split_name,
                seed=self.seed,
                queries_per_state=self.queries_per_state,
                epoch_index=self.epoch,
            )
            self.query_plans_by_state.update(
                {as_str(plan["state_id"]): plan for plan in plans}
            )

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch
        if self.is_replay_v1:
            self._build_replay_query_plans()

    def __getitem__(self, index: int) -> Example:
        row = self.rows[index]
        image_path = resolve_dataset_path(self.dataset_dir, as_str(row["image_path"]))
        label_path = resolve_dataset_path(self.dataset_dir, as_str(row["label_path"]))
        labels = load_json_cached(str(label_path.resolve()))
        with Image.open(image_path) as image_file:
            image: object = image_file.convert("RGB").copy()
        if self.image_transform is not None:
            image = self.image_transform(cast(Image.Image, image))
        if self.is_replay_v1:
            plan = self.query_plans_by_state[as_str(row["sample_id"])]
            queries = [as_dict(query) for query in as_list(plan["queries"])]
            source = as_dict(row["source"])
            group_id = as_str(source["trajectory_id"])
            role = as_str(source["kind"])
            target = None
            stage = row["density_bin"]
        else:
            base_row = self.base_rows_by_group[as_str(row["counterfactual_group_id"])]
            sampling_labels = load_json_cached(
                str(
                    resolve_dataset_path(
                        self.dataset_dir, as_str(base_row["label_path"])
                    ).resolve()
                )
            )
            queries = sample_state_queries(
                labels,
                spec=cast(JsonDict, self.spec),
                queries_per_state=self.queries_per_state,
                seed=self.seed,
                epoch=self.epoch,
                forced_query=as_dict(as_dict(row["counterfactual"])["target"]),
                sampling_key=as_str(row["counterfactual_group_id"]),
                sampling_dense_labels=sampling_labels,
            )
            group_id = as_str(row["counterfactual_group_id"])
            role = as_str(row["counterfactual_role"])
            target = as_dict(as_dict(row["counterfactual"])["target"])
            stage = row["stage"]
        item: Example = {
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


def _queries_of(example: Example) -> list[JsonDict]:
    """The query records of one dataset example."""

    queries = example["queries"]
    if not isinstance(queries, list):
        raise TypeError("dataset example queries must be a list")
    return [as_dict(query) for query in queries]


def collate_board_recognition_states(examples: Sequence[Example]) -> Batch:
    if not examples:
        raise ValueError("cannot collate an empty board-recognition batch")
    query_counts = {len(_queries_of(example)) for example in examples}
    if len(query_counts) != 1:
        raise ValueError("all states in a batch must have the same query count")
    images = [example["image"] for example in examples]
    if all(isinstance(image, torch.Tensor) for image in images):
        image_batch: object = torch.stack([image for image in images if isinstance(image, torch.Tensor)])
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


def query_tensor(examples: Sequence[Example], key: str) -> torch.Tensor:
    return torch.tensor(
        [[as_int(query[key]) for query in _queries_of(example)] for example in examples],
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
) -> DataLoader[Example]:
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
