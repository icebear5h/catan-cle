import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from data_pipeline.board_recognition.dataset import (
    BoardRecognitionStateDataset,
    make_board_recognition_dataloader,
    read_jsonl,
)
from data_pipeline.board_recognition.sft import export_qwen_sft


FIXTURE_DIR = Path("artifacts/fixtures/board_recognition/curriculum_smoke")


def image_to_tensor(image):
    array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1) / 255.0


def test_state_dataset_returns_balanced_queries_and_raw_image():
    dataset = BoardRecognitionStateDataset(
        FIXTURE_DIR,
        split="train",
        queries_per_state=16,
        seed=123,
        include_dense_labels=True,
    )

    item = dataset[0]

    assert len(dataset) == 28
    assert item["image"].size == (64, 64)
    assert item["image"].mode == "RGB"
    assert len(item["queries"]) == 16
    assert Counter(query["entity_type"] for query in item["queries"]) == {
        "tile": 4,
        "node": 4,
        "edge": 4,
        "port": 4,
    }
    assert all(
        query["class_vocabulary"][query["class_index"]] == query["class_name"]
        for query in item["queries"]
    )
    target = item["counterfactual_target"]
    assert any(
        query["entity_type"] == target["entity_type"]
        and query["attribute"] == target["attribute"]
        and query["slot"] == target["entity_id"]
        for query in item["queries"]
    )
    assert len(item["dense_labels"]["entities"]["tiles"]) == 19
    assert len(item["dense_labels"]["entities"]["nodes"]) == 54
    assert len(item["dense_labels"]["entities"]["edges"]) == 72
    assert len(item["dense_labels"]["entities"]["ports"]) == 9


def test_state_query_sampling_is_deterministic_and_epoch_variant():
    dataset = BoardRecognitionStateDataset(
        FIXTURE_DIR,
        split="validation",
        queries_per_state=16,
        seed=456,
    )

    epoch_zero = dataset[0]["queries"]
    assert epoch_zero == dataset[0]["queries"]
    dataset.set_epoch(1)
    epoch_one = dataset[0]["queries"]

    assert epoch_zero != epoch_one
    assert Counter(query["entity_type"] for query in epoch_one) == {
        "tile": 4,
        "node": 4,
        "edge": 4,
        "port": 4,
    }


def test_counterfactual_pair_samples_aligned_queries():
    dataset = BoardRecognitionStateDataset(
        FIXTURE_DIR,
        split="train",
        queries_per_state=16,
        seed=222,
    )
    first = dataset[0]
    second = next(
        dataset[index]
        for index in range(1, len(dataset))
        if dataset.rows[index]["counterfactual_group_id"] == first["counterfactual_group_id"]
    )

    first_signature = [
        (query["entity_type"], query["attribute"], query["slot"]) for query in first["queries"]
    ]
    second_signature = [
        (query["entity_type"], query["attribute"], query["slot"]) for query in second["queries"]
    ]
    changed = [
        (left, right)
        for left, right in zip(first["queries"], second["queries"], strict=True)
        if left["class_name"] != right["class_name"]
    ]

    assert first_signature == second_signature
    assert len(changed) == 1
    target = first["counterfactual_target"]
    assert (changed[0][0]["entity_type"], changed[0][0]["attribute"], changed[0][0]["slot"]) == (
        target["entity_type"],
        target["attribute"],
        target["entity_id"],
    )


def test_state_dataloader_collates_one_image_with_many_queries():
    dataset = BoardRecognitionStateDataset(
        FIXTURE_DIR,
        split="train",
        queries_per_state=16,
        seed=789,
        image_transform=image_to_tensor,
    )
    loader = make_board_recognition_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    assert batch["images"].shape == (2, 3, 64, 64)
    assert batch["entity_type_ids"].shape == (2, 16)
    assert batch["attribute_ids"].shape == (2, 16)
    assert batch["slot_indices"].shape == (2, 16)
    assert batch["class_indices"].shape == (2, 16)
    assert torch.equal(
        batch["entity_type_ids"][0],
        torch.tensor([0, 1, 2, 3] * 4),
    )


def test_worker_processes_receive_epoch_updates():
    dataset = BoardRecognitionStateDataset(
        FIXTURE_DIR,
        split="train",
        queries_per_state=16,
        seed=910,
        image_transform=image_to_tensor,
    )
    loader = make_board_recognition_dataloader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    epoch_zero = next(iter(loader))["queries"][0]
    dataset.set_epoch(1)
    epoch_one = next(iter(loader))["queries"][0]

    assert epoch_zero != epoch_one


def test_qwen_sft_export_has_official_conversation_shape(tmp_path: Path):
    output_dir = tmp_path / "qwen_sft"

    report = export_qwen_sft(
        FIXTURE_DIR,
        output_dir=output_dir,
        queries_per_state=16,
        seed=381_427,
    )

    assert report == {
        "valid": True,
        "states": 32,
        "rows": 512,
        "splits": {"train": 448, "validation": 32, "test": 32},
        "entity_types": {"edge": 128, "node": 128, "port": 128, "tile": 128},
        "output_dir": str(output_dir),
    }
    rows = read_jsonl(output_dir / "train.jsonl")
    audit = read_jsonl(output_dir / "audit/train.jsonl")
    assert len(rows) == len(audit) == 448
    assert set(rows[0]) == {"image", "conversations"}
    assert rows[0]["conversations"][0]["from"] == "human"
    assert rows[0]["conversations"][0]["value"].count("<image>") == 1
    assert rows[0]["conversations"][1] == {
        "from": "gpt",
        "value": audit[0]["class_name"],
    }
    assert "<image>" not in rows[0]["conversations"][1]["value"]
    assert (FIXTURE_DIR / rows[0]["image"]).is_file()


def test_qwen_sft_export_is_deterministic_except_timestamp(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for output_dir in (first, second):
        export_qwen_sft(
            FIXTURE_DIR,
            output_dir=output_dir,
            queries_per_state=16,
            seed=77,
        )

    first_files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
    second_files = sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
    assert first_files == second_files
    for relative_path in first_files:
        if relative_path == Path("metadata.json"):
            first_metadata = json.loads((first / relative_path).read_text())
            second_metadata = json.loads((second / relative_path).read_text())
            first_metadata.pop("generated_at")
            second_metadata.pop("generated_at")
            assert first_metadata == second_metadata
        else:
            assert (first / relative_path).read_bytes() == (second / relative_path).read_bytes()
