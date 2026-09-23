"""Dataset fingerprints and spatial eval variants."""

import json
from pathlib import Path

import pytest
from PIL import Image

from evals.catan_board_bench.tokens import recognition_token_inventory
from sft.qwen_series_vision_sft import (
    fingerprint_training_dataset,
    load_recognition_token_inventory,
)
from sft.scripts.eval.eval_qwen_vl_adapter import (
    evaluation_image,
    evaluation_metadata,
)


def test_dataset_fingerprint_hashes_referenced_images() -> None:
    dataset = Path("artifacts/fixtures/sft/modal_vision_sft_smoke/train.jsonl")

    first = fingerprint_training_dataset(dataset)
    second = fingerprint_training_dataset(dataset)

    assert first == second
    assert first["rows"] == 4
    assert first["unique_images"] == 4
    assert first["max_prompt_characters"] < 512
    assert first["max_answer_characters"] == len("<BLUE> <SETTLEMENT>")
    assert len(first["images"]) == 4
    assert len(first["combined_sha256"]) == 64


def test_replay_fingerprint_uses_explicit_image_root_and_exact_token_inventory(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "annotations"
    images = tmp_path / "images"
    annotations.mkdir()
    images.mkdir()
    (images / "board.png").write_bytes(b"board-image")
    dataset = annotations / "train.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "image": "board.png",
                "conversations": [
                    {"from": "human", "value": "<image>\n<N00><Q_NODE_OCCUPANCY>"},
                    {"from": "gpt", "value": "<A_NODE_OCCUPANCY_EMPTY>"},
                ],
            }
        )
        + "\n"
    )
    inventory_path = tmp_path / "trainable_tokens.json"
    inventory_path.write_text(json.dumps(recognition_token_inventory()))

    fingerprint = fingerprint_training_dataset(
        dataset,
        image_root=images,
        token_inventory=inventory_path,
    )

    assert fingerprint["rows"] == 1
    assert fingerprint["unique_images"] == 1
    assert fingerprint["image_root"] == str(images)
    assert fingerprint["annotation_root"] == str(annotations)
    assert fingerprint["token_inventory"]["counts"]["total"] == 220
    assert load_recognition_token_inventory(inventory_path)["tokens"] == (
        recognition_token_inventory()["tokens"]
    )
    with pytest.raises(FileNotFoundError):
        fingerprint_training_dataset(
            dataset,
            image_root=tmp_path / "wrong-images",
            token_inventory=inventory_path,
        )


def test_dataset_fingerprint_requires_one_pair_and_one_image(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps({"id": "text-only", "messages": []}) + "\n")

    with pytest.raises(ValueError, match="exactly one user/assistant pair"):
        fingerprint_training_dataset(dataset)


def test_dataset_fingerprint_rejects_long_generation_targets(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "long-answer",
                "image": "image.png",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "Read the board."},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "x" * 513}],
                    },
                ],
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="short-answer limit"):
        fingerprint_training_dataset(dataset)


def test_spatial_eval_variants_blank_or_occlude_equal_regions(tmp_path: Path) -> None:
    image_path = tmp_path / "board.png"
    Image.new("RGB", (100, 100), (255, 255, 255)).save(image_path)
    row = {
        "image": str(image_path),
        "row_id": "probe",
        "task_type": "neutral_probe_token_return",
        "entity_type": "node",
        "spatial_targets": [
            {
                "bbox": [0.1, 0.1, 0.2, 0.2],
                "control_bbox": [0.7, 0.7, 0.8, 0.8],
            }
        ],
    }

    blank = evaluation_image(
        row, variant="blank", shuffled_images=None, occlusion_margin=0.0
    )
    target = evaluation_image(
        row, variant="target_occlusion", shuffled_images=None, occlusion_margin=0.0
    )
    control = evaluation_image(
        row, variant="control_occlusion", shuffled_images=None, occlusion_margin=0.0
    )

    assert blank.getpixel((50, 50)) == (127, 127, 127)
    assert target.getpixel((15, 15)) == (42, 42, 42)
    assert target.getpixel((75, 75)) == (255, 255, 255)
    assert control.getpixel((75, 75)) == (42, 42, 42)
    assert evaluation_metadata(row, image_variant="target_occlusion") == {
        "entity_type": "node",
        "task_type": "neutral_probe_token_return",
        "eval_variant": "target_occlusion",
    }
