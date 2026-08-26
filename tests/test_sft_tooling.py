import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.build_catan_board_bench_piece_visuals import (
    DEFAULT_CONTRACT,
    DEFAULT_OUTPUT_DIR,
    DatasetWriter,
)
from scripts.verify_sft_layout import DEFAULT_MANIFEST, verify_layout
from sft.paths import (
    GENERATED_SFT_ROOT,
    PROJECT_ROOT,
    RENDERER_STYLE_CONFIG,
    SFT_FIXTURES_ROOT,
    resolve_dataset_asset,
)
from sft.scripts.build_vlm_sft_dataset import (
    DEFAULT_EXCLUDE,
    convert_row as convert_vlm_row,
    load_excluded_game_ids,
)
from sft.scripts.convert_to_qwen_series_sft import convert_file
from sft.scripts.render_contract_images import message_view


def test_sft_layout_migration_receipt():
    assert verify_layout(DEFAULT_MANIFEST) == {
        "groups": 10,
        "files": 1_429,
        "bytes": 259_913_133,
        "removed_files": 6,
        "portable_rows": 4_287,
    }


def test_canonical_sft_paths_are_outside_source_package():
    assert GENERATED_SFT_ROOT == PROJECT_ROOT / "artifacts/generated/sft"
    assert SFT_FIXTURES_ROOT == PROJECT_ROOT / "artifacts/fixtures/sft"
    assert RENDERER_STYLE_CONFIG == PROJECT_ROOT / "configs/sft/renderer_style.json"
    assert DEFAULT_OUTPUT_DIR == PROJECT_ROOT / "artifacts/generated/catan_board_bench/piece_recognition"
    assert DEFAULT_CONTRACT == SFT_FIXTURES_ROOT / "render_contracts/colonist_dummy_setup.json"


def test_piece_writer_supports_a_fresh_output_directory(tmp_path: Path):
    writer = DatasetWriter(tmp_path / "piece_probe", image_size=8, prompt_prefix="Answer.")
    writer.add_sample(
        sample_id="sample",
        category="test",
        image=Image.new("RGB", (8, 8)),
        question="Question?",
        answer="Answer",
        target={},
    )

    assert (tmp_path / "piece_probe/images/sample.png").is_file()


def test_resolve_dataset_asset_prefers_dataset_relative_path():
    dataset = SFT_FIXTURES_ROOT / "modal_vlm_smoke/train.jsonl"
    resolved = resolve_dataset_asset(dataset, "images/node_factor_n00_empty_v00.png")

    assert resolved.is_file()
    assert resolved.parent == dataset.parent / "images"


def test_missing_leakage_ledger_fails_closed(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Held-out game-ID ledger"):
        load_excluded_game_ids(tmp_path / "missing.json")


def test_default_leakage_ledger_is_populated():
    excluded = load_excluded_game_ids(DEFAULT_EXCLUDE)

    assert excluded
    assert all(game_id.isdigit() for game_id in excluded)


def test_invalid_leakage_ledger_schema_fails_closed(tmp_path: Path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"wrong_key": []}))

    with pytest.raises(ValueError, match="invalid schema"):
        load_excluded_game_ids(ledger)


def test_vlm_row_uses_repository_relative_image_path():
    image_root = PROJECT_ROOT / "artifacts/fixtures/sft/modal_vlm_smoke"
    row = {
        "id": "example",
        "sample_id": "sample",
        "image_path": "images/node_factor_n00_empty_v00.png",
        "question": "What is shown?",
        "answer": "EMPTY",
        "category": "node_occupancy",
        "target": {"building": None},
    }

    converted = convert_vlm_row(
        row,
        image_root=image_root,
        manifest_by_sample={},
        prompt_prefix="Answer exactly.",
    )

    assert converted["image"] == (
        "artifacts/fixtures/sft/modal_vlm_smoke/images/node_factor_n00_empty_v00.png"
    )
    assert not Path(converted["image"]).is_absolute()


def test_qwen_conversion_resolves_relative_images(tmp_path: Path):
    image = tmp_path / "images/example.png"
    image.parent.mkdir()
    image.write_bytes(b"image-bytes")
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "row-1",
                "image": "images/example.png",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": "Question"}],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": "Answer"}]},
                ],
            }
        )
        + "\n"
    )
    output = tmp_path / "converted/train.json"
    copied_images = tmp_path / "converted/images"

    rows, images = convert_file(source, output, copy_images_to=copied_images)
    converted = json.loads(output.read_text())

    assert (rows, images) == (1, 1)
    assert converted[0]["image"] == "000000.png"
    assert (copied_images / "000000.png").read_bytes() == b"image-bytes"


def test_rendered_message_rows_keep_relative_image_paths():
    row = {
        "id": "row-1",
        "sample_id": "sample-1",
        "image_path": "images/sample-1.png",
        "contract_path": "contracts/sample-1.json",
        "category": "node_occupancy",
        "question": "What is on the node?",
        "answer": "EMPTY",
    }

    message = message_view(row, prompt_prefix="Answer exactly.")

    assert message["image"] == "images/sample-1.png"
    assert not Path(message["image"]).is_absolute()
