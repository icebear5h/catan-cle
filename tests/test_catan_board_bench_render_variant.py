import json
from pathlib import Path

from PIL import Image
import pytest

from scripts import render_catan_board_bench_variant as variant


def make_source(root: Path) -> Path:
    source = root / "source"
    (source / "contracts").mkdir(parents=True)
    (source / "questions").mkdir()
    (source / "leakage").mkdir()
    (source / "contracts/sample_000.json").write_text("{}\n")
    (source / "questions/qa.jsonl").write_text('{"id": "q0"}\n')
    (source / "leakage/ids.json").write_text("[]\n")
    (source / "manifest.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample_000",
                "contract_path": "contracts/sample_000.json",
                "image_path": "images/sample_000.png",
            }
        )
        + "\n"
    )
    (source / "metadata.json").write_text(
        json.dumps(
            {
                "name": "fixture",
                "files": {"question_dir": "old/questions"},
            }
        )
        + "\n"
    )
    return source


def test_render_variant_preserves_contracts_and_changes_only_render(monkeypatch, tmp_path) -> None:
    source = make_source(tmp_path)
    output = tmp_path / "output"
    captured = {}

    def fake_render(contract, *, image_size, style):
        captured["contract"] = contract
        captured["image_size"] = image_size
        captured["padding"] = style.view_padding_factor
        return Image.new("RGB", (image_size, image_size), "blue")

    monkeypatch.setattr(variant, "render_contract_image", fake_render)
    metadata = variant.render_variant(
        source,
        output,
        image_size=512,
        view_padding_factor=0.9,
        target_board_canvas_fraction=0.9,
    )

    assert captured == {"contract": {}, "image_size": 512, "padding": 0.9}
    assert (output / "contracts/sample_000.json").read_text() == "{}\n"
    assert (output / "questions/qa.jsonl").read_text() == '{"id": "q0"}\n'
    assert Image.open(output / "images/sample_000.png").size == (512, 512)
    assert metadata["files"]["question_dir"] == "questions"
    assert metadata["render_variant"]["derived_from"] == str(source)


def test_render_variant_refuses_nonempty_output(tmp_path) -> None:
    source = make_source(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep.txt").write_text("user data")

    with pytest.raises(FileExistsError, match="not empty"):
        variant.render_variant(
            source,
            output,
            image_size=512,
            view_padding_factor=0.9,
            target_board_canvas_fraction=None,
        )
