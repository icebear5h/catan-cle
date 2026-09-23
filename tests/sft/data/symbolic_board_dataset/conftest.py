"""Package-scoped source and full-build fixtures; source assets stay read-only."""

from pathlib import Path

import pytest

from sft.scripts.builders.build_symbolic_board_dataset import (
    DEFAULT_ROOT,
    build_dataset,
    read_json,
    read_jsonl,
    validate_dataset,
)

from .support import BuiltDataset, RealExample


@pytest.fixture(scope="package")
def real_example() -> RealExample:
    if not DEFAULT_ROOT.is_dir():
        pytest.skip("full_board_diverse_v1 local source assets unavailable")
    manifest = read_jsonl(DEFAULT_ROOT / "manifest.jsonl")
    source = next(r for r in manifest if r["split"] == "train" and r["density_bin"] == "empty")
    rows = read_jsonl(DEFAULT_ROOT / "full_board_readout_v1/stage1/train.jsonl")
    readout = next(r for r in rows if r["state_id"] == source["sample_id"])
    contract = read_json(DEFAULT_ROOT / str(source["contract_path"]))
    return source, readout, contract


@pytest.fixture(scope="package")
def built_dataset(tmp_path_factory: pytest.TempPathFactory) -> BuiltDataset:
    if not DEFAULT_ROOT.is_dir():
        pytest.skip("full_board_diverse_v1 local source assets unavailable")
    output = tmp_path_factory.mktemp("symbolic-real") / "dataset"
    original_open = Path.open

    def no_images(path: Path, *args: object, **kwargs: object) -> object:
        assert path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"), "image bytes must never be loaded"
        return original_open(path, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "open", no_images)
        result = build_dataset(output)
        verified = validate_dataset(output)
    metadata = read_json(output / "metadata.json")
    files = {s: read_jsonl(output / f"{s}.jsonl") for s in metadata["counts"]}
    return output, result, verified, metadata, files
