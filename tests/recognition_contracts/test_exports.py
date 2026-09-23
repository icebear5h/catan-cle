"""Public namespace and complete exporter byte parity against the source oracle."""

import importlib
import pickle
import shutil
from pathlib import Path

import pytest

from .reference import HASHED_SOURCES, PACKAGES, ROOT, baseline, owned_names
from .witnesses import REPLAY


@pytest.mark.parametrize("name", PACKAGES)
def test_owned_exports_and_structural_limits(name: str) -> None:
    module = importlib.import_module(f"data_pipeline.board_recognition.{name}")
    assert owned_names(name) <= set(vars(module))
    package = ROOT / "data_pipeline/board_recognition" / name
    files = list(package.glob("*.py"))
    assert 1 < len(files) <= 15
    assert all(len(path.read_text().splitlines()) <= 300 for path in files)
    for symbol in owned_names(name):
        value = getattr(module, symbol)
        if callable(value) and getattr(value, "__module__", "").startswith(module.__name__):
            historical_reference = f"c{module.__name__}\n{symbol}\n.".encode()
            assert pickle.loads(historical_reference) is value


def test_package_root_and_source_hash_paths() -> None:
    package = ROOT / "data_pipeline/board_recognition"
    assert len(list(package.glob("*.py"))) == 11
    for name in HASHED_SOURCES:
        assert (package / f"{name}.py").is_file()
        assert not (package / name).is_dir()


@pytest.mark.parametrize(("name", "function", "source"), [
    ("replay_ms_swift", "export_replay_v1_ms_swift_semantic", REPLAY),
    ("inverse_grounding", "export_replay_v1_ms_swift_bidirectional", REPLAY),
    ("density_curriculum", "build_density_curriculum", REPLAY / "ms_swift_semantic_v1"),
    ("production_curriculum", "build_production_curriculum", REPLAY),
])
def test_export_file_bytes(name: str, function: str, source: Path, tmp_path: Path) -> None:
    output = tmp_path / "export"
    original = baseline(name)
    getattr(original, function)(source, output_dir=output)
    expected = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    shutil.rmtree(output)
    current = importlib.import_module(f"data_pipeline.board_recognition.{name}")
    getattr(current, function)(source, output_dir=output)
    actual = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert actual == expected
