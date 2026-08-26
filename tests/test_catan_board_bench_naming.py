import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from flask import Flask

from data_pipeline.catan_board_bench.eval.benchmark import (
    DEFAULT_BENCH_DIR,
    DEFAULT_PROBE_DIR,
)
from data_pipeline.catan_board_bench.eval.metadata import get_benchmark_metadata
from playground.game_viewer.routes.bench import bench_bp
from scripts.verify_catan_board_bench_rename import DEFAULT_MANIFEST, verify_rename


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_catan_board_bench_rename_receipt():
    assert verify_rename(DEFAULT_MANIFEST) == {
        "groups": 9,
        "files": 993,
        "bytes": 60_104_444,
        "raw_payloads": 60,
        "content_changed_files": 205,
    }


def test_canonical_benchmark_paths_use_new_namespace():
    assert DEFAULT_BENCH_DIR == (
        PROJECT_ROOT / "data_pipeline" / "catan_board_bench" / "datasets" / "catan_board_bench_100"
    )
    assert DEFAULT_PROBE_DIR == (
        PROJECT_ROOT / "artifacts" / "generated" / "catan_board_bench" / "piece_recognition"
    )
    old_slug = "catan" + "bench"
    assert not (PROJECT_ROOT / old_slug).exists()
    assert not (PROJECT_ROOT / "data_pipeline" / old_slug).exists()


def test_openbench_metadata_uses_public_name(monkeypatch):
    class BenchmarkMetadata(SimpleNamespace):
        pass

    openbench = ModuleType("openbench")
    openbench.__path__ = []
    utils = ModuleType("openbench.utils")
    utils.BenchmarkMetadata = BenchmarkMetadata
    monkeypatch.setitem(sys.modules, "openbench", openbench)
    monkeypatch.setitem(sys.modules, "openbench.utils", utils)

    metadata = get_benchmark_metadata()

    assert metadata.name == "CatanBoardBench"
    assert metadata.module_path == "catan_board_bench.eval.benchmark"
    assert metadata.function_name == "catan_board_bench"
    assert "visual-grounding" in metadata.tags


def test_verifier_api_uses_kebab_case_namespace():
    app = Flask(__name__)
    app.register_blueprint(bench_bp)
    routes = {rule.rule for rule in app.url_map.iter_rules()}

    benchmark_routes = {route for route in routes if route.startswith("/api/catan-board-bench")}
    assert benchmark_routes
    assert not any("catan_board_bench" in route for route in routes)


def test_frozen_dataset_identity_was_renamed():
    metadata = json.loads((DEFAULT_BENCH_DIR / "metadata.json").read_text())

    assert metadata["name"] == "CatanBoardBench-100"
    assert metadata["schema"] == "catan_board_bench/v1"


def test_leakage_builder_help_has_no_artifact_side_effects():
    guarded_paths = (
        DEFAULT_BENCH_DIR / "leakage/benchmark_game_ids.json",
        DEFAULT_BENCH_DIR / "leakage/benchmark_game_ids.md",
        PROJECT_ROOT / "reports/catan_board_bench/catan_board_bench_100_openrouter_baseline.md",
    )
    before = {path: path.read_bytes() for path in guarded_paths}

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.build_catan_board_bench_leakage_and_presft",
            "--help",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert {path: path.read_bytes() for path in guarded_paths} == before
