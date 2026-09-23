"""Every dataset, artifact, and model path the bench routes read."""

from __future__ import annotations

from pathlib import Path

from evals.catan_board_bench.paths import DATASETS_DIR

# One package level deeper than the old module, so the repo root is five parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
BENCH_DIR = DATASETS_DIR / "catan_board_bench_100"
QUESTION_DIR = BENCH_DIR / "questions"
QA_PATH = QUESTION_DIR / "qa.jsonl"
EVAL_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "catan_board_bench"
    / "catan_board_bench_100"
    / "openrouter"
)
TEXT_FORMAT_DIR = DATASETS_DIR / "text_format_optimization_probe"
TEXT_FORMAT_QA_PATH = TEXT_FORMAT_DIR / "qa.jsonl"
TEXT_FORMAT_MANIFEST_PATH = TEXT_FORMAT_DIR / "manifest.jsonl"
TEXT_FORMAT_METADATA_PATH = TEXT_FORMAT_DIR / "metadata.json"
TEXT_FORMAT_DEV_COMPARISON_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "catan_board_bench"
    / "text_format_optimization"
    / "openrouter"
    / "qwen3_8_27b_dev_full_20260824"
    / "summary.json"
)
TEXT_FORMAT_DEV_V3_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "catan_board_bench"
    / "text_format_optimization"
    / "openrouter"
    / "qwen3_8_27b_dev_v3_full_20260824"
    / "summary.json"
)
TEXT_FORMAT_TRANSFER_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "catan_board_bench"
    / "text_format_optimization"
    / "openrouter"
    / "qwen3_8_27b_transfer_20260824"
)
SFT_REPLAY_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "generated"
    / "board_recognition"
    / "replay_v1"
)
SFT_FORWARD_ROOT = SFT_REPLAY_ROOT / "ms_swift_semantic_v1"
SFT_BIDIRECTIONAL_ROOT = SFT_REPLAY_ROOT / "ms_swift_bidirectional_v1"
SFT_IMAGE_ROOT = SFT_REPLAY_ROOT / "images"
SFT_SPATIAL_LOCALIZATION_ROOT = SFT_REPLAY_ROOT / "spatial_localization_v1"
SFT_SPATIAL_LOCALIZATION_IMAGE_ROOT = SFT_SPATIAL_LOCALIZATION_ROOT / "images"
SFT_SPATIAL_LOCALIZATION_STAGES = {
    "stage1": ("train", "validation", "test"),
    "stage2": ("train", "validation", "test"),
    "probes": ("validation", "test"),
}
SFT_VIEWER_DATASETS = {
    "spatial_localization_v1": "Legacy markers and relations",
    "spatial_localization_v3": "v3 — single pieces",
    "terrain_readout_v2": "Stage 2 — terrain",
    "node_edge_readout_v1": "Stage 3 — pieces (original)",
    "node_edge_readout_reweighted_v1": "Stage 3 — pieces (reweighted training)",
}


def _viewer_dataset(dataset: str) -> tuple[Path, dict[str, tuple[str, ...]]]:
    if dataset not in SFT_VIEWER_DATASETS:
        raise ValueError("Unknown SFT dataset")
    if dataset == "spatial_localization_v1":
        return SFT_SPATIAL_LOCALIZATION_ROOT, SFT_SPATIAL_LOCALIZATION_STAGES
    root = SFT_REPLAY_ROOT / dataset
    splits = tuple(
        split for split in ("train", "validation", "test", "color_diagnostic")
        if (root / "stage1" / f"{split}.jsonl").is_file()
    )
    return root, {"stage1": splits}


SFT_SPLITS = ("train", "validation", "test", "color_diagnostic")
SFT_TOPOLOGY_PATH = (
    PROJECT_ROOT / "artifacts" / "generated" / "sft" / "atlas_topology"
    / "catan_atlas_topology.jsonl"
)
SFT_SMOKE_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "sft"
    / "2026-09-01-qwen38-spatial-robber-curriculum-smoke.json"
)
SFT_SPATIAL_ROBBER_ROOT = SFT_REPLAY_ROOT / "spatial_robber_v1"
SFT_CURRICULUM_STAGES = (
    "spatial_grounding",
    "clean_board_grounding",
    "pieces_and_colors",
    "real_game_distribution",
)
SFT_EVAL_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "catan_board_bench"
    / "sft"
    / "qwen38_spatial_sft_b32"
    / "final_validation_v1"
)
SFT_EVAL_SUMMARY_PATH = SFT_EVAL_ROOT / "summary.json"
SFT_EVAL_RECORDS_PATH = SFT_EVAL_ROOT / "records.jsonl"
SFT_EVAL_SUITE_PATH = SFT_REPLAY_ROOT / "evals" / "validation_v1.jsonl"



INITIAL_SETTLEMENT_REASONING_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "model_selection"
    / "openrouter_vlm_27b_72b_20260831"
)

CANONICAL_MODELS = {
    "qwen3-vl-8b": "Qwen3 (8B)",
    "qwen3.5-9b": "Qwen3.5 (9B)",
    "qwen3.6-flash": "Qwen3.6 Flash",
}

DEFAULT_EVAL_MODELS = ("qwen3-vl-8b", "qwen3.5-9b", "qwen3.6-flash")
