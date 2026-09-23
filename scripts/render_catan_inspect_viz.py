#!/usr/bin/env python
"""Render an Inspect Viz comparison from verified Catan archive-log index data."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict, cast

import pandas as pd
from inspect_ai.log import read_eval_log
from inspect_viz import Data
from inspect_viz.layout import vconcat
from inspect_viz.plot import write_html
from inspect_viz.table import table
from inspect_viz.view import scores_by_model

from evals.inspect_archives import DEFAULT_INSPECT_LOG_DIR


class VerificationRecord(TypedDict, total=False):
    """Verification block recorded for one archived Inspect log."""

    status: str
    source_scores_verified: bool
    scores: dict[str, float]


class ArchiveLogRecord(TypedDict, total=False):
    """One archived Inspect log as written by the archive importer."""

    archive_id: str
    input_mode: str
    verification: VerificationRecord
    expected_metrics: dict[str, int]
    model_id: str
    inspect_model_id: str
    log_path: str


# inspect_viz annotates score_stderr as str, but None is how these views are
# told the data has no stderr column; ci=False already disables the interval.
NO_STDERR_COLUMN: str = cast("str", None)

MODEL_NAMES = {
    "deepseek/deepseek-v4-flash-vision-exp": "DeepSeek V4 Vision",
    "google/gemma-4-31b-it": "Gemma 4 31B",
    "qwen/qwen3.8-max": "Qwen3.8 Max",
    "zai-org/glm-4.6v": "GLM-4.6V",
}
EXPECTED_MODEL_IDS = frozenset(MODEL_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INSPECT_LOG_DIR / "index.json",
        help="Verified archive-import index.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_INSPECT_LOG_DIR / "strict_vision_comparison.html",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.index.read_text())
    rows = strict_vision_rows(payload, log_root=args.index.parent)
    data = Data.from_dataframe(pd.DataFrame(rows))
    comparison = vconcat(
        table(
            data,
            columns=(
                "model_display_name",
                "exact_count",
                "json_valid_count",
                "protocol_exact_count",
            ),
            sorting=True,
            filtering="header",
            width=1000,
            height=150,
        ),
        scores_by_model(
            data,
            score_value="strict_exact",
            score_stderr=NO_STDERR_COLUMN,
            ci=False,
            sort="desc",
            score_label="Strict exact accuracy",
            title="Catan strict raw-image 60 · exact",
            color="#416AD0",
            width=1000,
            height=220,
            margin_left=190,
        ),
        scores_by_model(
            data,
            score_value="strict_json_valid",
            score_stderr=NO_STDERR_COLUMN,
            ci=False,
            sort="desc",
            score_label="Valid JSON rate",
            title="Output protocol · valid JSON",
            color="#2A9D8F",
            width=1000,
            height=220,
            margin_left=190,
        ),
        scores_by_model(
            data,
            score_value="strict_protocol_exact",
            score_stderr=NO_STDERR_COLUMN,
            ci=False,
            sort="desc",
            score_label="Protocol-exact rate",
            title="Output protocol · exact before semantic rescue",
            color="#E76F51",
            width=1000,
            height=220,
            margin_left=190,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_html(args.output, comparison)
    print(f"wrote {args.output}")
    return 0


def strict_vision_rows(
    payload: Mapping[str, object],
    *,
    log_root: Path | None = None,
) -> list[dict[str, str | float]]:
    if payload.get("schema") != "catan-inspect-archive/v1":
        raise ValueError("unsupported Inspect archive index schema")
    rows: list[dict[str, str | float]] = []
    model_ids: set[str] = set()
    contract_keys: set[str] = set()
    for record in cast("list[ArchiveLogRecord]", payload.get("logs", [])):
        if record.get("input_mode") != "raw_image":
            continue
        verification = record.get("verification") or {}
        if verification.get("status") != "success" or not verification.get(
            "source_scores_verified"
        ):
            raise ValueError(f"unverified strict-vision log: {record.get('archive_id')}")
        metrics = record["expected_metrics"]
        model_id = str(record["model_id"])
        if record.get("inspect_model_id") != f"archive/{model_id}":
            raise ValueError(f"unexpected archived Inspect model ID for {model_id}")
        if model_id not in EXPECTED_MODEL_IDS:
            raise ValueError(f"unexpected strict-vision model ID: {model_id}")
        if model_id in model_ids:
            raise ValueError(f"duplicate strict-vision model ID: {model_id}")
        model_ids.add(model_id)
        if log_root is not None:
            contract_keys.add(_verify_indexed_log(record, log_root))
        requests = int(metrics["requests"])
        if requests != 60:
            raise ValueError(
                f"strict-vision run is not the frozen 60-sample cohort: "
                f"{record.get('archive_id')} has {requests}"
            )
        verified_scores = verification["scores"]
        source_scores = {
            "strict_exact": int(metrics["exact"]) / requests,
            "strict_json_valid": int(metrics["json_valid"]) / requests,
            "strict_protocol_exact": int(metrics["protocol_exact"]) / requests,
        }
        for scorer_name, source_value in source_scores.items():
            if scorer_name not in verified_scores or not math.isclose(
                float(verified_scores[scorer_name]),
                source_value,
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"source/index score mismatch for {model_id} {scorer_name}"
                )
        rows.append(
            {
                "task_name": "catan_strict_raw_vision_60",
                "model": model_id,
                "model_display_name": MODEL_NAMES[model_id],
                "strict_exact": float(verified_scores["strict_exact"]),
                "strict_json_valid": float(verified_scores["strict_json_valid"]),
                "strict_protocol_exact": float(
                    verified_scores["strict_protocol_exact"]
                ),
                "exact_count": f"{int(metrics['exact'])}/{requests}",
                "json_valid_count": f"{int(metrics['json_valid'])}/{requests}",
                "protocol_exact_count": f"{int(metrics['protocol_exact'])}/{requests}",
                "inspect_model_id": record["inspect_model_id"],
                "log_path": record["log_path"],
            }
        )
    if model_ids != EXPECTED_MODEL_IDS:
        raise ValueError(
            "strict-vision model set differs from the frozen comparison: "
            f"expected={sorted(EXPECTED_MODEL_IDS)}, actual={sorted(model_ids)}"
        )
    if log_root is not None and len(contract_keys) != 1:
        raise ValueError("strict-vision logs do not share one benchmark contract")
    return rows


def _verify_indexed_log(record: ArchiveLogRecord, log_root: Path) -> str:
    root = log_root.resolve()
    relative = Path(str(record["log_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"invalid indexed Inspect log path: {relative}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"indexed Inspect log escapes log root: {relative}") from exc
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"indexed Inspect log is missing or unsafe: {relative}")

    log = read_eval_log(path, header_only=True)
    verification = record["verification"]
    if log.status != verification["status"]:
        raise ValueError(f"indexed Inspect status mismatch: {relative}")
    if str(log.eval.model) != record["inspect_model_id"]:
        raise ValueError(f"indexed Inspect model mismatch: {relative}")
    if log.results is None:
        raise ValueError(f"indexed Inspect log has no results: {relative}")
    actual = {}
    for score in log.results.scores:
        selected = score.metrics.get("accuracy") or score.metrics.get(
            "eligible_accuracy"
        )
        if selected is not None:
            actual[score.name] = float(selected.value)
    for scorer_name, expected in verification["scores"].items():
        if scorer_name not in actual or not math.isclose(
            actual[scorer_name],
            float(expected),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"indexed Inspect score mismatch for {scorer_name}: {relative}"
            )

    metadata = log.eval.metadata
    if metadata is None:
        raise ValueError(f"indexed Inspect log has no metadata: {relative}")
    contract = {
        "benchmark_contract": metadata.get("benchmark_contract"),
        "input_mode": metadata.get("input_mode"),
        "manifest_sha256": metadata.get("manifest_sha256"),
        "scorer": metadata.get("scorer"),
        "dataset_samples": log.eval.dataset.samples,
    }
    return json.dumps(contract, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    raise SystemExit(main())
