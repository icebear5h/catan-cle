"""Command line entry point for the board coverage probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.probes.probe_catan_board_coverage.model import (
    TILE_MODES,
    JsonDict,
    read_jsonl,
    read_manifest,
)
from scripts.probes.probe_catan_board_coverage.report import (
    build_report,
    filter_parts,
    render_text_report,
)

# The pre-split module path stays the advertised program name and description.
PROG = "probe_catan_board_coverage.py"
DESCRIPTION = """Probe a single board sample for atlas-part coverage and eval outcomes.

This script reports whether each board object (tile/node/edge/port) that appears in a
sample contract is covered by QA probes, and optionally whether those probes were
answered correctly in a provided eval responses file.
"""

__all__ = [
    "DESCRIPTION",
    "PROG",
    "build_parser",
    "load_response_map",
    "main",
    "resolve_sample",
]


def load_response_map(
    path: Path,
    sample_id: str | None = None,
    model: str | None = None,
) -> dict[str, JsonDict]:
    if not path.exists():
        raise FileNotFoundError(f"responses file not found: {path}")

    by_qid: dict[str, JsonDict] = {}
    for row in read_jsonl(path):
        if sample_id is not None and row.get("sample_id") != sample_id:
            continue
        if model is not None and row.get("model_key") != model:
            continue
        qid = row.get("question_id")
        if qid:
            by_qid[str(qid)] = row
    return by_qid


def resolve_sample(bench_dir: Path, sample_id: str | None, image: str | None) -> str:
    manifest = read_manifest(bench_dir)
    if sample_id:
        if sample_id not in manifest:
            raise KeyError(f"sample {sample_id} not found in {bench_dir / 'manifest.jsonl'}")
        return sample_id

    if image is None:
        return sorted(manifest)[0]

    image_norm = image.strip().replace("\\", "/")
    for sid, row in manifest.items():
        row_image = str(row.get("image_path"))
        if image_norm == row_image or image_norm == row_image.split("/")[-1]:
            return sid
    raise KeyError(f"could not find sample with image_path {image!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description=DESCRIPTION)
    parser.add_argument(
        "--bench-dir", type=Path, default=Path("evals/catan_board_bench/datasets/catan_board_bench_100")
    )
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Resolve sample by image path or file name",
    )
    parser.add_argument(
        "--qa",
        type=Path,
        default=Path("evals/catan_board_bench/datasets/catan_board_bench_100/questions/qa.jsonl"),
        help="QA rows JSONL (usually questions/qa.jsonl)",
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=None,
        help="Optional responses.jsonl path",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Directory containing responses.jsonl",
    )
    parser.add_argument("--model", type=str, default=None, help="Filter response rows by model_key")
    parser.add_argument("--top", type=int, default=20, help="Top N uncovered tokens per probe type to show")
    parser.add_argument(
        "--part",
        type=str,
        default="all",
        choices=["all", "tiles", "nodes", "edges", "ports", "robber"],
        help="Limit report to a single object class",
    )
    parser.add_argument(
        "--tile-mode",
        type=str,
        default="all",
        choices=sorted(TILE_MODES),
        help="For --part tiles: all/identity/values probe families",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output")
    return parser


def _load_contract(path: Path) -> JsonDict:
    contract = json.loads(path.read_text())
    if not isinstance(contract, dict):
        raise ValueError(f"{path} is not a JSON object")
    return contract


def main() -> int:
    args = build_parser().parse_args()

    bench_dir = args.bench_dir
    sample_id = resolve_sample(bench_dir, args.sample_id, args.image)
    manifest_map = read_manifest(bench_dir)
    sample_meta = manifest_map[sample_id]

    contract_path = bench_dir / str(sample_meta["contract_path"])
    contract = _load_contract(contract_path)

    all_questions = [row for row in read_jsonl(args.qa) if row.get("sample_id") == sample_id]
    if not all_questions:
        raise RuntimeError(f"no QA rows found for sample {sample_id} in {args.qa}")

    responses_path: Path | None = args.responses
    if responses_path is None and args.eval_dir is not None:
        responses_path = args.eval_dir / "responses.jsonl"

    response_map: dict[str, JsonDict] | None = None
    if responses_path is not None:
        response_map = load_response_map(responses_path, sample_id=sample_id, model=args.model)

    report = build_report(
        sample_id=sample_id,
        bench_dir=bench_dir,
        contract=contract,
        sample_meta=sample_meta,
        qa_rows=all_questions,
        responses_by_qid=response_map,
    )

    if args.part != "all":
        report = filter_parts(report, args.part, args.tile_mode)

    report_text = render_text_report(report, top_n=args.top)
    print(report_text)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")

    return 0
