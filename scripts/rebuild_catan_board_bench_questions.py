#!/usr/bin/env python
"""Rebuild CatanBoardBench question JSONL files from existing contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.catan_board_bench.builder import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QUESTION_DIR,
    CatanObservationSuite,
    _answer_view,
    _project_relative_path,
    _question_view,
    _write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--question-dir", type=Path, default=DEFAULT_QUESTION_DIR)
    parser.add_argument("--questions-per-sample", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract_dir = args.bench_dir / "contracts"
    contract_paths = sorted(contract_dir.glob("sample_*.json"))
    if not contract_paths:
        raise SystemExit(f"No contracts found in {contract_dir}")

    args.question_dir.mkdir(parents=True, exist_ok=True)
    questions_path = args.question_dir / "questions.jsonl"
    answer_key_path = args.question_dir / "answer_key.jsonl"
    qa_path = args.question_dir / "qa.jsonl"

    suite = CatanObservationSuite()
    qa_count = 0
    with (
        questions_path.open("w") as questions_f,
        answer_key_path.open("w") as answer_key_f,
        qa_path.open("w") as qa_f,
    ):
        for contract_path in contract_paths:
            contract = json.loads(contract_path.read_text())
            for qa in suite.qa_pairs(contract, questions_per_sample=args.questions_per_sample):
                _write_jsonl(qa_f, qa)
                _write_jsonl(questions_f, _question_view(qa))
                _write_jsonl(answer_key_f, _answer_view(qa))
                qa_count += 1

    metadata_path = args.bench_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        metadata["qa_count"] = qa_count
        metadata.setdefault("files", {})
        metadata["files"].update(
            {
                "question_dir": _project_relative_path(args.question_dir),
                "questions": questions_path.name,
                "answer_key": answer_key_path.name,
                "qa": qa_path.name,
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"Rebuilt {qa_count} QA rows from {len(contract_paths)} contracts")
    print(f"Question dir: {_project_relative_path(args.question_dir)}")


if __name__ == "__main__":
    main()
