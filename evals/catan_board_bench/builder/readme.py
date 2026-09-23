"""Dataset README text for the benchmark and its question suite."""

from __future__ import annotations

from pathlib import Path


def write_dataset_readme(output_dir: Path) -> None:
    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# CatanBoardBench-100",
                "",
                "Engine-oracle benchmark for Catan public-board perception and grounded reasoning.",
                "",
                "Files:",
                "- `manifest.jsonl`: one row per board sample.",
                "- `contracts/`: full public board contracts derived from the engine.",
                "- `images/`: optional 512x512 board renders when built with `--render-images`.",
                "",
                "Question-suite files live in `questions/`.",
                "That directory contains `questions.jsonl`, `answer_key.jsonl`, and `qa.jsonl`.",
                "",
                "The contract intentionally excludes hidden hands and hidden dev cards.",
                "It does include public board state, visible points, played knights,",
                "current prompt, ports, robber location, and Longest Road/Largest Army.",
                "",
            ]
        )
    )


def write_question_readme(question_dir: Path) -> None:
    readme = question_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# CatanBoardBench-100 Questions",
                "",
                "Engine-scored public-board QA rows for CatanBoardBench-100.",
                "",
                "Files:",
                "- `questions.jsonl`: promptable questions without answers.",
                "- `answer_key.jsonl`: deterministic answer targets for scoring.",
                "- `qa.jsonl`: questions plus answers and target metadata for local analysis.",
                "",
                "The referenced contracts and board images live in",
                "`evals/catan_board_bench/datasets/catan_board_bench_100/`.",
                "",
            ]
        )
    )
