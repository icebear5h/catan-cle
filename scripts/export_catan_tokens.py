"""Export Catan atlas tokens for tokenizer extension.

Usage:
    uv run python scripts/export_catan_tokens.py \
        --output configs/training_configs/catan_added_tokens.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from catan_board_bench.tokens import write_token_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Catan regular added tokens.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/training_configs/catan_added_tokens.json"),
    )
    args = parser.parse_args()

    output_path = write_token_manifest(args.output)
    print(output_path)


if __name__ == "__main__":
    main()
