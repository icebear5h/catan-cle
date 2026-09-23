"""Token-budgeted resampling of node_edge_readout_v1; no loss or label edits.

The default completion-token budget is 50% short occupied, 25% short empty,
25% complete readouts. Occupied mass is split road/settlement/city 50/25/25,
then uniformly across colours within each piece. Empty-kind and readout-density
proportions are retained within node/edge families. Evaluation files are copied
byte-for-byte. A checkpoint tokenizer (including atlas tokens) is required.

These are corpus token-exposure targets, NOT exact gradient/loss shares: batch
normalization, sequence difficulty and the training prefix also affect updates.
Reducing readout exposure means fewer readout images, not deleting empty items.
"""


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from data_pipeline.board_recognition.reweight_node_edge._config import (
    MixConfig,
)
from data_pipeline.board_recognition.reweight_node_edge._export import export_reweighted


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--occupied-share", type=float, default=0.5)
    parser.add_argument("--empty-share", type=float, default=0.25)
    parser.add_argument("--readout-share", type=float, default=0.25)
    parser.add_argument("--row-count", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-repeats", type=int, default=4)
    args = parser.parse_args(argv)
    result = export_reweighted(args.source, args.output_dir, args.tokenizer_json,
                               config=MixConfig(args.occupied_share, args.empty_share, args.readout_share, args.seed, args.max_repeats),
                               row_count=args.row_count)
    print(json.dumps({"before": result["before"], "after": result["after"], "output": str(args.output_dir)}, indent=2))
    return 0


__all__ = ["main"]
