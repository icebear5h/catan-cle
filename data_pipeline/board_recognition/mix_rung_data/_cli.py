"""Mix existing board-recognition exports into one rung by quota (``mix_rung_data``).

The ladder's rungs forget each other when trained one family at a time: the
terrain rung dropped the markers, the piece rung dropped terrain. A mixed
rung draws every head from the exports already on disk, with quotas that
say how many rows of each group come from each board density, so no head
starves and no density dominates. The recipe is a JSON file, not code:

    {
      "sources": {"pieces": "<export dir>", "terrain": "<export dir>"},
      "groups": [
        {"name": "piece_occupied", "source": "pieces", "categories": ["node.occupancy", "edge.owner"],
         "polarity": "positive", "rows": 6000, "density": {"setup": 1, "sparse": 1, "dense": 1},
         "balance": ["piece", "color"], "piece_shares": {"ROAD": 0.5, "SETTLEMENT": 0.25, "CITY": 0.25}},
        {"name": "piece_empty", ..., "polarity": "hard_negative", "kind_shares": {"adjacent": 0.5, ...}},
        {"name": "terrain_short", "source": "terrain", "categories": ["tile.resource", "tile.number", "port.port_type"], ...},
        {"name": "node_readout", "source": "pieces", "categories": ["node.readout"], "rows": 100, ...}
      ],
      "eval_sample": [{"source": "pieces", "split": "validation", "every": 8}, {"source": "terrain", "split": "validation", "every": 6}]
    }

Within a group, rows are apportioned to density bins by the ``density``
weights, then inside each bin spread evenly over the ``balance`` keys (each
colour, each piece type) as far as the pool allows, each cell in a stable
hashed order. A cell that runs short is topped up from the rest of its bin,
then from the group's other bins; a group with ``max_repeats`` above one
oversamples a short cell's own rows first (ids gain ``#r<n>``). Every
shortfall is reported in ``metadata.json``. Rows keep their source ids and metadata, so the panel and
the scorecard read the mixed run like any other. Images are hard-linked
into one root. Completion-token shares are estimated per group (atlas
tokens count one, other words and separators one each) so a recipe can be
sanity-checked without a tokenizer.
"""


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from data_pipeline.board_recognition.mix_rung_data._export import export_mixed_rung


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    metadata = export_mixed_rung(args.recipe, args.output_dir, overwrite=args.overwrite)
    print(json.dumps({key: value for key, value in metadata.items() if key != "files"}, indent=2, sort_keys=True))
    return 0


__all__ = ["main"]
