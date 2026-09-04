# Catan ASCII Variation Probe

A text-only, strict-scoring smoke benchmark for six information-equivalent renderings of a complete public Catan board graph.

## Design

- 12 density-stratified snapshots from 12 distinct replay games
- Deterministically permuted, board-local tile/node/edge/port IDs
- 19 tiles, 54 nodes, 72 edges, and 9 ports explicitly represented
- No player summaries or precomputed count/production answers
- Six lossless renderings, each round-tripped to the same per-board fact digest
- 60 questions: six rows in each of ten categories
- Balanced directions, occupied/empty state, connected/disconnected pairs, port joins, and positive/empty production
- Strict typed JSON scoring rejects prose, missing fields, extra keys, extra set members, duplicates, and broken tuple associations

The six renderings are `flat_sorted`, `flat_shuffled`, `sectioned`, `tile_rows`, `local_blocks`, and `topology_diagram`. The latter three add only derived/redundant views of facts already present in the canonical records.

Build deterministically:

```bash
uv run python scripts/build_catan_board_bench_ascii_variations.py
```

The evaluator supplies pointy-top cube-direction rules and Catan production rules equally to every format. It sends no image and reveals no expected answer.
