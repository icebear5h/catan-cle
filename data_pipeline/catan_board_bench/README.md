# CatanBoardBench Pipeline

This package owns the Catan public-board perception and grounded-reasoning
benchmark implementation and frozen benchmark inputs.

## Layout

- `builder.py`: builds replay-derived benchmark samples, contracts, images, and
  question files.
- `annotations.py`: derives frontend-aligned point and bbox annotations for
  tiles, nodes, edges, ports, resources, and dice numbers.
- `tokens.py`: canonical Catan token helpers for tiles, nodes, edges, ports, and
  board vocabulary.
- `scoring.py`: exact and component scoring for benchmark answers.
- `eval/`: benchmark harness integration.
- `datasets/`: frozen benchmark inputs only. Provider runs live under
  `artifacts/runs/catan_board_bench/`; reviewed reports live under
  `reports/catan_board_bench/`.

## Current Datasets

- `datasets/catan_board_bench_100/`: held-out CatanBoardBench-100 eval set.
- `datasets/resolution_sweep/`: small 256/512/768 image-size ablation set.

Keep CatanBoardBench-100 evaluation-only. New SFT/CPT data should come from fresh game
IDs that do not overlap `datasets/catan_board_bench_100/leakage/benchmark_game_ids.json`.
Do not write provider responses or reports back into a frozen dataset directory.
