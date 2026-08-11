# CatanBench Pipeline

This package owns the Catan visual-grounding benchmark and its supporting
artifacts.

## Layout

- `builder.py`: builds replay-derived benchmark samples, contracts, images, and
  question files.
- `annotations.py`: derives frontend-aligned point and bbox annotations for
  tiles, nodes, edges, ports, resources, and dice numbers.
- `tokens.py`: canonical Catan token helpers for tiles, nodes, edges, ports, and
  board vocabulary.
- `scoring.py`: exact and component scoring for benchmark answers.
- `eval/`: benchmark harness integration.
- `datasets/`: generated benchmark datasets and frozen reports.

## Current Datasets

- `datasets/catanbench_100/`: held-out CatanBench-100 eval set.
- `datasets/resolution_sweep/`: small 256/512/768 image-size ablation set.

Keep CatanBench-100 evaluation-only. New SFT/CPT data should come from fresh game
IDs that do not overlap `datasets/catanbench_100/leakage/benchmark_game_ids.json`.
