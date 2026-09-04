# CatanBoardBench

This package owns the Catan public-board perception and grounded-reasoning
benchmark implementation and frozen evaluation inputs.

## Layout

- `builder.py`: replay-derived benchmark contracts, images, and questions.
- `annotations.py`: frontend-aligned tile, node, edge, port, resource, and dice
  geometry.
- `tokens.py`: canonical Catan board-token helpers.
- `scoring.py`: exact and component answer scoring.
- `presentation.py`: typed wrappers for the frozen text and image board inputs;
  these preserve payload bytes, hashes, renderer provenance, and identifier
  spaces while sharing the runtime `BoardPresentation` contract.
- `benchmark.py` and `metadata.py`: OpenBench/Inspect integration.
- `datasets/`: frozen benchmark inputs only.

The standalone verifier UI lives at `evals/catan_board_bench_ui/`. Provider runs
live under `artifacts/runs/catan_board_bench/`; reviewed reports live under
`reports/catan_board_bench/`.

## Frozen datasets

- `datasets/catan_board_bench_100/`: held-out CatanBoardBench-100 evaluation set.
- `datasets/resolution_sweep/`: 256/512/768 image-size ablations.
- Other dataset directories are locked diagnostic format/perception probes.

Keep CatanBoardBench-100 evaluation-only. New SFT/CPT data must use fresh game
IDs that do not overlap
`datasets/catan_board_bench_100/leakage/benchmark_game_ids.json`. Do not write
provider responses or reports into a frozen dataset directory.
