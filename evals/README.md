# Evals

This directory now only contains compatibility wrappers for older OpenBench
entry points. Canonical CatanBoardBench code and data live under:

- `data_pipeline/catan_board_bench/eval/`: OpenBench/Inspect task code.
- `data_pipeline/catan_board_bench/datasets/catan_board_bench_100/`: contracts, images,
  questions, leakage ledger, annotations, reports, and provider run logs.
- `data_pipeline/catan_board_bench/datasets/resolution_sweep/`: resolution-sweep
  variants and run logs.

Prefer the registered benchmark or the canonical file:

```bash
uv run --extra eval bench eval catan_board_bench
uv run --extra eval bench eval data_pipeline/catan_board_bench/eval/benchmark.py
```

