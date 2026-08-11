# Evals

This directory now only contains compatibility wrappers for older OpenBench
entry points. Canonical CatanBench code and data live under:

- `data_pipeline/catanbench/eval/`: OpenBench/Inspect task code.
- `data_pipeline/catanbench/datasets/catanbench_100/`: contracts, images,
  questions, leakage ledger, annotations, reports, and provider run logs.
- `data_pipeline/catanbench/datasets/resolution_sweep/`: resolution-sweep
  variants and run logs.

Prefer the registered benchmark or the canonical file:

```bash
uv run --extra eval bench eval catanbench
uv run --extra eval bench eval data_pipeline/catanbench/eval/benchmark.py
```

