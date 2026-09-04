# Evals

Evaluation code and frozen evaluation inputs live under this package.

- `catan_board_bench/`: CatanBoardBench implementation, OpenBench/Inspect entry
  point, and frozen public-board datasets.
- `catan_board_bench_ui/`: standalone board-perception and agent-decision UI.
- Root modules: replay-policy comparisons, decision spot checks, and transcript
  evaluations.

Run CatanBoardBench through its registered name or canonical module:

```bash
uv run --extra eval bench eval catan_board_bench
uv run --extra eval bench eval evals/catan_board_bench/benchmark.py
```

Run the standalone eval UI:

```bash
cd evals/catan_board_bench_ui
npm ci
npm run dev
```
