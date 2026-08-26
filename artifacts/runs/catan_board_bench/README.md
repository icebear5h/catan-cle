# CatanBoardBench runs

Each suite directory contains provider run evidence separated from the frozen
benchmark inputs under `data_pipeline/catan_board_bench/datasets/`.

Use one immutable directory per run. Keep `plan.json`, append-only raw responses,
a deterministic summary, and any run-specific metadata together. Put reviewed
narrative conclusions in `reports/catan_board_bench/`.
