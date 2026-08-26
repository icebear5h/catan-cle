# CatanBoardBench-100

Engine-oracle benchmark for Catan public-board perception and grounded
reasoning across screenshot and text representations.

Files:
- `manifest.jsonl`: one row per board sample.
- `contracts/`: full public board contracts derived from the engine.
- `images/`: 1024x1024 board renders using `view_padding_factor=0.933134`,
  which keeps the complete board visible at approximately 90% canvas coverage.
- `annotations_512.jsonl`: inherited source annotations for the original 512px
  images; these coordinates do not apply to this render variant.

Question-suite files live in `questions/`, which contains `questions.jsonl`,
`answer_key.jsonl`, and `qa.jsonl`.

The contract intentionally excludes hidden hands and hidden dev cards.
It does include public board state, visible points, played knights,
current prompt, ports, robber location, and Longest Road/Largest Army.

OpenBench integration lives in `data_pipeline/catan_board_bench/eval/`. Prefer that
runner for model comparisons; keep the direct OpenRouter script for provider
debugging. Provider plans, responses, and summaries live under
`artifacts/runs/catan_board_bench/catan_board_bench_100/`; reviewed reports live under
`reports/catan_board_bench/`.

The paired Qwen3.8 OpenRouter run is under
`artifacts/runs/catan_board_bench/catan_board_bench_100_1024_board90/`.
