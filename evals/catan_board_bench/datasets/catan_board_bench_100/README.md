# CatanBoardBench-100

Engine-oracle benchmark for Catan public-board perception and grounded
reasoning across screenshot and text representations.

Files:
- `manifest.jsonl`: one row per board sample.
- `contracts/`: full public board contracts derived from the engine.
- `images/`: optional 512x512 board renders when built with `--render-images`.
- `annotations_512.jsonl`: frontend-aligned bbox/point annotations for tiles,
  tile numbers, edges, nodes, and ports in 512x512 image coordinates.

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
