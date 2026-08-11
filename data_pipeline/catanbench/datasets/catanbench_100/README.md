# CatanBench-100

Small engine-oracle benchmark for Catan board recognition.

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

OpenBench integration lives in `data_pipeline/catanbench/eval/`. Prefer that
runner for model comparisons; keep the direct OpenRouter script for provider
debugging and historical response logs.
