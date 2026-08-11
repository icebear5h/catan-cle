# CatanBench-100

Small engine-oracle benchmark for Catan board recognition.

Files:
- `manifest.jsonl`: one row per board sample.
- `contracts/`: full public board contracts derived from the engine.
- `images/`: optional 512x512 board renders when built with `--render-images`.

Question-suite files live in this dataset directory under `questions/`. That
directory contains `questions.jsonl`, `answer_key.jsonl`, and `qa.jsonl`.

The contract intentionally excludes hidden hands and hidden dev cards.
It does include public board state, visible points, played knights,
current prompt, ports, robber location, and Longest Road/Largest Army.
