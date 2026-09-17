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

## Symbolic board-fluency review

Open **http://localhost:5174/?tab=board-fluency** for the 200-example review:
40 examples per operator family, with family/operation filters, exact questions
and gold answers, source information, and shareable row links. The annotated
board is a human review aid; the model receives only the displayed symbolic input.

The bundle lives in `artifacts/generated/sft/symbolic_board_fluency_review_v1/`.
Build it from the repository root with:

```bash
PYTHONPATH=. .venv/bin/python sft/scripts/build_board_fluency_review.py
```

The builder reads at most 400 existing symbolic-v2 training rows, selects 200
distinct source states, checks their source contracts, and crosschecks answers.
Use `--verify-only` to check a byte-identical rebuild of an existing bundle.
Vite serves the three review artifacts directly and includes them in production
builds, so generate the bundle before building the eval UI.

### Quick text-only checkpoint evaluation

`sft.modal_board_fluency_eval` evaluates the unchanged 200 examples with greedy
generation, 512 completion tokens, and exact operation-aware scoring. It checks
the checkpoint/tokenizer and pinned base on CPU before a single H200 call capped
at 15 minutes. Run names must be new; receipts and predictions are downloaded to
`artifacts/runs/sft/<run-name>/`.

For the saved September 12 spatial checkpoint in the personal Modal workspace:

```bash
MODAL_PROFILE=icebear5h CATAN_HF_SECRET_NAME=huggingface-secret-2 \
  .venv/bin/python -m modal run -m sft.modal_board_fluency_eval \
  --adapter-dir /runs/catan-vision-sft/spatial-continuation-20260912-r01/checkpoints/checkpoint-128 \
  --run-name "spatial-fluency-$(date +%Y%m%d-%H%M%S)"
```

Use `--prepare-only` for CPU preparation. An HF bundle can instead be selected
with `--hf-repo` and an immutable `--hf-revision`; choose the matching Modal
workspace and HF secret. The review scorer compares unique unordered sets,
strict typed JSON, and integer/sentinel answers while retaining raw predictions.

The September 15 run of the latest saved spatial checkpoint scored **24/200
(12.0%)**. See [the result report](../reports/sft/2026-09-15-board-fluency-text-eval.md)
for grouped scores, raw-artifact paths, and error analysis.

The original September 2 HF checkpoint scored **16/200 (8.0%)** under the same
settings. See the [paired comparison](../reports/sft/2026-09-15-hf-spatial-board-fluency-comparison.md).
