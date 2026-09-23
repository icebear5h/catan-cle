# Reports

Reviewed human-readable experiment findings live here, grouped by domain.
Machine plans, raw provider responses, and deterministic summaries live under
`artifacts/runs/` and remain the evidence behind each report.

A report should link its exact run directory and identify the dataset version,
model/provider settings, scoring version, verification performed, and known
limitations.

- `catan_board_bench/`: benchmark and format-evaluation findings.
- `evals/`: evaluation-tooling, workflow, and migration findings.
- `model_selection/`: model-family audits and Catan-policy selection research.
- `sft/`: supervised fine-tuning run records and conclusions.
- `inference/`: serving latency analysis and self-play pacing findings.
- `infrastructure/`: training/sandbox infrastructure research and translations.
