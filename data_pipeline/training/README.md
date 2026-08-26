# Training Data Pipeline

This directory holds model-training data builders and outputs.

## Layout

- `pretraining/`: corpus-style Catan text builders for continued pretraining/CPT.
- `reasoning/`: active reasoning-data workspaces that have not yet been moved
  while a separate harness/UI change owns their paths.

Scrapers and importers live in `data_pipeline/ingestion/`; benchmark code and
frozen benchmark inputs live in `data_pipeline/catan_board_bench/`. Generated outputs
belong under the repository-level `artifacts/` tree.
