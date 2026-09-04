# Data pipeline

This package contains reusable acquisition and dataset-construction code. It is
not the storage root for raw payloads, benchmark inputs, or model runs.

## Ownership

- `ingestion/`: external text/source acquisition.
- `bootstrapping/`: Colonist indexing, capture, decoding, and storage helpers.
- `training/`: corpus and training-dataset builders.
- `board_recognition/`: board-recognition training-data loaders and projections.

CatanBoardBench implementation and frozen evaluation inputs live under
`evals/catan_board_bench/`. Data-pipeline code imports that canonical package
when it needs benchmark renderers, token definitions, or leakage ledgers.

## Repository artifacts

- `artifacts/raw/`: immutable captured inputs and source indexes.
- `artifacts/staging/`: unverified or rejected captures.
- `artifacts/manifests/`: deterministic split and queue manifests.
- `artifacts/fixtures/`: frozen integration fixtures.
- `artifacts/generated/`: generated corpora and intermediate datasets.
- `artifacts/runs/`: provider plans, raw responses, and machine summaries.
- `reports/`: reviewed human-readable findings.

Every retained run should keep its input hashes, exact settings, model/provider
identity, raw responses, and deterministic summary. Reports may cite runs but
do not replace their evidence.

Active narrator-reasoning pilots remain under
`data_pipeline/training/reasoning/pilots/` while a separate harness/UI change
owns those paths.

The old replay observation/action generator was removed because it applied the
target action before formatting the claimed pre-action observation. New replay
datasets must use the authoritative verified replay executor.
