# Data pipeline

This package contains reusable acquisition and dataset-construction code. It is
not the storage root for raw payloads or model runs.

## Ownership

- `ingestion/`: external text/source acquisition.
- `bootstrapping/`: Colonist indexing, capture, decoding, and storage helpers.
- `training/`: corpus and training-dataset builders.
- `catan_board_bench/`: benchmark implementation and frozen benchmark inputs.

`catan_board_bench/` remains available through the top-level `catan_board_bench` compatibility
boundary. A future replay-core extraction is intentionally outside this cleanup.

## Repository artifacts

- `artifacts/raw/`: immutable captured inputs and source indexes.
- `artifacts/staging/`: unverified or rejected captures.
- `artifacts/manifests/`: deterministic split and queue manifests.
- `artifacts/fixtures/`: frozen integration fixtures.
- `artifacts/generated/`: generated corpora and intermediate datasets.
- `artifacts/runs/`: provider plans, raw responses, and machine summaries.
- `reports/`: reviewed human-readable findings.

Every retained run should keep its input hashes, exact settings, model/provider
identity, raw responses, and deterministic summary. Reports may cite runs but do
not replace their evidence.

## Temporary exceptions

Frozen CatanBoardBench inputs remain under `data_pipeline/catan_board_bench/datasets/` until
the benchmark package is extracted. Active narrator-reasoning pilots remain
under `data_pipeline/training/reasoning/pilots/` while a separate harness/UI
change owns those paths.

The old replay observation/action generator was removed because it applied the
target action before formatting the claimed pre-action observation. New replay
datasets must use the authoritative verified replay executor.
