# Ingestion

Scrapers and importers for source material that later feeds training or analysis.

Current modules:

- `youtube_scraper.py`: fetches transcripts for Catan corpus building.

Keep raw acquisition code here. Training transforms should live under
`data_pipeline/training/`; benchmark construction should live under
`data_pipeline/catan_board_bench/`. Acquisition caches and generated transcript/corpus
outputs belong under `artifacts/`, not beside the Python modules.
