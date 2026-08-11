# Training Data Pipeline

This directory holds model-training data builders and outputs.

## Layout

- `pretraining/`: corpus-style Catan text data for continued pretraining/CPT.
- `continual_learning/`: self-play, RL, and population-training planning notes.
- `schema.py`: older scraped-strategy data structures for expert decisions and
  strategy-corpus entries.

Scrapers and importers live in `data_pipeline/ingestion/`; benchmark data lives in
`data_pipeline/catanbench/`.
