"""
Data Pipeline - Training data generation and management for Catan LLM agents.

Modules:
- bootstrapping: Generate training data from expert Colonist.io replays
- catanbench: Engine-backed board-recognition benchmark data and evals
- training: Continued-pretraining and continual-learning datasets
- ingestion: External source ingestion helpers

Usage:
    from data_pipeline.bootstrapping import process_replay, save_training_examples
"""
