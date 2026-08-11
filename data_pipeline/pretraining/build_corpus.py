"""Compatibility wrapper for the moved pretraining corpus builder."""

from data_pipeline.training.pretraining.build_corpus import *  # noqa: F401,F403
from data_pipeline.training.pretraining.build_corpus import main


if __name__ == "__main__":
    main()

