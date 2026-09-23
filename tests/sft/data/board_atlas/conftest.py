"""Shared fixtures for board atlas fact tables, sampling, and scoring contracts."""

import pytest

from sft.board.board_atlas import (
    FactTable,
    build_fact_tables,
)


@pytest.fixture(scope="module")
def tables() -> dict[str, FactTable]:
    return build_fact_tables()
