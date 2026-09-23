from typing import Any

import pytest

from data_pipeline.board_recognition.production_curriculum import (
    CURRICULUM_STAGES,
    ProductionCurriculumError,
    interleave_four_to_one,
    padding_needed,
    parse_stage_list,
)


def _records(prefix: str, count: int) -> list[dict[str, str]]:
    return [{"query_id": f"{prefix}-{index:03d}"} for index in range(count)]


def test_four_to_one_interleave_is_deterministic_and_lossless() -> None:
    base: Any = _records("base", 12)
    supplement: Any = _records("supplement", 3)

    first: Any = interleave_four_to_one(base, supplement, stage="clean_board_grounding")
    second = interleave_four_to_one(base, supplement, stage="clean_board_grounding")

    assert first == second
    assert len(first) == 15
    assert {row["query_id"] for row in first} == {row["query_id"] for row in base + supplement}
    assert [row["query_id"].split("-")[0] for row in first][4::5] == [
        "supplement",
        "supplement",
        "supplement",
    ]


def test_four_to_one_interleave_rejects_distribution_drift() -> None:
    with pytest.raises(ProductionCurriculumError, match="exact 4:1"):
        interleave_four_to_one(
            _records("base", 5),
            _records("supplement", 1),
            stage="pieces_and_colors",
        )


@pytest.mark.parametrize(
    ("rows", "expected"),
    [(0, 0), (1, 31), (1_032, 24), (3_885, 19), (3_405, 19), (8_070, 26)],
)
def test_padding_needed_aligns_stage_boundaries(rows: int, expected: int) -> None:
    assert padding_needed(rows) == expected


def test_padding_needed_accepts_an_explicit_alignment() -> None:
    assert padding_needed(1_032, multiple=8) == 0


def test_parse_stage_list_orders_and_rejects() -> None:
    assert parse_stage_list("pieces_and_colors,clean_board_grounding") == (
        "clean_board_grounding",
        "pieces_and_colors",
    )
    assert parse_stage_list(",".join(CURRICULUM_STAGES)) == CURRICULUM_STAGES
    with pytest.raises(ProductionCurriculumError):
        parse_stage_list("dense_boards")
    with pytest.raises(ProductionCurriculumError):
        parse_stage_list("pieces_and_colors,pieces_and_colors")
