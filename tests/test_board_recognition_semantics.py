import pytest

from data_pipeline.board_recognition.semantics import (
    QUERY_TEXT_BY_HEAD,
    semantic_answer,
    semantic_candidates,
    semantic_contract,
    semantic_prompt,
)
from evals.catan_board_bench.tokens import (
    RECOGNITION_CLASS_VOCABULARIES,
    atlas_tokens,
    recognition_token_inventory,
    semantic_recognition_token_inventory,
)


def test_semantic_projection_trains_154_bidirectional_location_rows():
    inventory = semantic_recognition_token_inventory()

    assert inventory["counts"] == {
        "node": 54,
        "edge": 72,
        "tile": 19,
        "port": 9,
        "atlas": 154,
        "total": 154,
    }
    assert inventory["tokens"] == inventory["atlas_tokens"] == atlas_tokens()
    assert len(inventory["tokens"]) == len(set(inventory["tokens"])) == 154
    assert inventory["trainable_side"] == "input_and_output_rows"
    assert not any(token.startswith(("<Q_", "<A_")) for token in inventory["tokens"])


def test_historical_220_token_inventory_is_unchanged():
    inventory = recognition_token_inventory()

    assert inventory["counts"] == {"atlas": 154, "query": 6, "answer": 60, "total": 220}
    assert inventory["tokens"][:154] == atlas_tokens()
    assert len(inventory["tokens"]) == 220


@pytest.mark.parametrize(
    ("head", "slot", "expected"),
    [
        ("tile.resource", "<T07>", "<T07> resource?"),
        ("tile.number", "<T07>", "<T07> number?"),
        ("tile.robber", "<T07>", "<T07> robber here?"),
        ("node.occupancy", "<N17>", "<N17> building?"),
        ("edge.owner", "<E17_39>", "<E17_39> road?"),
        ("port.port_type", "<P03>", "<P03> port?"),
    ],
)
def test_semantic_prompts_use_one_location_token_and_ordinary_language(head, slot, expected):
    prompt = semantic_prompt(head, slot)

    assert prompt == expected
    assert prompt.count("<") == prompt.count(">") == 1
    assert "<Q_" not in prompt


@pytest.mark.parametrize(
    ("head", "class_name", "expected"),
    [
        ("tile.resource", "SHEEP", "sheep"),
        ("tile.number", "NONE", "none"),
        ("tile.number", "12", "12"),
        ("tile.robber", "PRESENT", "yes"),
        ("tile.robber", "ABSENT", "no"),
        ("node.occupancy", "EMPTY", "empty"),
        ("node.occupancy", "MYSTIC_BLUE_CITY", "mystic blue city"),
        ("edge.owner", "RED", "red road"),
        ("edge.owner", "MYSTIC_BLUE", "mystic blue road"),
        ("port.port_type", "THREE_TO_ONE", "3:1 port"),
        ("port.port_type", "TWO_TO_ONE_ORE", "ore port"),
    ],
)
def test_semantic_answers_are_controlled_phrases(head, class_name, expected):
    assert semantic_answer(head, class_name) == expected


def test_every_engine_class_has_one_unique_legal_answer():
    contract = semantic_contract()

    assert set(contract["heads"]) == set(QUERY_TEXT_BY_HEAD)
    for head, classes in RECOGNITION_CLASS_VOCABULARIES.items():
        candidates = semantic_candidates(head)
        assert len(candidates) == len(classes) == len(set(candidates))
        assert contract["heads"][head]["candidate_answers"] == list(candidates)
        assert [row["class_name"] for row in contract["heads"][head]["classes"]] == list(
            classes
        )


def test_semantic_prompt_and_answer_reject_cross_head_or_unknown_values():
    with pytest.raises(ValueError, match="invalid for node.occupancy"):
        semantic_prompt("node.occupancy", "<T07>")
    with pytest.raises(ValueError, match="invalid for tile.resource"):
        semantic_prompt("tile.resource", "<T99>")
    with pytest.raises(ValueError, match="unknown class"):
        semantic_answer("edge.owner", "PURPLE")
    with pytest.raises(ValueError, match="unknown board-recognition head"):
        semantic_candidates("node.resource")
