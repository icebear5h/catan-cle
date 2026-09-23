import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from data_pipeline.board_recognition.full_board_readout import board_answer, row_for_state
from sft.board_state_readout import (
    board_keys,
    parse_state,
    score_board_state,
    select_validation,
    summarize_board_states,
)
from sft.scripts.eval.eval_qwen_vl_adapter import is_long_answer, score_response
from sft.scripts.train.train_trl_catan_vision import _message_pair

ROOT = Path("artifacts/fixtures/board_recognition/curriculum_smoke")


def fixture() -> dict[str, object]:
    states = [json.loads(line) for line in (ROOT / "manifest.jsonl").read_text().splitlines()]
    state = next(s for s in states if s["sample_id"] == "sparse_midgame_edge_p000_base")
    state = {**state, "density_bin": "sparse"}
    return state, json.loads((ROOT / state["contract_path"]).read_text())


def test_every_board_location_and_robber_match_source_contract() -> None:
    state, contract = fixture()
    row = row_for_state(state, contract)
    target = row["messages"][-1]["content"]
    parsed = parse_state(target)
    assert [key for key, _ in parsed["items"]] == board_keys()
    assert len(parsed["items"]) == 155
    for tile in contract["tiles"]:
        resource = str(tile["resource"]).lower() if tile["resource"] else "desert"
        number = str(tile["number"]) if tile["number"] is not None else "none"
        assert parsed["values"][tile["token"]] == f"{resource} {number}"
    for node in contract["nodes"]:
        expected = "empty" if node["building"] is None else (node["color"] + " " + node["building"]).lower().replace("_", " ")
        assert parsed["values"][node["token"]] == expected
    for edge in contract["edges"]:
        expected = "empty" if edge["road_color"] is None else edge["road_color"].lower().replace("_", " ") + " road"
        assert parsed["values"][edge["token"]] == expected
    assert parsed["values"]["robber"] == contract["robber"]["tile_token"]
    assert score_response(target, target)["correct"]
    assert is_long_answer(row)
    assert _message_pair(row, line_number=1)[1] == target
    assert "red settlement" not in row["messages"][0]["content"]


def test_duplicates_truncation_extras_and_order_are_not_exact() -> None:
    _, contract = fixture()
    target = board_answer(contract)
    entries = target.split("; ")
    for response in (target + "; " + entries[0], "; ".join(entries[:-1]), target + "; junk", target + "; <N99> empty"):
        assert not score_response(target, response)["correct"]
    duplicated = score_board_state(target, target + "; " + entries[0])
    assert duplicated["duplicates"] == ["<T00>"]
    assert duplicated["items_correct"] == 154
    swapped = score_board_state(target, "; ".join([entries[1], entries[0], *entries[2:]]))
    assert swapped["semantic_exact"] and not swapped["correct"]
    bad = copy.deepcopy(contract)
    bad["robber"]["tile_token"] = "<T99>"
    with pytest.raises(ValueError):
        board_answer(bad)


def test_all_empty_prediction_cannot_hide_missing_pieces() -> None:
    state, contract = fixture()
    target = board_answer(contract)
    response = "; ".join(f"{key} " + ("empty" if key.startswith(("<N", "<E")) else value)
                         for key, value in parse_state(target)["items"])
    score = score_board_state(target, response)
    assert score["occupied_items_total"] == 21
    assert score["occupied_items_correct"] == 0
    assert score["empty_items_correct"] == score["empty_items_total"]
    summary = summarize_board_states([{"score": score, "metadata": {"density_bin": state["density_bin"]}}])
    assert summary["groups"]["road"]["accuracy"] == 0
    assert summary["groups"]["node_occupied"]["accuracy"] == 0
    assert summary["groups"]["tile_resource"]["accuracy"] == 1


def test_full_board_training_guard_rejects_incomplete_targets() -> None:
    state, contract = fixture()
    row = row_for_state(state, contract)
    row["messages"][-1]["content"] = row["messages"][-1]["content"].rsplit(";", 1)[0]
    with pytest.raises(ValueError, match="invalid canonical"):
        _message_pair(row, line_number=1)


def test_validation_sample_covers_occupied_boards_in_every_layout() -> None:
    rows = [{"row_id": f"{layout}-{density}-{i}", "layout_id": layout, "density_bin": density,
             "image": f"{layout}-{density}-{i}.png", "answer": f"state-{layout}-{density}-{i}"}
            for layout in ("a", "b", "c", "d", "e")
            for density in ("empty", "setup", "sparse", "dense") for i in range(3)]
    selected = select_validation(rows, 16)
    counts = Counter(r["layout_id"] for r in selected)
    assert sorted(counts.values()) == [3, 3, 3, 3, 4]
    for layout in counts:
        assert {r["density_bin"] for r in selected if r["layout_id"] == layout} >= {"dense", "sparse", "setup"}
    assert selected == select_validation(list(reversed(rows)), 16)
    assert selected != select_validation(rows, 16, seed=44)
    assert all(r in rows for r in selected)
    assert len({r["row_id"] for r in select_validation(rows, len(rows))}) == len(rows)
    with pytest.raises(ValueError):
        select_validation(rows, len(rows) + 1)


def test_layout_macro_does_not_overweight_one_layout() -> None:
    _, contract = fixture()
    answer = board_answer(contract)
    good = score_board_state(answer, answer)
    bad = score_board_state(answer, "")
    records = [{"score": good, "metadata": {"layout_id": "a"}} for _ in range(9)]
    records.append({"score": bad, "metadata": {"layout_id": "b"}})
    summary = summarize_board_states(records)
    assert summary["occupied_layout_macro_accuracy"] == 0.5
    assert summary["by_layout"]["a"]["boards"] == 9
