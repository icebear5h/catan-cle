import copy

import pytest

from sft.analysis.marker_diagnostics import analyze, select_rows, selection_manifest


def source_rows() -> list[dict[str, object]]:
    rows = []
    positions = {"<N00>": [0, 0], "<N01>": [0, 1], "<N02>": [1, 1], "<E00_01>": [0, .5], "<E00_02>": [.5, .5]}
    markers = {"<N00>": "A", "<N01>": "B", "<N02>": "C", "<E00_01>": "A", "<E00_02>": "B"}
    for board in ["b0", "b1"]:
        for token, center in positions.items():
            for task in ["marker_to_token", "token_to_marker"]:
                rows.append({"row_id": f"{board}/{token}/{task}", "state_id": board,
                             "target_token": token, "task_type": task, "marker": markers[token],
                             "entity_type": "edge" if token.startswith("<E") else "node",
                             "images": [f"{board}/{token[1]}.png"], "marker_group": [t for t in positions if t[1] == token[1]],
                             "spatial_targets": [{"center": center}],
                             "messages": [{"role": "user", "content": "query"},
                                          {"role": "assistant", "content": token if task == "marker_to_token" else markers[token]}]})
    return rows


def record(row: dict[str, object], response: str | None = None) -> dict[str, object]:
    expected = row["messages"][1]["content"]
    return {"id": row["row_id"], "expected": expected, "response": (response or expected) + "<|im_end|>",
            "candidate_score": {"correct": True, "predicted": expected}}


def test_selection_is_deterministic_and_paired() -> None:
    rows = source_rows()
    selected = select_rows(rows)
    assert selected == select_rows(list(reversed(rows)))
    assert len(selected) == 10
    assert len(selection_manifest(rows, selected)["boards"]) == 2
    for a, b in zip(selected[::2], selected[1::2]):
        assert a["target_token"] == b["target_token"]
        assert a["images"] == b["images"]


def test_duplicates_and_missing_pairs_rejected() -> None:
    rows = source_rows()
    with pytest.raises(ValueError, match="duplicate"):
        select_rows(rows + [rows[0]])
    missing = select_rows(rows)[0]["row_id"]
    with pytest.raises(ValueError, match="missing"):
        select_rows([r for r in rows if r["row_id"] != missing])


def test_miss_modes_and_orientation() -> None:
    rows = source_rows()
    selected = select_rows(rows)
    records = [record(row) for row in selected]
    records[0] = record(selected[0], "<N00>")
    records[1] = record(selected[1], "B")
    records[2] = record(selected[2], "<E00_02><E00_02>")
    report = analyze(rows, selected, records)
    assert report["complete"]
    assert report["failure_modes"] == {"wrong_entity_type": 1, "wrong_marker_letter": 1, "repeated_token": 1}
    assert report["by"]["orientation"]["vertical"]["n"] == 2
    assert report["by"]["orientation"]["slanted"]["n"] == 2
    assert len(report["free_candidate_disagreement_ids"]) == 3
    assert report["failures"][1]["predicted_location"] == "<E00_02>"
    assert report["failures"][1]["other_marked_location"]


def test_partial_records_and_bad_ids() -> None:
    rows = source_rows()
    selected = select_rows(rows)
    assert not analyze(rows, selected, [])["complete"]
    first = record(selected[0])
    with pytest.raises(ValueError, match="duplicate"):
        analyze(rows, selected, [first, first])
    bad = copy.deepcopy(first)
    bad["expected"] = "bad"
    with pytest.raises(ValueError, match="target mismatch"):
        analyze(rows, selected, [bad])


def test_wrong_location_and_answer_type() -> None:
    rows = source_rows()
    selected = select_rows(rows)
    report = analyze(rows, selected, [record(selected[0], "<E00_02>"), record(selected[1], "<N00>")])
    assert report["failure_modes"] == {"wrong_location": 1, "wrong_answer_type": 1}


def test_mismatched_pair_images_rejected() -> None:
    rows = source_rows()
    chosen = select_rows(rows)[0]
    chosen["images"] = ["different.png"]
    with pytest.raises(ValueError, match="share a marked image"):
        select_rows(rows)


@pytest.mark.parametrize("response,mode", [("<E99_99>", "unknown_atlas_token"), ("road", "malformed_output"), ("A", "wrong_answer_type")])
def test_invalid_location_responses(response: str, mode: str) -> None:
    rows = source_rows()
    selected = select_rows(rows)
    report = analyze(rows, selected, [record(selected[0], response)])
    assert report["failure_modes"] == {mode: 1}
