import copy
import json
import re
from pathlib import Path

import pytest

from sft.scripts import failure_scorecard
from sft.scripts.failure_scorecard import eval_jsonl_for, main

FIXTURE = Path("artifacts/fixtures/board_recognition/curriculum_smoke")
STATE_ID = "empty_setup_node_p000_base"
SET_ID = "replay_v1-evals-validation_v1"
PAIRS_SET_ID = "spatial_localization_pairs_v2-stage1-validation"

# Red settlement at N01, red road on the touching edge E01_02, blue city at N40.
PIECES = {"nodes": {"<N01>": ("SETTLEMENT", "RED"), "<N40>": ("CITY", "BLUE")}, "edges": {"<E01_02>": "RED"}}
# (prompt, category, expected, response): one planted miss per mode plus one correct row.
CASES = [
    ("<N01> building?", "node.occupancy", "red settlement", "empty"),  # blindness
    ("<N02> building?", "node.occupancy", "empty", "red settlement"),  # neighbor: N02 is one hop from N01
    ("<E30_31> road?", "edge.owner", "empty", "red road"),  # far: the red road is five hops away (slanted edge)
    ("<T05> number?", "tile.number", "4", "wheat"),  # head flip: a resource word on the number head
    ("Where is the blue city?", "inverse.node", "<N40>", "<N40><N40>"),  # token glitch: doubled token
    ("<E01_02> road?", "edge.owner", "red road", "red road"),  # correct, vertical edge
]


@pytest.fixture
def fixture_contract() -> dict:
    for line in (FIXTURE / "manifest.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if entry["contract_path"].endswith(f"{STATE_ID}.json"):
            return json.loads((FIXTURE / entry["contract_path"]).read_text())
    raise AssertionError(f"{STATE_ID} missing from fixture manifest")


def jsonl(items: list[dict]) -> str:
    return "".join(json.dumps(item) + "\n" for item in items)


def write_contract(tmp_path: Path, contract: dict, *, pieces: dict) -> Path:
    board = copy.deepcopy(contract)
    for node in board["nodes"]:
        if node["token"] in pieces["nodes"]:
            node["building"], node["color"] = pieces["nodes"][node["token"]]
    for edge in board["edges"]:
        if edge["token"] in pieces["edges"]:
            edge["road_color"] = pieces["edges"][edge["token"]]
    contracts_dir = tmp_path / "contracts"
    contracts_dir.mkdir(exist_ok=True)
    (contracts_dir / f"{STATE_ID}.json").write_text(json.dumps(board))
    return contracts_dir


def write_set(panel_dir: Path, set_id: str, records: list[dict], *, variants: tuple[str, ...] = ("original", "blank")) -> None:
    for variant in variants:
        variant_dir = panel_dir / f"regression-panel-{set_id}-eval" / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        (variant_dir / "records.jsonl").write_text(jsonl(records))
        (variant_dir / "summary.json").write_text(json.dumps({"eval_set_id": set_id, "image_variant": variant}))


def record_for(row_id: str, index: int, expected: str, response: str, metadata: dict) -> dict:
    return {
        "id": row_id,
        "index": index,
        "expected": expected,
        "response": f"{response}<|im_end|>",
        "score": {"correct": expected == response, "expected_normalized": expected, "response_normalized": response},
        "candidate_score": None,
        "metadata": metadata,
    }


def write_panel(tmp_path: Path, contract: dict, cases: list[tuple[str, str, str, str]]) -> list[str]:
    """A replay-shaped panel set (original and blank variants) plus its eval jsonl and contract."""

    contracts_dir = write_contract(tmp_path, contract, pieces=PIECES)
    records, rows = [], []
    for index, (prompt, category, expected, response) in enumerate(cases, start=1):
        row_id = f"{STATE_ID}_q{index:03d}"
        metadata = {"category": category, "density_bin": "sparse", "state_id": STATE_ID}
        messages = [{"role": "user", "content": f"<image>\n{prompt}"}, {"role": "assistant", "content": expected}]
        rows.append({"id": row_id, "images": [f"{STATE_ID}.png"], "messages": messages, "metadata": metadata})
        records.append(record_for(row_id, index, expected, response, {**metadata, "eval_set_id": SET_ID, "eval_variant": "original"}))
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(jsonl(rows))
    panel_dir = tmp_path / "panel"
    write_set(panel_dir, SET_ID, records)
    return ["--panel-dir", str(panel_dir), "--contracts-dir", str(contracts_dir), "--set", f"{SET_ID}={eval_path}"]


def test_counts_one_planted_miss_per_mode(tmp_path: Path, fixture_contract: dict, capsys) -> None:
    output = tmp_path / "scorecard.json"
    assert main(write_panel(tmp_path, fixture_contract, CASES) + ["--output", str(output), "--quiet"]) == 0
    assert capsys.readouterr().out == ""
    report = json.loads(output.read_text())
    assert report["schema"] == "catan_failure_scorecard/v1"
    assert list(report["sets"]) == [SET_ID]
    assert report["row_entanglement"] is None
    entry = report["sets"][SET_ID]
    assert entry["counts"] == {
        "rows": 6,
        "errors": 5,
        "blindness": 1,
        "neighbor_confusion": 1,
        "far_false_positive": 1,
        "other_occupancy_miss": 0,
        "head_flip": 1,
        "token_glitch": 1,
        "orientation_ratio": 0.0,
        "colour_dropout_min_recall": 0.5,
    }
    modes = entry["modes"]
    assert modes["blindness"] == {"count": 1, "by_piece": {"settlement": 1}, "by_color": {"red": 1}}
    assert modes["neighbor_confusion"]["by_class"] == {"neighbor_false_positive_hop1": 1}
    assert modes["far_false_positive"]["by_class"] == {"false_positive_elsewhere": 1}
    assert modes["head_flip"] == {"count": 1, "by_head": {"number->resource": 1}, "tokens_with_3plus": {}}
    assert modes["token_glitch"] == {"count": 1, "by_kind": {"multiple_tokens": 1}, "tokens": {"<N40>": 1}}
    assert modes["orientation"] == {
        "vertical": {"errors": 0, "n": 1, "error_rate": 0.0},
        "slanted": {"errors": 1, "n": 1, "error_rate": 1.0},
        "ratio": 0.0,
    }
    assert modes["colour_dropout"] == {
        "recall_by_color": {"red": {"correct": 1, "total": 2, "recall": 0.5}},
        "min_color": "red",
        "min_recall": 0.5,
    }
    assert entry["occupied_recall_by_piece"] == {
        "road": {"correct": 1, "total": 1, "recall": 1.0},
        "settlement": {"correct": 0, "total": 1, "recall": 0.0},
    }
    per_token = entry["per_token"]
    assert set(per_token) == {"<N01>", "<N02>", "<E30_31>", "<T05>", "<N40>", "<E01_02>"}
    assert per_token["<N01>"]["dominant_wrong"] == "empty" and per_token["<N01>"]["error_rate"] == 1.0
    assert per_token["<N02>"]["dominant_wrong"] == "red settlement"
    assert per_token["<T05>"]["head_flips"] == 1
    assert per_token["<N40>"]["glitches"] == 1 and per_token["<N40>"]["dominant_wrong"] == "<N40><N40>"
    assert per_token["<E30_31>"]["orientation"] == "slanted" and per_token["<E30_31>"]["dominant_wrong"] == "red road"
    assert per_token["<E01_02>"] == {
        "n": 1, "errors": 0, "error_rate": 0.0, "dominant_wrong": None, "head_flips": 0, "glitches": 0, "orientation": "vertical",
    }


def test_synthetic_pair_rows_use_the_placed_pieces(tmp_path: Path, fixture_contract: dict) -> None:
    # The base contract is empty; only the row's target and partner pieces exist on the synthetic board.
    contracts_dir = write_contract(tmp_path, fixture_contract, pieces={"nodes": {}, "edges": {}})
    pair = {
        "state_id": STATE_ID, "target_token": "<N01>", "piece": "SETTLEMENT", "color": "red",
        "partner_token": "<N02>", "partner_piece": "CITY", "partner_color": "green", "partner_distance": 1,
    }
    cases = [
        ("<N01>", "occupancy_positive", "node.occupancy", "red settlement", "green city"),  # names the partner
        ("<N30>", "occupancy_negative_far", "node.occupancy", "empty", "red settlement"),  # target is five hops away
        ("<N06>", "occupancy_negative_adjacent", "node.occupancy", "empty", "empty"),  # correct
    ]
    records, rows = [], []
    for index, (token, task_type, category, expected, response) in enumerate(cases, start=1):
        row_id = f"{STATE_ID}_pair_{index}"
        row = {**pair, "row_id": row_id, "task_type": task_type, "category": category, "queried_token": token}
        row["messages"] = [{"role": "user", "content": f"<image>\n{token} building?"}, {"role": "assistant", "content": expected}]
        rows.append(row)
        metadata = {"category": category, "task_type": task_type, "eval_set_id": PAIRS_SET_ID, "eval_variant": "original"}
        records.append(record_for(row_id, index, expected, response, metadata))
    eval_path = tmp_path / "pairs.jsonl"
    eval_path.write_text(jsonl(rows))
    write_set(tmp_path / "panel", PAIRS_SET_ID, records, variants=("original",))
    output = tmp_path / "scorecard.json"
    args = ["--panel-dir", str(tmp_path / "panel"), "--contracts-dir", str(contracts_dir), "--set", f"{PAIRS_SET_ID}={eval_path}"]
    assert main(args + ["--output", str(output), "--quiet"]) == 0
    entry = json.loads(output.read_text())["sets"][PAIRS_SET_ID]
    assert entry["counts"]["blindness"] == 0
    assert entry["modes"]["neighbor_confusion"] == {"count": 1, "by_class": {"wrong_piece_neighbor_hop1": 1}}
    assert entry["modes"]["far_false_positive"] == {"count": 1, "by_class": {"false_positive_elsewhere": 1}}
    assert entry["modes"]["colour_dropout"]["recall_by_color"] == {"red": {"correct": 0, "total": 1, "recall": 0.0}}
    assert entry["occupied_recall_by_piece"] == {"settlement": {"correct": 0, "total": 1, "recall": 0.0}}


def test_baseline_deltas_are_reported(tmp_path: Path, fixture_contract: dict, capsys) -> None:
    first = tmp_path / "first"
    first.mkdir()
    baseline = first / "scorecard.json"
    assert main(write_panel(first, fixture_contract, CASES) + ["--output", str(baseline), "--quiet"]) == 0
    # Next checkpoint: the blind spot is fixed, but the number head keeps flipping to resources on T05.
    fixed = [(prompt, category, expected, expected if prompt.startswith("<N01>") else response) for prompt, category, expected, response in CASES]
    fixed += [("<T05> number?", "tile.number", "4", "ore")] * 3
    second = tmp_path / "second"
    second.mkdir()
    output = second / "scorecard.json"
    assert main(write_panel(second, fixture_contract, fixed) + ["--output", str(output), "--baseline", str(baseline)]) == 0
    report = json.loads(output.read_text())
    assert report["baseline"] == str(baseline)
    assert report["deltas"][SET_ID]["blindness"] == -1
    assert report["deltas"][SET_ID]["head_flip"] == 3
    assert report["deltas"][SET_ID]["rows"] == 3
    assert report["deltas"][SET_ID]["errors"] == 2
    assert report["sets"][SET_ID]["modes"]["head_flip"]["tokens_with_3plus"] == {"<T05>": 4}
    err = capsys.readouterr().err
    assert err.startswith(f"{SET_ID}  rows 9  errors 7")
    assert re.search(r"blindness\s+0\s+\(-1\)", err)
    assert re.search(r"head_flip\s+4\s+\(\+3\)", err)
    assert re.search(r"colour_dropout_min_recall\s+1\.000\s+\(\+0\.500\)", err)


def test_row_entanglement_uses_the_inspector_when_available(tmp_path: Path, fixture_contract: dict, monkeypatch) -> None:
    calls = []

    def fake_inspect(path, tokens, twin_threshold=0.25, family_floor=0.05):
        calls.append((path, tuple(tokens)))
        rows = {"tokens_with_twin": ["<N01>", "<N02>"], "twins": {"<N01>": "<N02>"}, "rows_below_family_floor": ["<E00_01>"], "own_family_loading": {}}
        return {"input": rows, "output": {"tokens_with_twin": [], "twins": {}, "rows_below_family_floor": [], "own_family_loading": {}}}

    monkeypatch.setattr(failure_scorecard, "inspect_adapter", fake_inspect)
    inventory = tmp_path / "tokens.json"
    inventory.write_text(json.dumps({"atlas_tokens": ["<N01>", "<N02>", "<E00_01>"]}))
    output = tmp_path / "scorecard.json"
    args = write_panel(tmp_path, fixture_contract, CASES)
    args += ["--output", str(output), "--adapter", str(tmp_path / "adapter"), "--token-inventory", str(inventory)]
    assert main(args) == 0
    assert calls == [(tmp_path / "adapter", ("<N01>", "<N02>", "<E00_01>"))]
    rows = json.loads(output.read_text())["row_entanglement"]
    assert rows["available"] is True
    assert rows["tokens_with_twin_count"] == 2 and rows["rows_below_family_floor_count"] == 1
    assert rows["twins"] == {"<N01>": "<N02>"}
    monkeypatch.setattr(failure_scorecard, "inspect_adapter", None)
    assert main(args) == 0
    assert json.loads(output.read_text())["row_entanglement"]["available"] is False


def test_eval_jsonl_for_inverts_set_ids(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "replay_v1"
    for relative in ("spatial_localization_v3/stage1/validation.jsonl", "evals/validation_v1.jsonl"):
        (root / relative).parent.mkdir(parents=True)
        (root / relative).write_text("")
    monkeypatch.setattr(failure_scorecard, "REPLAY_ROOT", root)
    assert eval_jsonl_for("spatial_localization_v3-stage1-validation", {}) == root / "spatial_localization_v3/stage1/validation.jsonl"
    assert eval_jsonl_for("replay_v1-evals-validation_v1", {}) == root / "evals/validation_v1.jsonl"
    override = tmp_path / "custom.jsonl"
    assert eval_jsonl_for("anything", {"anything": override}) == override
    with pytest.raises(FileNotFoundError, match="--set"):
        eval_jsonl_for("spatial_localization_v9-stage1-validation", {})
