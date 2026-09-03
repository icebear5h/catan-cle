import copy
import json
from pathlib import Path

import pytest

from sft.scripts.analyze_occupancy_misses import analyze, main

FIXTURE = Path("artifacts/fixtures/board_recognition/curriculum_smoke")
STATE_ID = "empty_setup_node_p000_base"


@pytest.fixture
def fixture_contract() -> dict:
    for line in (FIXTURE / "manifest.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if entry["contract_path"].endswith(f"{STATE_ID}.json"):
            return json.loads((FIXTURE / entry["contract_path"]).read_text())
    raise AssertionError(f"{STATE_ID} missing from fixture manifest")


def place(contract: dict, *, nodes: dict[str, tuple[str, str]], edges: dict[str, str]) -> dict:
    contract = copy.deepcopy(contract)
    for node in contract["nodes"]:
        if node["token"] in nodes:
            node["building"], node["color"] = nodes[node["token"]]
    for edge in contract["edges"]:
        if edge["token"] in edges:
            edge["road_color"] = edges[edge["token"]]
    return contract


def rows(cases: list[tuple[str, str, str]], *, density: str) -> tuple[list[dict], list[dict]]:
    records, eval_rows = [], []
    for index, (token, expected, response) in enumerate(cases, start=1):
        row_id = f"{STATE_ID}_q{index:03d}"
        category = "node.occupancy" if token.startswith("<N") else "edge.owner"
        metadata = {"category": category, "density_bin": density, "state_id": STATE_ID}
        noun = "building" if token.startswith("<N") else "road"
        eval_rows.append(
            {
                "id": row_id,
                "images": [f"{STATE_ID}.png"],
                "messages": [
                    {"role": "user", "content": f"<image>\n{token} {noun}?"},
                    {"role": "assistant", "content": expected},
                ],
                "metadata": metadata,
            }
        )
        records.append(
            {
                "id": row_id,
                "index": index,
                "expected": expected,
                "response": f"{response}<|im_end|>",
                "score": {
                    "correct": expected == response,
                    "expected_normalized": expected,
                    "response_normalized": response,
                },
                "candidate_score": None,
                "metadata": metadata,
            }
        )
    return records, eval_rows


# N01 touches E00_01, E01_02, E01_06; N02 is one hop from N01; N06 is one hop too.
# E01_02 (red road) shares N01 with E00_01 and N02 with E02_03/E02_07.
CASES = [
    ("<N01>", "red settlement", "empty"),  # false_negative
    ("<N02>", "empty", "red settlement"),  # neighbor_false_positive_hop1
    ("<E00_01>", "empty", "red settlement"),  # cross_type_false_positive (touches N01)
    ("<N01>", "red settlement", "blue settlement"),  # right_type_wrong_color
    ("<E01_02>", "red road", "red road"),  # correct
    ("<N30>", "empty", "blue city"),  # false_positive_elsewhere (blue city sits at N40)
    ("<N31>", "empty", "green road"),  # false_positive_absent
    ("<E01_02>", "red road", "orange city"),  # wrong_piece_other
]


def write_inputs(tmp_path: Path, contract: dict) -> tuple[Path, Path, Path]:
    board = place(contract, nodes={"<N01>": ("SETTLEMENT", "RED"), "<N40>": ("CITY", "BLUE")}, edges={"<E01_02>": "RED"})
    contracts_dir = tmp_path / "contracts"
    contracts_dir.mkdir()
    (contracts_dir / f"{STATE_ID}.json").write_text(json.dumps(board))
    records, eval_rows = rows(CASES, density="sparse")
    records_path = tmp_path / "records.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    records_path.write_text("".join(json.dumps(item) + "\n" for item in records))
    eval_path.write_text("".join(json.dumps(item) + "\n" for item in eval_rows))
    return records_path, eval_path, contracts_dir


def test_classifies_each_miss_kind(tmp_path: Path, fixture_contract: dict) -> None:
    records_path, eval_path, contracts_dir = write_inputs(tmp_path, fixture_contract)
    report = analyze(
        [json.loads(line) for line in records_path.read_text().splitlines()],
        [json.loads(line) for line in eval_path.read_text().splitlines()],
        contracts_dir,
    )
    assert report["total"] == len(CASES)
    assert report["misses"] == len(CASES) - 1
    assert report["by_class"] == {
        "false_negative": 1,
        "neighbor_false_positive_hop1": 1,
        "cross_type_false_positive": 1,
        "false_positive_elsewhere": 1,
        "false_positive_absent": 1,
        "right_type_wrong_color": 1,
        "wrong_piece_other": 1,
    }
    assert report["by_class_by_density"]["false_negative"] == {"sparse": 1}
    assert report["occupied_recall_by_color"] == {"RED": {"correct": 1, "total": 4, "recall": 0.25}}
    assert report["occupied_recall_by_piece"] == {
        "ROAD": {"correct": 1, "total": 2, "recall": 0.5},
        "SETTLEMENT": {"correct": 0, "total": 2, "recall": 0.0},
    }
    # N01 touches the red road; E01_02 touches the red settlement: one touching piece each.
    assert report["occupied_recall_by_touching"] == {"1": {"correct": 1, "total": 4, "recall": 0.25}}
    assert report["accuracy_by_category_density"]["edge.owner/sparse"] == {"correct": 1, "total": 3, "accuracy": round(1 / 3, 4)}
    assert report["accuracy_by_category_density"]["node.occupancy/sparse"] == {"correct": 0, "total": 5, "accuracy": 0.0}
    example = report["examples"]["cross_type_false_positive"][0]
    assert example["token"] == "<E00_01>" and example["response"] == "red settlement"


def test_cli_writes_output_and_skips_other_categories(tmp_path: Path, fixture_contract: dict, capsys) -> None:
    records_path, eval_path, contracts_dir = write_inputs(tmp_path, fixture_contract)
    with records_path.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "id": "tile", "index": 99, "expected": "ore", "response": "ore",
                    "score": {"correct": True, "expected_normalized": "ore", "response_normalized": "ore"},
                    "metadata": {"category": "tile.resource", "density_bin": "sparse", "state_id": STATE_ID},
                }
            )
            + "\n"
        )
    output = tmp_path / "report.json"
    args = ["--records", str(records_path), "--eval-jsonl", str(eval_path), "--contracts-dir", str(contracts_dir), "--output", str(output)]
    assert main(args) == 0
    captured = capsys.readouterr()
    assert json.loads(output.read_text()) == json.loads(captured.out)
    assert json.loads(captured.out)["total"] == len(CASES)
    assert "miss classes:" in captured.err
