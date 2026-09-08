import copy
import json
from collections import Counter

import pytest

from data_pipeline.board_recognition.node_edge_readout import EXPORT_SCHEMA, rows_for_state
from data_pipeline.board_recognition.replay_dataset import file_sha256
from data_pipeline.board_recognition.reweight_node_edge import (
    ALL_COLORS, EVAL_SPLITS, MixConfig, answer, apportion, audit, export_reweighted,
    group, parse_readout, resample, training_pool,
)
from data_pipeline.board_recognition.spatial_localization import _write_json, _write_jsonl


@pytest.fixture
def source_rows():
    from pathlib import Path

    root = Path("artifacts/fixtures/board_recognition/curriculum_smoke")
    manifest = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    state = next(r for r in manifest if r["sample_id"] == "sparse_midgame_edge_p000_base")
    contract = json.loads((root / state["contract_path"]).read_text())
    result = []
    for index, colour in enumerate(ALL_COLORS):
        board = copy.deepcopy(contract)
        for node in board["nodes"]:
            if node["building"]:
                node["color"] = colour
        for edge in board["edges"]:
            if edge["road_color"]:
                edge["road_color"] = colour
        rows = rows_for_state(dict(state, sample_id=f"train_{index}", split="train"), board, full_coverage=False)
        result.extend(rows)
    return result


def count_tokens(text):
    # A cheap deterministic test tokenizer; integration below uses tokenizers.
    return len(text.split()) + 2


def test_pool_expands_all_existing_labels_without_mutating_inputs(source_rows):
    before = copy.deepcopy(source_rows)
    pool = training_pool(source_rows)
    by_board = {(r["state_id"], r["task_type"].removesuffix("_readout")): parse_readout(r)
                for r in source_rows if group(r) == "readout"}
    positives = [r for r in pool if group(r) == "occupied"]
    assert len(positives) == sum(value != "empty" for b in by_board.values() for value in b.values())
    assert len(positives) > sum(group(r) == "occupied" for r in source_rows)
    assert len({r["row_id"] for r in pool}) == len(pool)
    for row in positives:
        assert answer(row) == by_board[(row["state_id"], row["entity_type"])][row["target_token"]]
    assert source_rows == before


@pytest.mark.parametrize("fault", ["eval", "duplicate", "label", "metadata", "incomplete", "doubled_token"])
def test_pool_rejects_bad_inputs(source_rows, fault):
    rows = copy.deepcopy(source_rows)
    if fault == "eval":
        rows[0]["split"] = "validation"
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "label":
        rows[0]["messages"][1]["content"] = "not the answer"
    elif fault == "metadata":
        next(r for r in rows if group(r) == "occupied")["color"] = "INVALID"
    elif fault == "incomplete":
        rows = [r for r in rows if r["task_type"] != "edge_readout"]
    else:
        row = next(r for r in rows if group(r) == "readout")
        row["messages"][1]["content"] += "; " + answer(row).split("; ")[0]
    with pytest.raises(ValueError):
        training_pool(rows)


def test_token_budget_balances_colour_piece_and_preserves_full_answers(source_rows):
    pool = training_pool(source_rows)
    config = MixConfig(max_repeats=1000)
    sampled, recipe = resample(pool, count_tokens, row_count=30000, config=config)
    again, _ = resample(list(reversed(pool)), count_tokens, row_count=30000, config=config)
    assert sampled == again
    assert len(sampled) == 30000 == len({r["row_id"] for r in sampled})
    result = audit(sampled, count_tokens)
    for kind, target in [("occupied", 0.5), ("empty", 0.25), ("readout", 0.25)]:
        assert result["completion_token_shares"][kind] == pytest.approx(target, abs=0.01)
    for piece in ("ROAD", "CITY", "SETTLEMENT"):
        masses = [result["short_positive_colour_piece_tokens"][f"{c}/{piece}"] for c in ALL_COLORS]
        assert max(masses) - min(masses) <= 7
    sources = {r["row_id"]: r for r in pool}
    for row in sampled:
        assert row["messages"] == sources[row["source_query_id"]]["messages"]
    assert recipe["unique_selected_queries"] <= len(pool)
    assert Counter(r["source_query_id"] for r in sampled).most_common(1)[0][1] == recipe["max_query_repeats"]


@pytest.mark.parametrize("config", [MixConfig(occupied_share=float("nan")), MixConfig(empty_share=-1),
                                   MixConfig(readout_share=0), MixConfig(occupied_share=0.6), MixConfig(max_repeats=0)])
def test_invalid_config_fails(config):
    with pytest.raises(ValueError):
        config.validate()


def test_missing_buckets_and_repeat_cap_fail_closed(source_rows):
    pool = training_pool(source_rows)
    with pytest.raises(ValueError, match="missing colour x piece"):
        resample([r for r in pool if r["color"] != "WHITE"], count_tokens, row_count=30000)
    with pytest.raises(ValueError, match="max_repeats"):
        resample(pool, count_tokens, row_count=30000)
    with pytest.raises(ValueError, match="drops stratum"):
        resample(pool, count_tokens, row_count=1)
    with pytest.raises(ValueError, match="missing empty/node"):
        resample([r for r in pool if not (group(r) == "empty" and r["entity_type"] == "node")], count_tokens, row_count=30000)


def test_apportion_is_exact_and_stable():
    assert apportion({"b": 1.0, "a": 1.0, "c": 1.0}, 5) == {"b": 2, "a": 2, "c": 1}


def _export_fixture(tmp_path, source_rows):
    from tokenizers import Tokenizer, models, pre_tokenizers

    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    for row in source_rows:
        (source / "images" / row["images"][0]).write_bytes(b"test-image")
    splits = {"train": source_rows}
    for split in EVAL_SPLITS:
        splits[split] = [dict(row, split=split, layout_id=f"{split}-layout", state_id=f"{split}-state")
                         for row in source_rows[:18]]
    files = {}
    for split, rows in splits.items():
        path = source / "stage1" / f"{split}.jsonl"
        _write_jsonl(path, rows)
        files[f"stage1/{split}.jsonl"] = {"sha256": file_sha256(path)}
    _write_json(source / "metadata.json", {"schema": EXPORT_SCHEMA, "files": files})
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    atlas = {t for r in source_rows if group(r) == "readout" for t in parse_readout(r)}
    tokenizer.add_special_tokens(["<|im_end|>", *sorted(atlas)])
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    return source, tokenizer_path


def test_export_preserves_evaluation_and_source_bytes_and_image_contract(tmp_path, source_rows):
    from sft.scripts.train_trl_catan_vision import inspect_jsonl_contract

    source, tokenizer = _export_fixture(tmp_path, source_rows)
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*.json*")}
    output = tmp_path / "weighted"
    result = export_reweighted(source, output, tokenizer, row_count=30000, config=MixConfig(max_repeats=1000))
    assert result["after"]["rows"] == 30000
    for path, content in before.items():
        assert (source / path).read_bytes() == content
    for split in EVAL_SPLITS:
        assert (source / "stage1" / f"{split}.jsonl").read_bytes() == (output / "stage1" / f"{split}.jsonl").read_bytes()
    contract = inspect_jsonl_contract(output / "stage1/train.jsonl", output / "images", require_curriculum=False)
    assert contract["rows"] == 30000
    with pytest.raises(FileExistsError):
        export_reweighted(source, output, tokenizer)
    with pytest.raises(ValueError, match="separate"):
        export_reweighted(source, source / "nested", tokenizer)


def test_export_checks_fingerprints_before_writing(tmp_path, source_rows):
    source, tokenizer = _export_fixture(tmp_path, source_rows)
    path = source / "stage1/validation.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="fingerprint"):
        export_reweighted(source, tmp_path / "weighted", tokenizer, row_count=30000, config=MixConfig(max_repeats=1000))
    assert not (tmp_path / "weighted").exists()
