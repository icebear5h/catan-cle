"""Real-contract offline integration tests; source assets are read-only and image-free."""

import copy
import json
import random
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import pytest

from data_pipeline.board_recognition.sources import file_sha256
from evals.catan_board_bench.tokens import atlas_tokens
from sft.scripts.build_symbolic_board_dataset import (
    DEFAULT_OUTPUT, DEFAULT_ROOT, SOURCE_COUNTS, TRAIN_QUOTAS, audit_source_state,
    build_dataset, directional_pair_manifest, known_direction_exposure, read_json, read_jsonl,
    static_queries, transfer_projection, validate_dataset, validate_row_declarations, validate_rows,
)
from sft.symbolic_board_tasks import (
    STATIC_TASKS, TRAIN_TASKS, TRANSFER_TASKS, PhysicalStateError, atlas_geometry,
    decode_state, detached_board, score_symbolic_task, symbolic_answer, validate_contract,
)


@pytest.fixture(scope="module")
def real_example():
    if not DEFAULT_ROOT.is_dir():
        pytest.skip("full_board_diverse_v1 local source assets unavailable")
    manifest = read_jsonl(DEFAULT_ROOT / "manifest.jsonl")
    source = next(r for r in manifest if r["split"] == "train" and r["density_bin"] == "empty")
    rows = read_jsonl(DEFAULT_ROOT / "full_board_readout_v1/stage1/train.jsonl")
    readout = next(r for r in rows if r["state_id"] == source["sample_id"])
    contract = read_json(DEFAULT_ROOT / source["contract_path"])
    return source, readout, contract


@pytest.fixture(scope="module")
def built_dataset(tmp_path_factory):
    if not DEFAULT_ROOT.is_dir():
        pytest.skip("full_board_diverse_v1 local source assets unavailable")
    output = tmp_path_factory.mktemp("symbolic-real") / "dataset"
    original_open = Path.open

    def no_images(path, *args, **kwargs):
        assert path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"), "image bytes must never be loaded"
        return original_open(path, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "open", no_images)
        result = build_dataset(output)
        verified = validate_dataset(output)
    metadata = read_json(output / "metadata.json")
    files = {s: read_jsonl(output / f"{s}.jsonl") for s in metadata["counts"]}
    return output, result, verified, metadata, files


def test_real_contract_readout_dense_smoke_and_no_source_mutation(real_example):
    source, readout, contract = real_example
    before = copy.deepcopy((source, readout, contract))
    digest = file_sha256(DEFAULT_ROOT / source["contract_path"])
    state, provenance = audit_source_state(DEFAULT_ROOT, source, readout)
    assert state == validate_contract(contract)
    assert len(state["board"].split(";")) == 155
    assert len(state["colors"]) == 4
    assert set(provenance["paths"]) == {"contract", "labels"}
    board = detached_board(state)
    assert len(board.map.land_nodes) == 54
    assert state["board"] == readout["messages"][-1]["content"]
    assert (source, readout, contract) == before
    assert file_sha256(DEFAULT_ROOT / source["contract_path"]) == digest


@pytest.mark.parametrize("family,field,value", [
    ("tiles", "coord", [True, 0, 0]),
    ("tiles", "number", "6"),
    ("tiles", "node_tokens", ["<N00>"] * 6),
    ("tiles", "edges", [[0, 2]] * 6),
    ("nodes", "token", "<N01>"),
    ("nodes", "adjacent_edges", [[0, 2]]),
    ("nodes", "port_ids", [8]),
    ("edges", "nodes", [0, 2]),
    ("edges", "road_color", "NOT_A_PLAYER"),
    ("ports", "attached_nodes", [0, 1]),
    ("ports", "direction", "NORTH"),
])
def test_contract_exact_topology_and_identity_rejected(real_example, family, field, value):
    contract = copy.deepcopy(real_example[2])
    contract[family][0][field] = value
    with pytest.raises(ValueError):
        validate_contract(contract)


def test_duplicate_ownership_and_source_hash_checks(real_example):
    source, readout, original = real_example
    contract = copy.deepcopy(original)
    contract["nodes"][1] = copy.deepcopy(contract["nodes"][0])
    with pytest.raises(ValueError):
        validate_contract(contract)
    contract = copy.deepcopy(original)
    contract["nodes"][0]["color"] = contract["players"][0]["color"]
    with pytest.raises(ValueError, match="ownership"):
        validate_contract(contract)
    changed = copy.deepcopy(source)
    changed["sha256"]["contract"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        audit_source_state(DEFAULT_ROOT, changed, readout)
    changed = copy.deepcopy(readout)
    changed["messages"][-1]["content"] += " junk"
    with pytest.raises(ValueError, match="answer"):
        audit_source_state(DEFAULT_ROOT, source, changed)
    # Derived source road caches may be historically stale; never use them as gold.
    changed = copy.deepcopy(original)
    changed["achievements"]["longest_road"]["length"] = 999
    changed["players"][0]["longest_road_length"] = 999
    assert validate_contract(changed) == validate_contract(original)


def test_source_physical_supply_and_distance_are_not_silently_repaired(real_example):
    original = real_example[2]
    contract = copy.deepcopy(original)
    color = contract["players"][0]["color"]
    for node in contract["nodes"][:2]:
        node.update(building="SETTLEMENT", color=color,
                    building_token="<SETTLEMENT>", color_token=f"<{color}>")
    with pytest.raises(PhysicalStateError, match="distance"):
        validate_contract(contract)
    contract = copy.deepcopy(original)
    for edge in contract["edges"][:16]:
        edge.update(road_color=color, road_color_token=f"<{color}>")
    with pytest.raises(PhysicalStateError, match="supply"):
        validate_contract(contract)


def test_direction_pair_partition_covers_all_and_balances_choices_and_inverses():
    atlas = atlas_geometry()
    pairs = directional_pair_manifest(46)
    sets = {s: {tuple(p) for p in values} for s, values in pairs.items()}
    all_pairs = {p for family in "NT" for p in combinations(sorted(t for t in atlas["positions"] if t[1] == family), 2)}
    assert set.union(*sets.values()) == all_pairs
    for a, b in combinations(sets, 2):
        assert not sets[a] & sets[b]
    for split, assigned in pairs.items():
        for task, count in (("symbolic_direction", 400), ("symbolic_direction_choice", 400)):
            queries = static_queries(task, assigned, count, random.Random(46))
            assert queries == static_queries(task, assigned, count, random.Random(46))
            bins = Counter()
            for query in queries:
                assert tuple(sorted((query["a"], query["b"]))) in sets[split]
                gold = symbolic_answer(task, {"state": None, "query": query})
                bins[(query["a"][1], query["choices"].index(gold) if task.endswith("choice") else gold)] += 1
                inverse = dict(query, a=query["b"], b=query["a"])
                inverse["direction"] = {"left": "right", "right": "left", "above": "below", "below": "above"}[query["direction"]]
                inverse_gold = symbolic_answer(task, {"state": None, "query": inverse})
                assert inverse_gold == (gold if not task.endswith("choice") else query["b"] if gold == query["a"] else query["a"])
            assert set(bins.values()) == {100}


def test_full_real_build_counts_split_exclusions_and_transfer_only(built_dataset):
    output, result, verified, metadata, files = built_dataset
    assert verified["valid"] and result["counts"] == metadata["counts"]
    assert len(files["train"]) == 3200
    assert len(files["validation"]) == len(files["test"]) == 480
    assert Counter(r["task_type"] for r in files["train"]) == TRAIN_QUOTAS
    assert metadata["static_training_presentations"] == metadata["state_training_presentations"] == 1600
    audit = metadata["source_audit"]
    assert audit["source_counts"] == SOURCE_COUNTS
    assert len(audit["benchmark_excluded_game_ids"]) == 13
    assert audit["exclusion_reason_counts"]["diagnostic_reserved"] == 64
    assert "NOT certified" in audit["certification"]
    all_ids = {r["sample_id"] for r in read_jsonl(DEFAULT_ROOT / "manifest.jsonl")}
    excluded_ids = {r["state_id"] for r in audit["exclusions"]}
    assert excluded_ids <= all_ids
    for split, rows in files.items():
        task_set = {r["task_type"] for r in rows}
        assert task_set <= (TRANSFER_TASKS if split.startswith("transfer_") else TRAIN_TASKS)
        for row in rows:
            p = row["metadata"].get("provenance")
            if p:
                assert p["state_id"] in all_ids - excluded_ids
                assert p["source"]["game_id"] not in audit["benchmark_excluded_game_ids"]
                assert p["split"] == split.removeprefix("transfer_")
                assert p["source"]["kind"] in ("engine_rollout", "colonist_replay")
    assert metadata["source_kind_counts"]["train"]["colonist_replay"] > 0
    assert metadata["source_kind_counts"]["train"]["engine_rollout"] > 0
    assert metadata["historical_exposure_audit"]["status"] == "known_sources_audited_history_incomplete"
    assert "Compositional transfer" in metadata["transfer_semantics"]
    inputs = read_json(output / "dataset_inputs.json")
    for key in ("train_jsonl", "validation_jsonl", "test_jsonl", "transfer_validation_jsonl", "transfer_test_jsonl", "metadata", "manifest", "token_inventory"):
        assert Path(inputs[key]).is_absolute() and Path(inputs[key]).is_file()
    assert not inputs["proposed_paired_experiment_inputs"]["final_sft_budget_approved"]
    assert read_json(output / "token_inventory.json")["tokens"] == atlas_tokens()


def test_record_schema_no_leakage_full_state_recomputation_and_active_ports(built_dataset):
    _, _, _, metadata, files = built_dataset
    assert set(metadata["coverage"]["active_query_tokens"]["train"]) == set(atlas_tokens())
    for rows in files.values():
        for row in rows:
            assert "image" not in json.dumps(row).lower()
            assert len(row["messages"]) == 2
            prompt = row["messages"][0]["content"]
            for forbidden in ("adjacent_tiles", "node_tokens", "coord", "road_lengths", "has_longest_road", "legal_nodes"):
                assert forbidden not in prompt
            state = row["metadata"]["target"]["state"]
            if state:
                assert set(state) == {"board", "colors"}
                assert len(state["board"].split(";")) == 155
                assert state["board"] in prompt
    for task in TRAIN_TASKS - STATIC_TASKS:
        row = next(r for r in files["train"] if r["task_type"] == task)
        gold = row["messages"][1]["content"]
        assert score_symbolic_task("bogus cached expected", gold, row["metadata"])["correct"]
        bad = copy.deepcopy(row["metadata"])
        bad["target"]["state"]["leaked_gold"] = gold
        with pytest.raises(ValueError):
            score_symbolic_task(gold, gold, bad)


def test_real_routes_and_atomic_constraints_have_measured_polarity(built_dataset):
    files = built_dataset[-1]
    for split in ("train", "validation", "test"):
        rows = files[split]
        for task in ("symbolic_direction", "symbolic_near", "symbolic_local_constraint"):
            counts = Counter(r["messages"][1]["content"] for r in rows if r["task_type"] == task)
            assert counts["yes"] == counts["no"]
        routes = [json.loads(r["messages"][1]["content"]) for r in rows if r["task_type"] == "symbolic_shortest_route"]
        assert sum(r["nodes"] is None for r in routes) == len(routes) * 3 // 8
        assert sum(r["edges"] == [] for r in routes) == len(routes) // 8
        assert any(r["edges"] and len(r["edges"]) >= 2 for r in routes)


def test_real_settlement_transfer_against_independent_complete_predicate(built_dataset):
    rows = built_dataset[-1]["transfer_test"]
    atlas = atlas_geometry()
    checked = 0
    for row in rows:
        if row["task_type"] != "symbolic_settlement_locations":
            continue
        target = row["metadata"]["target"]
        q = target["query"]
        if q["near"] is not None:
            continue
        data = decode_state(target["state"])
        occupied = set(data["buildings"])
        expected = set()
        for node, neighbors in atlas["graph"].items():
            has_road = any(data["roads"].get(edge) == q["color"]
                           for edge, endpoints in atlas["edges"].items() if node in endpoints)
            if node not in occupied and not neighbors & occupied and (q["phase"] == "setup" or has_road):
                expected.add(node)
        assert set(row["messages"][1]["content"].split()) - {"NONE"} == expected
        checked += 1
    assert checked > 0


def test_full_validation_catches_prompt_leak_or_target_mismatch(built_dataset):
    _, _, _, metadata, files = built_dataset
    pairs = directional_pair_manifest(metadata["seed"],
                                      [p.split() for p in metadata["historical_exposure_audit"]["pair_sources"]])
    changed = {k: list(v) for k, v in files.items()}
    changed["train"][0] = copy.deepcopy(changed["train"][0])
    changed["train"][0]["messages"][0]["content"] += " Endpoint table: <E00_01> -> <N00> <N01>"
    with pytest.raises(ValueError, match="prompt"):
        validate_rows(changed, pairs)
    changed["train"][0] = copy.deepcopy(files["train"][0])
    changed["train"][0]["metadata"]["row_position"] = 5
    with pytest.raises(ValueError, match="ordering"):
        validate_rows(changed, pairs)


def test_immutable_output_and_parent_precondition(tmp_path):
    assert DEFAULT_OUTPUT.name == "symbolic_board_v2"
    with pytest.raises(FileExistsError):
        build_dataset(tmp_path)
    destination = tmp_path / "missing-parent" / "dataset"
    with pytest.raises(FileNotFoundError, match="parent"):
        build_dataset(destination, dry_run=True)
    assert not destination.parent.exists()


def test_rendered_roster_position_cannot_predict_atomic_labels(built_dataset):
    # Deliberately derive both queried participant and predicate from the rendered
    # question; a balanced metadata table cannot mask a leaky model presentation.
    for split in ("train", "validation", "test"):
        tables = {}
        for row in built_dataset[-1][split]:
            task = row["task_type"]
            if task not in ("symbolic_owned_incident_roads", "symbolic_local_constraint", "symbolic_owned_roads"):
                continue
            prompt, response = [m["content"] for m in row["messages"]]
            roster = re.search(r"Participants: ([A-Z_ ]+)\.\n", prompt).group(1).split()
            question = prompt.rsplit("\n", 1)[-1]
            mode = "all"
            if task == "symbolic_local_constraint":
                mode = ("has_owned_incident_road" if "incident road" in question else
                        "no_adjacent_building" if "edge-adjacent" in question else "empty")
            match = re.search(r"existing ([A-Z_]+) (?:incident road|road edges|road edge)", question)
            position = roster.index(match.group(1)) if match else None
            key = (task, mode, position)
            tables.setdefault(key, Counter())[response not in ("no", "NONE")] += 1
        for counts in tables.values():
            assert counts[True] == counts[False] > 0
        for task, mode in (("symbolic_owned_incident_roads", "all"),
                           ("symbolic_local_constraint", "has_owned_incident_road")):
            assert {p for t, m, p in tables if (t, m) == (task, mode)} == {0, 1, 2, 3}


def test_global_dynamic_variety_and_conditional_density_support(built_dataset):
    metadata, files = built_dataset[-2:]
    train = [r for r in files["train"] if r["task_type"] not in STATIC_TASKS]
    ids = {r["metadata"]["provenance"]["state_id"] for r in train}
    assert len(ids) >= 1000  # v1 reused only 306 states across 1600 presentations.
    assert len(ids) == metadata["component_sampling"]["train"]["unique_states"]
    roads = [r for r in train if r["task_type"] == "symbolic_owned_roads"]
    positives = [r for r in roads if r["messages"][1]["content"] != "NONE"]
    negatives = [r for r in roads if r["messages"][1]["content"] == "NONE"]
    assert {r["metadata"]["provenance"]["density_bin"] for r in positives} == {"setup", "sparse", "dense"}
    assert all(r["metadata"]["provenance"]["density_bin"] != "empty" for r in negatives)
    assert all(set(c["eligible_states_by_density"]) == {"empty", "setup", "sparse", "dense"}
               for c in metadata["component_sampling"]["train"]["conditional_support"])
    # Cross-tabs retain the actual source-density condition as well as the visible roster.
    assert all("density" in c for c in metadata["component_profiles"]["train"]["rendered_roster_crosstabs"])


@pytest.mark.parametrize("level", ["row", "metadata", "both"])
@pytest.mark.parametrize("field,value", [
    ("task_role", "train"), ("training_family", "symbolic_owned_roads"),
    ("task_type", "symbolic_owned_roads"), ("split", "train"),
])
def test_all_transfer_declarations_fail_closed(built_dataset, level, field, value):
    row = copy.deepcopy(built_dataset[-1]["transfer_test"][0])
    if level in ("row", "both"):
        row[field] = value
    if level in ("metadata", "both"):
        row["metadata"][field] = value
    with pytest.raises(ValueError):
        validate_row_declarations(row, "transfer_test")


def test_missing_role_and_forged_transfer_weights_rejected(built_dataset):
    files = built_dataset[-1]
    row = copy.deepcopy(files["transfer_test"][0])
    row.pop("task_role")
    with pytest.raises(ValueError):
        validate_row_declarations(row, "transfer_test")
    row = copy.deepcopy(files["transfer_test"][0])
    row["metadata"]["task_role"] = "train"
    with pytest.raises(ValueError):
        score_symbolic_task("", row["messages"][1]["content"], row["metadata"])


def test_transfer_dedup_selection_weights_and_missing_hard_cases(built_dataset):
    metadata, files = built_dataset[-2:]
    for split in ("transfer_validation", "transfer_test"):
        rows = files[split]
        projections = [transfer_projection(r["task_type"], r["metadata"]["target"]["state"],
                                           r["metadata"]["target"]["query"]) for r in rows]
        assert len(projections) == len(set(projections))
        report = metadata["transfer_selection"][split]
        assert report["status"] == "transfer_pilot_limited"
        assert report["population_rows"] > report["unique_relevant_queries"] > report["selected_rows"]
        assert report["selected_rows"] == len(rows) < 1000
        assert not report["aggregation"]["pooled_micro_headline"]
        for group in report["groups"]:
            pos, neg = group["positive"], group["negative"]
            if not group["mode"].startswith("setup/"):
                assert pos["selected"] == pos["unique"]
            assert neg["selected"] == min(neg["unique"], pos["selected"])
        for task in TRANSFER_TASKS:
            selected = [r for r in rows if r["task_type"] == task]
            if selected:
                assert sum(r["metadata"]["macro_weight"] for r in selected) == pytest.approx(1)
        coverage = report["population_graph_cases"]
        assert coverage["states"] == 64
        assert set(coverage["missing_coverage"]) == {"cycle", "effective_blocker", "tied_max_ge5"}
        assert all(coverage["counts"][key] == 0 for key in coverage["missing_coverage"])
    diagnostic = metadata["reserved_color_diagnostic_coverage"]
    assert not diagnostic["included_in_transfer"] and diagnostic["graph_cases"]["states"] == 64
    reserved = {p["state_id"] for p in diagnostic["provenance"]}
    assert all(r["metadata"]["provenance"]["state_id"] not in reserved
               for split in ("transfer_validation", "transfer_test") for r in files[split])


def test_known_history_pair_union_filtered_and_v1_preserved(built_dataset):
    output, _, _, metadata, files = built_dataset
    history = known_direction_exposure()
    known = {tuple(k.split()) for k in history["pair_sources"]}
    assert known
    pairs = read_json(output / "directional_pairs.json")["splits"]
    assert known <= {tuple(p) for p in pairs["train"]}
    for split in ("validation", "test"):
        assert not known & {tuple(p) for p in pairs[split]}
        assert all(not r["metadata"]["known_training_exposure"] for r in files[split]
                   if "directional_pair" in r["metadata"])
    assert "New-corpus holdout only" in history["claim"]
    receipt = metadata["preserved_v1"]
    if receipt["status"] == "preserved":
        for info in receipt["files"].values():
            assert file_sha256(Path(info["path"])) == info["sha256"]
