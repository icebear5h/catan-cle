"""Byte/order witnesses captured against the original monolithic format modules."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from evals.catan_board_bench import ascii_variations as ascii_formats
from evals.catan_board_bench import full_graph_formats as graph_formats
from evals.catan_board_bench import text_format_optimization as indexed_formats

SOURCE = Path("evals/catan_board_bench/datasets/ascii_variation_probe")
CONTRACTS = Path("evals/catan_board_bench/datasets/catan_board_bench_100/contracts")
SCORER_HASH = "b80971f02ef525a56f5e84bd97801276f953620d82211cabc4a716baf9b9ed3e"
SOURCE_HASHES = {
    "score_strict_json_answer": "b4a5d72fd0fb262ece5eaed3908bd3f5c76c79e1e3e3f4d1435a8e39a9c0da51",
    "_reject_duplicate_object_pairs": "84d44193a0a53adab820fca0212e8cedea4ea4bacea17c24b2b4ad31553bc33e",
    "_semantic_equal": "0bb2a93bb89eb731b5a3fb0d64fea1fabdf5d9e64e9a8d8fd32dd0cbb69b7bcb",
    "_stable_value": "0a4ae1a2f09d8715b2a2bd990e471a6b9d2c9be1f16d6bfe765b8d1775d7fabd",
    "canonical_answer_text": "e0231a3c5f5437eb2279196185df8e41c14ceb0f442ea1f05001335a827d3d64",
    "strict_scorer_digest": "e1f1eccb3713cfee2449f1ca8a2f14de3faa2482a8a3d670e94eec14473c5a13",
}
REPRESENTATION_HASHES = {
    "minimal": "9f2d19d7a2d0cece96555f0997e1c857e972de27f86587d5bb87fd00b1951722",
    "ascii/flat_sorted": "931d0d86eaa7179ad3e07331a3bd80f3fa5a177f1e2b2b07bb7f9a18d17a9e4a",
    "ascii/flat_shuffled": "467ba634fa93e206f6ac9deaffcb196f929a62ce5768d35a91cdc73ec096d043",
    "ascii/sectioned": "9eda18d70fa2d9dc6eec3ae553e17c899ccd605952e5dbf09318539c3430a695",
    "ascii/tile_rows": "391c3a75f4b0ec3420167d2870f0e2ea034f93f2ab640c1f72afe8c4043651ba",
    "ascii/local_blocks": "b8a83d1c99429c8443ae65fed642576470d6df8ca1075a88109f5ee26084044a",
    "ascii/topology_diagram": "88e43ad7d11da71373f2c19950855552347978ccb6f6b144e1b88bde9ac3ec55",
    "graph/optimized_html": "63af0595d2eca9540bab9f92418fbf5164aa3adc560199992000f6ace2739049",
    "graph/full_graph_json": "662ee49ee0921f3d88ee344bd7f98bede0d739760307117bc97a0b05e62a9570",
    "graph/datalog": "ffafcbc2cee939a4b90dfe5ec37dfabd57fe209198f25d521aab5de807911305",
    "graph/sql_relational": "bbb2cb67b262b675e25d4ed3049f33787d9f410b172d4b7f35fda16b54600192",
    "graph/integrated_ascii": "3b45ddf41fb10771ef41deafc82fb27239ac3ecdbb550b059f9504e892b607a2",
    "graph/tile_rows": "391c3a75f4b0ec3420167d2870f0e2ea034f93f2ab640c1f72afe8c4043651ba",
    "indexed/tile_rows": "391c3a75f4b0ec3420167d2870f0e2ea034f93f2ab640c1f72afe8c4043651ba",
    "indexed/indexed_records": "70b99855fba873399024d70450e02e28170d2e45497a9eb1cf4e3a252dcbcee0",
    "indexed/indexed_tile_rows": "b7dc12974725cb7772678b75c984438dc29ba78fb4791ff867d258848af10637",
    "indexed/indexed_json": "53e1f36138aa9fa512b2674512dfbb65ed3e883a75b119f72a6f163e990b606a",
}


def _digest(value: object) -> str:
    # Deliberately preserve insertion order, including nested incidence mappings.
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def test_scoring_sources_and_saved_manifest_identity() -> None:
    assert ascii_formats.STRICT_SCORER_VERSION == "strict_typed_json/v2"
    assert ascii_formats.strict_scorer_digest() == SCORER_HASH
    for name, expected in SOURCE_HASHES.items():
        source = inspect.getsource(getattr(ascii_formats, name))
        assert hashlib.sha256(source.encode()).hexdigest() == expected, name
    metadata = json.loads((SOURCE / "metadata.json").read_text())
    assert metadata["strict_scorer_sha256"] == SCORER_HASH


def test_all_representation_bytes_and_parsed_order() -> None:
    witnesses: dict[str, list[tuple[str, str, str]]] = {}
    fact_paths = sorted((SOURCE / "facts").glob("*.json"))
    assert len(fact_paths) == 12
    for path in fact_paths:
        facts = json.loads(path.read_text())
        minimal = graph_formats.minimal_graph_facts(facts)
        expanded = graph_formats.expand_minimal_graph(minimal)
        assert expanded == facts
        witnesses.setdefault("minimal", []).append((path.stem, _digest(minimal), _digest(expanded)))
        for name in ascii_formats.ASCII_VARIANTS:
            text = ascii_formats.render_ascii_variant(name, facts, sample_id=path.stem)
            parsed = ascii_formats.parse_ascii_variant(text)
            assert parsed == facts
            witnesses.setdefault(f"ascii/{name}", []).append((path.stem, _digest(text), _digest(parsed)))
        for name in graph_formats.FORMAT_NAMES:
            text = graph_formats.render_full_graph_format(name, facts, sample_id=path.stem)
            parsed = graph_formats.parse_full_graph_format(name, text)
            assert parsed == facts
            witnesses.setdefault(f"graph/{name}", []).append((path.stem, _digest(text), _digest(parsed)))
        for name in indexed_formats.FORMAT_NAMES:
            text = indexed_formats.render_text_format(name, facts, sample_id=path.stem)
            parsed = indexed_formats.parse_text_format(name, text)
            assert parsed == facts
            witnesses.setdefault(f"indexed/{name}", []).append((path.stem, _digest(text), _digest(parsed)))
    assert {name: _digest(rows) for name, rows in witnesses.items()} == REPRESENTATION_HASHES


def test_complete_dataset_file_bytes(tmp_path: Path) -> None:
    metadata = ascii_formats.build_ascii_variation_dataset(tmp_path, contract_dir=CONTRACTS)
    assert metadata["strict_scorer_sha256"] == SCORER_HASH
    files = {
        str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }
    assert len(files) == 99
    assert _digest(files) == "1dd64503791cfb05f34fd0a7a3d3e427f411412874155025565f44c0baf55603"
