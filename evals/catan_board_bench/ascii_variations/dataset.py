"""Density-stratified source selection and reproducible probe artifacts."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from evals.catan_board_bench.ascii_variations.codec import _json_digest, write_json, write_jsonl
from evals.catan_board_bench.ascii_variations.facts import AsciiBoard
from evals.catan_board_bench.ascii_variations.graph import (
    _dynamic_density,
    full_fact_digest,
    full_public_graph_facts,
)
from evals.catan_board_bench.ascii_variations.questions import (
    build_ascii_smoke_questions,
    validate_ascii_smoke_questions,
)
from evals.catan_board_bench.ascii_variations.records import parse_ascii_variant
from evals.catan_board_bench.ascii_variations.rendering import render_ascii_variant
from evals.catan_board_bench.ascii_variations.schema import (
    ASCII_VARIANTS,
    DATASET_SCHEMA,
    FACT_SCHEMA,
)
from evals.catan_board_bench.ascii_variations.scoring import (
    STRICT_SCORER_VERSION,
    strict_scorer_digest,
)
from evals.json_types import JsonDict, JsonValue, as_dict, as_str


def select_smoke_contract_paths(
    contract_dir: Path,
    *,
    board_count: int = 12,
) -> list[Path]:
    """Select density-stratified snapshots from distinct games."""

    by_game: dict[str, list[tuple[Path, JsonDict]]] = defaultdict(list)
    for path in sorted(contract_dir.glob("*.json")):
        contract = json.loads(path.read_text())
        game_id = str(contract.get("source", {}).get("game_id", path.stem))
        by_game[game_id].append((path, contract))

    ranked_games = sorted(
        by_game.items(),
        key=lambda item: (
            -max(_dynamic_density(contract) for _path, contract in item[1]),
            item[0],
        ),
    )
    if len(ranked_games) < board_count:
        raise ValueError(f"need {board_count} distinct games, found {len(ranked_games)}")

    selected = []
    stage_fractions = (1.0, 0.55, 0.0)
    for game_index, (_game_id, candidates) in enumerate(
        sorted(ranked_games[:board_count], key=lambda item: item[0])
    ):
        ordered = sorted(
            candidates,
            key=lambda item: (
                _dynamic_density(item[1]),
                _integer(_source(item[1]).get("replay_step", 0)),
            ),
        )
        fraction = stage_fractions[game_index % len(stage_fractions)]
        candidate_index = round((len(ordered) - 1) * fraction)
        selected.append(ordered[candidate_index][0])
    return selected


def build_ascii_variation_dataset(
    output_dir: Path,
    *,
    contract_dir: Path,
    board_count: int = 12,
) -> JsonDict:
    """Build facts, six renderings, and sixty paired diagnostic questions."""

    output_dir.mkdir(parents=True, exist_ok=True)
    facts_dir = output_dir / "facts"
    alias_dir = output_dir / "aliases"
    representation_dir = output_dir / "representations"
    facts_dir.mkdir(exist_ok=True)
    alias_dir.mkdir(exist_ok=True)
    representation_dir.mkdir(exist_ok=True)

    boards: list[AsciiBoard] = []
    manifest_rows: list[JsonDict] = []
    for board_index, contract_path in enumerate(
        select_smoke_contract_paths(contract_dir, board_count=board_count)
    ):
        contract = json.loads(contract_path.read_text())
        sample_id = f"ascii_board_{board_index:02d}"
        facts, aliases = full_public_graph_facts(contract, sample_id=sample_id)
        digest = full_fact_digest(facts)
        board: AsciiBoard = {
            "sample_id": sample_id,
            "facts": facts,
            "contract": contract,
            "contract_path": str(contract_path),
            "digest": digest,
        }
        boards.append(board)

        write_json(facts_dir / f"{sample_id}.json", facts)
        write_json(alias_dir / f"{sample_id}.json", aliases)
        sample_representation_dir = representation_dir / sample_id
        sample_representation_dir.mkdir(exist_ok=True)
        prompt_metrics: JsonDict = {}
        for variant in ASCII_VARIANTS:
            text = render_ascii_variant(variant, facts, sample_id=sample_id)
            parsed = parse_ascii_variant(text)
            if full_fact_digest(parsed) != digest:
                raise ValueError(f"round-trip mismatch for {sample_id}/{variant}")
            path = sample_representation_dir / f"{variant}.txt"
            path.write_text(text + "\n")
            prompt_metrics[variant] = {
                "characters": len(text),
                "lines": len(text.splitlines()),
            }

        source = contract.get("source", {})
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "fact_digest": digest,
                "source_contract": str(contract_path),
                "source_game_id": str(source.get("game_id")),
                "source_replay_step": source.get("replay_step"),
                "original_sample_id": contract.get("sample", {}).get("id"),
                "dynamic_density": _dynamic_density(contract),
                "alias_sha256": _json_digest(aliases),
                "representation_metrics": prompt_metrics,
            }
        )

    questions = build_ascii_smoke_questions(boards)
    validate_ascii_smoke_questions(questions)
    write_jsonl(output_dir / "manifest.jsonl", manifest_rows)
    write_jsonl(output_dir / "qa.jsonl", questions)

    metadata: JsonDict = {
        "schema": DATASET_SCHEMA,
        "fact_schema": FACT_SCHEMA,
        "board_count": len(boards),
        "source_game_count": len({row["source_game_id"] for row in manifest_rows}),
        "question_count": len(questions),
        "questions_per_variant": len(questions),
        "request_count": len(questions) * len(ASCII_VARIANTS),
        "variants": list(ASCII_VARIANTS),
        "categories": dict(
            Counter(as_str(row["category"], "question category") for row in questions)
        ),
        "entity_ids": "deterministically permuted and board-local",
        "strict_json_answers": True,
        "strict_scorer_version": STRICT_SCORER_VERSION,
        "strict_scorer_sha256": strict_scorer_digest(),
        "images": False,
        "precomputed_player_counts_in_facts": False,
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def _source(contract: JsonDict) -> JsonDict:
    return as_dict(contract.get("source", {}), "contract source")


def _integer(value: JsonValue) -> int:
    if not isinstance(value, (int, float, str)):
        raise TypeError(f"replay_step is not an integer: {value!r}")
    return int(value)
