"""Reconstructing board-state candidates from Colonist replays."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Sequence

from cle.replay.runtime.audit import ensure_replay_audit_state
from data_pipeline.board_recognition.replay_impl._config import (
    DEFAULT_SEED,
    DENSITY_BINS,
    BoardStateCandidate,
    ReplayDatasetBuildError,
)
from data_pipeline.board_recognition.replay_impl._facts import (
    board_density,
    candidate_key,
    stable_seed,
    static_board_facts,
)
from data_pipeline.board_recognition.sources import (
    PROJECT_ROOT,
    canonical_sha256,
    diagnostic_is_board_safe,
    file_sha256,
    validate_public_board_contract,
    visible_board_facts,
)
from data_pipeline.json_coerce import as_str
from data_pipeline.json_types import JsonDict
from evals.catan_board_bench.builder import (
    CatanObservationSuite,
    load_colonist_replay,
    step_replay,
)
from playground.game_viewer.state import ServerState


def _candidate(
    contract: JsonDict,
    *,
    trajectory_id: str,
    source: JsonDict,
) -> BoardStateCandidate:
    validate_public_board_contract(contract)
    facts = visible_board_facts(contract)
    board_hash = canonical_sha256(facts)
    buildings, roads, density = board_density(contract)
    return BoardStateCandidate(
        trajectory_id=trajectory_id,
        board_fact_sha256=board_hash,
        board_map_sha256=canonical_sha256(static_board_facts(contract)),
        density_bin=density,
        building_count=buildings,
        road_count=roads,
        contract=contract,
        source=source,
    )


def _capture_replay_contract(
    suite: CatanObservationSuite,
    state: ServerState,
    source_row: JsonDict,
) -> BoardStateCandidate:
    game_id = as_str(source_row["game_id"])
    replay_step = int(state.replay_index)
    game = state.current_game
    if game is None:
        raise ReplayDatasetBuildError(f"replay state for {game_id} carries no game")
    source = {
        "kind": "colonist_replay",
        "game_id": game_id,
        "trajectory_id": f"replay:{game_id}",
        "replay_path": source_row["path"],
        "replay_step": replay_step,
        "source_sha256": source_row["sha256"],
        "engine_action_count": len(game.state.actions),
        "diagnostic_count": len(getattr(state, "replay_semantic_issues", [])),
    }
    contract = suite.public_board_contract(
        game,
        sample={
            "id": f"replay_{game_id}_s{replay_step:06d}",
            "index": replay_step,
        },
        source=source,
    )
    return _candidate(contract, trajectory_id=as_str(source["trajectory_id"]), source=source)


def reconstruct_replay_candidates(source_row: JsonDict) -> list[BoardStateCandidate]:
    """Walk one accepted replay once and retain unique visible board states."""

    path = PROJECT_ROOT / as_str(source_row["path"])
    if not path.is_file() or file_sha256(path) != source_row["sha256"]:
        raise ReplayDatasetBuildError(f"locked replay payload changed: {path}")
    state = load_colonist_replay(path, quiet=True)
    ensure_replay_audit_state(state)
    suite = CatanObservationSuite()
    if state.replay_data is None:
        raise ReplayDatasetBuildError(f"replay state for {path} carries no replay data")
    parsed_actions = state.replay_data.get("parsed_actions", [])
    if not isinstance(parsed_actions, list):
        raise ReplayDatasetBuildError(f"replay parsed_actions must be a list: {path}")
    total_steps = len(parsed_actions)
    candidates: list[BoardStateCandidate] = []
    seen: set[str] = set()

    def capture() -> None:
        candidate = _capture_replay_contract(suite, state, source_row)
        if candidate.board_fact_sha256 not in seen:
            seen.add(candidate.board_fact_sha256)
            candidates.append(candidate)

    capture()
    while state.replay_index < total_steps:
        before = state.replay_index
        result = step_replay(state, quiet=True)
        if isinstance(result, tuple) or not isinstance(result, dict) or result.get("error"):
            raise ReplayDatasetBuildError(
                f"locked replay {source_row['game_id']} failed at {before}: {result}"
            )
        if state.replay_index <= before and not result.get("finished"):
            raise ReplayDatasetBuildError(
                f"locked replay {source_row['game_id']} made no progress at {before}"
            )
        capture()

    blocking = [
        issue
        for issue in getattr(state, "replay_semantic_issues", [])
        if not diagnostic_is_board_safe(json.loads(json.dumps(issue, default=str)))
    ]
    if blocking:
        raise ReplayDatasetBuildError(
            f"locked replay {source_row['game_id']} developed board-unsafe diagnostics: "
            f"{blocking[:3]}"
        )
    if state.replay_index != total_steps:
        raise ReplayDatasetBuildError(
            f"locked replay {source_row['game_id']} stopped at "
            f"{state.replay_index}/{total_steps}"
        )
    if not candidates:
        raise ReplayDatasetBuildError(f"locked replay has no visible board states: {path}")
    return candidates


def split_replay_games(
    source_rows: Sequence[JsonDict],
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[JsonDict]]:
    """Assign whole games to stable 80/10/10 source partitions."""

    ordered = sorted(
        source_rows,
        key=lambda row: (stable_seed(seed, "game-split", row["game_id"]), row["game_id"]),
    )
    if len(ordered) < 40:
        raise ReplayDatasetBuildError(
            f"need at least 40 accepted replay games, received {len(ordered)}"
        )
    validation_count = max(5, round(len(ordered) * 0.1))
    test_count = max(5, round(len(ordered) * 0.1))
    return {
        "validation": ordered[:validation_count],
        "test": ordered[validation_count : validation_count + test_count],
        "train": ordered[validation_count + test_count :],
    }


def _cap_trajectory_candidates(
    candidates: Sequence[BoardStateCandidate],
    *,
    cap: int,
    seed: int,
) -> list[BoardStateCandidate]:
    if len(candidates) <= cap:
        return list(candidates)
    by_density: dict[str, list[BoardStateCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_density[candidate.density_bin].append(candidate)
    for density, rows in by_density.items():
        random.Random(stable_seed(seed, candidate_key(rows), density)).shuffle(rows)
    selected: list[BoardStateCandidate] = []
    while len(selected) < cap:
        progressed = False
        for density in DENSITY_BINS:
            rows = by_density[density]
            if rows and len(selected) < cap:
                selected.append(rows.pop())
                progressed = True
        if not progressed:
            break
    return selected

