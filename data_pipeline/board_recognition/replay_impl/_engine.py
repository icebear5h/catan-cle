"""Engine trajectory generation and balanced training selection."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action
from cle.sandbox.palette import balanced_datagen_colors
from data_pipeline.board_recognition.replay_impl._config import (
    ALL_COLOR_VALUES,
    PER_TRAJECTORY_CAP,
    BoardStateCandidate,
    ReplayDatasetBuildError,
)
from data_pipeline.board_recognition.replay_impl._facts import (
    stable_seed,
)
from data_pipeline.board_recognition.replay_impl._policy import (
    choose_legal_datagen_action,
)
from data_pipeline.board_recognition.replay_impl._replay import (
    _candidate,
)
from data_pipeline.board_recognition.sources import (
    canonical_sha256,
)
from data_pipeline.json_coerce import as_dict, as_list
from evals.catan_board_bench.builder import (
    CatanObservationSuite,
)


def generate_engine_trajectory_candidates(
    *,
    trajectory_index: int,
    seed: int,
    split: str,
    max_actions: int = 1_000,
) -> list[BoardStateCandidate]:
    colors = balanced_datagen_colors(trajectory_index, seed=seed)
    game_seed = stable_seed(seed, split, trajectory_index) % (2**31)
    policy_seed = stable_seed(seed, split, trajectory_index, "policy")
    engine = GameEngine(colors, seed=game_seed, shuffle_players=False, vps_to_win=13)
    policy_rng = random.Random(policy_seed)
    suite = CatanObservationSuite()
    trajectory_id = f"engine:{split}:{game_seed}"
    policy_payload = {
        "schema": "catan_board_recognition_engine_policy/v2",
        "game_seed": game_seed,
        "policy_seed": policy_seed,
        "colors": [color.value for color in colors],
        "legal_actions_only": True,
        "victory_points_to_win": 13,
        "action_weights": {
            "BUILD_CITY": 20,
            "BUILD_SETTLEMENT": 8,
            "BUILD_ROAD": 5,
            "BUY_DEVELOPMENT_CARD": 2,
            "MARITIME_TRADE": 2,
            "END_TURN": 1,
        },
        "forced_resources": False,
        "forced_board_mutation": False,
        "trajectory_action_limit": max_actions,
    }
    source_sha = canonical_sha256(policy_payload)
    candidates: list[BoardStateCandidate] = []
    seen: set[str] = set()

    def capture(last_action: Action | None) -> None:
        source = {
            "kind": "engine_rollout",
            "game_id": None,
            "trajectory_id": trajectory_id,
            "replay_path": None,
            "replay_step": None,
            "source_sha256": source_sha,
            "engine_action_count": len(engine.state.actions),
            "engine_seed": game_seed,
            "policy_seed": policy_seed,
            "last_action_type": last_action.action_type.value if last_action else None,
            "legal_actions_only": True,
            "trajectory_action_limit": max_actions,
        }
        contract = suite.public_board_contract(
            engine,
            sample={
                "id": f"engine_{split}_{game_seed}_a{len(engine.state.actions):06d}",
                "index": len(engine.state.actions),
            },
            source=source,
        )
        candidate = _candidate(contract, trajectory_id=trajectory_id, source=source)
        if candidate.board_fact_sha256 not in seen:
            seen.add(candidate.board_fact_sha256)
            candidates.append(candidate)

    capture(None)
    for _ in range(max_actions):
        if engine.winning_color() is not None:
            break
        advertised = tuple(engine.state.playable_actions)
        action = choose_legal_datagen_action(engine, policy_rng)
        if action not in advertised or not engine.is_action_valid(action):
            raise ReplayDatasetBuildError(f"datagen selected a non-playable action: {action}")
        transition = engine.step(action)
        if transition.requested_action != action:
            raise ReplayDatasetBuildError("engine transition changed the requested action identity")
        capture(transition.resolved_action)
    return candidates


def required_dynamic_classes() -> set[str]:
    return {
        *(
            f"node:{color.value}_{building}"
            for color in ALL_COLOR_VALUES
            for building in ("SETTLEMENT", "CITY")
        ),
        *(f"edge:{color.value}" for color in ALL_COLOR_VALUES),
    }


def candidate_dynamic_classes(candidate: BoardStateCandidate) -> set[str]:
    classes: set[str] = set()
    for raw_node in as_list(candidate.contract["nodes"]):
        node = as_dict(raw_node)
        if node["color"] and node["building"]:
            classes.add(f"node:{node['color']}_{node['building']}")
    for raw_edge in as_list(candidate.contract["edges"]):
        edge = as_dict(raw_edge)
        if edge["road_color"]:
            classes.add(f"edge:{edge['road_color']}")
    return classes


def select_balanced_engine_training_candidates(
    candidates: Sequence[BoardStateCandidate],
    *,
    target: int,
    seed: int,
    minimum_class_support: int = 48,
    per_trajectory_cap: int = PER_TRAJECTORY_CAP,
) -> list[BoardStateCandidate]:
    required = required_dynamic_classes()
    ordered = sorted(
        candidates,
        key=lambda row: (stable_seed(seed, row.board_fact_sha256), row.board_fact_sha256),
    )
    selected: list[BoardStateCandidate] = []
    selected_hashes: set[str] = set()
    trajectory_counts: Counter[str] = Counter()
    support: Counter[str] = Counter()
    while len(selected) < target:
        choices = [
            row
            for row in ordered
            if row.board_fact_sha256 not in selected_hashes
            and trajectory_counts[row.trajectory_id] < per_trajectory_cap
        ]
        if not choices:
            break

        def score(row: BoardStateCandidate) -> tuple[int, int, int, int]:
            classes = candidate_dynamic_classes(row) & required
            under_target = [name for name in classes if support[name] < minimum_class_support]
            return (
                len(under_target),
                sum(minimum_class_support - support[name] for name in under_target),
                row.building_count + row.road_count,
                stable_seed(seed, "train-balance", row.board_fact_sha256),
            )

        best = max(choices, key=score)
        selected.append(best)
        selected_hashes.add(best.board_fact_sha256)
        trajectory_counts[best.trajectory_id] += 1
        support.update(candidate_dynamic_classes(best) & required)
    if len(selected) != target:
        raise ReplayDatasetBuildError(f"selected {len(selected)}/{target} engine train states")
    missing_support = {
        name: support[name] for name in sorted(required) if support[name] < minimum_class_support
    }
    if missing_support:
        raise ReplayDatasetBuildError(
            f"engine training supplement lacks dynamic class support: {missing_support}"
        )
    return selected

