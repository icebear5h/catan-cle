"""Engine-backed replay_v1 board-recognition state corpus."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx
from PIL import Image

from cle.replay.runtime.audit import ensure_replay_audit_state
from cle.sandbox.palette import balanced_datagen_colors
from data_pipeline.board_recognition.sources import (
    DEFAULT_SOURCE_LOCK,
    PROJECT_ROOT,
    canonical_sha256,
    diagnostic_is_board_safe,
    file_sha256,
    repository_relative,
    source_lock_matches_metadata,
    validate_public_board_contract,
    validate_replay_source_lock,
    visible_board_facts,
)
from evals.catan_board_bench.builder import (
    CatanObservationSuite,
    load_colonist_replay,
    step_replay,
)
from evals.catan_board_bench.render import RenderStyle, render_contract_image
from evals.catan_board_bench.tokens import (
    RECOGNITION_CLASS_VOCABULARIES,
    base_edges,
    edge_token,
    node_token,
    port_token,
    tile_token,
)
from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color


JsonDict = dict[str, Any]
PRIMARY_SPLITS = ("train", "validation", "test")
ALL_SPLITS = (*PRIMARY_SPLITS, "color_diagnostic")
DATASET_SCHEMA = "catan_board_recognition_dataset/v2"
SAMPLE_SCHEMA = "catan_board_recognition_sample/v2"
LABEL_SCHEMA = "catan_board_recognition_dense_labels/v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "generated" / "board_recognition" / "replay_v1"
DEFAULT_STYLE_PATH = PROJECT_ROOT / "configs" / "sft" / "renderer_style.json"
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_SEED = 381_427
REPLAY_TARGETS = {"train": 688, "validation": 64, "test": 64}
ENGINE_TRAIN_TARGET = 336
ENGINE_DIAGNOSTIC_TARGET = 64
PER_TRAJECTORY_CAP = 16
DENSITY_BINS = ("empty", "setup", "sparse", "dense")
ALL_COLOR_VALUES = tuple(Color)
TRADE_ACTIONS = {
    ActionType.OFFER_TRADE,
    ActionType.COUNTER_OFFER,
    ActionType.ACCEPT_TRADE,
    ActionType.REJECT_TRADE,
    ActionType.CONFIRM_TRADE,
    ActionType.CANCEL_TRADE,
}


@dataclass(frozen=True)
class BoardStateCandidate:
    trajectory_id: str
    board_fact_sha256: str
    board_map_sha256: str
    density_bin: str
    building_count: int
    road_count: int
    contract: JsonDict
    source: JsonDict


class ReplayDatasetBuildError(RuntimeError):
    """Raised when replay_v1 cannot satisfy a fail-closed corpus gate."""


def stable_seed(*parts: Any) -> int:
    payload = ":".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)


def load_render_style(path: Path = DEFAULT_STYLE_PATH) -> RenderStyle:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    values = payload.get("style", payload)
    keys = RenderStyle.__dataclass_fields__.keys()
    return RenderStyle(**{key: values[key] for key in keys if key in values})


def board_density(contract: JsonDict) -> tuple[int, int, str]:
    buildings = sum(node.get("building") is not None for node in contract["nodes"])
    roads = sum(edge.get("road_color") is not None for edge in contract["edges"])
    total = buildings + roads
    if total == 0:
        density = "empty"
    elif total <= 16:
        density = "setup"
    elif total <= 31:
        density = "sparse"
    else:
        density = "dense"
    return buildings, roads, density


def static_board_facts(contract: JsonDict) -> JsonDict:
    return {
        "tiles": [
            {
                "id": tile["id"],
                "resource": tile["resource"],
                "number": tile["number"],
            }
            for tile in contract["tiles"]
        ],
        "ports": [
            {
                "id": port["id"],
                "kind": port["kind"],
                "resource": port["resource"],
            }
            for port in contract["ports"]
        ],
    }


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
    state: Any,
    source_row: JsonDict,
) -> BoardStateCandidate:
    game_id = source_row["game_id"]
    replay_step = int(state.replay_index)
    source = {
        "kind": "colonist_replay",
        "game_id": game_id,
        "trajectory_id": f"replay:{game_id}",
        "replay_path": source_row["path"],
        "replay_step": replay_step,
        "source_sha256": source_row["sha256"],
        "engine_action_count": len(state.current_game.state.actions),
        "diagnostic_count": len(getattr(state, "replay_semantic_issues", [])),
    }
    contract = suite.public_board_contract(
        state.current_game,
        sample={
            "id": f"replay_{game_id}_s{replay_step:06d}",
            "index": replay_step,
        },
        source=source,
    )
    return _candidate(contract, trajectory_id=source["trajectory_id"], source=source)


def reconstruct_replay_candidates(source_row: JsonDict) -> list[BoardStateCandidate]:
    """Walk one accepted replay once and retain unique visible board states."""

    path = PROJECT_ROOT / source_row["path"]
    if not path.is_file() or file_sha256(path) != source_row["sha256"]:
        raise ReplayDatasetBuildError(f"locked replay payload changed: {path}")
    state = load_colonist_replay(path, quiet=True)
    ensure_replay_audit_state(state)
    suite = CatanObservationSuite()
    total_steps = len(state.replay_data.get("parsed_actions", []))
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
    selected = []
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


def candidate_key(candidates: Sequence[BoardStateCandidate]) -> str:
    return candidates[0].trajectory_id if candidates else "empty"


def select_split_candidates(
    candidates: Sequence[BoardStateCandidate],
    *,
    target: int,
    per_trajectory_cap: int = PER_TRAJECTORY_CAP,
    seed: int = DEFAULT_SEED,
) -> list[BoardStateCandidate]:
    """Select deterministically across density and trajectory without leakage."""

    by_trajectory: dict[str, list[BoardStateCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_trajectory[candidate.trajectory_id].append(candidate)
    capped = {
        trajectory: _cap_trajectory_candidates(rows, cap=per_trajectory_cap, seed=seed)
        for trajectory, rows in by_trajectory.items()
    }
    available = sum(len(rows) for rows in capped.values())
    if available < target:
        raise ReplayDatasetBuildError(
            f"candidate pool has {available} states after per-game cap; requires {target}"
        )

    trajectory_order = sorted(
        capped,
        key=lambda trajectory: (stable_seed(seed, "trajectory", trajectory), trajectory),
    )
    for trajectory in trajectory_order:
        capped[trajectory].sort(
            key=lambda row: (
                DENSITY_BINS.index(row.density_bin),
                stable_seed(seed, row.board_fact_sha256),
            )
        )

    selected: list[BoardStateCandidate] = []
    density_cursor = 0
    while len(selected) < target:
        progressed = False
        preferred_density = DENSITY_BINS[density_cursor % len(DENSITY_BINS)]
        density_cursor += 1
        for trajectory in trajectory_order:
            rows = capped[trajectory]
            index = next(
                (i for i, row in enumerate(rows) if row.density_bin == preferred_density),
                0 if rows else None,
            )
            if index is not None and len(selected) < target:
                selected.append(rows.pop(index))
                progressed = True
        if not progressed:
            break
    if len(selected) != target:
        raise ReplayDatasetBuildError(f"selected {len(selected)}/{target} states")
    return selected


def _canonical_payload_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _canonical_action_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_canonical_action_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_action_value(item) for item in value]
        return sorted(normalized, key=_canonical_payload_bytes)
    if isinstance(value, dict):
        return {
            str(key): _canonical_action_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    to_payload = getattr(value, "to_payload", None)
    if callable(to_payload):
        return _canonical_action_value(to_payload())
    raise TypeError(f"unsupported legal-action value for canonical ordering: {value!r}")


def legal_action_sort_key(action: Action) -> tuple[str, str, bytes]:
    return (
        action.color.value,
        action.action_type.value,
        _canonical_payload_bytes(_canonical_action_value(action.value)),
    )


def _weighted_choice(
    actions: Sequence[Action],
    policy_rng: random.Random,
) -> Action:
    weighted = []
    weights = {
        ActionType.BUILD_CITY: 20,
        ActionType.BUILD_SETTLEMENT: 8,
        ActionType.BUILD_ROAD: 5,
        ActionType.BUY_DEVELOPMENT_CARD: 2,
        ActionType.MARITIME_TRADE: 2,
        ActionType.END_TURN: 1,
    }
    for action in actions:
        weighted.extend([action] * weights.get(action.action_type, 1))
    return policy_rng.choice(weighted)


def choose_legal_datagen_action(engine: GameEngine, policy_rng: random.Random) -> Action:
    """Choose one advertised action without force or handcrafted state mutation."""

    actions = sorted(
        (
            action
            for action in engine.state.playable_actions
            if action.action_type not in TRADE_ACTIONS
        ),
        key=legal_action_sort_key,
    )
    if not actions:
        raise ReplayDatasetBuildError("engine datagen policy has no non-domestic legal action")
    prompt = engine.state.current_prompt
    if prompt != ActionPrompt.PLAY_TURN or engine.state.is_road_building:
        return policy_rng.choice(actions)
    roll = next((action for action in actions if action.action_type == ActionType.ROLL), None)
    if roll is not None:
        pre_roll_development = [
            action
            for action in actions
            if action.action_type
            in {
                ActionType.PLAY_KNIGHT_CARD,
                ActionType.PLAY_YEAR_OF_PLENTY,
                ActionType.PLAY_MONOPOLY,
                ActionType.PLAY_ROAD_BUILDING,
            }
        ]
        if pre_roll_development and policy_rng.random() < 0.15:
            return policy_rng.choice(pre_roll_development)
        return roll
    return _weighted_choice(actions, policy_rng)


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
    classes = set()
    for node in candidate.contract["nodes"]:
        if node["color"] and node["building"]:
            classes.add(f"node:{node['color']}_{node['building']}")
    for edge in candidate.contract["edges"]:
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


def dynamic_class_coverage_states(
    candidates: Sequence[BoardStateCandidate],
    *,
    required: set[str],
    seed: int,
    per_state_capacity: int = 2,
) -> list[BoardStateCandidate]:
    source = "source"
    sink = "sink"
    graph = nx.DiGraph()
    graph.add_node(source, demand=-len(required))
    graph.add_node(sink, demand=len(required))
    candidates_by_hash = {row.board_fact_sha256: row for row in candidates}
    for class_name in sorted(required):
        class_node = ("class", class_name)
        graph.add_node(class_node, demand=0)
        graph.add_edge(source, class_node, capacity=1, weight=0)
        matches = [row for row in candidates if class_name in candidate_dynamic_classes(row)]
        if not matches:
            raise ReplayDatasetBuildError(
                f"engine color diagnostic lacks source class: {class_name}"
            )
        for row in sorted(matches, key=lambda item: item.board_fact_sha256):
            state_node = ("state", row.board_fact_sha256)
            graph.add_node(state_node, demand=0)
            graph.add_edge(
                class_node,
                state_node,
                capacity=1,
                weight=stable_seed(seed, class_name, row.board_fact_sha256) % 10_000,
            )
            graph.add_edge(state_node, sink, capacity=per_state_capacity, weight=0)
    try:
        flow = nx.min_cost_flow(graph)
    except (nx.NetworkXUnfeasible, nx.NetworkXError) as exc:
        raise ReplayDatasetBuildError(
            "engine color diagnostic classes are not schedulable at "
            f"capacity {per_state_capacity} per state"
        ) from exc
    selected_hashes = {
        state_node[1]
        for class_name in required
        for state_node, amount in flow[("class", class_name)].items()
        if amount
    }
    return [candidates_by_hash[digest] for digest in sorted(selected_hashes)]


def select_color_diagnostic_candidates(
    candidates: Sequence[BoardStateCandidate],
    *,
    target: int,
    seed: int,
    per_trajectory_cap: int = PER_TRAJECTORY_CAP,
) -> list[BoardStateCandidate]:
    required = required_dynamic_classes()
    ordered = sorted(
        candidates,
        key=lambda row: (stable_seed(seed, row.board_fact_sha256), row.board_fact_sha256),
    )
    node_required = {name for name in required if name.startswith("node:")}
    edge_required = {name for name in required if name.startswith("edge:")}
    mandatory = {
        row.board_fact_sha256: row
        for row in (
            *dynamic_class_coverage_states(
                ordered,
                required=node_required,
                seed=stable_seed(seed, "node-coverage"),
            ),
            *dynamic_class_coverage_states(
                ordered,
                required=edge_required,
                seed=stable_seed(seed, "edge-coverage"),
            ),
        )
    }
    selected = sorted(
        mandatory.values(),
        key=lambda row: (stable_seed(seed, row.board_fact_sha256), row.board_fact_sha256),
    )
    if len(selected) > target:
        raise ReplayDatasetBuildError(
            f"diagnostic class coverage requires {len(selected)}/{target} states"
        )
    selected_hashes = set(mandatory)
    trajectory_counts = Counter(row.trajectory_id for row in selected)
    over_cap = {
        trajectory: count
        for trajectory, count in trajectory_counts.items()
        if count > per_trajectory_cap
    }
    if over_cap:
        raise ReplayDatasetBuildError(f"diagnostic coverage exceeds trajectory cap: {over_cap}")
    for density in DENSITY_BINS:
        for row in ordered:
            if len(selected) >= target:
                break
            if (
                row.density_bin == density
                and row.board_fact_sha256 not in selected_hashes
                and trajectory_counts[row.trajectory_id] < per_trajectory_cap
            ):
                selected.append(row)
                selected_hashes.add(row.board_fact_sha256)
                trajectory_counts[row.trajectory_id] += 1
        if len(selected) >= target:
            break
    if len(selected) < target:
        for row in ordered:
            if len(selected) >= target:
                break
            if (
                row.board_fact_sha256 not in selected_hashes
                and trajectory_counts[row.trajectory_id] < per_trajectory_cap
            ):
                selected.append(row)
                selected_hashes.add(row.board_fact_sha256)
                trajectory_counts[row.trajectory_id] += 1
    if len(selected) != target:
        raise ReplayDatasetBuildError(f"selected {len(selected)}/{target} color diagnostic states")
    return selected


def collect_engine_candidates(
    *,
    target: int,
    split: str,
    seed: int,
    start_index: int,
    max_trajectories: int = 256,
) -> list[BoardStateCandidate]:
    pool: list[BoardStateCandidate] = []
    for offset in range(max_trajectories):
        trajectory_index = start_index + offset
        rows = generate_engine_trajectory_candidates(
            trajectory_index=trajectory_index,
            seed=seed,
            split=split,
        )
        pool.extend(rows)
        if split == "train":
            enough = len(pool) >= target * 20 and offset >= 63
        elif split == "color_diagnostic":
            required = required_dynamic_classes()
            pool_classes = {
                name for candidate in pool for name in candidate_dynamic_classes(candidate)
            }
            coverage_ready = False
            if required <= pool_classes:
                try:
                    dynamic_class_coverage_states(
                        pool,
                        required={name for name in required if name.startswith("node:")},
                        seed=stable_seed(seed, split, "node-pool-coverage"),
                    )
                    dynamic_class_coverage_states(
                        pool,
                        required={name for name in required if name.startswith("edge:")},
                        seed=stable_seed(seed, split, "edge-pool-coverage"),
                    )
                except ReplayDatasetBuildError:
                    coverage_ready = False
                else:
                    coverage_ready = True
            enough = len(pool) >= target * 3 and offset >= 15 and coverage_ready
        else:
            enough = len(pool) >= target * 3 and offset >= 15
        if enough:
            break
    selection_seed = stable_seed(seed, split)
    if split == "train":
        return select_balanced_engine_training_candidates(
            pool,
            target=target,
            seed=selection_seed,
        )
    if split == "color_diagnostic":
        return select_color_diagnostic_candidates(pool, target=target, seed=selection_seed)
    return select_split_candidates(pool, target=target, seed=selection_seed)


def normalize_class_name(attribute: str, value: Any) -> str:
    if attribute == "number":
        return "NONE" if value is None else str(value)
    if attribute == "robber":
        return "PRESENT" if value else "ABSENT"
    return str(value)


def dense_labels(contract: JsonDict, *, sample_id: str) -> JsonDict:
    labels = {
        "schema": LABEL_SCHEMA,
        "sample_id": sample_id,
        "entities": {
            "tiles": [
                {
                    "id": tile_token(tile["id"]),
                    "resource": "DESERT" if tile["resource"] is None else tile["resource"],
                    "number": tile["number"],
                    "robber": bool(tile["has_robber"]),
                }
                for tile in sorted(contract["tiles"], key=lambda row: int(row["id"]))
            ],
            "nodes": [
                {
                    "id": node_token(node["id"]),
                    "occupancy": (
                        "EMPTY"
                        if node["color"] is None or node["building"] is None
                        else f"{node['color']}_{node['building']}"
                    ),
                }
                for node in sorted(contract["nodes"], key=lambda row: int(row["id"]))
            ],
            "edges": [
                {
                    "id": edge_token(tuple(edge["id"])),
                    "owner": edge["road_color"] or "EMPTY",
                }
                for edge in sorted(contract["edges"], key=lambda row: tuple(row["id"]))
            ],
            "ports": [
                {
                    "id": port_token(port["id"]),
                    "port_type": (
                        "THREE_TO_ONE"
                        if port["resource"] is None
                        else f"TWO_TO_ONE_{port['resource']}"
                    ),
                }
                for port in sorted(contract["ports"], key=lambda row: int(row["id"]))
            ],
        },
    }
    validate_dense_labels(labels)
    return labels


def validate_dense_labels(labels: JsonDict) -> None:
    entities = labels.get("entities", {})
    expected_lengths = {"tiles": 19, "nodes": 54, "edges": 72, "ports": 9}
    actual_lengths = {name: len(entities.get(name, [])) for name in expected_lengths}
    if actual_lengths != expected_lengths:
        raise ReplayDatasetBuildError(
            f"dense label topology mismatch: {actual_lengths} != {expected_lengths}"
        )
    expected_ids = {
        "tiles": {tile_token(index) for index in range(19)},
        "nodes": {node_token(index) for index in range(54)},
        "ports": {port_token(index) for index in range(9)},
        "edges": {edge_token(edge) for edge in base_edges()},
    }
    for collection, identifiers in expected_ids.items():
        if {row["id"] for row in entities[collection]} != identifiers:
            raise ReplayDatasetBuildError(f"dense label {collection} IDs are incomplete")
    checks = (
        ("tiles", "resource", "tile.resource"),
        ("tiles", "number", "tile.number"),
        ("tiles", "robber", "tile.robber"),
        ("nodes", "occupancy", "node.occupancy"),
        ("edges", "owner", "edge.owner"),
        ("ports", "port_type", "port.port_type"),
    )
    for collection, attribute, head in checks:
        vocabulary = set(RECOGNITION_CLASS_VOCABULARIES[head])
        for row in entities[collection]:
            class_name = normalize_class_name(attribute, row[attribute])
            if class_name not in vocabulary:
                raise ReplayDatasetBuildError(
                    f"unknown dense class {class_name!r} for {head} in {labels['sample_id']}"
                )
    tiles = entities["tiles"]
    if sum(tile["robber"] for tile in tiles) != 1:
        raise ReplayDatasetBuildError("dense labels require exactly one robber")
    if any((tile["resource"] == "DESERT") != (tile["number"] is None) for tile in tiles):
        raise ReplayDatasetBuildError("dense tile resource/number labels disagree")


def prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {path}; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_candidate(
    output_dir: Path,
    candidate: BoardStateCandidate,
    *,
    split: str,
    sample_index: int,
    image_size: int,
    style: RenderStyle,
    style_path: Path,
) -> JsonDict:
    source_prefix = "r" if candidate.source["kind"] == "colonist_replay" else "e"
    source_position = (
        candidate.source["replay_step"]
        if candidate.source["kind"] == "colonist_replay"
        else candidate.source["engine_action_count"]
    )
    trajectory_slug = candidate.trajectory_id.replace(":", "_")
    sample_id = f"{source_prefix}_{trajectory_slug}_s{int(source_position or 0):06d}"
    contract_rel = Path("contracts") / f"{sample_id}.json"
    label_rel = Path("dense_labels") / f"{sample_id}.json"
    image_rel = Path("images") / f"{sample_id}.png"
    contract = copy.deepcopy(candidate.contract)
    contract["sample"] = {
        "id": sample_id,
        "index": sample_index,
        "contract_path": str(contract_rel),
        "image_path": str(image_rel),
        "image_size": [image_size, image_size],
    }
    contract["source"] = {**candidate.source, "split": split}
    labels = dense_labels(contract, sample_id=sample_id)
    image = render_contract_image(contract, image_size=image_size, style=style)
    write_json(output_dir / contract_rel, contract)
    write_json(output_dir / label_rel, labels)
    image.save(output_dir / image_rel)
    colors = sorted(
        {node["color"] for node in contract["nodes"] if node["color"] is not None}
        | {edge["road_color"] for edge in contract["edges"] if edge["road_color"] is not None}
    )
    return {
        "schema": SAMPLE_SCHEMA,
        "sample_id": sample_id,
        "split": split,
        "view": "raw_full_board",
        "contract_path": str(contract_rel),
        "label_path": str(label_rel),
        "image_path": str(image_rel),
        "image_size": [image_size, image_size],
        "source": contract["source"],
        "board_fact_sha256": candidate.board_fact_sha256,
        "board_map_sha256": candidate.board_map_sha256,
        "building_count": candidate.building_count,
        "road_count": candidate.road_count,
        "density_bin": candidate.density_bin,
        "color_palette": [player["color"] for player in contract["players"]],
        "visible_piece_colors": colors,
        "render": {
            "renderer": "evals.catan_board_bench.render",
            "style_config": repository_relative(style_path),
            "image_annotation": None,
        },
        "sha256": {
            "contract": file_sha256(output_dir / contract_rel),
            "labels": file_sha256(output_dir / label_rel),
            "image": file_sha256(output_dir / image_rel),
        },
    }


def _assert_unique_fact_hashes(rows: Sequence[BoardStateCandidate]) -> None:
    counts = Counter(row.board_fact_sha256 for row in rows)
    duplicates = [digest for digest, count in counts.items() if count > 1]
    if duplicates:
        raise ReplayDatasetBuildError(
            f"board-fact hashes are not globally unique: {duplicates[:10]}"
        )


def build_replay_v1_dataset(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    source_lock_path: Path = DEFAULT_SOURCE_LOCK,
    image_size: int = DEFAULT_IMAGE_SIZE,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> JsonDict:
    if image_size < 64 or image_size % 16:
        raise ValueError("image_size must be at least 64 and divisible by 16")
    source_lock = json.loads(source_lock_path.read_text())
    validate_replay_source_lock(source_lock, minimum_accepted=40)
    source_splits = split_replay_games(source_lock["accepted"], seed=seed)

    selected_by_split: dict[str, list[BoardStateCandidate]] = {}
    for split, sources in source_splits.items():
        candidates = []
        for source in sources:
            candidates.extend(reconstruct_replay_candidates(source))
        selected_by_split[split] = select_split_candidates(
            candidates,
            target=REPLAY_TARGETS[split],
            seed=stable_seed(seed, split),
        )

    engine_train = collect_engine_candidates(
        target=ENGINE_TRAIN_TARGET,
        split="train",
        seed=seed,
        start_index=0,
    )
    diagnostic = collect_engine_candidates(
        target=ENGINE_DIAGNOSTIC_TARGET,
        split="color_diagnostic",
        seed=seed,
        start_index=10_000,
    )
    selected_by_split["train"].extend(engine_train)
    selected_by_split["color_diagnostic"] = diagnostic
    all_candidates = [row for rows in selected_by_split.values() for row in rows]
    _assert_unique_fact_hashes(all_candidates)

    if {split: len(selected_by_split[split]) for split in PRIMARY_SPLITS} != {
        "train": 1_024,
        "validation": 64,
        "test": 64,
    }:
        raise ReplayDatasetBuildError("primary state quotas were not met")
    if len(diagnostic) != ENGINE_DIAGNOSTIC_TARGET:
        raise ReplayDatasetBuildError("color diagnostic state quota was not met")

    prepare_output_dir(output_dir, overwrite=overwrite)
    for name in ("contracts", "dense_labels", "images", "splits", "diagnostics"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    style_path = DEFAULT_STYLE_PATH
    style = load_render_style(style_path)
    manifest_rows = []
    for split in ALL_SPLITS:
        candidates = selected_by_split[split]
        candidates.sort(
            key=lambda row: (
                row.trajectory_id,
                (
                    row.source.get("replay_step")
                    if row.source["kind"] == "colonist_replay"
                    else row.source["engine_action_count"]
                ),
            )
        )
        for candidate in candidates:
            manifest_rows.append(
                _write_candidate(
                    output_dir,
                    candidate,
                    split=split,
                    sample_index=len(manifest_rows),
                    image_size=image_size,
                    style=style,
                    style_path=style_path,
                )
            )

    manifest_rows.sort(key=lambda row: row["sample_id"])
    write_jsonl(output_dir / "manifest.jsonl", manifest_rows)
    for split in PRIMARY_SPLITS:
        write_jsonl(
            output_dir / "splits" / f"{split}.jsonl",
            (row for row in manifest_rows if row["split"] == split),
        )
    write_jsonl(
        output_dir / "diagnostics" / "color_diagnostic.jsonl",
        (row for row in manifest_rows if row["split"] == "color_diagnostic"),
    )

    metadata = {
        "schema": DATASET_SCHEMA,
        "seed": seed,
        "sample_count": 1_152,
        "diagnostic_sample_count": 64,
        "image_size": [image_size, image_size],
        "source_lock_path": repository_relative(source_lock_path),
        "source_lock_sha256": source_lock["lock_sha256"],
        "source_lock_file_sha256": file_sha256(source_lock_path),
        "leakage_ledger_sha256": source_lock["leakage"]["ledger_sha256"],
        "split_counts": dict(sorted(Counter(row["split"] for row in manifest_rows).items())),
        "source_counts": dict(
            sorted(Counter(row["source"]["kind"] for row in manifest_rows).items())
        ),
        "density_counts": dict(
            sorted(Counter(row["density_bin"] for row in manifest_rows).items())
        ),
        "color_counts": dict(
            sorted(
                Counter(
                    color for row in manifest_rows for color in row["visible_piece_colors"]
                ).items()
            )
        ),
        "replay_split_games": {
            split: sorted(source["game_id"] for source in sources)
            for split, sources in source_splits.items()
        },
        "render": {
            "renderer": "evals.catan_board_bench.render",
            "style_config": repository_relative(style_path),
            "style_sha256": file_sha256(style_path),
            "image_annotation": None,
        },
        "files": {
            "manifest": "manifest.jsonl",
            "contracts_dir": "contracts",
            "labels_dir": "dense_labels",
            "images_dir": "images",
            "splits_dir": "splits",
            "color_diagnostic": "diagnostics/color_diagnostic.jsonl",
        },
    }
    write_json(output_dir / "metadata.json", metadata)
    return validate_replay_v1_dataset(output_dir, rerender=True)


def validate_replay_v1_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    rerender: bool = True,
) -> JsonDict:
    metadata = json.loads((output_dir / "metadata.json").read_text())
    if metadata.get("schema") != DATASET_SCHEMA:
        raise ReplayDatasetBuildError("dataset metadata schema mismatch")
    lock_path = PROJECT_ROOT / metadata["source_lock_path"]
    lock = json.loads(lock_path.read_text())
    validate_replay_source_lock(lock, minimum_accepted=40)
    if not source_lock_matches_metadata(
        lock,
        lock_sha256=metadata["source_lock_sha256"],
        file_sha256_value=metadata["source_lock_file_sha256"],
    ):
        if metadata["source_lock_sha256"] != lock["lock_sha256"]:
            raise ReplayDatasetBuildError("source lock identity changed")
        raise ReplayDatasetBuildError("source lock file changed")

    rows = read_jsonl(output_dir / "manifest.jsonl")
    if len(rows) != 1_216 or len({row["sample_id"] for row in rows}) != len(rows):
        raise ReplayDatasetBuildError("manifest state count or IDs are invalid")
    expected_counts = {
        "train": 1_024,
        "validation": 64,
        "test": 64,
        "color_diagnostic": 64,
    }
    if Counter(row["split"] for row in rows) != Counter(expected_counts):
        raise ReplayDatasetBuildError("manifest split counts are invalid")
    if len({row["board_fact_sha256"] for row in rows}) != len(rows):
        raise ReplayDatasetBuildError("board facts are duplicated")
    if len({row["sha256"]["image"] for row in rows}) != len(rows):
        raise ReplayDatasetBuildError("rendered images are duplicated")

    replay_games: dict[str, set[str]] = defaultdict(set)
    board_maps: dict[str, set[str]] = defaultdict(set)
    engine_seeds: dict[str, set[int]] = defaultdict(set)
    style = load_render_style(DEFAULT_STYLE_PATH)
    diagnostic_dynamic_classes: set[str] = set()
    diagnostic_candidates: list[BoardStateCandidate] = []
    for row in rows:
        if row.get("schema") != SAMPLE_SCHEMA or row.get("view") != "raw_full_board":
            raise ReplayDatasetBuildError(f"sample contract mismatch: {row.get('sample_id')}")
        for digest_key, path_key in (
            ("contract", "contract_path"),
            ("labels", "label_path"),
            ("image", "image_path"),
        ):
            path = output_dir / row[path_key]
            if not path.is_file() or file_sha256(path) != row["sha256"][digest_key]:
                raise ReplayDatasetBuildError(f"artifact hash mismatch: {path}")
        contract = json.loads((output_dir / row["contract_path"]).read_text())
        labels = json.loads((output_dir / row["label_path"]).read_text())
        validate_public_board_contract(contract)
        validate_dense_labels(labels)
        if labels != dense_labels(contract, sample_id=row["sample_id"]):
            raise ReplayDatasetBuildError(f"labels disagree with contract: {row['sample_id']}")
        if canonical_sha256(visible_board_facts(contract)) != row["board_fact_sha256"]:
            raise ReplayDatasetBuildError(f"board fact hash mismatch: {row['sample_id']}")
        if canonical_sha256(static_board_facts(contract)) != row["board_map_sha256"]:
            raise ReplayDatasetBuildError(f"board map hash mismatch: {row['sample_id']}")
        board_maps[row["split"]].add(row["board_map_sha256"])
        if board_density(contract) != (
            row["building_count"],
            row["road_count"],
            row["density_bin"],
        ):
            raise ReplayDatasetBuildError(f"density metadata mismatch: {row['sample_id']}")
        if row["split"] == "color_diagnostic":
            diagnostic_candidates.append(
                BoardStateCandidate(
                    trajectory_id=row["source"]["trajectory_id"],
                    board_fact_sha256=row["board_fact_sha256"],
                    board_map_sha256=row["board_map_sha256"],
                    density_bin=row["density_bin"],
                    building_count=row["building_count"],
                    road_count=row["road_count"],
                    contract=contract,
                    source=row["source"],
                )
            )
            diagnostic_dynamic_classes.update(
                f"node:{node['color']}_{node['building']}"
                for node in contract["nodes"]
                if node["color"] and node["building"]
            )
            diagnostic_dynamic_classes.update(
                f"edge:{edge['road_color']}" for edge in contract["edges"] if edge["road_color"]
            )
        source = row["source"]
        if source["kind"] == "colonist_replay":
            replay_games[row["split"]].add(source["game_id"])
            if source["game_id"] in lock["leakage"]["excluded_game_ids"]:
                raise ReplayDatasetBuildError(f"benchmark game leaked: {source['game_id']}")
        elif source["kind"] == "engine_rollout":
            engine_seeds[row["split"]].add(int(source["engine_seed"]))
            if row["split"] not in {"train", "color_diagnostic"}:
                raise ReplayDatasetBuildError("engine rollout entered replay validation/test")
            if not source.get("legal_actions_only"):
                raise ReplayDatasetBuildError("engine rollout lacks legal-action evidence")
        else:
            raise ReplayDatasetBuildError(f"unknown source kind: {source['kind']}")
        if rerender:
            expected = render_contract_image(contract, image_size=row["image_size"][0], style=style)
            with Image.open(output_dir / row["image_path"]) as image_file:
                actual = image_file.convert("RGB")
                if (
                    actual.size != tuple(row["image_size"])
                    or actual.tobytes() != expected.tobytes()
                ):
                    raise ReplayDatasetBuildError(
                        f"image disagrees with contract: {row['sample_id']}"
                    )

    replay_split_sets = [replay_games[split] for split in PRIMARY_SPLITS]
    if any(
        left & right
        for index, left in enumerate(replay_split_sets)
        for right in replay_split_sets[index + 1 :]
    ):
        raise ReplayDatasetBuildError("source replay game crossed dataset splits")
    if engine_seeds["train"] & engine_seeds["color_diagnostic"]:
        raise ReplayDatasetBuildError("engine diagnostic seeds overlap training")
    split_map_sets = [board_maps[split] for split in ALL_SPLITS]
    if any(
        left & right
        for index, left in enumerate(split_map_sets)
        for right in split_map_sets[index + 1 :]
    ):
        raise ReplayDatasetBuildError("static board map crossed dataset splits")
    missing_dynamic = sorted(required_dynamic_classes() - diagnostic_dynamic_classes)
    if missing_dynamic:
        raise ReplayDatasetBuildError(f"color diagnostic lacks dynamic classes: {missing_dynamic}")
    required = required_dynamic_classes()
    dynamic_class_coverage_states(
        diagnostic_candidates,
        required={name for name in required if name.startswith("node:")},
        seed=stable_seed(metadata["seed"], "validation", "node-coverage"),
    )
    dynamic_class_coverage_states(
        diagnostic_candidates,
        required={name for name in required if name.startswith("edge:")},
        seed=stable_seed(metadata["seed"], "validation", "edge-coverage"),
    )
    if set(metadata["split_counts"]) != set(expected_counts):
        raise ReplayDatasetBuildError("metadata split counts are stale")
    return {
        "valid": True,
        "samples": 1_152,
        "diagnostic_samples": 64,
        "splits": expected_counts,
        "sources": dict(sorted(Counter(row["source"]["kind"] for row in rows).items())),
        "density": dict(sorted(Counter(row["density_bin"] for row in rows).items())),
        "replay_games": {split: len(replay_games[split]) for split in PRIMARY_SPLITS},
    }
