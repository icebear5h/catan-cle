"""Dynamic-class coverage and colour-diagnostic candidates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import TypeAlias

import networkx as nx

from data_pipeline.board_recognition.replay_impl._config import (
    DENSITY_BINS,
    PER_TRAJECTORY_CAP,
    BoardStateCandidate,
    ReplayDatasetBuildError,
)
from data_pipeline.board_recognition.replay_impl._engine import (
    candidate_dynamic_classes,
    generate_engine_trajectory_candidates,
    required_dynamic_classes,
    select_balanced_engine_training_candidates,
)
from data_pipeline.board_recognition.replay_impl._facts import (
    stable_seed,
)
from data_pipeline.board_recognition.replay_impl._selection import (
    select_split_candidates,
)

GraphNode: TypeAlias = "str | tuple[str, str]"
CoverageGraph: TypeAlias = "nx.DiGraph[GraphNode, dict[str, object], dict[str, object]]"


def dynamic_class_coverage_states(
    candidates: Sequence[BoardStateCandidate],
    *,
    required: set[str],
    seed: int,
    per_state_capacity: int = 2,
) -> list[BoardStateCandidate]:
    source = "source"
    sink = "sink"
    graph: CoverageGraph = nx.DiGraph()
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

