"""Benchmark game inventory, training candidates, and stored eval results."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from cle.players.data import JsonValue
from scripts.board_bench.builders.build_catan_board_bench_leakage_and_presft.paths import (
    INDEX_FILES,
    MANIFEST_PATH,
    METADATA_PATH,
    OPENROUTER_EVAL_DIR,
    QWEN_KEY,
    RAW_REPLAY_DIR,
)
from scripts.board_bench.shapes import (
    JsonDict,
    obj,
    objs,
    read_json,
    read_json_object,
    read_jsonl,
)

__all__ = [
    "build_training_candidates",
    "load_benchmark_games",
    "load_eval_summaries",
    "load_qwen_failures",
    "load_raw_replay_ids",
    "norm_game_id",
    "pct",
]


def norm_game_id(value: JsonValue) -> str:
    raw = str(value)
    match = re.search(r"(\d+)", raw)
    return match.group(1) if match else raw


def pct(value: JsonValue) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    return f"{float(value) * 100:.1f}%"


def load_benchmark_games() -> tuple[dict[str, JsonDict], Counter[str]]:
    samples = read_jsonl(MANIFEST_PATH)
    sample_counts: Counter[str] = Counter()
    steps_by_game: dict[str, list[int]] = defaultdict(list)
    replay_files_by_game: dict[str, set[str]] = defaultdict(set)

    for sample in samples:
        source = obj(sample.get("source", {}), "sample source")
        game_id = norm_game_id(source.get("game_id") or source.get("replay_id") or "")
        if not game_id:
            continue
        sample_counts[game_id] += 1
        replay_step = source.get("replay_step")
        if replay_step is not None:
            steps_by_game[game_id].append(int(str(replay_step)))
        replay_file = source.get("replay_file") or source.get("path")
        if replay_file:
            replay_files_by_game[game_id].add(str(replay_file))

    metadata = read_json_object(METADATA_PATH)
    for key, count in obj(metadata.get("counts", {}), "metadata counts").items():
        if not key.startswith("replay:"):
            continue
        replay_file = key.removeprefix("replay:")
        game_id = norm_game_id(replay_file)
        sample_counts.setdefault(game_id, int(str(count)))
        replay_files_by_game[game_id].add(replay_file)

    games: dict[str, JsonDict] = {}
    for game_id in sorted(sample_counts, key=int):
        games[game_id] = {
            "game_id": game_id,
            "sample_count": sample_counts[game_id],
            "replay_steps": list(sorted(set(steps_by_game.get(game_id, [])))),
            "replay_files": list(sorted(replay_files_by_game.get(game_id, []))),
        }
    return games, sample_counts


def load_raw_replay_ids() -> list[str]:
    if not RAW_REPLAY_DIR.exists():
        return []
    return sorted({norm_game_id(path.name) for path in RAW_REPLAY_DIR.glob("*.json")}, key=int)


def build_training_candidates(
    benchmark_ids: set[str],
) -> tuple[list[JsonDict], dict[str, int]]:
    seen: set[str] = set()
    candidates: list[JsonDict] = []
    source_counts: dict[str, int] = {}

    for index_path in INDEX_FILES:
        if not index_path.exists():
            continue
        index_rows = objs(read_json(index_path), str(index_path))
        source_counts[index_path.name] = len(index_rows)
        for row in index_rows:
            game_id = norm_game_id(row.get("game_id") or row.get("id") or "")
            if not game_id or game_id in benchmark_ids or game_id in seen:
                continue
            seen.add(game_id)
            candidate = dict(row)
            candidate["game_id"] = game_id
            candidate["source_index"] = index_path.name
            candidates.append(candidate)

    return candidates, source_counts


def load_eval_summaries() -> list[JsonDict]:
    summaries: list[JsonDict] = []
    for summary_path in sorted(OPENROUTER_EVAL_DIR.glob("*/summary.json")):
        run_dir = summary_path.parent
        plan_path = run_dir / "plan.json"
        summary = read_json_object(summary_path)
        plan: JsonDict = read_json_object(plan_path) if plan_path.exists() else {}
        plan_models = obj(plan.get("models", {}), "plan models")
        for model_key, metrics_value in sorted(obj(summary.get("models", {}), "summary models").items()):
            metrics = obj(metrics_value, "model metrics")
            raw_categories = plan.get("categories")
            categories: JsonValue = raw_categories or list(
                sorted(obj(metrics.get("categories", {}), "metric categories"))
            )
            summaries.append(
                {
                    "run_id": run_dir.name,
                    "model_key": model_key,
                    "model_id": plan_models.get(model_key, model_key),
                    "question_count": plan.get("question_count", metrics.get("requests")),
                    "categories": categories,
                    "metrics": metrics,
                }
            )
    return summaries


def load_qwen_failures() -> list[JsonDict]:
    failures: list[JsonDict] = []
    for response_path in sorted(OPENROUTER_EVAL_DIR.glob("*/responses.jsonl")):
        run_id = response_path.parent.name
        for row in read_jsonl(response_path):
            if row.get("model_key") != QWEN_KEY:
                continue
            score = obj(row.get("score") or {}, "response score")
            if score.get("correct") is True:
                continue
            failures.append(
                {
                    "run_id": run_id,
                    "category": row.get("category"),
                    "question_id": row.get("question_id"),
                    "expected": row.get("expected"),
                    "response": row.get("response"),
                    "component_accuracy": score.get("component_accuracy"),
                }
            )
    return failures
