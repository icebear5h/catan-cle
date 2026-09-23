"""Load Colonist game-index rows and group them into deduplicated games."""

import json
from collections import Counter
from pathlib import Path

from data_pipeline.bootstrapping.scrapers.build_replay_splits._config import (
    COLONIST_COLOR_NAMES,
    ROOT,
)
from data_pipeline.bootstrapping.scrapers.build_replay_splits._models import (
    GameEntry,
    SourceRecord,
    StepPlan,
)
from data_pipeline.json_types import JsonValue


def load_json(path: Path) -> JsonValue:
    payload: JsonValue = json.loads(path.read_text())
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def portable_path(value: str | Path) -> str:
    """Return a repository-relative path when the target is inside the repo."""

    path = Path(value).expanduser()
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def norm_game_id(value: JsonValue) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.removesuffix(".json").removesuffix("_sample")


def load_excluded_game_ids(path: Path) -> set[str]:
    """Read the CatanBoardBench holdout ids that must never enter a split."""

    if not path.exists():
        return set()
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    raw_ids = payload.get("benchmark_game_ids", [])
    ids = raw_ids if isinstance(raw_ids, list) else []
    return {game_id for game_id in (norm_game_id(value) for value in ids) if game_id}


def parse_modes(raw: str | None) -> set[str] | None:
    if not raw or raw.lower() == "all":
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def _source_index_value(row: dict[str, JsonValue], path: Path) -> str:
    candidate = row.get("source_index")
    if isinstance(candidate, str) and candidate:
        return portable_path(candidate)
    return portable_path(path)


def _new_entry(game_id: str, row: dict[str, JsonValue], mode: JsonValue) -> GameEntry:
    return {
        "game_id": game_id,
        "source_index_files": [],
        "source_records": [],
        "turnCount": row.get("turnCount"),
        "mode": mode,
        "date": row.get("date"),
        "replay_url": row.get("replay_url"),
    }


def _source_record(row: dict[str, JsonValue], player_color: int, path: Path) -> SourceRecord:
    return {
        "username": row.get("username"),
        "player_rank": row.get("player_rank"),
        "player_rating": row.get("player_rating"),
        "player_color": player_color,
        "player_color_name": COLONIST_COLOR_NAMES.get(player_color, f"COLOR_{player_color}"),
        "result": row.get("result"),
        "mode": row.get("mode"),
        "turnCount": row.get("turnCount"),
        "date": row.get("date"),
        "source_index": _source_index_value(row, path),
    }


def _carry_forward(entry: GameEntry, row: dict[str, JsonValue], mode: JsonValue) -> None:
    """Fill entry fields that the first row for this game left empty."""

    if entry.get("turnCount") is None and row.get("turnCount") is not None:
        entry["turnCount"] = row.get("turnCount")
    if entry.get("mode") is None and mode is not None:
        entry["mode"] = mode
    if entry.get("date") is None and row.get("date") is not None:
        entry["date"] = row.get("date")
    if entry.get("replay_url") is None and row.get("replay_url") is not None:
        entry["replay_url"] = row.get("replay_url")


def _finalize_colors(
    grouped: dict[str, GameEntry], colors: dict[str, set[int]]
) -> None:
    """Assign each game the globally rarest of the colours it was indexed under."""

    color_counter: Counter[int] = Counter()
    for game_id in grouped:
        color_counter.update(colors[game_id] or {0})

    for game_id, entry in grouped.items():
        color_ids = sorted(colors[game_id] or {0})
        balance_color = min(color_ids, key=lambda color_id: (color_counter[color_id], color_id))
        entry["indexed_player_color_ids"] = color_ids
        entry["indexed_player_color_names"] = [
            COLONIST_COLOR_NAMES.get(color_id, f"COLOR_{color_id}") for color_id in color_ids
        ]
        entry["balance_color_id"] = balance_color
        entry["balance_color_name"] = COLONIST_COLOR_NAMES.get(
            balance_color, f"COLOR_{balance_color}"
        )
        entry["source_index_files"] = sorted(set(entry["source_index_files"]))


def load_index_records(paths: list[Path], modes: set[str] | None) -> dict[str, GameEntry]:
    """Group index rows from every file into one entry per game id."""

    grouped: dict[str, GameEntry] = {}
    colors: dict[str, set[int]] = {}
    for path in paths:
        rows = load_json(path)
        if not isinstance(rows, list):
            raise ValueError(f"{path} must contain a JSON list")

        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"{path} must contain a JSON list of objects")
            game_id = norm_game_id(row.get("game_id") or row.get("id"))
            if not game_id:
                continue

            mode = row.get("mode")
            if modes is not None and mode is not None and mode not in modes:
                continue

            raw_color = row.get("player_color", 0) or 0
            player_color = int(raw_color) if isinstance(raw_color, (int, float, str)) else 0
            if game_id not in grouped:
                grouped[game_id] = _new_entry(game_id, row, mode)
                colors[game_id] = set()
            entry = grouped[game_id]
            entry["source_index_files"].append(portable_path(path))
            entry["source_records"].append(_source_record(row, player_color, path))
            if player_color:
                colors[game_id].add(player_color)

            _carry_forward(entry, row, mode)

    _finalize_colors(grouped, colors)
    return grouped


def replay_event_count(raw_replay_dir: Path, game_id: str) -> int | None:
    """Count decoded events for a downloaded replay, or None when absent."""

    path = raw_replay_dir / f"{game_id}.json"
    if not path.exists():
        return None
    payload = load_json(path)
    data = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not isinstance(data, dict):
        return None
    event_history = data.get("eventHistory", data)
    if not isinstance(event_history, dict):
        return None
    events = event_history.get("events")
    return len(events) if isinstance(events, list) else None


def planned_steps(fractions: dict[str, list[float]], event_count: int | None) -> StepPlan:
    payload: StepPlan = {"fractions": fractions}
    if event_count:
        payload["event_indices"] = {
            band: sorted(
                {max(0, min(event_count - 1, round(frac * (event_count - 1)))) for frac in values}
            )
            for band, values in fractions.items()
        }
    return payload


__all__ = [
    "load_excluded_game_ids",
    "load_index_records",
    "load_json",
    "norm_game_id",
    "parse_modes",
    "planned_steps",
    "portable_path",
    "replay_event_count",
    "write_json",
]
