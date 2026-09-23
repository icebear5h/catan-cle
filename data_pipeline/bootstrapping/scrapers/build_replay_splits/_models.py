"""Typed shapes for the replay split manifest and its intermediate records."""

from typing import TypedDict

from data_pipeline.json_types import JsonValue


class SourceRecord(TypedDict):
    """One index row flattened onto the game it belongs to."""

    username: JsonValue
    player_rank: JsonValue
    player_rating: JsonValue
    player_color: int
    player_color_name: str
    result: JsonValue
    mode: JsonValue
    turnCount: JsonValue
    date: JsonValue
    source_index: str


class StepPlan(TypedDict, total=False):
    """Step fractions for one band family, with resolved indices when known."""

    fractions: dict[str, list[float]]
    event_indices: dict[str, list[int]]


class GameEntry(TypedDict, total=False):
    """A deduplicated game, built up across grouping, enrichment, and splitting."""

    game_id: str
    source_index_files: list[str]
    source_records: list[SourceRecord]
    indexed_player_color_ids: list[int]
    indexed_player_color_names: list[str]
    balance_color_id: int
    balance_color_name: str
    turnCount: JsonValue
    mode: JsonValue
    date: JsonValue
    replay_url: JsonValue
    raw_replay_present: bool
    raw_replay_path: str | None
    event_count: int | None
    step_plans: dict[str, StepPlan]
    split: str


class SplitSummary(TypedDict):
    """Per-split counts reported in the manifest and the markdown report."""

    games: int
    raw_replays_present: int
    balance_color_counts: dict[str, int]
    mode_counts: dict[str, int]


class Summary(TypedDict):
    """Aggregate counts across every split."""

    total_games: int
    splits: dict[str, SplitSummary]


class Manifest(TypedDict):
    """The `colonist_replay_splits/v0` document written to disk."""

    schema: str
    generated_at: str
    source_index_files: list[str]
    exclude_game_ids: str
    excluded_game_count: int
    short_game_count: int
    modes: list[str] | str
    min_turn_count: int
    seed: int
    ratios: dict[str, float]
    step_plans: dict[str, dict[str, list[float]]]
    summary: Summary
    splits: dict[str, list[GameEntry]]


__all__ = [
    "GameEntry",
    "Manifest",
    "SourceRecord",
    "SplitSummary",
    "StepPlan",
    "Summary",
]
