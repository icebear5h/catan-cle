"""Structured replay event and replay dataclasses."""

from dataclasses import asdict, dataclass

from data_pipeline.json_types import JsonDict, JsonValue


@dataclass
class ReplayEvent:
    """A single event in the replay."""
    event_index: int
    delta_seconds: float
    state_change: JsonDict

    # Parsed fields
    action_type: str | None = None
    acting_player: int | None = None
    dice_roll: tuple[JsonValue, JsonValue] | None = None
    resources_gained: dict[int, JsonValue] | None = None  # player -> resources
    building_placed: JsonDict | None = None
    road_placed: JsonDict | None = None
    trade_offer: JsonDict | None = None


@dataclass
class ParsedReplay:
    """Fully parsed replay data."""
    game_id: str
    player_color: int
    total_events: int
    events: list[ReplayEvent]
    initial_state: JsonValue = None
    final_state: JsonValue = None
    winner: int | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "game_id": self.game_id,
            "player_color": self.player_color,
            "total_events": self.total_events,
            "events": [asdict(e) for e in self.events],
            "initial_state": self.initial_state,
            "final_state": self.final_state,
            "winner": self.winner,
        }


__all__ = ["ParsedReplay", "ReplayEvent"]
