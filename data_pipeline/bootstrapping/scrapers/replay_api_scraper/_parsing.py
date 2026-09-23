"""Turn raw replay payloads into structured events and a parsed replay."""

from data_pipeline.bootstrapping.scrapers.replay_api_scraper._config import (
    BUILDING_TYPE,
    EDGE_TYPE,
    RESOURCE_ENUM,
)
from data_pipeline.bootstrapping.scrapers.replay_api_scraper._models import (
    ParsedReplay,
    ReplayEvent,
)
from data_pipeline.json_coerce import as_dict, as_float, as_int, as_list
from data_pipeline.json_types import JsonDict, JsonValue


def _section(container: JsonDict, key: str) -> JsonDict:
    """Read a nested object, treating an absent or null section as empty."""
    value = container.get(key)
    if value is None:
        return {}
    return as_dict(value)


def _resource_names(values: JsonValue) -> list[JsonValue]:
    """Map Colonist resource ids to names, leaving unknown ids untouched."""
    return [RESOURCE_ENUM.get(r, r) if isinstance(r, int) else r for r in as_list(values)]


def _parse_buildings(parsed: ReplayEvent, map_state: JsonDict) -> None:
    corner_states = _section(map_state, "tileCornerStates")
    if corner_states:
        for corner_id, raw_corner in corner_states.items():
            corner_data = as_dict(raw_corner)
            if "owner" in corner_data and "buildingType" in corner_data:
                building_type = as_int(corner_data["buildingType"])
                parsed.building_placed = {
                    "corner_id": int(corner_id),
                    "owner": corner_data["owner"],
                    "type": BUILDING_TYPE.get(building_type, "unknown"),
                }
                parsed.action_type = (
                    "place_settlement" if building_type == 1 else "place_city"
                )
                parsed.acting_player = as_int(corner_data["owner"])


def _parse_roads(parsed: ReplayEvent, map_state: JsonDict) -> None:
    edge_states = _section(map_state, "tileEdgeStates")
    if edge_states:
        for edge_id, raw_edge in edge_states.items():
            edge_data = as_dict(raw_edge)
            if "owner" in edge_data and "type" in edge_data:
                parsed.road_placed = {
                    "edge_id": int(edge_id),
                    "owner": edge_data["owner"],
                    "type": EDGE_TYPE.get(as_int(edge_data["type"]), "unknown"),
                }
                parsed.action_type = "place_road"
                parsed.acting_player = as_int(edge_data["owner"])


def _parse_resources(parsed: ReplayEvent, state_change: JsonDict) -> None:
    player_states = _section(state_change, "playerStates")
    if player_states:
        resources_gained: dict[int, JsonValue] = {}
        for player_id, raw_state in player_states.items():
            resource_cards = _section(as_dict(raw_state), "resourceCards")
            if "cards" in resource_cards:
                resources_gained[int(player_id)] = resource_cards["cards"]
        if resources_gained:
            parsed.resources_gained = resources_gained


def _parse_trades(parsed: ReplayEvent, state_change: JsonDict) -> None:
    trade_state = _section(state_change, "tradeState")
    active_offers = _section(trade_state, "activeOffers")
    if active_offers:
        for offer_id, raw_offer in active_offers.items():
            if not raw_offer:
                continue
            offer_data = as_dict(raw_offer)
            if "creator" in offer_data:
                parsed.trade_offer = {
                    "offer_id": offer_id,
                    "creator": offer_data["creator"],
                    "offered": _resource_names(offer_data.get("offeredResources", [])),
                    "wanted": _resource_names(offer_data.get("wantedResources", [])),
                    "responses": offer_data.get("playerResponses", {}),
                }
                parsed.action_type = "trade_offer"
                parsed.acting_player = as_int(offer_data["creator"])


def parse_replay_event(index: int, event: JsonDict) -> ReplayEvent:
    """Parse a single replay event into structured format."""
    delta_s = as_float(_section(event, "input").get("deltaS", 0))
    state_change = _section(event, "stateChange")

    parsed = ReplayEvent(
        event_index=index,
        delta_seconds=delta_s,
        state_change=state_change,
    )

    # Parse current state changes
    current_state = _section(state_change, "currentState")
    if "currentTurnPlayerColor" in current_state:
        parsed.acting_player = as_int(current_state["currentTurnPlayerColor"])

    # Parse dice roll
    dice_state = _section(state_change, "diceState")
    if dice_state.get("diceThrown") and "dice1" in dice_state and "dice2" in dice_state:
        parsed.dice_roll = (dice_state["dice1"], dice_state["dice2"])
        parsed.action_type = "roll_dice"

    map_state = _section(state_change, "mapState")
    _parse_buildings(parsed, map_state)
    _parse_roads(parsed, map_state)
    _parse_resources(parsed, state_change)
    _parse_trades(parsed, state_change)

    return parsed


def _winner_of(events: list[ReplayEvent]) -> int | None:
    """Read the winner off the last event's victory-point totals."""
    if not events:
        return None
    player_states = _section(events[-1].state_change, "playerStates")
    for player_id, raw_state in player_states.items():
        vp_state = as_dict(raw_state).get("victoryPointsState", {})
        total_vp = (
            sum(as_float(value) for value in vp_state.values())
            if isinstance(vp_state, dict)
            else 0.0
        )
        if total_vp >= 10:
            return int(player_id)
    return None


def parse_replay(game_id: str, player_color: int, raw_data: JsonDict) -> ParsedReplay:
    """Parse raw API response into structured replay."""
    data = as_dict(raw_data.get("data", raw_data))
    event_history = _section(data, "eventHistory")
    raw_events = as_list(event_history.get("events", []))

    events = [parse_replay_event(i, as_dict(raw_event)) for i, raw_event in enumerate(raw_events)]

    return ParsedReplay(
        game_id=game_id,
        player_color=player_color,
        total_events=len(events),
        events=events,
        initial_state=data.get("initialState"),
        final_state=data.get("finalState"),
        winner=_winner_of(events),
    )


__all__ = ["parse_replay", "parse_replay_event"]
