"""Turn raw leaderboard and history payloads into typed entries."""

from datetime import datetime

from data_pipeline.bootstrapping.scrapers.colonist_api._models import (
    GameHistoryEntry,
    LeaderboardEntry,
    logger,
)
from data_pipeline.json_coerce import (
    as_dict,
    as_dict_list,
    as_float_or_none,
    as_int,
    as_int_or_none,
    as_list,
    as_str,
)
from data_pipeline.json_types import JsonDict, JsonValue

# Game type mapping (from setting.gameType)
GAME_TYPES = {
    0: "Classic4P",
    1: "Classic3P",
    2: "1v1",
    3: "CitiesAndKnights4P",
    4: "CitiesAndKnights3P",
    5: "Seafarers4P",
    6: "Seafarers",  # 1v1 seafarers or general
    7: "Custom",
    8: "Tournament",
}


def _leaderboard_items(data: JsonValue) -> list[JsonValue]:
    """Accept either a bare array or an object keyed by players/rankings."""
    if isinstance(data, list):
        return data
    container = as_dict(data)
    if "players" in container:
        return as_list(container["players"])
    return as_list(container.get("rankings", []))


def parse_leaderboard_response(data: JsonValue) -> list[LeaderboardEntry]:
    """Parse leaderboard API response into LeaderboardEntry objects."""
    entries: list[LeaderboardEntry] = []

    # Debug: print first item to see structure
    if isinstance(data, (list, dict)) and len(data) > 0:
        logger.debug(f"Sample leaderboard item: {data[0] if isinstance(data, list) else data}")

    items = _leaderboard_items(data)

    for i, raw_item in enumerate(items):
        try:
            item = as_dict(raw_item)
            entry = LeaderboardEntry(
                rank=as_int(item.get("rank", i + 1)),
                username=as_str(item.get("username", "")),
                rating=as_int(item.get("skillRating", 0)),
                games_played=as_int(item.get("totalGamesPlayed", 0)),
                win_rate=as_float_or_none(item.get("winRate")),
            )
            entries.append(entry)
        except Exception as e:
            logger.warning(f"Failed to parse leaderboard entry: {raw_item}, error: {e}")

    return entries


def _game_date(value: JsonValue) -> datetime:
    """Read Colonist's millisecond start time, falling back to the current time."""
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise TypeError("startTime must be a JSON number or string")
        return datetime.fromtimestamp(int(value) / 1000)
    except (ValueError, TypeError):
        return datetime.now()


def _seat_of(players: list[JsonDict], username: str) -> tuple[int, str]:
    """Return the requested player's colour and win/loss result in this game."""
    for player in players:
        if as_str(player.get("username", "")).lower() == username.lower():
            # rank=1 means winner
            return as_int(player.get("playerColor", 0)), (
                "win" if player.get("rank") == 1 else "loss"
            )
    return 0, "unknown"


def _history_items(data: JsonValue) -> list[JsonValue]:
    """History endpoint returns {"profileUserId": ..., "gameDatas": [...]}."""
    if isinstance(data, dict):
        return as_list(data.get("gameDatas", []))
    return as_list(data)


def parse_games_response(data: JsonValue, username: str) -> list[GameHistoryEntry]:
    """Parse games API response into GameHistoryEntry objects."""
    entries: list[GameHistoryEntry] = []

    for raw_item in _history_items(data):
        try:
            item = as_dict(raw_item)
            game_id = str(item.get("id", ""))
            if not game_id:
                continue

            # Skip games without replay
            if not item.get("hasReplay", True):
                continue

            players = as_dict_list(item.get("players", []))
            player_color, result = _seat_of(players, username)

            game_date = _game_date(item.get("startTime", "0"))

            # Get game mode from setting
            setting = as_dict(item.get("setting", {}))
            game_type = as_int(setting.get("gameType", 0))
            mode = GAME_TYPES.get(game_type, f"Unknown_{game_type}")

            entry = GameHistoryEntry(
                game_id=game_id,
                date=game_date,
                mode=mode,
                player_color=player_color,
                result=result,
                players=players,
                replay_url=f"https://colonist.io/replay?gameId={game_id}&playerColor={player_color}",
                turn_count=as_int_or_none(item.get("turnCount")),
            )
            entries.append(entry)
        except Exception as e:
            logger.warning(f"Failed to parse game entry: {raw_item}, error: {e}")

    return entries


__all__ = ["GAME_TYPES", "parse_games_response", "parse_leaderboard_response"]
