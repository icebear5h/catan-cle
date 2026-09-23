"""Colonist.io wire enumerations used by the replay decoder."""

from enum import IntEnum


class TileType(IntEnum):
    DESERT = 0
    WHEAT = 1
    BRICK = 2
    SHEEP = 3
    WOOD = 4
    ORE = 5


class ResourceType(IntEnum):
    WHEAT = 1
    BRICK = 2
    SHEEP = 3
    WOOD = 4
    ORE = 5
    ANY = 9  # Used in trade offers for 3:1 ports


class PortType(IntEnum):
    GENERIC = 1  # 3:1
    WHEAT = 2
    BRICK = 3
    ORE = 4
    SHEEP = 5
    WOOD = 6


class BuildingType(IntEnum):
    NONE = 0
    SETTLEMENT = 1
    CITY = 2


class PieceEnum(IntEnum):
    ROAD = 0
    SHIP = 1
    SETTLEMENT = 2
    CITY = 3
    WALL = 4
    ROBBER = 5


class ActionState(IntEnum):
    """Game action states - what the current player can/must do"""
    MAIN_TURN = 0  # Can roll, trade, build, end turn
    SETUP_PLACE_SETTLEMENT = 1  # Must place settlement (setup phase)
    SETUP_PLACE_ROAD = 3  # Must place road (setup phase)
    MUST_ROLL_DICE = 24  # Must roll dice
    MUST_MOVE_ROBBER = 27  # Must move robber (after rolling 7)
    ROBBER_STEAL = 28  # Must steal from player
    ROAD_BUILDING_1 = 30  # Playing road building dev card
    ROAD_BUILDING_2 = 31  # Second road from road building


class DevCardType(IntEnum):
    KNIGHT = 11
    YEAR_OF_PLENTY = 13
    ROAD_BUILDING = 14
    MONOPOLY = 15
    VICTORY_POINT = 16  # Guess


class GameLogType(IntEnum):
    TURN_START = 1
    BUILD_PIECE = 4
    BUY_PIECE = 5
    DICE_ROLL = 10
    ROBBER_MOVED = 11
    CARDS_RECEIVED = 14
    CARDS_DISCARDED = 15
    ROBBER_STEAL = 16
    DEV_CARD_PLAYED = 20
    YEAR_OF_PLENTY = 21
    GAME_END = 44
    GAME_WINNER = 45
    RESOURCE_DISTRIBUTION = 47
    ROBBER_ON_TILE = 49
    DISCARD_ON_7 = 55
    ACHIEVEMENT = 66
    MONOPOLY = 86
    EMBARGO = 113
    REMOVE_EMBARGO = 114
    PLAYER_TRADE = 115
    BANK_TRADE = 116
    COUNTER_OFFER = 117
    TRADE_OFFER = 118


__all__ = [
    "ActionState",
    "BuildingType",
    "DevCardType",
    "GameLogType",
    "PieceEnum",
    "PortType",
    "ResourceType",
    "TileType",
]
