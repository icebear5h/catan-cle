"""Populate a `GameState` from a replay's `initialState` block."""

from data_pipeline.bootstrapping.replay_decoder._enums import ActionState, PortType, TileType
from data_pipeline.bootstrapping.replay_decoder._models import (
    Corner,
    Edge,
    GameState,
    HexTile,
    PlayerState,
    Port,
)
from data_pipeline.json_coerce import as_dict, as_int, as_list
from data_pipeline.json_types import JsonDict


def initialize_state(state: GameState, init: JsonDict, play_order: list[int]) -> None:
    """Initialize game state from initialState"""
    # Initialize board
    map_state = as_dict(init.get('mapState', {}))

    # Tiles
    for tid, raw_tile in as_dict(map_state.get('tileHexStates', {})).items():
        tdata = as_dict(raw_tile)
        state.tiles[int(tid)] = HexTile(
            index=int(tid),
            x=as_int(tdata['x']),
            y=as_int(tdata['y']),
            tile_type=TileType(as_int(tdata['type'])),
            dice_number=as_int(tdata['diceNumber'])
        )

    # Ports
    for eid, raw_port in as_dict(map_state.get('portEdgeStates', {})).items():
        pdata = as_dict(raw_port)
        state.ports[int(eid)] = Port(
            edge_id=int(eid),
            x=as_int(pdata['x']),
            y=as_int(pdata['y']),
            z=as_int(pdata['z']),
            port_type=PortType(as_int(pdata['type']))
        )

    # Corners
    for cid, raw_corner in as_dict(map_state.get('tileCornerStates', {})).items():
        cdata = as_dict(raw_corner)
        state.corners[int(cid)] = Corner(
            index=int(cid),
            x=as_int(cdata['x']),
            y=as_int(cdata['y']),
            z=as_int(cdata['z'])
        )

    # Edges
    for eid, raw_edge in as_dict(map_state.get('tileEdgeStates', {})).items():
        edata = as_dict(raw_edge)
        state.edges[int(eid)] = Edge(
            index=int(eid),
            x=as_int(edata['x']),
            y=as_int(edata['y']),
            z=as_int(edata['z'])
        )

    # Players
    for raw_player in as_dict(init.get('playerStates', {})).values():
        pdata = as_dict(raw_player)
        color = as_int(pdata['color'])
        ratios = as_dict(pdata.get('bankTradeRatiosState', {}))
        state.players[color] = PlayerState(
            color=color,
            is_connected=bool(pdata.get('isConnected', True)),
            resources=[as_int(c) for c in as_list(
                as_dict(pdata.get('resourceCards', {})).get('cards', []))],
            bank_trade_ratios={int(k): as_int(v) for k, v in ratios.items()}
        )

    # Current state
    curr = as_dict(init.get('currentState', {}))
    state.current_player = as_int(curr.get(
        'currentTurnPlayerColor', play_order[0] if play_order else 0))
    state.action_state = ActionState(as_int(curr.get('actionState', 1)))
    state.completed_turns = as_int(curr.get('completedTurns', 0))

    # Robber
    robber = as_dict(init.get('mechanicRobberState', {}))
    state.robber_tile = as_int(robber.get('locationTileIndex', 0))

    # Bank
    bank = as_dict(init.get('bankState', {}))
    for rid, count in as_dict(bank.get('resourceCards', {})).items():
        state.bank_resources[int(rid)] = as_int(count)

    # Dev cards bank
    dev_bank = as_dict(as_dict(init.get(
        'mechanicDevelopmentCardsState', {})).get('bankDevelopmentCards', {}))
    state.bank_dev_cards = [as_int(c) for c in as_list(dev_bank.get('cards', []))]


__all__ = ["initialize_state"]
