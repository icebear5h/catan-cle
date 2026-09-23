"""Serializable views of the decoder's dynamic state and static board."""

from data_pipeline.bootstrapping.replay_decoder._geometry import corner_tiles
from data_pipeline.bootstrapping.replay_decoder._models import GameState
from data_pipeline.bootstrapping.replay_decoder._types import (
    BoardState,
    BuildingSnapshot,
    PlayerSnapshot,
    PortSnapshot,
    RoadSnapshot,
    StateSnapshot,
    TileSnapshot,
)


def snapshot_state(state: GameState) -> StateSnapshot:
    """Create a serializable snapshot of current state"""
    players: dict[int, PlayerSnapshot] = {
        color: {
            'resources': list(p.resources),
            'victory_points': dict(p.victory_points),
            'bank_trade_ratios': dict(p.bank_trade_ratios)
        }
        for color, p in state.players.items()
    }
    buildings: dict[int, BuildingSnapshot] = {
        cid: {'owner': c.owner, 'type': c.building_type.name}
        for cid, c in state.corners.items()
        if c.owner is not None
    }
    roads: dict[int, RoadSnapshot] = {
        eid: {'owner': e.owner}
        for eid, e in state.edges.items()
        if e.owner is not None
    }
    return {
        'current_player': state.current_player,
        'action_state': state.action_state.name,
        'completed_turns': state.completed_turns,
        'dice': state.dice_value,
        'robber_tile': state.robber_tile,
        'players': players,
        'buildings': buildings,
        'roads': roads,
    }


def board_state(state: GameState) -> BoardState:
    """Get the static board layout for replay inspection."""
    tiles: dict[int, TileSnapshot] = {
        tid: {
            'x': t.x,
            'y': t.y,
            'type': t.tile_type.name,
            'dice_number': t.dice_number
        }
        for tid, t in state.tiles.items()
    }
    ports: dict[int, PortSnapshot] = {
        pid: {
            'x': p.x,
            'y': p.y,
            'z': p.z,
            'type': p.port_type.name,
            'ratio': p.trade_ratio()
        }
        for pid, p in state.ports.items()
    }
    return {
        'tiles': tiles,
        'ports': ports,
        'corner_tiles': {cid: corner_tiles(state, cid) for cid in state.corners},
    }


__all__ = ["board_state", "snapshot_state"]
