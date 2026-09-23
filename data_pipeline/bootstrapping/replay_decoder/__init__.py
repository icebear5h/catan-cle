"""
Colonist.io Replay Decoder

Decodes replay JSON files and reconstructs game state step by step.
Extracts valid moves at each game state for training data generation.
"""

import sys

from data_pipeline.bootstrapping.replay_decoder._decoder import ReplayDecoder
from data_pipeline.bootstrapping.replay_decoder._enums import (
    ActionState,
    BuildingType,
    DevCardType,
    GameLogType,
    PieceEnum,
    PortType,
    ResourceType,
    TileType,
)
from data_pipeline.bootstrapping.replay_decoder._models import (
    Corner,
    Edge,
    GameState,
    HexTile,
    PlayerState,
    Port,
)
from data_pipeline.bootstrapping.replay_decoder._types import (
    ActionInfo,
    BoardState,
    Move,
    ReplayStep,
    StateSnapshot,
)


def main() -> None:

    if len(sys.argv) < 2:
        print("Usage: python replay_decoder.py <replay_file.json>")
        sys.exit(1)

    decoder = ReplayDecoder(sys.argv[1])

    print("=== GAME INFO ===")
    print(f"Play order: {decoder.play_order}")
    print(f"Total events: {len(decoder.events)}")
    print()

    print("=== BOARD LAYOUT ===")
    for tid, tile in sorted(decoder.state.tiles.items()):
        print(f"Tile {tid}: {tile.tile_type.name} ({tile.dice_number}) at ({tile.x}, {tile.y})")

    print()
    print("=== REPLAYING GAME ===")

    states = decoder.replay_with_states()
    print(f"Generated {len(states)} state snapshots")

    # Show first few
    for i, s in enumerate(states[:5]):
        print(f"\n--- State {i} (event {s['event_idx']}) ---")
        print(f"Current player: {s['state']['current_player']}")
        print(f"Action state: {s['state']['action_state']}")
        print(f"Valid moves: {len(s['valid_moves'])}")
        if s['action_taken']:
            print(f"Action taken: {s['action_taken']}")


__all__ = [
    "ActionInfo",
    "ActionState",
    "BoardState",
    "BuildingType",
    "Corner",
    "DevCardType",
    "Edge",
    "GameLogType",
    "GameState",
    "HexTile",
    "Move",
    "PieceEnum",
    "PlayerState",
    "Port",
    "PortType",
    "ReplayDecoder",
    "ReplayStep",
    "ResourceType",
    "StateSnapshot",
    "TileType",
    "main",
]


if __name__ == '__main__':
    main()
