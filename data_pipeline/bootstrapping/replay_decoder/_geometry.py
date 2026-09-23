"""Hex-grid adjacency between the corner, edge and tile records of a replay."""

from data_pipeline.bootstrapping.replay_decoder._models import Corner, Edge, GameState


def corners_adjacent(corner1: Corner, corner2: Corner) -> bool:
    """Check if two corners are adjacent (share an edge)"""
    cx, cy, cz = corner1.x, corner1.y, corner1.z
    ocx, ocy, ocz = corner2.x, corner2.y, corner2.z

    if cz == 0:
        # z=0 corner is adjacent to specific z=1 corners
        return (ocz == 1 and ocx == cx and ocy == cy) or \
               (ocz == 1 and ocx == cx + 1 and ocy == cy - 1) or \
               (ocz == 1 and ocx == cx - 1 and ocy == cy)
    else:  # cz == 1
        # z=1 corner is adjacent to specific z=0 corners
        return (ocz == 0 and ocx == cx and ocy == cy) or \
               (ocz == 0 and ocx == cx - 1 and ocy == cy + 1) or \
               (ocz == 0 and ocx == cx + 1 and ocy == cy)


def edge_connects_corner(edge: Edge, corner: Corner) -> bool:
    """Check if edge connects to corner using hex grid geometry"""
    ex, ey, ez = edge.x, edge.y, edge.z
    cx, cy, cz = corner.x, corner.y, corner.z

    if cz == 0:
        # z=0 corner (upper corner) is touched by:
        # - ez=0 edge at same position
        # - ez=1 edge at (cx+1, cy-1)
        # - ez=2 edge at same position OR at (cx+1, cy-1)
        return (ez == 0 and ex == cx and ey == cy) or \
               (ez == 1 and ex == cx + 1 and ey == cy - 1) or \
               (ez == 2 and ((ex == cx and ey == cy) or (ex == cx + 1 and ey == cy - 1)))
    else:  # cz == 1
        # z=1 corner (lower corner) is touched by:
        # - ez=0 edge at (cx, cy+1)
        # - ez=1 edge at same position
        # - ez=2 edge at same position
        return (ez == 0 and ex == cx and ey == cy + 1) or \
               (ez == 1 and ex == cx and ey == cy) or \
               (ez == 2 and ex == cx and ey == cy)


def edges_connected(state: GameState, e1: int, e2: int) -> bool:
    """Check if two edges share a corner"""
    # Two edges are connected if they both touch the same corner
    for corner in state.corners.values():
        if edge_connects_corner(state.edges[e1], corner) and edge_connects_corner(
            state.edges[e2], corner
        ):
            return True
    return False


def corner_tiles(state: GameState, corner_id: int) -> list[int]:
    """Get tile indices that a corner touches (for resource production)"""
    corner = state.corners[corner_id]
    cx, cy, cz = corner.x, corner.y, corner.z
    touching_tiles = []

    for tid, tile in state.tiles.items():
        tx, ty = tile.x, tile.y
        # A corner touches a tile based on hex grid geometry
        if cz == 0:  # Upper corner type
            # Touches tiles at: (cx, cy), (cx-1, cy), (cx, cy-1)
            if (tx == cx and ty == cy) or \
               (tx == cx - 1 and ty == cy) or \
               (tx == cx and ty == cy - 1):
                touching_tiles.append(tid)
        else:  # cz == 1, Lower corner type
            # Touches tiles at: (cx, cy), (cx+1, cy), (cx, cy+1)
            if (tx == cx and ty == cy) or \
               (tx == cx + 1 and ty == cy) or \
               (tx == cx and ty == cy + 1):
                touching_tiles.append(tid)

    return touching_tiles


__all__ = ["corner_tiles", "corners_adjacent", "edge_connects_corner", "edges_connected"]
