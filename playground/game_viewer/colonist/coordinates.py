"""Hex math, port parsing, and map creation from Colonist data."""

from engine.models.map import CatanMap, BASE_MAP_TEMPLATE, LandTile, initialize_tiles, WOOD, BRICK, SHEEP, WHEAT, ORE

from .constants import (
    COLONIST_RESOURCE, COLONIST_PORT_RESOURCE, ENGINE_PORT_MAP, HEX_DIRECTIONS,
)


def rotate_60_cw(coord):
    """Rotate cube coordinates 60 degrees clockwise."""
    return (-coord[2], -coord[0], -coord[1])


def reflect_x(coord):
    """Reflect across the x axis (swap y and z, negate x)."""
    return (-coord[0], -coord[2], -coord[1])


def add_coords(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

# converts colonist coordinates to engine coordinates for ports
def parse_colonist_ports(port_edge_states):
    """Parse Colonist portEdgeStates into engine port order.

    Colonist uses a coordinate system rotated 60 degrees from the engine.
    We rotate Colonist coords 60 CW to convert to engine coords.
    """
    if not port_edge_states:
        # Fallback to default port resources
        return [WOOD, BRICK, SHEEP, WHEAT, ORE, None, None, None, None]

    print("\n" + "=" * 60)
    print("PARSING COLONIST PORTS")
    print("=" * 60)

    engine_port_resources = [None] * 9

    for pid, port in sorted(port_edge_states.items(), key=lambda x: int(x[0])):
        cx, cy = port['x'], port['y']
        raw_type = port.get('type')
        colonist_cube = (cx, cy, -cx - cy)
        ptype = COLONIST_PORT_RESOURCE.get(raw_type)

        # Rotate 180 CW (3x60) then reflect to convert to engine coordinate system
        engine_cube = reflect_x(rotate_60_cw(rotate_60_cw(rotate_60_cw(colonist_cube))))

        engine_idx = None
        if engine_cube in ENGINE_PORT_MAP:
            # Direct water hex reference
            engine_idx = ENGINE_PORT_MAP[engine_cube]
            engine_port_resources[engine_idx] = ptype
        else:
            # Land tile reference - find adjacent engine port
            for offset in HEX_DIRECTIONS.values():
                neighbor = add_coords(engine_cube, offset)
                if neighbor in ENGINE_PORT_MAP:
                    engine_idx = ENGINE_PORT_MAP[neighbor]
                    engine_port_resources[engine_idx] = ptype
                    break

        print(f"  Colonist port {pid}: ({cx},{cy}) type={raw_type} -> Engine port {engine_idx} = {ptype}")

    print()
    print("Final engine port resources:")
    for i, res in enumerate(engine_port_resources):
        print(f"  Engine {i}: {res if res else '3:1'}")
    print("=" * 60 + "\n")

    return engine_port_resources


def create_map_from_colonist(initial_state):
    """Create a CatanMap matching the Colonist game's board layout."""
    map_state = initial_state.get('mapState', {})
    tile_hex_states = map_state.get('tileHexStates', {})

    if not tile_hex_states:
        return None  # Fall back to random map

    # Build Colonist coord -> (resource, number) mapping
    # Colonist uses (x, y), we convert to cube and apply rotation+reflection
    colonist_tiles = {}
    for t in tile_hex_states.values():
        cx, cy = t['x'], t['y']
        colonist_cube = (cx, cy, -cx - cy)
        # Apply same transform as ports: rotate 180 CW then reflect
        engine_coord = reflect_x(rotate_60_cw(rotate_60_cw(rotate_60_cw(colonist_cube))))
        colonist_tiles[engine_coord] = (COLONIST_RESOURCE.get(t['type']), t['diceNumber'])

    # Get engine topology iteration order for land tiles
    engine_land_coords = [coord for coord, tt in BASE_MAP_TEMPLATE.topology.items() if tt == LandTile]

    # Build resource and number arrays in topology order
    resources_order = []
    numbers_order = []

    for coord in engine_land_coords:
        if coord in colonist_tiles:
            res, num = colonist_tiles[coord]
            resources_order.append(res)
            if res is not None:  # Non-desert tiles have numbers
                numbers_order.append(num)
        else:
            # Fallback if coordinate not found
            resources_order.append(None)

    # Parse port resources from Colonist data
    port_edge_states = map_state.get('portEdgeStates', {})
    port_resources = parse_colonist_ports(port_edge_states)

    # Create tiles with the Colonist layout
    # Note: initialize_tiles pops from arrays (LIFO), so we reverse them
    tiles = initialize_tiles(
        BASE_MAP_TEMPLATE,
        shuffled_numbers_param=list(reversed(numbers_order)),
        shuffled_port_resources_param=list(reversed(port_resources)),
        shuffled_tile_resources_param=list(reversed(resources_order)),
    )

    return CatanMap.from_tiles(tiles)
