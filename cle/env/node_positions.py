"""
Static node position classifications for spatial reasoning.
Helps LLMs understand board topology: inward (center) vs outward (coastal/edge).

Generated from analysis of standard Catan board layout.

NODE NUMBERING SCHEME (standard Catan board):
- Nodes 0-5: Corners of the center hexagon (6 nodes)
- Nodes 6-23: Corners of the inner ring (first 6 hexagons around center)
- Nodes 24-53: Corners of the outer ring (12 hexagons on the edge)
"""

# Node classifications based on distance from board center
INWARD_NODES = frozenset([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 37, 38, 39, 40,
    43, 44, 45, 46, 47, 48, 49, 52, 53
])

MIDDLE_NODES = frozenset([
    25, 26, 35, 36, 41, 42, 50, 51, 54, 57, 59, 61, 64, 66, 68, 71, 73,
    75, 76, 79, 81, 84, 87, 89, 92, 94
])

OUTWARD_NODES = frozenset([
    55, 56, 58, 60, 62, 63, 65, 67, 69, 70, 72, 74, 77, 78, 80, 82, 83,
    85, 86, 88, 90, 91, 93, 95
])


def get_node_position_label(node_id: int) -> str:
    """
    Get human-readable position label for a node.

    Args:
        node_id: Node ID (0-95)

    Returns:
        "center" | "mid-board" | "coastal"
    """
    if node_id in INWARD_NODES:
        return "center"
    elif node_id in MIDDLE_NODES:
        return "mid-board"
    elif node_id in OUTWARD_NODES:
        return "coastal"
    else:
        return "unknown"


def is_inward_direction(from_node: int, to_node: int) -> bool:
    """
    Check if moving from one node to another is moving toward the center.

    Args:
        from_node: Starting node ID
        to_node: Destination node ID

    Returns:
        True if moving toward center, False otherwise
    """
    from_pos = get_node_position_label(from_node)
    to_pos = get_node_position_label(to_node)

    position_values = {"coastal": 0, "mid-board": 1, "center": 2}

    from_val = position_values.get(from_pos, 1)
    to_val = position_values.get(to_pos, 1)

    return to_val > from_val


def get_position_hint(node_id: int) -> str:
    """
    Get a brief strategic hint about a node's position.

    Args:
        node_id: Node ID

    Returns:
        Strategic hint string
    """
    label = get_node_position_label(node_id)

    hints = {
        "center": "interior",
        "mid-board": "middle",
        "coastal": "edge"
    }

    return hints.get(label, "")
