"""Legality of settlement/road placement and the build-cost checks."""

from data_pipeline.bootstrapping.replay_decoder._enums import BuildingType, ResourceType
from data_pipeline.bootstrapping.replay_decoder._geometry import (
    corners_adjacent,
    edge_connects_corner,
    edges_connected,
)
from data_pipeline.bootstrapping.replay_decoder._models import GameState, PlayerState


def can_place_settlement(state: GameState, corner_id: int, is_setup: bool = False) -> bool:
    """Check if settlement can be placed at corner"""
    corner = state.corners.get(corner_id)
    if not corner or corner.building_type != BuildingType.NONE:
        return False

    # Check distance rule - no adjacent settlements
    for other_cid, other in state.corners.items():
        if other_cid != corner_id and other.building_type != BuildingType.NONE:
            if corners_adjacent(corner, other):
                return False

    # During normal play, must connect to own road
    if not is_setup:
        has_connected_road = False
        for eid, edge in state.edges.items():
            if edge.owner == state.current_player:
                if edge_connects_corner(state.edges[eid], corner):
                    has_connected_road = True
                    break
        if not has_connected_road:
            return False

    return True


def can_place_road(state: GameState, edge_id: int) -> bool:
    """Check if road can be placed at edge"""
    edge = state.edges.get(edge_id)
    if not edge or edge.road_type != 0:
        return False

    # Must connect to own settlement/city or road
    player_color = state.current_player
    has_connection = False

    # Check corners
    for cid, corner in state.corners.items():
        if corner.owner == player_color:
            if edge_connects_corner(edge, state.corners[cid]):
                has_connection = True
                break

    # Check other roads
    if not has_connection:
        for eid, other_edge in state.edges.items():
            if other_edge.owner == player_color and eid != edge_id:
                if edges_connected(state, edge_id, eid):
                    has_connection = True
                    break

    return has_connection


def can_afford_settlement(player: PlayerState) -> bool:
    return (player.resource_count(ResourceType.WHEAT) >= 1 and
            player.resource_count(ResourceType.BRICK) >= 1 and
            player.resource_count(ResourceType.SHEEP) >= 1 and
            player.resource_count(ResourceType.WOOD) >= 1)


def can_afford_city(player: PlayerState) -> bool:
    return (player.resource_count(ResourceType.WHEAT) >= 2 and
            player.resource_count(ResourceType.ORE) >= 3)


def can_afford_road(player: PlayerState) -> bool:
    return (player.resource_count(ResourceType.BRICK) >= 1 and
            player.resource_count(ResourceType.WOOD) >= 1)


def can_afford_dev_card(player: PlayerState) -> bool:
    return (player.resource_count(ResourceType.WHEAT) >= 1 and
            player.resource_count(ResourceType.SHEEP) >= 1 and
            player.resource_count(ResourceType.ORE) >= 1)


__all__ = [
    "can_afford_city",
    "can_afford_dev_card",
    "can_afford_road",
    "can_afford_settlement",
    "can_place_road",
    "can_place_settlement",
]
