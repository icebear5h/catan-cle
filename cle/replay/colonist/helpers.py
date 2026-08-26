"""Helper functions for Colonist data conversion."""

from .constants import (
    COLONIST_RES_TO_ENGINE_IDX, COLONIST_PLAYER_COLORS, ENGINE_RESOURCES,
)


def format_resources(resource_tuple):
    """Format a resource tuple (WOOD, BRICK, SHEEP, WHEAT, ORE) as readable string."""
    if not resource_tuple:
        return "nothing"
    parts = []
    for i, count in enumerate(resource_tuple):
        if count > 0:
            parts.append(f"{count} {ENGINE_RESOURCES[i].lower()}")
    return ", ".join(parts) if parts else "nothing"


def format_trade(offered, wanted):
    """Format a trade display, handling 'any card' cases.

    When one side is all zeros, it means 'any card(s)' where the count
    matches the total cards on the other side.
    """
    offered_sum = sum(offered) if offered else 0
    wanted_sum = sum(wanted) if wanted else 0

    # Format offered side
    if offered_sum == 0:
        offered_str = "nothing"
    elif all(x == 0 for x in offered):
        offered_str = "nothing"
    else:
        offered_str = format_resources(offered)

    # Format wanted side
    if wanted_sum == 0:
        if offered_sum == 1:
            wanted_str = "any 1 card"
        else:
            wanted_str = f"any {offered_sum} cards"
    elif all(x == 0 for x in wanted):
        if offered_sum == 1:
            wanted_str = "any 1 card"
        else:
            wanted_str = f"any {offered_sum} cards"
    else:
        wanted_str = format_resources(wanted)

    return f"{offered_str} for {wanted_str}"


def get_player_color_name(player_id):
    """Get player color name from Colonist player ID."""
    if player_id is None:
        return None
    return COLONIST_PLAYER_COLORS.get(player_id, f"Player{player_id}")


def colonist_resources_to_tuple(resource_list):
    """Convert Colonist resource list to engine 5-tuple (WOOD, BRICK, SHEEP, WHEAT, ORE).

    Returns (tuple, any_count) where any_count is how many "any" (resource 9) were present.
    """
    counts = [0, 0, 0, 0, 0]
    any_count = 0
    for res_id in resource_list:
        if res_id == 9:
            any_count += 1
        elif res_id in COLONIST_RES_TO_ENGINE_IDX:
            counts[COLONIST_RES_TO_ENGINE_IDX[res_id]] += 1
    return tuple(counts), any_count


def colonist_cards_to_freqdeck(card_list):
    """Convert Colonist card list [card_ids...] to engine freqdeck [WOOD, BRICK, SHEEP, WHEAT, ORE].

    Colonist uses: 1=WOOD, 2=BRICK, 3=SHEEP, 4=WHEAT, 5=ORE (verified from road builds)
    Engine uses index: 0=WOOD, 1=BRICK, 2=SHEEP, 3=WHEAT, 4=ORE
    """
    counts = [0, 0, 0, 0, 0]
    for card_id in card_list:
        if card_id in COLONIST_RES_TO_ENGINE_IDX:
            counts[COLONIST_RES_TO_ENGINE_IDX[card_id]] += 1
    return counts


def get_engine_player_resources(game, player_idx):
    """Get engine's current resources for a player as freqdeck [WOOD, BRICK, SHEEP, WHEAT, ORE]."""
    state = game.state.player_state
    return [
        state.get(f"P{player_idx}_WOOD_IN_HAND", 0),
        state.get(f"P{player_idx}_BRICK_IN_HAND", 0),
        state.get(f"P{player_idx}_SHEEP_IN_HAND", 0),
        state.get(f"P{player_idx}_WHEAT_IN_HAND", 0),
        state.get(f"P{player_idx}_ORE_IN_HAND", 0),
    ]


def validate_resources_match(game, expected_resources, colonist_to_engine_idx, step_info=""):
    """Compare engine resources to Colonist's expected state. Returns list of mismatches.

    Args:
        game: Current game instance
        expected_resources: Dict {colonist_player_id: [card_ids...]} from Colonist
        colonist_to_engine_idx: Dict {colonist_color_id: engine_player_idx}
        step_info: String describing what step was just executed (for logging)

    Returns:
        List of (player_idx, engine_resources, expected_resources, diff) tuples for mismatches
    """
    mismatches = []

    for colonist_id_str, card_list in expected_resources.items():
        colonist_id = int(colonist_id_str) if isinstance(colonist_id_str, str) else colonist_id_str
        engine_idx = colonist_to_engine_idx.get(str(colonist_id))

        if engine_idx is None:
            continue  # Unknown player mapping

        expected = colonist_cards_to_freqdeck(card_list)
        actual = get_engine_player_resources(game, engine_idx)

        if expected != actual:
            diff = [a - e for a, e in zip(actual, expected)]
            mismatches.append({
                "player_idx": engine_idx,
                "colonist_id": colonist_id,
                "engine_has": actual,
                "colonist_expects": expected,
                "diff": diff,  # positive = engine has more, negative = engine has less
            })

    return mismatches
