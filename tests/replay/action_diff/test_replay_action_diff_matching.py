"""Human action matching normalizes randomness and keeps controlled values."""



from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from evals.replay_action_diff import (
    canonicalize_policy_action_order,
    match_human_action,
)

from .support import (
    _matcher_state,
)


def test_canonicalize_policy_order_swaps_only_same_event_steal_move_pair() -> None:
    actions = [
        {"type": "STEAL", "player": 5, "index": 10},
        {"type": "MOVE_ROBBER", "player": 5, "index": 10},
        {"type": "STEAL", "player": 5, "index": 11},
        {"type": "MOVE_ROBBER", "player": 5, "index": 12},
    ]

    canonical, changes = canonicalize_policy_action_order(actions)

    assert [row["type"] for row in canonical] == [
        "MOVE_ROBBER",
        "STEAL",
        "STEAL",
        "MOVE_ROBBER",
    ]
    assert canonical[0]["_source_replay_index"] == 1
    assert canonical[1]["_source_replay_index"] == 0
    assert len(changes) == 1
    assert actions[0].get("_source_replay_index") is None


def test_match_human_action_normalizes_random_outcomes_but_keeps_controlled_values() -> None:
    state = _matcher_state()
    actions = [
        Action(Color.BLACK, ActionType.ROLL, None),
        Action(Color.BLACK, ActionType.STEAL, (Color.BLUE, None)),
        Action(Color.BLACK, ActionType.BUY_DEVELOPMENT_CARD, None),
        Action(Color.BLACK, ActionType.PLAY_MONOPOLY, "WOOD"),
    ]

    roll = match_human_action(
        actions,
        {"type": "ROLL", "dice": (6, 4)},
        state,
    )
    steal = match_human_action(
        actions,
        {"type": "STEAL", "victim": 2, "stolen_resource": "ORE"},
        state,
    )
    buy = match_human_action(
        actions,
        {"type": "BUY_DEVELOPMENT_CARD", "card_type": "KNIGHT"},
        state,
    )
    monopoly = match_human_action(
        actions,
        {"type": "MONOPOLY_RESOURCE", "resource": "WOOD"},
        state,
    )

    assert roll["action_index"] == 0
    assert "dice outcome omitted" in roll["normalization"]
    assert steal["action_index"] == 1
    assert "stolen resource omitted" in steal["normalization"]
    assert buy["action_index"] == 2
    assert "card identity omitted" in buy["normalization"]
    assert monopoly["action_index"] == 3


def test_match_human_action_maps_geometry_and_maritime_trade_exactly() -> None:
    state = _matcher_state()
    actions = [
        Action(Color.BLACK, ActionType.BUILD_SETTLEMENT, 42),
        Action(Color.BLACK, ActionType.BUILD_ROAD, (3, 9)),
        Action(
            Color.BLACK,
            ActionType.MARITIME_TRADE,
            ("WOOD", "WOOD", None, None, "ORE"),
        ),
    ]

    settlement = match_human_action(
        actions,
        {"type": "BUILD_SETTLEMENT", "colonist_corner": 7},
        state,
    )
    road = match_human_action(
        actions,
        {"type": "BUILD_ROAD", "colonist_edge": 8},
        state,
    )
    maritime = match_human_action(
        actions,
        {
            "type": "MARITIME_TRADE",
            "given": (2, 0, 0, 0, 0),
            "received": (0, 0, 0, 0, 1),
        },
        state,
    )

    assert settlement["action_index"] == 0
    assert road["action_index"] == 1
    assert maritime["action_index"] == 2


def test_match_human_action_excludes_missing_controlled_parameters() -> None:
    state = _matcher_state()
    actions = [
        Action(Color.BLACK, ActionType.DISCARD, None),
        Action(Color.BLACK, ActionType.OFFER_TRADE, "format"),
    ]

    discard = match_human_action(
        actions,
        {"type": "DISCARD", "cards": (1, 0, 0, 1, 0)},
        state,
    )
    offer = match_human_action(
        actions,
        {"type": "OFFER_TRADE", "trade_tuple": (1,) * 10},
        state,
    )

    assert discard["status"] == "coarse"
    assert "card selection" in discard["reason"]
    assert offer["status"] == "coarse"
    assert "trade terms" in offer["reason"]
