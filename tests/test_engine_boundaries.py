"""Isolated engine-boundary regressions using reduced, bank-balanced positions.

Score/phase overrides exercise API boundaries, not full-game reachability.
"""

import json
import pickle
from copy import deepcopy

import pytest

from cle.game_engine.communication import CommitmentStatus
from cle.game_engine.events import event_from_action, project_event
from cle.game_engine.game import GameEngine
from cle.game_engine.json import GameEncoder
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.board import STATIC_GRAPH, Board
from cle.game_engine.models.enums import (
    CITY, ROAD, SETTLEMENT, VICTORY_POINT, Action, ActionPrompt, ActionType,
)
from cle.game_engine.models.map import CatanMap
from cle.game_engine.models.player import Color
from cle.game_engine.observation import observe_state
from cle.game_engine.state import GameState, ensure_trade_window, new_trade_window, validate_discard
from cle.game_engine.state_functions import (
    build_road,
    build_settlement,
    get_player_freqdeck,
    maintain_longest_road,
    player_key,
)
from cle.game_engine.trading import TradeOffer, TradeWindowStatus


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


@pytest.fixture
def engine():
    game = GameEngine(COLORS, seed=7, shuffle_players=False, capture_history=True)
    state = game.state
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.player_state["P0_HAS_ROLLED"] = True
    state.player_state["P0_WOOD_IN_HAND"] = 5
    state.player_state["P0_BRICK_IN_HAND"] = 4
    state.player_state["P1_ORE_IN_HAND"] = 3
    state.resource_freqdeck = [14, 15, 19, 19, 16]
    # Populate lazy building caches before checking read-boundary mutations.
    for color in COLORS:
        for kind in (SETTLEMENT, CITY, ROAD):
            state.buildings_by_color[color][kind] = []
    state.playable_actions = generate_playable_actions(state)
    return game


@pytest.fixture
def offer_action():
    return Action(
        Color.RED,
        ActionType.OFFER_TRADE,
        TradeOffer(Color.RED, frozenset({Color.BLUE}), (1, 0, 0, 0, 0), (0, 0, 0, 0, 1)),
    )


def _message(engine, **changes):
    arguments = {
        "speaker": Color.RED,
        "text": "Leave the robber elsewhere and I will offer ORE.",
        "audience": (Color.BLUE,),
        "causation_id": "talk:boundary",
        "commitment": ("Leave the robber elsewhere", "Offer ORE", 2),
    }
    arguments.update(changes)
    return engine.append_message(**arguments)


def test_caller_owned_forced_dice_cannot_mutate_resolved_state(engine):
    dice = [1, 2]
    transition = engine.step(Action(Color.RED, ActionType.ROLL, dice), force=True)
    dice[0] = 6

    assert engine.state.last_dice_roll == [1, 2]
    assert engine.state.actions[-1].value == [1, 2]
    assert transition.resolved_action.value == [1, 2]
    assert engine.events[-1].public_payload == [1, 2]


def test_rejected_forced_discard_does_not_capture_an_undo_entry(engine):
    roll = Action(Color.RED, ActionType.ROLL, (3, 4))
    engine.step(roll, force=True)
    before = pickle.dumps(engine)

    with pytest.raises(ValueError):
        engine.step(Action(Color.RED, ActionType.DISCARD, ("ORE",) * 4), force=True)

    assert pickle.dumps(engine) == before
    assert engine.undo() == roll


def test_observation_does_not_populate_empty_building_caches():
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    before = pickle.dumps(engine)
    for color in COLORS:
        observation = engine.observe(color)
        assert observation.my_roads == []
    assert pickle.dumps(engine) == before


@pytest.mark.parametrize("validate_action", [True, False])
@pytest.mark.parametrize("action_type", [ActionType.ROLL, ActionType.END_TURN])
def test_terminal_guard_precedes_history_and_rng(engine, monkeypatch, validate_action, action_type):
    engine.state.player_state["P0_ACTUAL_VICTORY_POINTS"] = engine.vps_to_win
    engine.state.player_state["P0_HAS_ROLLED"] = action_type == ActionType.END_TURN
    engine.state.playable_actions = generate_playable_actions(engine.state)
    action = Action(Color.RED, action_type, None)
    assert action in engine.state.playable_actions
    before = pickle.dumps(engine)

    def reject_state_copy(self, memo):
        pytest.fail("Terminal rejection must precede history capture")

    monkeypatch.setattr(GameState, "__deepcopy__", reject_state_copy, raising=False)
    with pytest.raises(ValueError, match="terminal"):
        engine.step(action, validate_action=validate_action)

    assert pickle.dumps(engine) == before
    assert engine.rng is engine.state.rng
    assert not engine.is_action_valid(action)


@pytest.mark.parametrize(
    "action",
    [
        Action(Color.RED, ActionType.ROLL, None),
        Action(Color.RED, ActionType.DISCARD, None),
        Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None),
        Action(Color.RED, ActionType.STEAL, (Color.BLUE, None)),
    ],
)
def test_terminal_force_still_requires_explicit_outcomes_before_history(engine, action):
    engine.state.player_state["P0_ACTUAL_VICTORY_POINTS"] = engine.vps_to_win
    before = pickle.dumps(engine)

    with pytest.raises(ValueError, match="Forced"):
        engine.step(action, force=True)

    assert pickle.dumps(engine) == before


def test_terminal_force_explicit_bypass_publishes_without_rng_and_can_undo(engine):
    engine.state.player_state["P0_ACTUAL_VICTORY_POINTS"] = engine.vps_to_win
    action = Action(Color.RED, ActionType.ROLL, (1, 2))
    rng = engine.rng.getstate()
    assert not engine.is_action_valid(action)

    transition = engine.step(action, force=True)

    assert (transition.before_revision, transition.after_revision) == (0, 1)
    assert transition.resolved_action == action
    assert transition.winner == Color.RED
    assert engine.events[0].public_payload == (1, 2)
    assert engine.rng.getstate() == rng
    assert len(engine.history) == 1
    assert engine.undo() == action
    assert engine.events == []
    assert engine.state.actions == []
    assert engine.winning_color() == Color.RED
    assert engine.rng is engine.state.rng
    assert engine.rng.getstate() == rng


@pytest.mark.parametrize(
    "owner_points, responder_points, winner",
    [(9, 10, None), (10, 9, Color.RED), (10, 12, Color.RED)],
)
def test_winner_is_turn_owner_not_discarding_actor(engine, owner_points, responder_points, winner):
    engine.state.current_player_index = 1
    engine.state.current_prompt = ActionPrompt.DISCARD
    engine.state.is_discarding = True
    engine.state.player_state["P0_ACTUAL_VICTORY_POINTS"] = owner_points
    engine.state.player_state["P1_ACTUAL_VICTORY_POINTS"] = responder_points
    assert engine.state.current_color() == Color.BLUE
    assert engine.state.colors[engine.state.current_turn_index] == Color.RED

    assert engine.winning_color() == winner


def test_end_turn_declares_next_turn_owner_not_the_responding_actor(engine):
    engine.state.player_state["P1_ACTUAL_VICTORY_POINTS"] = engine.vps_to_win
    assert engine.winning_color() is None
    action = Action(Color.RED, ActionType.END_TURN, None)
    assert engine.is_action_valid(action)

    transition = engine.step(action)

    assert transition.resolved_action.color == Color.RED
    assert transition.winner == engine.winning_color() == Color.BLUE
    assert engine.state.player_state["P1_HAS_ROLLED"] is False
    assert engine.observe(Color.BLUE).valid_actions == []


def test_exact_discard_validation_step_and_private_event_agree(engine):
    engine.step(Action(Color.RED, ActionType.ROLL, (3, 4)), force=True)
    cards = ["WOOD", "BRICK", "WOOD", "BRICK"]
    action = Action(Color.RED, ActionType.DISCARD, cards)
    assert action not in engine.state.playable_actions
    assert engine.state.playable_actions == [Action(Color.RED, ActionType.DISCARD, None)]
    before = pickle.dumps(engine)

    assert validate_discard(engine.state, action) == tuple(cards)
    assert engine.is_action_valid(action)
    assert engine.is_action_valid(action)
    assert pickle.dumps(engine) == before
    rng = engine.rng.getstate()
    transition = engine.step(action)

    assert transition.requested_action.value == cards
    assert transition.resolved_action.value == tuple(cards)
    assert get_player_freqdeck(engine.state, Color.RED) == [3, 2, 0, 0, 0]
    assert engine.state.resource_freqdeck == [16, 17, 19, 19, 16]
    assert engine.state.current_prompt == ActionPrompt.MOVE_ROBBER
    assert engine.project_events(Color.RED)[-1].payload == tuple(cards)
    assert all(engine.project_events(color)[-1].payload == 4 for color in COLORS[1:])
    assert engine.rng.getstate() == rng

    cards[0] = "ORE"
    transition.requested_action.value[1] = "ORE"
    assert engine.history[-1][1].value == ["WOOD", "BRICK", "WOOD", "BRICK"]
    assert engine.state.actions[-1].value == ("WOOD", "BRICK", "WOOD", "BRICK")
    assert engine.project_events(Color.RED)[-1].payload == engine.state.actions[-1].value


@pytest.mark.parametrize(
    "cards",
    [
        "WOOD",
        {"WOOD": 4},
        [True] * 4,
        [[]] * 4,
        ["GOLD"] * 4,
        ["WOOD"] * 3,
        ["WOOD"] * 5,
        ["ORE"] * 4,
    ],
)
def test_invalid_exact_discard_is_rejected_before_engine_history(engine, cards):
    engine.step(Action(Color.RED, ActionType.ROLL, (3, 4)), force=True)
    before = pickle.dumps(engine)
    action = Action(Color.RED, ActionType.DISCARD, cards)

    with pytest.raises(ValueError):
        validate_discard(engine.state, action)
    assert not engine.is_action_valid(action)
    with pytest.raises(ValueError, match="not playable"):
        engine.step(action)

    assert pickle.dumps(engine) == before


@pytest.mark.parametrize("closed", [False, True], ids=["missing", "closed"])
def test_window_preview_is_pure_and_matches_materialized_identity(engine, offer_action, closed):
    if closed:
        engine.step(offer_action)
        engine.state.trade_window.close()
    old_window = engine.state.trade_window
    before = pickle.dumps(engine)

    first = new_trade_window(engine.state)
    second = new_trade_window(engine.state)
    assert first is not second
    assert first == second
    assert first.id == f"turn-{engine.state.num_turns}-trade-{len(engine.state.actions)}"
    assert first.turn_player == Color.RED
    assert first.participants == COLORS
    assert engine.is_action_valid(offer_action)
    assert engine.is_action_valid(offer_action)
    assert engine.state.trade_window is old_window
    assert pickle.dumps(engine) == before

    _message(engine)
    assert new_trade_window(engine.state).id == first.id
    first.close()
    materialized = ensure_trade_window(engine.state)
    assert materialized.id == second.id
    assert materialized.status == TradeWindowStatus.OPEN
    assert ensure_trade_window(engine.state) is materialized
    transition = engine.step(offer_action)
    assert transition.resolved_action.value.id == f"{second.id}:o1"
    if closed:
        assert materialized.id != old_window.id


def test_step_and_message_share_the_canonical_publish_seam(engine, offer_action, monkeypatch):
    publish = engine.publish_event
    published = []

    def record_publish(*args, **kwargs):
        event = publish(*args, **kwargs)
        published.append(event)
        return event

    monkeypatch.setattr(engine, "publish_event", record_publish)
    external = engine.publish_event("REPLAY_FACT", Color.RED, {"resources": {"WOOD": [1]}})
    message = _message(engine)
    transition = engine.step(offer_action)

    assert [event.event_type for event in published] == [
        "REPLAY_FACT",
        "MESSAGE_SENT",
        "OFFER_TRADE",
    ]
    assert [event.sequence for event in published] == [0, 1, 2]
    assert [event.causation_id for event in published] == ["action:0", "talk:boundary", "action:2"]
    assert (transition.before_revision, transition.after_revision) == (2, 3)
    assert transition.events == (published[-1],)
    assert engine.events == [external, message, transition.events[0]]
    assert len(engine.state.actions) == len(engine.history) == 1
    assert engine.history[0][2] == 2


def test_publish_event_detaches_nested_inputs_returns_and_every_projection(engine):
    public = {"cards": [{"WOOD": [1, 2]}]}
    private = {"cards": [{"ORE": [3]}]}
    state_before = pickle.dumps(engine.state)
    returned = engine.publish_event(
        "REPLAY_FACT",
        Color.RED,
        public,
        private_overlays=((Color.BLUE, private),),
        causation_id="replay:7",
    )
    canonical = deepcopy(engine.events[0])

    public["cards"][0]["WOOD"].append(9)
    private["cards"][0]["ORE"].append(9)
    returned.public_payload["cards"][0]["WOOD"].append(8)
    returned.private_overlays[0][1]["cards"][0]["ORE"].append(8)
    red = engine.project_events(Color.RED)[0]
    blue = engine.project_events(Color.BLUE)[0]
    white = engine.project_events(Color.WHITE)[0]
    red.payload["cards"][0]["WOOD"].append(7)
    blue.payload["cards"][0]["ORE"].append(7)

    assert engine.events == [canonical]
    assert white.payload == {"cards": [{"WOOD": [1, 2]}]}
    assert engine.project_events(Color.BLUE)[0].payload == {"cards": [{"ORE": [3]}]}
    assert engine.revision == 1
    assert engine.history == []
    assert pickle.dumps(engine.state) == state_before


@pytest.mark.parametrize(
    "event_type, actor", [("", Color.RED), (None, Color.RED), ("FACT", Color.BLACK)]
)
def test_invalid_published_event_is_rejected_without_appending(engine, event_type, actor):
    before = pickle.dumps(engine)
    with pytest.raises(ValueError):
        engine.publish_event(event_type, actor, {"cards": []})
    assert pickle.dumps(engine) == before


def test_event_from_action_and_projection_detach_mutable_roll_payload():
    dice = [2, 3]
    action = Action(Color.RED, ActionType.ROLL, dice)
    event = event_from_action(action, 4)
    projected = project_event(event, Color.RED)
    dice[0] = 6
    projected.payload[1] = 6

    assert event.public_payload == [2, 3]
    assert project_event(event, Color.BLUE).payload == [2, 3]
    assert (event.sequence, event.causation_id) == (4, "action:4")


def test_transition_actions_event_and_request_do_not_alias_live_trade(engine, offer_action):
    transition = engine.step(offer_action)
    expected_requested = deepcopy(transition.requested_action)
    expected_resolved = deepcopy(transition.resolved_action)
    expected_event = deepcopy(transition.events[0])
    before = pickle.dumps(engine)

    offer_action.value.give = (4, 0, 0, 0, 0)
    assert transition.requested_action == expected_requested
    transition.requested_action.value.willing_by.add(Color.WHITE)
    transition.resolved_action.value.give = (3, 0, 0, 0, 0)
    transition.resolved_action.value.willing_by.add(Color.ORANGE)
    transition.events[0].public_payload["give"]["WOOD"] = 9
    transition.events[0].public_payload["audience"].append("WHITE")

    assert engine.state.actions[-1] == expected_resolved
    assert engine.events[-1] == expected_event
    assert pickle.dumps(engine) == before

    engine.step(Action(Color.BLUE, ActionType.ACCEPT_TRADE, expected_resolved.value.id))
    assert engine.state.trade_window.offers[expected_resolved.value.id].willing_by == {Color.BLUE}
    assert engine.state.actions[0] == expected_resolved
    assert engine.events[0] == expected_event


@pytest.mark.parametrize("field", ["requested_action", "resolved_action"])
def test_returned_roll_action_does_not_alias_materialized_state(engine, field):
    transition = engine.step(Action(Color.RED, ActionType.ROLL, [1, 2]), force=True)
    before = pickle.dumps(engine)

    returned_dice = getattr(transition, field).value
    assert tuple(returned_dice) == (1, 2)
    if isinstance(returned_dice, list):
        returned_dice[0] = 6
    else:
        assert isinstance(returned_dice, tuple)

    assert tuple(engine.state.last_dice_roll) == (1, 2)
    assert tuple(engine.state.actions[-1].value) == (1, 2)
    assert tuple(transition.events[0].public_payload) == (1, 2)
    assert pickle.dumps(engine) == before


@pytest.mark.parametrize(
    "commitment",
    [
        True,
        1,
        "abc",
        {},
        (),
        ("condition", "promise"),
        ("condition", "promise", 2, "extra"),
        (None, "promise", 2),
        (" ", "promise", 2),
        ("condition", None, 2),
        ("condition", 1, 2),
        ("condition", [], 2),
        ("condition", " ", 2),
        ("condition", "promise", True),
        ("condition", "promise", False),
        ("condition", "promise", -1),
        ("condition", "promise", 2.0),
        ("condition", "promise", "2"),
        ("condition", "promise", None),
    ],
)
def test_invalid_direct_commitment_never_appends_a_partial_message(engine, commitment):
    _message(engine)
    before = pickle.dumps(engine)

    with pytest.raises(ValueError, match="Commitment"):
        _message(engine, commitment=commitment)

    assert engine.revision == len(engine.commitments) == 1
    assert pickle.dumps(engine) == before


@pytest.mark.parametrize(
    "changes",
    [
        {"speaker": Color.BLACK},
        {"text": None},
        {"text": " \n"},
        {"audience": None},
        {"audience": "BLUE"},
        {"audience": ("BLUE",)},
        {"audience": ([],)},
        {"audience": (Color.BLACK,)},
        {"causation_id": None},
        {"causation_id": ""},
    ],
)
def test_invalid_direct_message_rejects_before_events_or_commitments(engine, changes):
    _message(engine)
    before = pickle.dumps(engine)

    with pytest.raises(ValueError):
        _message(engine, **changes)

    assert pickle.dumps(engine) == before


def test_private_message_and_commitment_are_detached_views_not_live_handles(engine):
    audience = [Color.BLUE]
    commitment = ["Leave the robber elsewhere", "Offer ORE", 0]
    returned = _message(engine, audience=audience, commitment=commitment)
    active = engine.active_commitments(Color.BLUE)
    projected = engine.project_messages(Color.BLUE)
    audience.append(Color.WHITE)
    commitment[1] = "Give everything"
    returned.private_overlays[0][1]["text"] = "Rewritten"
    projected[0].payload["text"] = "Also rewritten"
    active[0].promise = "Not the stored promise"

    assert engine.project_messages(Color.WHITE) == ()
    assert engine.active_commitments(Color.WHITE) == ()
    assert engine.events[0].public_payload is None
    assert engine.events[0].visible_to == (Color.RED, Color.BLUE)
    assert engine.project_messages(Color.RED)[0].payload["text"].startswith("Leave the robber")
    assert engine.commitments[0].audience == (Color.BLUE,)
    assert engine.commitments[0].promise == "Offer ORE"
    assert engine.commitments[0].source_message_sequence == returned.sequence

    engine.step(Action(Color.RED, ActionType.END_TURN, None))

    assert engine.commitments[0].status == CommitmentStatus.EXPIRED
    assert engine.active_commitments(Color.BLUE) == ()
    assert active[0].status == CommitmentStatus.ACTIVE
    engine.undo()
    assert engine.commitments[0].status == CommitmentStatus.ACTIVE
    assert engine.commitments[0].promise == "Offer ORE"


def test_message_window_zero_means_empty_not_default(engine):
    _message(engine)
    _message(engine, text="Second message")
    assert engine.project_messages(Color.BLUE, limit=0) == ()
    assert len(engine.project_messages(Color.BLUE, limit=None)) == 2
    assert [event.payload["text"] for event in engine.project_messages(Color.BLUE, limit=1)] == [
        "Second message"
    ]


@pytest.mark.parametrize("limit", [True, False, -1, 1.5, "1"])
def test_message_window_rejects_non_integer_or_negative_limits(engine, limit):
    _message(engine)
    before = pickle.dumps(engine)
    with pytest.raises(ValueError, match="non-negative integer"):
        engine.project_messages(Color.BLUE, limit=limit)
    assert pickle.dumps(engine) == before


@pytest.mark.parametrize("boundary", ["snapshot", "copy", "restore"])
def test_branch_mutations_cannot_change_source_or_saved_snapshot(engine, offer_action, boundary):
    _message(engine)
    engine.step(offer_action)
    engine.step(Action(Color.BLUE, ActionType.ACCEPT_TRADE, engine.state.actions[-1].value.id))
    saved = engine.snapshot()
    original_bytes = pickle.dumps(engine)
    saved_bytes = pickle.dumps(saved)
    if boundary == "snapshot":
        branch = engine.snapshot()
    elif boundary == "copy":
        branch = engine.copy()
    else:
        branch = GameEngine(COLORS, seed=9, shuffle_players=False)
        branch.restore(saved)
    assert branch.state.rng is not engine.rng
    assert branch.state.rng.getstate() == engine.rng.getstate()
    if boundary != "snapshot":
        assert branch.rng is branch.state.rng

    coordinate = next(iter(branch.state.board.map.land_tiles))
    tile = branch.state.board.map.land_tiles[coordinate]
    assert branch.state.board.map.tiles[coordinate] is tile
    assert branch.state.board.map.tiles_by_id[tile.id] is tile
    tile.nodes.clear()
    tile.number = 12
    branch.state.board.map.node_production.clear()
    branch.state.actions[0].value.willing_by.add(Color.WHITE)
    branch.state.trade_window.offers[branch.state.actions[0].value.id].give = (4, 0, 0, 0, 0)
    branch.events[1].public_payload["give"]["WOOD"] = 9
    branch.events[0].private_overlays[0][1]["text"] = "Branch-only text"
    branch.commitments[0].promise = "Branch-only promise"
    branch.history[0][1].value.give = (3, 0, 0, 0, 0)
    branch.history[1][0].actions[0].value.declined_by.add(Color.WHITE)
    branch.history[0][0].board.map.land_tiles[coordinate].edges.clear()
    branch.history[0][3][0].status = CommitmentStatus.VIOLATED
    branch.state.rng.random()

    assert pickle.dumps(engine) == original_bytes
    assert pickle.dumps(saved) == saved_bytes


def test_history_capture_owns_nested_map_actions_and_commitments(engine, offer_action):
    _message(engine)
    engine.step(offer_action)
    offer_id = engine.state.actions[0].value.id
    accept = Action(Color.BLUE, ActionType.ACCEPT_TRADE, offer_id)
    engine.step(accept)
    coordinate = next(iter(engine.state.board.map.land_tiles))
    original_nodes = dict(engine.state.board.map.land_tiles[coordinate].nodes)
    history_before = pickle.dumps(engine.history)

    engine.state.board.map.land_tiles[coordinate].nodes.clear()
    engine.state.actions[0].value.give = (4, 0, 0, 0, 0)
    offer_action.value.give = (3, 0, 0, 0, 0)
    engine.commitments[0].promise = "Rewritten after history capture"

    assert pickle.dumps(engine.history) == history_before
    assert engine.undo() == accept
    assert engine.state.board.map.land_tiles[coordinate].nodes == original_nodes
    assert engine.state.actions[0].value.give == (1, 0, 0, 0, 0)
    assert engine.state.trade_window.offers[offer_id].willing_by == set()
    assert engine.commitments[0].promise == "Offer ORE"
    assert engine.revision == 2
    assert engine.rng is engine.state.rng


def test_restore_detaches_from_later_snapshot_mutation_and_replays_rng(engine):
    engine.step(Action(Color.RED, ActionType.END_TURN, None))
    saved = engine.snapshot()
    expected = engine.step(Action(Color.BLUE, ActionType.ROLL, None))
    expected_rng = engine.rng.getstate()
    engine.restore(saved)
    before = pickle.dumps(engine)
    saved.state.board.map.land_tiles.clear()
    saved.state.actions.clear()
    saved.history[0][0].resource_freqdeck[0] = 0
    saved.state.rng.random()

    assert pickle.dumps(engine) == before
    assert engine.rng is engine.state.rng
    assert engine.step(Action(Color.BLUE, ActionType.ROLL, None)) == expected
    assert engine.rng.getstate() == expected_rng


@pytest.mark.parametrize("projection", ["engine", "state"])
def test_observation_detaches_nested_map_actions_history_and_trade(
    engine, offer_action, projection
):
    engine.step(offer_action)
    # Lists are supported forced replay dice and exercise nested action values.
    engine.step(Action(Color.RED, ActionType.ROLL, [1, 2]), force=True)
    engine.state.playable_actions = [offer_action]
    recent = engine.state.actions
    observation = (
        engine.observe(Color.RED)
        if projection == "engine"
        else observe_state(engine.state, Color.RED, recent_events=recent)
    )
    other = engine.observe(Color.BLUE)
    before = pickle.dumps(engine)
    other_bytes = pickle.dumps(other)

    next(iter(observation.board_map.land_tiles.values())).nodes.clear()
    next(iter(observation.board_map.node_production.values())).clear()
    next(iter(observation.board_map.adjacent_tiles.values())).clear()
    observation.buildings_dict[0] = (Color.RED, CITY)
    observation.my_settlements.append(0)
    observation.opponent_roads[Color.BLUE].append((0, 1))
    observation.my_resources["WOOD"] = 99
    observation.valid_actions[0].value.willing_by.add(Color.WHITE)
    observation.trade_window.offers[recent[0].value.id].give = (4, 0, 0, 0, 0)
    if isinstance(observation.last_dice_roll, list):
        observation.last_dice_roll[0] = 6
    else:
        assert observation.last_dice_roll == (1, 2)
    if projection == "state":
        observation.recent_events[0].value.declined_by.add(Color.WHITE)
        if isinstance(observation.recent_events[-1].value, list):
            observation.recent_events[-1].value[1] = 6
        else:
            assert observation.recent_events[-1].value == (1, 2)

    assert pickle.dumps(engine) == before
    assert pickle.dumps(other) == other_bytes
    assert tuple(engine.observe(Color.RED).last_dice_roll) == (1, 2)


@pytest.mark.parametrize("terminal", [False, True])
def test_serialization_and_observation_agree_on_terminal_menu(engine, terminal):
    engine.step(Action(Color.RED, ActionType.ROLL, (1, 2)), force=True)
    engine.state.player_state["P0_ACTUAL_VICTORY_POINTS"] = 10 if terminal else 9
    before = pickle.dumps(engine)
    raw_menu = deepcopy(engine.state.playable_actions)
    assert raw_menu

    payload = json.loads(json.dumps(engine, cls=GameEncoder))

    assert payload["winning_color"] == ("RED" if terminal else None)
    assert payload["current_playable_actions"] == json.loads(
        json.dumps([] if terminal else raw_menu, cls=GameEncoder)
    )
    assert engine.observe(Color.RED).valid_actions == ([] if terminal else raw_menu)
    assert all(engine.observe(color).valid_actions == [] for color in COLORS[1:])
    assert payload["actions"] == [["RED", "ROLL", [1, 2]]]
    assert engine.state.playable_actions == raw_menu
    assert pickle.dumps(engine) == before


def test_history_disabled_step_does_not_deepcopy_state_or_board(engine, offer_action, monkeypatch):
    engine.capture_history = False

    def reject_full_copy(self, memo):
        pytest.fail("A history-disabled step must not deepcopy state, board, or map")

    for cls in (GameState, Board, CatanMap):
        monkeypatch.setattr(cls, "__deepcopy__", reject_full_copy, raising=False)

    engine.step(offer_action)
    _message(engine)
    assert engine.revision == 2
    assert engine.history == []


def _snapshot_road_position(engine, paths, settlements=()):
    """Place bank-balanced free pieces in a reduced position, not a replay."""
    state = engine.state
    for color, node in settlements:
        state.board.build_settlement(color, node, initial_build_phase=True)
        build_settlement(state, color, node, is_free=True)
    for color, path in paths:
        for left, right in zip(path, path[1:]):
            assert STATIC_GRAPH.has_edge(left, right)
            assert (left, right) not in state.board.roads
            state.board.roads[left, right] = state.board.roads[right, left] = color
            build_road(state, color, tuple(sorted((left, right))), is_free=True)
    maintain_longest_road(state, *state.board.recompute_road_state())
    state.playable_actions = generate_playable_actions(state)


def _assert_road_restore_preserves_material(restored, saved, expected_player_state):
    state = restored.state
    assert state.player_state == expected_player_state
    assert {
        key: value for key, value in vars(state).items()
        if key not in {"board", "rng", "player_state", "playable_actions"}
    } == {
        key: value for key, value in vars(saved.state).items()
        if key not in {"board", "rng", "player_state", "playable_actions"}
    }
    for name in ("buildings", "roads", "board_buildable_ids", "robber_coordinate"):
        assert getattr(state.board, name) == getattr(saved.state.board, name)
    assert pickle.dumps(state.board.map) == pickle.dumps(saved.state.board.map)
    assert set(state.board.buildable_subgraph.edges) == set(saved.state.board.buildable_subgraph.edges)
    assert restored.rng is state.rng
    assert restored.rng is not saved.state.rng
    assert restored.rng.getstate() == saved.state.rng.getstate()
    assert (restored.id, restored.seed, restored.vps_to_win) == (
        saved.engine_id, saved.seed, saved.vps_to_win,
    )
    assert tuple(restored.events) == saved.events
    assert tuple(restored.commitments) == saved.commitments
    assert restored.capture_history == saved.capture_history
    assert restored.communication_limits == saved.communication_limits
    assert pickle.dumps(tuple(restored.history)) == pickle.dumps(saved.history)


@pytest.mark.parametrize("stale_part", ["cache", "menu", "both", "inactive-cache"])
def test_restore_removes_legacy_enemy_crossing_road_candidates(engine, stale_part):
    _snapshot_road_position(
        engine, ((Color.RED, (37, 14, 15)),), ((Color.RED, 37), (Color.BLUE, 15)),
    )
    if stale_part == "inactive-cache":
        engine.step(Action(Color.RED, ActionType.END_TURN, None))
    state = engine.state
    blocked = Action(Color.RED, ActionType.BUILD_ROAD, (4, 15))
    expected_actions = deepcopy(state.playable_actions)
    assert blocked not in expected_actions
    assert state.board.road_lengths[Color.RED] == 2
    if stale_part in {"cache", "both", "inactive-cache"}:
        state.board.buildable_edges_cache[Color.RED].append(blocked.value)
    if stale_part in {"menu", "both"}:
        state.playable_actions.append(blocked)
    saved = pickle.loads(pickle.dumps(engine.snapshot()))
    saved_bytes = pickle.dumps(saved)
    source_bytes = pickle.dumps(engine)
    restored = GameEngine(COLORS, initialize=False)

    restored.restore(saved)

    assert blocked.value not in restored.state.board.buildable_edges(Color.RED)
    assert restored.state.playable_actions == expected_actions
    assert not restored.is_action_valid(blocked)
    _assert_road_restore_preserves_material(restored, saved, saved.state.player_state)
    assert pickle.dumps(saved) == saved_bytes
    assert pickle.dumps(engine) == source_bytes
    before_rejection = pickle.dumps(restored)
    with pytest.raises(ValueError, match="not playable"):
        restored.step(blocked)
    assert pickle.dumps(restored) == before_rejection


@pytest.mark.parametrize("stale_board", [False, True], ids=["counters-only", "board-and-counters"])
def test_restore_revokes_legacy_below_five_award_without_rewriting_history(
    engine, offer_action, stale_board,
):
    _snapshot_road_position(
        engine,
        ((Color.RED, (29, 30, 31, 32, 33, 34)), (Color.BLUE, (12, 11, 32))),
        ((Color.RED, 29), (Color.BLUE, 12), (Color.BLUE, 32)),
    )
    state = engine.state
    state.development_listdeck.remove(VICTORY_POINT)
    state.player_state["P0_VICTORY_POINT_IN_HAND"] += 1
    state.player_state["P0_ACTUAL_VICTORY_POINTS"] += 1
    expected_player_state = state.player_state.copy()
    assert state.board.road_lengths[Color.RED] == 3
    assert state.board.road_color is None
    if stale_board:
        state.board.connected_components[Color.RED] = [{29, 30, 31, 32, 33, 34}]
        state.board.road_lengths[Color.RED] = 5
        state.board.road_length = 5
        state.board.road_color = Color.RED
    state.player_state["P0_LONGEST_ROAD_LENGTH"] = 5
    state.player_state["P0_HAS_ROAD"] = True
    state.player_state["P0_VICTORY_POINTS"] += 2
    state.player_state["P0_ACTUAL_VICTORY_POINTS"] += 2
    _message(engine)
    engine.step(offer_action)
    saved = pickle.loads(pickle.dumps(engine.snapshot()))
    saved_bytes = pickle.dumps(saved)
    restored = GameEngine(COLORS, initialize=False)

    restored.restore(saved)

    assert restored.state.board.road_color is None
    assert restored.state.board.road_length == 3
    assert restored.state.board.road_lengths[Color.RED] == 3
    assert restored.state.board.connected_components[Color.RED] == [
        {29, 30, 31, 32}, {32, 33, 34},
    ]
    assert restored.state.player_state["P0_VICTORY_POINTS"] == 1
    assert restored.state.player_state["P0_ACTUAL_VICTORY_POINTS"] == 2
    assert restored.state.playable_actions == generate_playable_actions(restored.state)
    _assert_road_restore_preserves_material(restored, saved, expected_player_state)
    assert restored.history[-1][0].player_state["P0_HAS_ROAD"] is True
    assert pickle.dumps(saved) == saved_bytes
    normalized = restored.snapshot()
    restored.restore(normalized)
    assert pickle.dumps(restored.snapshot()) == pickle.dumps(normalized)


@pytest.mark.parametrize("incumbent", [None, Color.RED, Color.BLUE])
def test_restore_uses_saved_board_incumbent_for_corrected_five_road_tie(engine, incumbent):
    _snapshot_road_position(
        engine,
        ((Color.RED, (29, 30, 31, 32, 33, 34)), (Color.BLUE, (49, 50, 51, 52, 23, 6))),
    )
    state = engine.state
    state.board.road_color = incumbent
    state.board.road_lengths[Color.RED] = 6
    state.board.road_length = 6
    state.player_state["P0_LONGEST_ROAD_LENGTH"] = 6
    # Award flags are derived counters, not evidence of a different historical incumbent.
    wrong_flag = Color.RED if incumbent != Color.RED else Color.BLUE
    key = player_key(state, wrong_flag)
    state.player_state[f"{key}_HAS_ROAD"] = True
    state.player_state[f"{key}_VICTORY_POINTS"] += 2
    state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] += 2
    saved = engine.snapshot()
    restored = GameEngine(COLORS, initialize=False)

    restored.restore(saved)

    assert restored.state.board.road_color == incumbent
    assert restored.state.board.road_length == 5
    for color in COLORS:
        key = player_key(state, color)
        assert restored.state.player_state[f"{key}_LONGEST_ROAD_LENGTH"] == (
            5 if color in {Color.RED, Color.BLUE} else 0
        )
        assert restored.state.player_state[f"{key}_HAS_ROAD"] == (color == incumbent)
        assert restored.state.player_state[f"{key}_VICTORY_POINTS"] == 2 * (color == incumbent)
        assert restored.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] == (
            2 * (color == incumbent)
        )


@pytest.mark.parametrize("opening_steps", [0, 1, 16])
def test_current_snapshot_preserves_material_menu_order_and_forward_suffix(opening_steps):
    engine = GameEngine(COLORS, seed=7, shuffle_players=False, capture_history=True)
    for _ in range(opening_steps):
        engine.step(engine.state.playable_actions[0])
    _message(engine)
    # Equivalent cache order and absent zero-length entries must not trigger normalization.
    for color in COLORS:
        engine.state.board.buildable_edges(color).reverse()
    engine.state.playable_actions = generate_playable_actions(engine.state)
    saved = pickle.loads(pickle.dumps(engine.snapshot()))
    saved_bytes = pickle.dumps(saved)
    restored = GameEngine(COLORS, initialize=False)

    restored.restore(saved)

    assert pickle.dumps(restored.snapshot()) == saved_bytes
    assert restored.state.playable_actions == engine.state.playable_actions
    _assert_road_restore_preserves_material(restored, saved, saved.state.player_state)
    for _ in range(20):
        assert restored.state.playable_actions == engine.state.playable_actions
        action = engine.state.playable_actions[0]
        assert restored.step(action) == engine.step(action)
        assert json.loads(json.dumps(restored, cls=GameEncoder)) == json.loads(
            json.dumps(engine, cls=GameEncoder)
        )
        assert restored.state.playable_actions == engine.state.playable_actions
        assert restored.state.player_state == engine.state.player_state
        assert restored.state.development_listdeck == engine.state.development_listdeck
        assert restored.state.resource_freqdeck == engine.state.resource_freqdeck
        assert restored.rng.getstate() == engine.rng.getstate()
        assert restored.events == engine.events
        assert restored.commitments == engine.commitments
    assert pickle.dumps(saved) == saved_bytes


@pytest.mark.parametrize("difference", ["reordered", "missing-menu", "duplicate-menu", "missing-cache"])
def test_restore_stale_counter_preserves_only_equivalent_menu_and_cache_order(
    engine, monkeypatch, difference,
):
    _snapshot_road_position(
        engine, ((Color.RED, (37, 14, 15)),), ((Color.RED, 37), (Color.BLUE, 15)),
    )
    state = engine.state
    for color in COLORS:
        state.board.buildable_edges(color).reverse()
    state.playable_actions = list(reversed(generate_playable_actions(state)))
    assert state.playable_actions != generate_playable_actions(state)
    expected = engine.copy()
    expected_actions = deepcopy(state.playable_actions)
    state.player_state["P0_LONGEST_ROAD_LENGTH"] = 5
    road_index = next(
        index for index, action in enumerate(state.playable_actions)
        if action.action_type == ActionType.BUILD_ROAD
    )
    if difference == "missing-menu":
        state.playable_actions.pop(road_index)
    elif difference == "duplicate-menu":
        state.playable_actions[road_index] = state.playable_actions[0]
    elif difference == "missing-cache":
        state.board.buildable_edges_cache[Color.RED].pop()
    if difference in {"missing-menu", "duplicate-menu"}:
        expected_actions = generate_playable_actions(expected.state)
    saved = pickle.loads(pickle.dumps(engine.snapshot()))
    saved_bytes = pickle.dumps(saved)
    restored = GameEngine(COLORS, initialize=False)
    # Menu identity comparison must work even when Actions cannot be hashed.
    monkeypatch.setattr(Action, "__hash__", None)

    restored.restore(saved)

    assert restored.state.player_state["P0_LONGEST_ROAD_LENGTH"] == 2
    assert restored.state.playable_actions == expected_actions
    for color, edges in expected.state.board.buildable_edges_cache.items():
        restored_edges = restored.state.board.buildable_edges_cache[color]
        if difference == "missing-cache" and color == Color.RED:
            assert set(restored_edges) == set(edges)
            assert len(restored_edges) == len(edges)
        else:
            assert restored_edges == edges
    _assert_road_restore_preserves_material(restored, saved, expected.state.player_state)
    assert pickle.dumps(saved) == saved_bytes

    if difference == "reordered":
        road = next(
            action for action in expected_actions if action.action_type == ActionType.BUILD_ROAD
        )
        assert restored.step(road) == expected.step(road)
        for _ in range(20):
            assert restored.state.playable_actions == expected.state.playable_actions
            action = expected.state.playable_actions[0]
            assert restored.step(action) == expected.step(action)
            assert json.loads(json.dumps(restored, cls=GameEncoder)) == json.loads(
                json.dumps(expected, cls=GameEncoder)
            )
            assert restored.rng.getstate() == expected.rng.getstate()
            assert restored.state.development_listdeck == expected.state.development_listdeck
            assert restored.events == expected.events
            assert restored.commitments == expected.commitments
