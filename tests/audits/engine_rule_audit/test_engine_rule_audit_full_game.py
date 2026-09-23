"""Opt-in full-game conservation and continuity audit."""
import os
import random
from collections import Counter
from typing import Any

import pytest

from cle.game_engine.events import project_event
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.decks import (
    DEVELOPMENT_CARD_COUNTS,
)
from cle.game_engine.models.enums import (
    RESOURCES,
    Action,
    ActionType,
)
from cle.game_engine.state_functions import (
    get_player_freqdeck,
    player_key,
)
from cle.sandbox import CatanSandbox, TerminalSandboxError

from .players import _AuditPlayer
from .support import AUDIT_SEED_COUNT, COLORS, _assert_inventory, _assert_road_scores


@pytest.mark.filterwarnings("ignore:Prompt suite .* is deprecated:DeprecationWarning")
@pytest.mark.skipif(
    os.environ.get("CATAN_FULL_GAME_AUDIT") != "1",
    reason="Opt in with CATAN_FULL_GAME_AUDIT=1; CATAN_AUDIT_SEEDS defaults to 32",
)
@pytest.mark.parametrize("seed", range(AUDIT_SEED_COUNT))
@pytest.mark.asyncio
async def test_full_game_inventory_and_seat_continuity(seed: int) -> None:
    game: Any = GameEngine(COLORS, seed=seed, shuffle_players=bool(seed % 2))
    rng = random.Random(100000 + seed)
    players: Any = {color: _AuditPlayer(color, seed, rng) for color in COLORS}
    sandbox: Any = CatanSandbox(game, players)
    accepted: Any = Counter()
    actions: Any = Counter()
    dev_actions: Any = {
        ActionType.PLAY_KNIGHT_CARD: "KNIGHT",
        ActionType.PLAY_YEAR_OF_PLENTY: "YEAR_OF_PLENTY",
        ActionType.PLAY_MONOPOLY: "MONOPOLY",
        ActionType.PLAY_ROAD_BUILDING: "ROAD_BUILDING",
    }
    owned_at_start: Any = {color: Counter() for color in COLORS}
    played_this_turn: Any = Counter()
    _assert_inventory(game)
    _assert_road_scores(game, None)
    for _ in range(3000):
        if game.winning_color() is not None:
            break
        revision = game.revision
        cursors: Any = {color: player.event_cursor for color, player in players.items()}
        menu: Any = tuple(game.state.playable_actions)
        previous_road_holder: Any = game.state.board.road_color
        result: Any = await sandbox.step()
        assert len(result.transitions) == 1 and not result.messages
        transition: Any = result.transitions[0]
        action: Any = transition.requested_action
        menu_action: Any = (
            Action(action.color, action.action_type, None)
            if action.action_type == ActionType.DISCARD else action
        )
        assert menu_action in menu
        assert (result.before_revision, result.after_revision) == (revision, revision + 1)
        assert game.revision == len(game.state.actions) == revision + 1
        actions[transition.requested_action.action_type.value] += 1
        if action.action_type in dev_actions:
            card: Any = dev_actions[action.action_type]
            assert owned_at_start[action.color][card] > 0, (seed, revision, action)
            assert played_this_turn[action.color] == 0, (seed, revision, action)
            owned_at_start[action.color][card] -= 1
            played_this_turn[action.color] += 1
        if action.action_type == ActionType.END_TURN:
            next_color: Any = game.state.current_color()
            key: Any = player_key(game.state, next_color)
            owned_at_start[next_color] = Counter(
                {
                    card: game.state.player_state[f"{key}_{card}_IN_HAND"]
                    for card in DEVELOPMENT_CARD_COUNTS
                }
            )
            played_this_turn[next_color] = 0
        _assert_inventory(game)
        if action.action_type in (ActionType.BUILD_ROAD, ActionType.BUILD_SETTLEMENT):
            _assert_road_scores(game, previous_road_holder)
        if game.winning_color() is None:
            assert game.state.playable_actions == generate_playable_actions(game.state)
        assert not sandbox.decision_trace

        for context, attempt in zip(result.contexts, result.attempts):
            accepted[context.actor] += 1
            assert tuple(event.sequence for event in context.events) == tuple(range(revision))
            assert all(action.color == context.actor for action in context.legal_actions)
            assert context.legal_actions == menu
            assert context.action_at(attempt.choice.action_index) == menu_action
            if action.action_type == ActionType.DISCARD:
                assert action == Action(context.actor, ActionType.DISCARD, attempt.choice.discard_cards)
                assert Counter(action.value) <= Counter(context.observation.my_resources)
                assert len(action.value) == sum(context.observation.my_resources.values()) // 2
            player: Any = players[context.actor]
            assert player.session.receipts[context.context_id].after_revision == revision + 1
            assert player.session.strategic_memory == attempt.choice.game_plan
        for color, player in players.items():
            assert cursors[color] <= player.event_cursor <= game.revision
            assert len(player.session.receipts) == accepted[color]
            assert len(player.session.messages) == 2 * accepted[color]
            observation: Any = game.observe(color)
            if game.winning_color() is None:
                assert observation.valid_actions == (
                    game.state.playable_actions if color == game.state.current_color() else []
                )
            assert [
                observation.my_resources[resource] for resource in RESOURCES
            ] == get_player_freqdeck(game.state, color)
            assert observation.opponent_resource_counts == {
                other: sum(get_player_freqdeck(game.state, other))
                for other in COLORS
                if other != color
            }
            event: Any = project_event(transition.events[0], color)
            resolved: Any = transition.resolved_action
            if resolved.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                assert event.payload == (resolved.value if color == resolved.color else None)
            elif resolved.action_type == ActionType.STEAL:
                victim, resource = resolved.value
                assert event.payload == (
                    victim,
                    resource if color in (resolved.color, victim) else None,
                )
            elif resolved.action_type == ActionType.DISCARD:
                assert event.payload == (
                    tuple(resolved.value) if color == resolved.color else len(resolved.value)
                )

    assert game.winning_color() is not None, f"seed={seed} exceeded 3000 steps: {actions}"
    winner = game.winning_color()
    assert winner == game.state.colors[game.state.current_turn_index]
    assert game.state.player_state[f"{player_key(game.state, winner)}_ACTUAL_VICTORY_POINTS"] >= (
        game.vps_to_win
    )
    final_road_lengths = _assert_road_scores(game, game.state.board.road_color)
    assert all(not sandbox.view(color).legal_actions for color in COLORS)
    assert all(not game.observe(color).valid_actions for color in COLORS)
    revision = game.revision
    with pytest.raises(TerminalSandboxError):
        await sandbox.step()
    assert game.revision == revision
    print(
        f"seed={seed} actions={revision} winner={winner.value} accepted={dict(accepted)} "
        f"road_lengths={ {color.value: length for color, length in final_road_lengths.items()} } "
        f"discards={actions[ActionType.DISCARD.value]}"
    )
