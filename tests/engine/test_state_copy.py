"""Optimized state-copy and replay-checkpoint payload isolation regressions."""
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.decks import freqdeck_subtract
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import player_freqdeck_add
from cle.game_engine.trading import TradeOffer
from cle.replay.runtime.checkpoint import ReplayStepCheckpoint

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def _engine_with_mutable_history() -> GameEngine:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False, capture_history=True)
    state = engine.state
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.PLAY_TURN
    for color, cards in ((Color.RED, [2, 2, 0, 0, 0]), (Color.BLUE, [0, 0, 0, 0, 2])):
        state.resource_freqdeck = freqdeck_subtract(state.resource_freqdeck, cards)
        player_freqdeck_add(state, color, cards)
    state.playable_actions = generate_playable_actions(state)
    engine.step(Action(Color.RED, ActionType.ROLL, [1, 2]), force=True)
    root = engine.step(
        Action(Color.RED, ActionType.OFFER_TRADE, TradeOffer(
            offered_by=Color.RED, audience=frozenset(COLORS[1:]),
            give=(1, 0, 0, 0, 0), receive=(0, 0, 0, 0, 1),
        ))
    ).resolved_action.value
    engine.step(
        Action(Color.BLUE, ActionType.COUNTER_OFFER, TradeOffer(
            offered_by=Color.BLUE, audience=frozenset({Color.RED}),
            give=(0, 0, 0, 0, 1), receive=(0, 1, 0, 0, 0), parent_offer_id=root.id,
        ))
    )
    assert state.actions[0].value is state.last_dice_roll
    return engine


@pytest.mark.parametrize("mutate_original", [False, True], ids=["copy-mutation", "original-mutation"])
def test_state_copy_detaches_historical_trade_offers_and_list_dice(mutate_original: bool) -> None:
    original = _engine_with_mutable_history().state
    branch = original.copy()
    expected_actions = deepcopy(original.actions)
    expected_window = deepcopy(original.trade_window)
    changed, unchanged = (original, branch) if mutate_original else (branch, original)
    changed.actions[0].value[0] = 6
    changed.last_dice_roll[1] = 6
    changed.actions[1].value.give = (2, 0, 0, 0, 0)
    changed.actions[1].value.willing_by.add(Color.WHITE)
    changed.actions[2].value.declined_by.add(Color.RED)
    changed.actions[2].value.parent_offer_id = "changed-parent"
    assert unchanged.actions == expected_actions
    assert unchanged.last_dice_roll == [1, 2]
    assert unchanged.trade_window == expected_window
    assert branch.last_dice_roll is branch.actions[0].value
    assert branch.actions[1].value is not original.actions[1].value
    assert branch.actions[2].value is not original.actions[2].value


def test_state_copy_detaches_mutable_playable_payloads_including_nested_tuples() -> None:
    original = _engine_with_mutable_history().state
    nested = ([1, 0, 0, 0, 0], [0, 0, 0, 0, 1])
    original.playable_actions = [
        original.actions[1],
        Action(Color.RED, ActionType.MARITIME_TRADE, nested),
    ]
    expected = deepcopy(original.playable_actions)
    branch = original.copy()
    branch.playable_actions[0].value.willing_by.add(Color.ORANGE)
    branch.playable_actions[1].value[0][0] = 3
    branch.playable_actions[1].value[1][4] = 3
    assert original.playable_actions == expected
    assert not original.actions[1].value.willing_by
    assert branch.playable_actions[0].value is branch.actions[1].value


def test_state_copy_keeps_static_map_shared_without_copying_all_state(monkeypatch: pytest.MonkeyPatch) -> None:
    original = _engine_with_mutable_history().state

    def reject_map_deepcopy(self: object, memo: dict[int, object]) -> None:
        raise AssertionError("Optimized state copy must not deep-copy the static map")

    monkeypatch.setattr(type(original.board.map), "__deepcopy__", reject_map_deepcopy, raising=False)
    branch = original.copy()
    assert branch.board is not original.board
    assert branch.board.map is original.board.map
    assert branch.rng is not original.rng
    assert branch.rng.getstate() == original.rng.getstate()
    assert branch.actions == original.actions
    assert branch.trade_window == original.trade_window
    branch.color_to_index[Color.RED] = 3
    assert original.color_to_index[Color.RED] == 0


def test_replay_checkpoint_reuse_detaches_list_dice_and_historical_offers() -> None:
    engine: Any = _engine_with_mutable_history()
    runtime: Any = SimpleNamespace(
        current_game=engine,
        replay_index=3,
        replay_actions_per_step=[1, 1, 1],
        game_log=[],
        replay_semantic_issues=[],
        first_divergence_step={},
        game_running=True,
        replay_final_state_synced=False,
        replay_pending_dev_card=None,
        replay_trade_ledger={},
    )
    expected_actions: Any = deepcopy(engine.state.actions)
    expected_rng: Any = engine.rng.getstate()
    checkpoint: Any = ReplayStepCheckpoint.capture(runtime)
    for _ in range(2):
        engine.state.actions[0].value[0] = 6
        engine.state.last_dice_roll[1] = 6
        engine.state.actions[1].value.willing_by.add(Color.ORANGE)
        engine.state.actions[2].value.give = (0, 0, 0, 0, 2)
        engine.state.actions[2].value.parent_offer_id = "mutated-parent"
        engine.rng.random()
        assert checkpoint.game_state.actions == expected_actions
        assert checkpoint.game_state.last_dice_roll == [1, 2]
        checkpoint.restore(runtime)
        assert engine.state.actions == expected_actions
        assert engine.state.last_dice_roll == [1, 2]
        assert engine.state is not checkpoint.game_state
        assert engine.state.actions[1].value is not checkpoint.game_state.actions[1].value
        assert engine.state.actions[2].value is not checkpoint.game_state.actions[2].value
        assert engine.rng is engine.state.rng
        assert engine.rng.getstate() == expected_rng
