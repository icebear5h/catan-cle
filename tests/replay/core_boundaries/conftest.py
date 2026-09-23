"""Opening replay sandbox shared by the replay core boundary tests."""

from types import SimpleNamespace

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.sandbox.replay import ReplaySandbox


@pytest.fixture
def opening_replay() -> tuple[SimpleNamespace, ReplaySandbox]:
    colors = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)
    game = GameEngine(
        colors, seed=3, shuffle_players=False, capture_history=True,
        discard_limit=5, vps_to_win=12,
    )
    game.append_message(
        speaker=Color.RED, text="initial promise", audience=(Color.BLUE,),
        causation_id="opening",
        commitment=("offer wood", "return brick", 1),
    )
    runtime = SimpleNamespace(
        replay_revision=0, replay_index=0, replay_mode=True, game_running=True,
        replay_actions_per_step=[], replay_step_checkpoints=[],
        replay_trade_ledger={}, replay_semantic_issues=[], first_divergence_step={},
        replay_final_state_synced=False, replay_pending_dev_card=None,
        game_log=[{"message": "initial log"}],
        corner_to_node_map={}, edge_to_edge_map={},
    )
    sandbox = ReplaySandbox(runtime, game)
    runtime.current_sandbox = sandbox
    trajectory = game.copy()
    actions = []
    for index in range(18):
        if index < 16:
            action = trajectory.state.playable_actions[0]
        elif index == 16:
            action = Action(Color.RED, ActionType.ROLL, (1, 1))
        else:
            action = Action(Color.RED, ActionType.END_TURN, None)
        hint = {
            "index": index, "type": action.action_type.value,
            "player": colors.index(action.color) + 1,
        }
        if action.action_type == ActionType.BUILD_SETTLEMENT:
            hint["colonist_corner"] = str(index)
            runtime.corner_to_node_map[f"_{index}"] = action.value
        elif action.action_type == ActionType.BUILD_ROAD:
            hint["colonist_edge"] = str(index)
            runtime.edge_to_edge_map[f"_{index}"] = tuple(sorted(action.value))
        elif action.action_type == ActionType.ROLL:
            hint["dice"] = action.value
        trajectory.step(action, force=True)
        actions.append(hint)
    runtime.replay_data = {
        "game_id": "opening", "parsed_actions": actions,
        "events": [{} for _ in actions], "end_game_state": {},
        "colonist_color_to_engine_idx": {str(index + 1): index for index in range(4)},
    }
    return runtime, sandbox
