"""Auto-play game runner."""

import time

from cle.agents.llm_player import LLMPlayer
from .game_logging import analyze_action, post_analyze_action


def run_game_auto(state, delay, broadcast_fn):
    """Run game automatically with delays."""
    while state.game_running and state.current_game and state.auto_play_running:
        game = state.current_game
        current_player = game.state.current_player()

        decision_info = {
            "color": str(current_player.color),
            "is_llm": isinstance(current_player, LLMPlayer),
            "timestamp": time.time()
        }

        if not game.state.playable_actions:
            break

        decision_info["available_actions"] = [str(a) for a in game.state.playable_actions]

        if isinstance(current_player, LLMPlayer):
            state.llm_processing = True

        try:
            action = current_player.decide(game, game.state.playable_actions)
        finally:
            if isinstance(current_player, LLMPlayer):
                state.llm_processing = False

        decision_info["action"] = str(action)

        if isinstance(current_player, LLMPlayer):
            if hasattr(current_player, 'last_reasoning') and current_player.last_reasoning:
                decision_info["reasoning"] = current_player.last_reasoning
            if hasattr(current_player, 'last_game_plan') and current_player.last_game_plan:
                decision_info["game_plan"] = current_player.last_game_plan
            if hasattr(current_player, 'last_observation') and current_player.last_observation:
                decision_info["observation"] = current_player.last_observation
            if hasattr(current_player, 'strategic_notes') and current_player.strategic_notes:
                decision_info["strategic_notes"] = current_player.strategic_notes

        pre_state = analyze_action(state, action, game.state)

        game.execute(action)

        post_analyze_action(state, pre_state, game.state)

        winner = game.winning_color()
        if winner:
            state.game_running = False
            decision_info["game_over"] = True
            decision_info["winner"] = str(winner)

        state.llm_thinking.append(decision_info)

        broadcast_fn()

        time.sleep(delay)

        if winner or not state.auto_play_running:
            break

    state.auto_play_running = False
