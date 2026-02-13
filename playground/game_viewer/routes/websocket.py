"""WebSocket handlers and broadcast function."""

import json

from flask_socketio import emit

from engine.json import GameEncoder
from cle.agents.llm_player import LLMPlayer
from ..live.game_logging import get_player_resources, get_player_dev_cards


def broadcast_game_state(socketio, state):
    """Broadcast current game state to all connected clients."""
    game = state.current_game
    if not game:
        return

    all_resources = get_player_resources(game.state)
    all_dev_cards = get_player_dev_cards(game.state)

    current_player = game.state.current_player()
    is_current_player_llm = isinstance(current_player, LLMPlayer)

    player_types = {}
    for player in state.current_players:
        color_str = player.color.name if hasattr(player.color, 'name') else str(player.color)
        if isinstance(player, LLMPlayer):
            player_types[color_str] = "LLM"
        else:
            player_types[color_str] = "Random"

    trade_state = None
    if game.state.is_resolving_trade:
        trade_tuple = game.state.current_trade
        offered = list(trade_tuple[:5])
        requested = list(trade_tuple[5:10])
        offering_player_idx = trade_tuple[10]
        offering_color = game.state.colors[offering_player_idx]

        trade_state = {
            "offering_player": offering_color.value,
            "offered_resources": {
                "WOOD": offered[0], "BRICK": offered[1], "SHEEP": offered[2],
                "WHEAT": offered[3], "ORE": offered[4],
            },
            "requested_resources": {
                "WOOD": requested[0], "BRICK": requested[1], "SHEEP": requested[2],
                "WHEAT": requested[3], "ORE": requested[4],
            },
            "acceptees": {
                color.value: accepted
                for color, accepted in zip(game.state.colors, game.state.acceptees)
            },
            "current_prompt": game.state.current_prompt.value,
        }

    state_data = {
        "game": json.loads(json.dumps(game, cls=GameEncoder)),
        "running": state.game_running,
        "llm_thinking": state.llm_thinking[-5:] if state.llm_thinking else [],
        "game_log": state.game_log[-50:] if state.game_log else [],
        "all_player_resources": all_resources,
        "all_player_dev_cards": all_dev_cards,
        "is_current_player_llm": is_current_player_llm,
        "player_types": player_types,
        "trade_state": trade_state,
        "replay_mode": state.replay_mode,
        "last_dice_roll": getattr(game.state, 'last_dice_roll', None),
    }

    if state.replay_mode and state.replay_data:
        state_data["replay"] = {
            "game_id": state.replay_data.get("game_id"),
            "event_index": state.replay_index,
            "total_events": state.replay_data.get("total_events", 0),
            "progress": f"{state.replay_index}/{state.replay_data.get('total_events', 0)}",
            "colonist_players": state.replay_data.get("colonist_players", []),
            "play_order": state.replay_data.get("play_order", []),
        }

    print(f"\n[BROADCAST] Sending game state to clients:")
    print(f"  Running: {state.game_running}")
    print(f"  Turn: {game.state.num_turns}")
    print(f"  Current player: {current_player.color} ({'LLM' if is_current_player_llm else 'Random'})")
    print(f"  Game log entries: {len(state.game_log)}")

    socketio.emit('game_state', state_data)


def register_websocket_handlers(socketio, state):
    """Register SocketIO connect/disconnect handlers."""

    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        print("Client connected")
        if state.current_game:
            broadcast_game_state(socketio, state)

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnect."""
        print("Client disconnected")
