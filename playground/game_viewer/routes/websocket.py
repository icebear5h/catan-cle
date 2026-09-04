"""WebSocket handlers and broadcast function."""

import json

from cle.game_engine.json import GameEncoder
from ..live.game_logging import get_player_resources, get_player_dev_cards
from ..replay.model_traces import build_paired_model_trace_window
from ..replay.narrator_reasoning import build_paired_narrator_reasoning_window
from cle.replay.runtime.trade_ledger import replay_trade_ledger_payload
from ..replay.transcript import build_paired_transcript_window


def build_game_state_snapshot(state):
    sandbox = getattr(state, "current_sandbox", None)
    if sandbox is None:
        return
    game = sandbox.game_engine

    private_resources = get_player_resources(game.state)
    private_dev_cards = get_player_dev_cards(game.state)
    all_resources = {
        color: {"TOTAL": sum(resources.values())}
        for color, resources in private_resources.items()
    }
    all_dev_cards = {
        color: {
            "total_in_hand": cards["total_in_hand"],
            "played": cards["played"],
        }
        for color, cards in private_dev_cards.items()
    }

    if sandbox.players:
        player_statuses = {
            color.name: player.status()
            for color, player in sandbox.players.items()
        }
    else:
        player_statuses = {}
    player_types = {
        color.name: (
            "LLM"
            if player_statuses.get(color.name, {}).get("kind") == "agent"
            else "Random"
        )
        for color in game.state.colors
    }

    trade_state = None
    window = game.state.trade_window
    if window is not None and window.active_offers:
        trade_state = {
            "id": window.id,
            "turn_player": window.turn_player.value,
            "round": window.round,
            "remaining_root_slots": window.remaining_root_slots,
            "remaining_counter_slots": window.remaining_counter_slots,
            "offers": [offer.to_payload() for offer in window.active_offers],
        }

    game_payload = json.loads(json.dumps(game, cls=GameEncoder))
    public_events = [
        {
            "sequence": event.sequence,
            "causation_id": event.causation_id,
            "actor": event.actor.value,
            "event_type": event.event_type,
            "payload": event.public_payload,
        }
        for event in game.events
        if event.visible_to is None
    ]
    game_payload["actions"] = public_events
    game_payload["player_state"] = {
        key: value
        for key, value in game_payload["player_state"].items()
        if not key.endswith("_IN_HAND")
        and not key.endswith("_ACTUAL_VICTORY_POINTS")
        and not key.endswith("_OWNED_AT_START")
    }

    state_data = {
        "game": game_payload,
        "events": public_events,
        "running": state.game_running,
        "game_log": state.game_log[-50:] if state.game_log else [],
        "all_player_resources": all_resources,
        "all_player_dev_cards": all_dev_cards,
        "player_types": player_types,
        "sandbox_players": player_statuses,
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
            "trade_ledger": replay_trade_ledger_payload(state),
        }
        transcript_window = build_paired_transcript_window(
            state.replay_data, state.replay_index
        )
        if transcript_window is not None:
            state_data["replay"]["paired_transcript"] = transcript_window
        model_trace_window = build_paired_model_trace_window(
            state.replay_data, state.replay_index
        )
        if model_trace_window is not None:
            state_data["replay"]["paired_model_trace"] = model_trace_window
        narrator_reasoning_window = build_paired_narrator_reasoning_window(
            state.replay_data, state.replay_index
        )
        if narrator_reasoning_window is not None:
            state_data["replay"]["paired_narrator_reasoning"] = (
                narrator_reasoning_window
            )

    return json.loads(json.dumps(state_data, cls=GameEncoder))


def _broadcast_game_state_snapshot(socketio, state):
    state_data = build_game_state_snapshot(state)
    socketio.emit('game_state', state_data)
    return state_data


def broadcast_game_state(socketio, state):
    """Broadcast one cursor-consistent game/replay snapshot."""
    mutation_lock = getattr(state, "replay_mutation_lock", None)
    if mutation_lock is None:
        return _broadcast_game_state_snapshot(socketio, state)
    with mutation_lock:
        return _broadcast_game_state_snapshot(socketio, state)


def register_websocket_handlers(socketio, state):
    """Register SocketIO connect/disconnect handlers."""

    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        print("Client connected")
        if state.current_sandbox:
            broadcast_game_state(socketio, state)

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnect."""
        print("Client disconnected")
