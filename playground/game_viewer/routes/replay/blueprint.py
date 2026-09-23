"""The replay blueprint and the two accessors every endpoint shares."""

from flask import Blueprint, current_app
from flask_socketio import SocketIO

from ...state import ServerState
from ..websocket import broadcast_game_state

__all__ = ["_broadcast", "_get_state", "replay_bp"]

replay_bp = Blueprint('replay', __name__)


def _get_state() -> ServerState:
    state: ServerState = current_app.config['SERVER_STATE']
    return state


def _broadcast() -> None:
    state = _get_state()
    socketio: SocketIO = current_app.config['SOCKETIO']
    broadcast_game_state(socketio, state)
