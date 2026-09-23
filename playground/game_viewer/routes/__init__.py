"""Route registration using Flask Blueprints."""

from flask import Flask
from flask_socketio import SocketIO

from ..state import ServerState
from .bench import bench_bp as bench_bp
from .decision_evals import decision_evals_bp
from .health import health_bp
from .inject import inject_bp
from .live_game import live_game_bp
from .mapping import mapping_bp
from .prompt_suite import prompt_suite_bp
from .replay import replay_bp
from .websocket import register_websocket_handlers


def register_routes(app: Flask, socketio: SocketIO, state: ServerState) -> None:
    """Register all route blueprints and websocket handlers."""
    # Store state on app for blueprint access
    app.config['SERVER_STATE'] = state
    app.config['SOCKETIO'] = socketio

    app.register_blueprint(health_bp)
    app.register_blueprint(inject_bp)
    app.register_blueprint(mapping_bp)
    app.register_blueprint(replay_bp)
    app.register_blueprint(live_game_bp)
    app.register_blueprint(prompt_suite_bp)
    app.register_blueprint(bench_bp)
    app.register_blueprint(decision_evals_bp)

    register_websocket_handlers(socketio, state)
