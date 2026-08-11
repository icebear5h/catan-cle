"""Route registration using Flask Blueprints."""

from .health import health_bp
from .inject import inject_bp
from .mapping import mapping_bp
from .replay import replay_bp
from .live_game import live_game_bp
from .current_state import current_state_bp
from .bench import bench_bp
from .websocket import register_websocket_handlers, broadcast_game_state


def register_routes(app, socketio, state):
    """Register all route blueprints and websocket handlers."""
    # Store state on app for blueprint access
    app.config['SERVER_STATE'] = state
    app.config['SOCKETIO'] = socketio

    app.register_blueprint(health_bp)
    app.register_blueprint(inject_bp)
    app.register_blueprint(mapping_bp)
    app.register_blueprint(replay_bp)
    app.register_blueprint(live_game_bp)
    app.register_blueprint(current_state_bp)
    app.register_blueprint(bench_bp)

    register_websocket_handlers(socketio, state)
