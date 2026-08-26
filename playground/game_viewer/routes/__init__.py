"""Route registration using Flask Blueprints."""

from .health import health_bp
from .inject import inject_bp
from .mapping import mapping_bp
from .replay import replay_bp
from .live_game import live_game_bp
from .bench import bench_bp
from .decision_evals import decision_evals_bp
from .websocket import register_websocket_handlers


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
    app.register_blueprint(bench_bp)
    app.register_blueprint(decision_evals_bp)

    register_websocket_handlers(socketio, state)
