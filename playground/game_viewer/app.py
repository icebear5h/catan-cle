"""Flask app + SocketIO creation, __main__ entry point."""

import os
import sys

from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

from cle.traces import SQLiteLiveTraceStore

from .state import server_state
from .routes import register_routes


load_dotenv()
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def create_app():
    """Flask application factory."""
    app = Flask(__name__)
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*")

    if server_state.live_trace_store is None:
        trace_path = os.getenv(
            "CATAN_LIVE_TRACE_DB",
            ".cle/live_traces.sqlite3",
        )
        server_state.live_trace_store = SQLiteLiveTraceStore(trace_path)

    register_routes(app, socketio, server_state)

    return app, socketio


app, socketio = create_app()


def _environment_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def server_run_options() -> dict[str, bool]:
    """Return safe defaults for the stateful viewer development server."""
    return {
        "debug": _environment_flag("CATAN_VIEWER_DEBUG"),
        "use_reloader": _environment_flag("CATAN_VIEWER_RELOAD"),
    }


def run_server() -> None:
    socketio.run(
        app,
        port=5001,
        allow_unsafe_werkzeug=True,
        **server_run_options(),
    )


if __name__ == '__main__':
    print("=" * 60)
    print("Catan Game Viewer Server")
    print("=" * 60)
    print()
    print("Server starting on http://localhost:5001")
    print("UI should connect from http://localhost:5173")
    print()
    print("Endpoints:")
    print("  POST /api/start-game - Start one sandbox")
    print("  POST /api/step - Execute one complete sandbox step")
    print("  GET  /api/state - Get current projected state")
    print()

    run_server()
