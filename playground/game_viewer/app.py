"""Flask app + SocketIO creation, __main__ entry point."""

import sys
sys.path.insert(0, '/Users/henry/CascadeProjects/windsurf-project-4')
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from dotenv import load_dotenv
load_dotenv()

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

from .state import server_state
from .routes import register_routes


def create_app():
    """Flask application factory."""
    app = Flask(__name__)
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*")

    register_routes(app, socketio, server_state)

    return app, socketio


app, socketio = create_app()


if __name__ == '__main__':
    print("=" * 60)
    print("Catan Game Viewer Server")
    print("=" * 60)
    print()
    print("Server starting on http://localhost:5001")
    print("UI should connect from http://localhost:3000")
    print()
    print("Endpoints:")
    print("  POST /api/start-game - Start new game")
    print("  POST /api/step - Execute one step")
    print("  POST /api/auto-play - Auto-play to completion")
    print("  GET  /api/state - Get current state")
    print()

    socketio.run(app, debug=True, port=5001, use_reloader=True, allow_unsafe_werkzeug=True)
