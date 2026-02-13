"""Game viewer server entry point.

Usage:
    python -m playground.game_viewer_server
"""

from playground.game_viewer.colonist.event_parser import parse_colonist_events_to_actions
from playground.game_viewer.app import app, socketio

if __name__ == '__main__':
    from playground.game_viewer.app import socketio, app
    print("=" * 60)
    print("Catan Game Viewer Server")
    print("=" * 60)
    print()
    print("Server starting on http://localhost:5001")
    print("UI should connect from http://localhost:3000")
    print()
    socketio.run(app, debug=True, port=5001, use_reloader=True, allow_unsafe_werkzeug=True)
