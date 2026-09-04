"""Game viewer server entry point.

Usage:
    python -m playground.game_viewer_server
"""

from playground.game_viewer.app import run_server

if __name__ == '__main__':
    print("=" * 60)
    print("Catan Game Viewer Server")
    print("=" * 60)
    print()
    print("Server starting on http://localhost:5001")
    print("UI should connect from http://localhost:5173")
    print()
    run_server()
