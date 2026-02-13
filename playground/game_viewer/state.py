"""ServerState singleton replacing all global variables."""

import json
import time
from pathlib import Path


class ServerState:
    """Central server state - replaces all global variable declarations."""

    def __init__(self):
        # Game state
        self.current_game = None
        self.current_players = []
        self.game_running = False
        self.auto_play_running = False
        self.llm_thinking = []
        self.game_log = []
        self.llm_processing = False

        # Replay mode state
        self.replay_data = None
        self.replay_index = 0
        self.replay_mode = False
        self.replay_actions_per_step = []
        self.first_divergence_step = {}

        # Mapping files
        self._data_dir = Path(__file__).parent
        self.CORNER_MAP_FILE = self._data_dir / "corner_to_node_map.json"
        self.EDGE_MAP_FILE = self._data_dir / "edge_to_edge_map.json"
        self.corner_to_node_map = {}
        self.edge_to_edge_map = {}

        # Load maps on init
        self.load_corner_map()
        self.load_edge_map()

    def load_corner_map(self):
        if self.CORNER_MAP_FILE.exists():
            with open(self.CORNER_MAP_FILE) as f:
                self.corner_to_node_map = json.load(f)
        return self.corner_to_node_map

    def load_edge_map(self):
        if self.EDGE_MAP_FILE.exists():
            with open(self.EDGE_MAP_FILE) as f:
                self.edge_to_edge_map = json.load(f)
        return self.edge_to_edge_map

    def save_corner_map(self):
        with open(self.CORNER_MAP_FILE, 'w') as f:
            json.dump(self.corner_to_node_map, f, indent=2)

    def reset(self):
        """Reset all game and replay state."""
        self.current_game = None
        self.current_players = []
        self.game_running = False
        self.auto_play_running = False
        self.llm_thinking = []
        self.game_log = []
        self.llm_processing = False
        self.replay_data = None
        self.replay_index = 0
        self.replay_mode = False
        self.replay_actions_per_step = []
        self.first_divergence_step = {}

    def log_event(self, event_type, message, color=None, details=None):
        """Add an event to the game log."""
        self.game_log.append({
            "type": event_type,
            "timestamp": time.time(),
            "message": message,
            "color": color,
            "details": details,
        })


# Module-level singleton
server_state = ServerState()
