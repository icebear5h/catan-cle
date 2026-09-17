"""ServerState singleton replacing all global variables."""

import json
import time
from pathlib import Path
from threading import Lock, RLock

from cle.replay.runtime.revision import bump_replay_revision as bump_replay_revision


class ServerState:
    """Central server state - replaces all global variable declarations."""

    def __init__(self):
        # Game state
        self.current_sandbox = None
        self.game_running = False
        self.game_log = []
        self.step_processing = False
        self.live_trace_store = None
        self.live_trace_game_id = None
        self.live_inference = None
        # Runtime selection outlives game reset/load; never comes from a trace.
        self.active_live_config = None
        self.last_live_step_error = None

        # Replay mode state
        self.replay_data = None
        self.replay_index = 0
        self.replay_mode = False
        self.replay_actions_per_step = []
        self.first_divergence_step = {}
        self.replay_semantic_issues = []
        self.replay_final_state_synced = False
        self.replay_pending_dev_card = None
        self.replay_step_checkpoints = []
        self.replay_trade_ledger = {}
        self.replay_revision = 0
        self.replay_mutation_lock = RLock()
        self.replay_llm_lock = Lock()

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
        self.replay_revision += 1
        self.current_sandbox = None
        self.game_running = False
        self.game_log = []
        self.step_processing = False
        self.live_trace_game_id = None
        self.live_inference = None
        self.last_live_step_error = None
        self.replay_data = None
        self.replay_index = 0
        self.replay_mode = False
        self.replay_actions_per_step = []
        self.first_divergence_step = {}
        self.replay_semantic_issues = []
        self.replay_final_state_synced = False
        self.replay_pending_dev_card = None
        self.replay_step_checkpoints = []
        self.replay_trade_ledger = {}

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
