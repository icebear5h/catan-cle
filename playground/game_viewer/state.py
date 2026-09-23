"""ServerState singleton replacing all global variables."""

import json
import time
from pathlib import Path
from threading import Lock, RLock

from cle.game_engine.game import GameEngine
from cle.game_engine.public_board import JsonValue
from cle.replay.colonist.types import TradeLedgerRecord
from cle.replay.contracts import ReplayArchive, ReplayIssue
from cle.replay.runtime.checkpoint import ReplayStepCheckpoint
from cle.replay.runtime.revision import bump_replay_revision as bump_replay_revision
from cle.sandbox.catan import CatanSandbox
from cle.sandbox.factory import LiveSandboxConfig
from cle.sandbox.replay import ReplaySandbox
from cle.traces import SQLiteLiveTraceStore


class ServerState:
    """Central server state - replaces all global variable declarations."""

    def __init__(self) -> None:
        # Game state
        self.current_sandbox: CatanSandbox | ReplaySandbox | None = None
        self.current_game: GameEngine | None = None
        self.game_running = False
        self.game_log: list[dict[str, object]] = []
        self.step_processing = False
        self.live_trace_store: SQLiteLiveTraceStore | None = None
        self.live_trace_game_id: str | None = None
        self.live_inference: dict[str, JsonValue] | None = None
        # Runtime selection outlives game reset/load; never comes from a trace.
        self.active_live_config: LiveSandboxConfig | None = None
        self.last_live_step_error: dict[str, JsonValue] | None = None

        # Replay mode state
        self.replay_data: ReplayArchive | None = None
        self.replay_index = 0
        self.replay_mode = False
        self.replay_actions_per_step: list[int] = []
        self.first_divergence_step: dict[str, int] = {}
        self.replay_semantic_issues: list[ReplayIssue] = []
        self.replay_final_state_synced = False
        self.replay_pending_dev_card: dict[str, object] | None = None
        self.replay_step_checkpoints: list[ReplayStepCheckpoint] = []
        self.replay_trade_ledger: dict[object, TradeLedgerRecord] = {}
        self.replay_revision = 0
        self.replay_mutation_lock = RLock()
        self.replay_llm_lock = Lock()

        # Mapping files
        self._data_dir = Path(__file__).parent
        self.CORNER_MAP_FILE = self._data_dir / "corner_to_node_map.json"
        self.EDGE_MAP_FILE = self._data_dir / "edge_to_edge_map.json"
        self.corner_to_node_map: dict[str, int] = {}
        self.edge_to_edge_map: dict[str, list[int]] = {}

        # Load maps on init
        self.load_corner_map()
        self.load_edge_map()

    def load_corner_map(self) -> dict[str, int]:
        if self.CORNER_MAP_FILE.exists():
            with open(self.CORNER_MAP_FILE) as f:
                self.corner_to_node_map = json.load(f)
        return self.corner_to_node_map

    def load_edge_map(self) -> dict[str, list[int]]:
        if self.EDGE_MAP_FILE.exists():
            with open(self.EDGE_MAP_FILE) as f:
                self.edge_to_edge_map = json.load(f)
        return self.edge_to_edge_map

    def save_corner_map(self) -> None:
        with open(self.CORNER_MAP_FILE, 'w') as f:
            json.dump(self.corner_to_node_map, f, indent=2)

    def reset(self) -> None:
        """Reset all game and replay state."""
        self.replay_revision += 1
        self.current_sandbox = None
        self.current_game = None
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

    def log_event(
        self,
        event_type: str,
        message: str,
        color: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
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
