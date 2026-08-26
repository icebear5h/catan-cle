"""Replay audit helpers for semantic alignment beyond resource ledgers."""

from cle.replay.runtime.access import get_game_engine
from copy import deepcopy

from game_engine.models.actions import generate_playable_actions


SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2}


def ensure_replay_audit_state(state):
    """Initialize replay audit fields on older ServerState instances."""
    if not hasattr(state, "replay_semantic_issues"):
        state.replay_semantic_issues = []
    if not hasattr(state, "replay_final_state_synced"):
        state.replay_final_state_synced = False
    if not hasattr(state, "replay_pending_dev_card"):
        state.replay_pending_dev_card = None


def record_replay_issue(
    state,
    *,
    kind,
    action_hint=None,
    message,
    severity="error",
    details=None,
):
    """Record a replay semantic issue without necessarily stopping playback."""
    ensure_replay_audit_state(state)
    action_hint = action_hint or {}
    issue = {
        "step": state.replay_index,
        "raw_event_index": action_hint.get("index"),
        "action_type": action_hint.get("type"),
        "kind": kind,
        "severity": severity,
        "message": message,
    }
    if details:
        issue["details"] = details
    state.replay_semantic_issues.append(issue)
    return issue


def replay_issues_since(state, start_index, min_severity="error"):
    ensure_replay_audit_state(state)
    min_rank = SEVERITY_RANK[min_severity]
    return [
        issue
        for issue in state.replay_semantic_issues[start_index:]
        if SEVERITY_RANK.get(issue.get("severity", "error"), 2) >= min_rank
    ]


def colonist_victory_points(victory_points):
    """Convert Colonist victory-point categories to public and actual totals.

    Colonist stores a point breakdown by category:
    0 settlement points, 1 city count, 2 VP-card count, 3 largest army,
    4 longest road. City, largest army, and longest road categories need
    weighting to become actual Catan points.
    """
    victory_points = victory_points or {}

    def _count(key):
        value = victory_points.get(key, victory_points.get(str(key), 0))
        return int(value or 0)

    settlements = _count(0)
    cities = _count(1)
    vp_cards = _count(2)
    largest_army = _count(3)
    longest_road = _count(4)

    public = settlements + (2 * cities) + (2 * largest_army) + (2 * longest_road)
    actual = public + vp_cards
    return {
        "public": public,
        "actual": actual,
        "settlements": settlements,
        "cities": cities,
        "vp_cards": vp_cards,
        "largest_army": largest_army,
        "longest_road": longest_road,
    }


def _merged_final_mechanic_state(replay_data, state_key):
    initial_state = replay_data.get("initial_state") or {}
    merged = deepcopy(initial_state.get(state_key, {}) or {})
    for event in replay_data.get("events", []):
        update = event.get("stateChange", {}).get(state_key)
        if not isinstance(update, dict):
            continue
        for player_id, player_update in update.items():
            if player_update is None:
                merged.pop(str(player_id), None)
                continue
            current = merged.get(str(player_id), {})
            if not isinstance(current, dict):
                current = {}
            current.update(player_update)
            merged[str(player_id)] = current
    return merged


def expected_final_state_by_engine_index(replay_data):
    """Build final Colonist scoreboard/mechanic expectations by engine index."""
    end_game_state = replay_data.get("end_game_state") or {}
    players = end_game_state.get("players") or {}
    if not players:
        return {}

    colonist_to_engine = replay_data.get("colonist_color_to_engine_idx", {})
    largest_army = _merged_final_mechanic_state(replay_data, "mechanicLargestArmyState")
    longest_road = _merged_final_mechanic_state(replay_data, "mechanicLongestRoadState")

    expected = {}
    for colonist_id, player_state in players.items():
        engine_idx = colonist_to_engine.get(str(colonist_id))
        if engine_idx is None:
            continue

        vp = colonist_victory_points(player_state.get("victoryPoints", {}))
        army_state = largest_army.get(str(colonist_id), {}) or {}
        road_state = longest_road.get(str(colonist_id), {}) or {}
        expected[engine_idx] = {
            "colonist_id": str(colonist_id),
            "public_vp": vp["public"],
            "actual_vp": vp["actual"],
            "has_largest_army": army_state.get("hasLargestArmy") is True,
            "has_longest_road": road_state.get("hasLongestRoad") is True,
            "longest_road_length": int(road_state.get("longestRoad") or 0),
            "winning_player": bool(player_state.get("winningPlayer")),
            "vp_breakdown": vp,
        }
    return expected


def validate_final_replay_state(state):
    """Return final-state mismatches after replay completion."""
    ensure_replay_audit_state(state)
    replay_data = state.replay_data or {}
    game = get_game_engine(state)
    if not game:
        return []

    expected_by_idx = expected_final_state_by_engine_index(replay_data)
    mismatches = []
    for engine_idx, expected in expected_by_idx.items():
        if engine_idx >= len(game.state.colors):
            mismatches.append({
                "kind": "final_player_mapping_mismatch",
                "engine_idx": engine_idx,
                "expected": expected,
                "actual": None,
            })
            continue

        key = f"P{engine_idx}"
        actual = {
            "public_vp": game.state.player_state.get(f"{key}_VICTORY_POINTS", 0),
            "actual_vp": game.state.player_state.get(f"{key}_ACTUAL_VICTORY_POINTS", 0),
            "has_largest_army": bool(game.state.player_state.get(f"{key}_HAS_ARMY", False)),
            "has_longest_road": bool(game.state.player_state.get(f"{key}_HAS_ROAD", False)),
            "longest_road_length": int(
                game.state.player_state.get(f"{key}_LONGEST_ROAD_LENGTH", 0)
            ),
        }

        for field in (
            "public_vp",
            "actual_vp",
            "has_largest_army",
            "has_longest_road",
            "longest_road_length",
        ):
            if actual[field] != expected[field]:
                mismatches.append({
                    "kind": f"final_{field}_mismatch",
                    "engine_idx": engine_idx,
                    "colonist_id": expected["colonist_id"],
                    "field": field,
                    "expected": expected[field],
                    "actual": actual[field],
                    "expected_state": expected,
                    "actual_state": actual,
                })

    return mismatches


def sync_final_replay_state(state):
    """Force final score/award fields to Colonist's end-game state.

    This is replay-mode bookkeeping, not proof that the engine organically
    derived the awards. Any changed field is recorded as a warning.
    """
    ensure_replay_audit_state(state)
    if state.replay_final_state_synced:
        return []

    replay_data = state.replay_data or {}
    game = get_game_engine(state)
    if not game or not replay_data.get("end_game_state"):
        state.replay_final_state_synced = True
        return []

    before = validate_final_replay_state(state)
    for mismatch in before:
        if "field" not in mismatch:
            record_replay_issue(
                state,
                kind="final_state_mapping_mismatch",
                message=f"Could not sync final state for engine player {mismatch['engine_idx']}",
                severity="error",
                details=mismatch,
            )
            continue
        record_replay_issue(
            state,
            kind="forced_final_state_sync",
            message=(
                f"Forced final {mismatch['field']} for engine player "
                f"{mismatch['engine_idx']} from {mismatch['actual']} to {mismatch['expected']}"
            ),
            severity="warning",
            details=mismatch,
        )

    expected_by_idx = expected_final_state_by_engine_index(replay_data)
    for engine_idx, expected in expected_by_idx.items():
        if engine_idx >= len(game.state.colors):
            continue
        key = f"P{engine_idx}"
        game.state.player_state[f"{key}_VICTORY_POINTS"] = expected["public_vp"]
        game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] = expected["actual_vp"]
        game.state.player_state[f"{key}_HAS_ARMY"] = expected["has_largest_army"]
        game.state.player_state[f"{key}_HAS_ROAD"] = expected["has_longest_road"]
        game.state.player_state[f"{key}_LONGEST_ROAD_LENGTH"] = expected[
            "longest_road_length"
        ]

    game.state.playable_actions = generate_playable_actions(game.state)
    state.replay_final_state_synced = True
    return before
