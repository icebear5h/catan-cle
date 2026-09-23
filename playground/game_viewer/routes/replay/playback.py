"""Loading a replay, and stepping, undoing, or jumping through it."""

from flask import Response, jsonify, request

from .blueprint import _broadcast, _get_state, replay_bp
from .loading import _load_replay_transaction

__all__ = [
    "load_replay",
    "replay_goto_divergence",
    "replay_goto_fast",
    "replay_goto_sequential",
    "replay_step",
    "replay_undo",
]


@replay_bp.route('/api/load-replay', methods=['POST'])
def load_replay() -> Response | tuple[Response, int]:
    """Load a Colonist replay for playback - runs through our engine."""
    state = _get_state()
    with state.replay_mutation_lock:
        return _load_replay_transaction(state)


@replay_bp.route('/api/replay-step', methods=['POST'])
def replay_step() -> Response | tuple[Response, int]:
    """Step through the loaded replay."""
    state = _get_state()
    sandbox = getattr(state, "current_sandbox", None)
    if sandbox is None:
        return jsonify({"error": "No replay sandbox loaded"}), 400
    result = sandbox.step(_broadcast)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


@replay_bp.route('/api/replay-undo', methods=['POST'])
def replay_undo() -> Response | tuple[Response, int]:
    """Undo the last replay step."""
    state = _get_state()
    sandbox = getattr(state, "current_sandbox", None)
    if sandbox is None:
        return jsonify({"error": "No replay sandbox loaded"}), 400
    result = sandbox.undo(_broadcast)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


@replay_bp.route('/api/replay-goto-fast', methods=['POST'])
def replay_goto_fast() -> Response | tuple[Response, int]:
    """Jump to a specific replay step using fast skip logic."""
    state = _get_state()
    data = request.get_json() or {}
    target_step = data.get("step", 0)
    sandbox = getattr(state, "current_sandbox", None)
    if sandbox is None:
        return jsonify({"error": "No replay sandbox loaded"}), 400
    result = sandbox.goto_fast(target_step, _broadcast)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


@replay_bp.route('/api/replay-goto-sequential', methods=['POST'])
def replay_goto_sequential() -> Response | tuple[Response, int]:
    """Jump to a specific replay step using sequential stepping."""
    state = _get_state()
    data = request.get_json() or {}
    target_step = data.get("step", 0)
    sandbox = getattr(state, "current_sandbox", None)
    if sandbox is None:
        return jsonify({"error": "No replay sandbox loaded"}), 400
    result = sandbox.goto_sequential(target_step, _broadcast)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


@replay_bp.route('/api/replay-goto-divergence', methods=['POST'])
def replay_goto_divergence() -> Response | tuple[Response, int]:
    """Step sequentially until divergence is detected."""
    state = _get_state()
    data = request.get_json() or {}
    max_steps = data.get("max_steps", 500)
    sandbox = getattr(state, "current_sandbox", None)
    if sandbox is None:
        return jsonify({"error": "No replay sandbox loaded"}), 400
    result = sandbox.goto_divergence(max_steps, _broadcast)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)
