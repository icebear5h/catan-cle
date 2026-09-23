"""Corner-to-node mapping and engine-nodes CRUD endpoints."""


from flask import Blueprint, Response, current_app, jsonify, request

from ..state import ServerState

mapping_bp = Blueprint('mapping', __name__)


def _get_state() -> ServerState:
    state: ServerState = current_app.config['SERVER_STATE']
    return state


@mapping_bp.route('/api/corner-map', methods=['GET'])
def get_corner_map() -> Response | tuple[Response, int]:
    """Get the current Colonist corner -> Engine node mapping."""
    state = _get_state()
    return jsonify({
        "corner_to_node": state.corner_to_node_map,
        "total_mappings": len(state.corner_to_node_map),
    })


@mapping_bp.route('/api/corner-map', methods=['POST'])
def set_corner_mapping() -> Response | tuple[Response, int]:
    """Add a Colonist corner -> Engine node mapping."""
    state = _get_state()
    data = request.json
    colonist_corner = str(data.get("colonist_corner"))
    engine_node = data.get("engine_node")

    if colonist_corner is None or engine_node is None:
        return jsonify({"error": "Missing colonist_corner or engine_node"}), 400

    state.corner_to_node_map[colonist_corner] = engine_node
    state.save_corner_map()

    return jsonify({
        "status": "ok",
        "mapping": {colonist_corner: engine_node},
        "total_mappings": len(state.corner_to_node_map),
    })


@mapping_bp.route('/api/corner-map/<colonist_corner>', methods=['DELETE'])
def delete_corner_mapping(colonist_corner: str) -> Response | tuple[Response, int]:
    """Delete a corner mapping."""
    state = _get_state()
    if colonist_corner in state.corner_to_node_map:
        del state.corner_to_node_map[colonist_corner]
        state.save_corner_map()
        return jsonify({"status": "deleted", "colonist_corner": colonist_corner})
    return jsonify({"error": "Mapping not found"}), 404


@mapping_bp.route('/api/engine-nodes', methods=['GET'])
def get_engine_nodes() -> Response | tuple[Response, int]:
    """Get all engine node IDs from the current game."""
    state = _get_state()
    with state.replay_mutation_lock:
        if not state.current_sandbox:
            return jsonify({"error": "No game loaded"}), 400

        engine = state.current_sandbox.game_engine
        node_ids = sorted(list(engine.state.board.map.land_nodes))

        nodes = {}
        for node_id in node_ids:
            building = (
                engine.state.board.get_node_building(node_id)
                if hasattr(engine.state.board, "get_node_building")
                else None
            )
            nodes[node_id] = {
                "id": node_id,
                "building": building.building_type.value if building else None,
                "color": building.color.value if building else None,
            }

    return jsonify({
        "nodes": nodes,
        "total": len(nodes),
    })
