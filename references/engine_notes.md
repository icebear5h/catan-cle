# Engine Notes for Board QA

## Useful Existing Structure

The engine is already suitable for generating board-QA labels.

- `cle.game_engine.game.GameEngine` owns the game loop and validates actions through `is_valid_action`.
- `cle.game_engine.state.GameState` owns mutable game state and exposes `state.playable_actions`.
- `cle.game_engine.models.actions.generate_playable_actions` centralizes legal action generation.
- `cle.game_engine.models.map.CatanMap` builds fixed topology for land tiles, water, ports, nodes, and edges.
- `cle.game_engine.models.board.Board` owns public board occupancy: buildings, roads, robber coordinate, buildable nodes, buildable edges, port access, and longest-road cache.
- `cle.game_engine.json.GameEncoder` serializes public board and legal-action state.
- `playground.board_renderer.CatanBoardRenderer` can render engine states to images for VLM data.

This means the first benchmark can be generated from engine state without asking a model or a human to label data.

## Board Atlas

The standard 4-player map already has fixed topology:

- 19 land tiles
- 54 land nodes
- 72 land edges
- 9 port slots

The following are stable for atlas questions:

- tile ID to cube coordinate
- tile ID to node IDs
- tile ID to edge node-pairs
- node ID to adjacent tile IDs
- edge node-pair to endpoint nodes
- port ID to port nodes

The following change per game:

- tile resource
- tile number
- port resource
- robber coordinate
- road ownership
- building ownership/type
- player private resources and dev cards

## Label Sources

Recommended label source by benchmark family:

- robber: `state.board.robber_coordinate` plus `state.board.map.tiles_by_id`
- tile resource/number: `state.board.map.tiles_by_id`
- node owner/type: `state.board.buildings`
- edge owner: `state.board.roads`, canonicalized as sorted node-pairs
- port type/nodes: `state.board.map.ports_by_id`
- node adjacency: `state.board.map.adjacent_tiles`
- legal actions: `state.playable_actions`
- legal settlement nodes: `state.board.buildable_node_ids(color, initial_build_phase=...)`
- legal road edges: `state.board.buildable_edges(color)`
- visible longest road length: `get_longest_road_length(state, color)`
- official Longest Road holder: `get_longest_road_color(state)`

## Engine Observations

The implementation is fairly good for this project because it keeps board topology and legality centralized. A few details matter before relying on generated trajectories:

1. `apply_action()` appends the original action object after dispatch.

   Some handlers create a more fully specified local action, such as dice values for `ROLL`, card identity for `BUY_DEVELOPMENT_CARD`, discarded cards for `DISCARD`, and stolen resource for `STEAL`. Since the local variable is not returned to `apply_action()`, the action log may keep the original underspecified action.

   This is important for replay/export fidelity, but not a blocker for static board-QA labels if labels are read directly from state after applying actions.

2. Roads are stored bidirectionally in `Board.roads`.

   Dataset labels should canonicalize every edge as `tuple(sorted(edge))` to avoid duplicate answer variants.

3. Port access is cached by resource, not by port slot.

   For VLM QA, use `state.board.map.ports_by_id` for port-slot questions and `state.board.get_player_port_resources(color)` for player-access questions.

4. Longest road has two useful labels.

   - `Board.continuous_roads_by_player(color)` / graph traversal supports visible-chain questions.
   - `get_longest_road_color(state)` supports official award-holder questions.

   Keep these labels separate because official Longest Road uses Catan award rules and thresholds.

5. Player-to-player trading has async state.

   For early board-understanding QA, skip trade-specific questions. Add them only after perception and graph benchmarks are stable.

## First Code Target

Create a board-QA dataset builder that:

1. Samples engine games or random public board states.
2. Renders each state with `CatanBoardRenderer`.
3. Emits deterministic QA rows from engine labels.
4. Saves a manifest JSONL with image path, question, exact answer, answer type, tier, and source label.

Do not start with expert strategy SFT. Start with atlas and board-state QA so we can prove the VLM is carrying usable board facts.

