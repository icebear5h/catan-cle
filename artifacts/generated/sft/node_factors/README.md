# Post-Atlas Node Visual Grounding Dataset

Controlled post-atlas visual QA for node grounding. This dataset assumes
Phase 0 text atlas topology has already established stable tokens like
`<N11>`, then asks the model to bind those stable node handles to visible
coordinates and transient local board facts.

This is contract-only output. QA rows have `image_path: null` until a frontend render step fills images.

Curriculum:
- Stage: `phase_1_post_atlas_visual_grounding`
- Requires: `phase_0_text_atlas_topology`
- Role: `bind_stable_node_tokens_to_transient_visual_facts`

Category groups:
- `atlas_coordinate_grounding`: locate the stable node handle.
- `full_board_atlas_bbox_map`: return all tile/node/edge/port bboxes.
- `transient_node_state_readout`: read occupancy at that handle.
- `local_tile_readout`: read neighboring tile resource/number facts.
- `local_edge_readout`: read road ownership around the node.
- `local_state_composition`: compose the local node state as JSON.

Files:
- `contracts/`: public board contracts.
- `annotations.jsonl`: frontend-aligned tile/node/edge/port bboxes for each contract.
- `manifest.jsonl`: one row per synthetic sample.
- `questions/qa.jsonl`: QA rows with answers and targets.
- `questions/questions.jsonl`: promptable questions without answers.
- `questions/answer_key.jsonl`: deterministic answer targets.
- `messages.jsonl`: chat-style rows for text-only or pending-image smoke tests.

Samples: 594
QA rows: 3564
