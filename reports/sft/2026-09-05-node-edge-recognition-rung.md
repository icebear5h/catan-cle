# Node and edge recognition rung (gaussian ladder, stage 3)

Status, 2026-09-05 02:30 PDT: dataset built, exported and verified
(`node_edge_readout_v1`); evaluator, panel and scorecard updated; pairs_v2
baseline scoring the new validation sets; training launch pending the
user's go. Working checklist: `tasks/todo.md`. Numbers land in
`reports/sft/2026-09-03-qwen38-single-piece-v2.json` under
`gaussian_ladder_20260904.stage3`.

## Context

The board reader has to reconstruct real replay boards exactly. The ladder
so far, all from the base model with `vocab_gaussian` token rows:

| Rung | Bundle | Result |
|---|---|---|
| 1, entity-shaped markers, 256 steps | `gauss-s1-markers-entity` final | markers 99.1% on its own set; 0 twins |
| 2, terrain readout, stopped at step 405 | `gauss-s2-terrain` checkpoint-384 | tile number, resource, port 100% on 5 unseen layouts; readouts 64 of 64; 0 twins |

Rung 2 forgot the marker heads (marker-to-token 77% to 2%). Decision: the
markers were scaffolding, forget them, do not gate on them. Every earlier
panel set is a side-effect check from here on, not a gate.

Rung 3 was going to be adjacent-pair data. Rejected: pairs are synthetic
two-piece boards, one of the three pair kinds is illegal under the distance
rule, and the rung's job is node and edge recognition on real boards. The
old v1 rows are out too: this rung has its own exporter and validation
set, modelled on the terrain rung ("what is at each tile and port?"), and
does not reuse the production curriculum mix or `evals/validation_v1.jsonl`
as its gate.

Carried over: ladder only, one H200 at a time, no hand-edited adapters,
loss is never a gate signal, gates are failure-mode scorecards.

## The question the rung answers

"What is at each node and edge?" on a real board screenshot. Token as
query only. No inverse rows ("where is the red road?"): they interfere with
the forward heads and the mapping is not bijective on a real board.

| Head | Prompt | Answer |
|---|---|---|
| node occupancy | `<N17> building?` | `red settlement`, `red city`, or `empty` |
| edge owner | `<E17_18> road?` | `blue road` or `empty` |
| node readout | see below | all 54 nodes in token order |
| edge readout | see below | all 72 edges in token order |

Prompt and answer strings for the two heads are the production ones from
`data_pipeline/board_recognition/semantics.py`: colour words lowercased
with underscores as spaces (`mystic blue city`), piece word last. That keeps
the rows comparable with the pairs_v2 bundle, `analyze_occupancy_misses.py`
and the full-board panel set. Nothing else from the old rows is inherited.

## Source states

`artifacts/generated/board_recognition/replay_v1/manifest.jsonl`. Real
boards, 1024 by 1024, contracts beside the images.

| Split | States | Layouts | Pieces per state (min / median / max) | empty / setup / sparse / dense |
|---|---|---|---|---|
| train | 1,024 | 77 (43 replay, 34 engine rollouts) | 0 / 33 / 74 | 43 / 216 / 227 / 538 |
| validation | 64 | 5 replay | 0 / 18 / 42 | 5 / 26 / 18 / 15 |
| test | 64 | 5 replay | 0 / 20 / 46 | 5 / 25 / 19 / 15 |
| color_diagnostic | 64 | 16 engine | 0 / 14 / 71 | 16 / 20 / 8 / 20 |

Splits are by whole game, so every state of a layout sits in one split; the
exporter re-checks that pairwise.

### Colour validation

Colour dropout is a known failure (bronze recall 0 of 8 on the v3 bundle, 7
of 8 after pairs_v2) and the replay splits cannot measure it alone. States
in which each colour is visible:

| Colour | train | validation | test | color_diagnostic |
|---|---|---|---|---|
| red | 751 | 56 | 56 | 15 |
| blue | 728 | 43 | 47 | 17 |
| black | 584 | 45 | 43 | 18 |
| orange | 541 | 41 | 35 | 20 |
| green | 373 | 11 | 44 | 16 |
| pink | 172 | 0 | 0 | 16 |
| bronze | 145 | 11 | 0 | 14 |
| mystic blue | 139 | 9 | 0 | 13 |
| silver | 134 | 0 | 0 | 17 |
| gold | 113 | 0 | 0 | 18 |
| white | 82 | 0 | 0 | 19 |

Validation covers 7 colours, test only the 5 base colours; gold, pink,
silver and white never appear in either. `color_diagnostic` is 64 states
over 16 engine layouts built so all 11 colours appear in 13 to 20 states,
same renderer as the engine layouts in train. It is exported eval-only with
the full-coverage rows. The gate reads from it: per-colour recall on
occupied nodes and edges (the minimum is the colour-dropout number, bar set
by pairs_v2 on the same rows), a colour confusion table (which colour was
named instead), and readout colour accuracy per colour. Train is skewed
(red in 751 states, white in 82); occupied tokens are sampled uniformly
under the cap, so rare colours still get a few hundred short rows plus full
exposure in every readout. A rare-colour sampling weight is parked until
the diagnostic asks for it.

## Every row

### 1. Node occupancy, occupied (`node.occupancy`, polarity `positive`)

- Prompt `<image>\n<N17> building?`; answer `red settlement` or `red city`
  from the contract node's colour and building.
- Train: up to 4 occupied nodes per image by stable hash. Eval splits:
  every occupied node.
- Metadata: `target_token`, `queried_token`, `entity_type: node`, `piece:
  SETTLEMENT | CITY`, `color: RED` (raw contract colour), `state_id`,
  `layout_id`, `split`, `density_bin`, `piece_count`, `color_heldout: false`.
- Row id `<state_id>_N17_node_occupancy`.

### 2. Node occupancy, empty (`node.occupancy`, polarity `hard_negative`)

- Prompt `<image>\n<N18> building?`; answer `empty`.
- Train: up to 4 empty nodes per image, hardest first, quota adjacent 2 /
  cross-type 1 / far 1, leftovers filled from the ranked pool in kind order:
  1. `adjacent`: one hop from a building (the N17 versus N18 confusion).
  2. `cross_type`: the empty end of a road (the road-end hallucination).
  3. `hop2`, `hop3`: two or three hops from a building.
  4. `far`: beyond three hops, or any node on an empty board.
- Eval splits: every empty node, a full classification of all 54 nodes.
- Metadata adds `negative_kind` and `negative_distance` (`1 | 2 | 3 |
  "far"`), `piece: EMPTY`, `color: none`.

### 3. Edge owner, road (`edge.owner`, polarity `positive`)

- Prompt `<image>\n<E17_18> road?`; answer `blue road`.
- Train: up to 4 roads per image; eval splits: every road.
- Metadata as row 1 with `entity_type: edge`, `piece: ROAD`; row id
  `<state_id>_E17_18_edge_owner`.

### 4. Edge owner, empty (`edge.owner`, polarity `hard_negative`)

- Prompt `<image>\n<E18_40> road?`; answer `empty`.
- Train ranking: `adjacent` (shares a node with a road: continuation and the
  slanted-versus-vertical turn), `cross_type` (touches a building), `hop2`,
  `hop3`, `far`; same quota.
- Eval splits: every empty edge, all 72 edges covered.

### 5. Node readout (`node.readout`, task type `node_readout`)

Prompt:

    List every node <N00> to <N53> as "token building", where building is "empty", "colour settlement" or "colour city", in token order, separated by "; ".

Answer, 54 items, empties explicit:

    <N00> empty; <N01> red settlement; <N02> empty; ...; <N53> blue city

One per image in every split; empty boards stay in (all items `empty`).
Every location is listed because an occupied-only list has no stopping
criterion, and the sequence skip was the one failure the terrain readouts
still showed. It also supervises every empty node on every image, which is
why the short rows spend their budget on hard cases. About 230 to 300
generated tokens. Metadata: `entity_type: board`, `piece: BOARD`, `color:
none`, `target_token: <N00>`, `item_count: 54`, `occupied_count`.

### 6. Edge readout (`edge.readout`, task type `edge_readout`)

Prompt:

    List every edge <E00_01> to <E52_53> as "token road", where road is "empty" or "colour road", in token order, separated by "; ".

Answer, 72 items in atlas order:

    <E00_01> empty; <E00_20> red road; ...; <E52_53> empty

About 300 to 400 generated tokens on the densest boards; the eval launch
uses `--long-max-new-tokens 640`. `target_token: <E00_01>`, `item_count: 72`.

### Not in the data

- No inverse, localization or marker rows.
- No tile, port or robber rows: the rung is scoped to nodes and edges. The
  terrain heads are watched as a side effect; if they collapse the way the
  markers did, that is the measured cost of a rung without rehearsal and it
  informs the production-rung mix, not this rung.
- No synthetic boards; train already holds 34 engine layouts.

### Row counts, as exported

| Split | Images | Short rows per image | Readouts per image | Rows |
|---|---|---|---|---|
| train | 1,024 | up to 16 (4 occupied + 4 empty per family) | 2 | 17,708 |
| validation | 64 | 126 | 2 | 8,192 |
| test | 64 | 126 | 2 | 8,192 |
| color_diagnostic | 64 | 126 | 2 | 8,192 |

886 of 1,024 train images carry the full 18 rows. Train empties: 6,123
adjacent, 830 cross-type, 1,239 far. Readout answers run 700 to 1,525
characters; the trainer's row guard is now 2,048. One epoch at batch 16 x 2
is about 553 steps, so 512 steps sees the data about once.

## Implementation, done

- `data_pipeline/board_recognition/node_edge_readout.py` (exporter, CLI
  `python -m data_pipeline.board_recognition.node_edge_readout <replay_v1>
  --rows-per-family 4 --readouts-per-family 1`), tests in
  `tests/test_board_recognition_node_edge_readout.py` (6).
- `sft/scripts/eval_qwen_vl_adapter.py`: `node_readout` and `edge_readout`
  take the long budget; metadata keeps `state_id`, `layout_id`,
  `piece_count`, `item_count`, `occupied_count`.
- `sft/scripts/eval_regression_panel.py`: sets `node-edge` and
  `node-edge-colors`.
- `sft/scripts/failure_scorecard.py`: `readouts` mode (exact and item rates,
  dropped tokens, values shifted onto the previous token) and
  `sequence_skips`.
- `sft/scripts/train_trl_catan_vision.py`: `MAX_ANSWER_CHARACTERS = 2048`.

## Launch

- Baseline first: pairs_v2 final on `node-edge` and `node-edge-colors`,
  original variant, label `pairs-v2-final-node-edge` (running).
- Rung: `sft/modal_catan_vision_sft.py` from
  `/runs/catan-vision-sft/catan-qwen38-gauss-s2-terrain-20260904/398f0a023ec9/checkpoints/checkpoint-384`,
  train and validation from `node_edge_readout_v1/stage1`, batch 16 x 2,
  512 steps, eval and save every 128, `--no-require-curriculum`, run name
  `catan-qwen38-gauss-s3-nodes-edges-20260905`, output
  `/runs/catan-vision-sft/catan-qwen38-gauss-s3-nodes-edges-20260905/e64dcbc2cc4c`.
  About 5 hours. Dry run clean.
- Checkpoint evals at 128, 256, 384 on the full validation set, original
  variant, `--long-max-new-tokens 640 --long-batch-size 8`, under
  `qwen-series-eval/gauss-ladder/s3-nodes-edges-ck<N>`.
- Final: panel with the two new sets, scorecard, row inspector.

## Gate

Node and edge heads only, on the 5 unseen validation layouts, every node
and edge of every image, plus the colour-diagnostic split:

| Mode | Measure | Bar |
|---|---|---|
| accuracy | node occupancy and edge owner, per density bin | above pairs_v2 on the same rows in every bin |
| blindness | occupied answered `empty` | below pairs_v2, by piece type |
| neighbor confusion | empty answered with a piece at hop 1 or 2 | below pairs_v2 |
| cross-type | road end answered as a building, building corner as a road | below pairs_v2 |
| far false positive | empty answered with a piece from elsewhere or absent | below pairs_v2 |
| colour dropout | min per-colour recall on `color_diagnostic` | no colour below pairs_v2's minimum |
| head flip | building phrase on an edge query, road phrase on a node query | zero |
| glitch | malformed or doubled answers | zero |
| readouts | exact and item rates, sequence skips | reported; no baseline exists |
| rows | twins, anti-aligned rows | zero |

Side effects, reported not gated, as deltas against the terrain
checkpoint-384 panel (`reports/sft/scorecards/gauss-s2-terrain-ck384.json`):
markers 0.369, probes 0.000, single-v2 0.001, single-v3 0.304, single-v7
0.264, pairs 0.211, pairs-control 0.209, terrain 1.000, full-board tile and
port heads 1.000.

A rung that beats pairs_v2 on every gated mode is the bundle for the
production rungs; a loss on any mode names the fix before anything else
launches (blindness or neighbor confusion: caps and quota; colour: colour
balance in the sample; head flip or rows: init or row anchoring).

## What runs when

| When | What |
|---|---|
| now | baseline eval on the H200; launch waits on the user |
| every 128 steps | trainer eval (crash and divergence only); side eval on the full validation set; scorecard delta against the previous checkpoint |
| final | panel with the new sets; scorecard against the pairs_v2 baseline; row inspector; gate table in `tasks/todo.md`; results into the report JSON |
