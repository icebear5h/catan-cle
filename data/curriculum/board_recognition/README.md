# Catan board-recognition curriculum

This folder is the source-controlled contract for training a direct visual
parser of a standard Catan board. It defines state-complexity stages, closed
classes, dense labels, counterfactual pairing, rendering, sampling, splits, and
leakage controls.

This is **direct slot-classifier data, not generative QA**. One image is stored
with a complete symbolic label payload. At training time, choose an entity type,
attribute, and canonical slot, then read its class from that payload. The model
returns a class distribution; it does not narrate or serialize the board.

## Source layout

- `curriculum.json`: authoritative `catan_board_recognition_curriculum/v1`
  contract.
- `schemas/manifest.schema.json`: one rendered-state manifest row.
- `schemas/dense_labels.schema.json`: complete labels for one board state.

Generated data belongs under
`artifacts/generated/board_recognition/curriculum/`, never in this folder:

```text
metadata.json
manifest.jsonl
contracts/<state_id>.json
dense_labels/<state_id>.json
images/<state_id>.png
splits/{train,validation,test}.jsonl
```

## Images

Every example is an ordinary raw full-board render. The training default is
1024×1024 with `configs/sft/renderer_style.json`. Images contain no labels,
boxes, coordinate overlays, crops, phase shifts, query text, or answer state.
The engine contract remains the rendering oracle and is never painted into the
image.

## Dense labels

Every state contains exactly:

- 19 tiles: `resource`, `number`, and `robber`;
- 54 nodes: atomic `occupancy`;
- 72 edges: atomic `owner`;
- 9 ports: atomic `port_type`.

Canonical slot handles are `<Txx>`, `<Nxx>`, `<Exx_yy>`, and `<Pxx>`. The dense payload is the
single answer source for all sampled slot queries. Four-player colors are RED,
BLUE, WHITE, and ORANGE.

## Curriculum

State complexity is the primary axis:

1. `empty_setup`: no buildings or roads.
2. `initial_placements`: one target settlement, three distractor buildings,
   and four roads.
3. `sparse_midgame`: one target city, eight distractor buildings, and twelve
   roads.
4. `dense_endgame`: one target settlement, fourteen distractor buildings, and
   twenty-eight roads.

Earlier stages remain in later mixtures using the weights in `curriculum.json`.
Renderer variation is not a curriculum stage. In this first pilot, stage names
are controlled piece-density proxies, not claims that every synthetic contract
is reachable through a legal game trajectory. Production expansion should use
engine-generated or leakage-cleared replay states; one-label counterfactuals
may intentionally leave the natural setup distribution to provide causal
controls.

## Equal entity sampling

Do not flatten the 154 slots into one sampling pool: nodes and edges would
silently dominate. Training samples hierarchically:

1. choose `tile`, `node`, `edge`, or `port` uniformly;
2. choose an attribute for that entity type uniformly;
3. choose one canonical slot uniformly;
4. read its class from `dense_labels`.

Each pilot stage also contains an equal number of one-slot counterfactual groups
targeting each entity type.

## Counterfactuals and splits

A pair differs in exactly one declared dense label; every other dense label must
remain identical. Pair members share one split. Split assignment occurs at the
counterfactual-group/state level before any slot queries are sampled.

Replay-derived sources must be checked against
`data_pipeline/catan_board_bench/datasets/catan_board_bench_100/leakage/benchmark_game_ids.json`.
The build fails if that ledger is missing or changed. Synthetic rows include a
deterministic seed and explicit source kind.

## Build and validate

Build the working pilot:

```bash
uv run python scripts/build_catan_board_recognition_curriculum.py \
  --overwrite
```

Validate an existing build without changing it:

```bash
uv run python scripts/build_catan_board_recognition_curriculum.py \
  --validate-only
```

Use `--pairs-per-entity-type` to scale the number of pairs per stage and
`--image-size` only for infrastructure smoke fixtures. Training data defaults
to 1024px.

Validation enforces artifact hashes, canonical slot coverage, class
vocabularies, exactly one robber, one-label counterfactual deltas, same-split
pair grouping, raw-image policy, equal entity-type targets, and the held-out
benchmark ledger.
