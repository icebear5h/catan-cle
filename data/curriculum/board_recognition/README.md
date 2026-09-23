# Catan board-recognition data

This directory contains the source-controlled contracts for direct visual
classification of canonical Catan board slots. The active corpus is
`replay_v1`; the older synthetic stage curriculum remains a legacy pilot and
rendering oracle.

The task is deliberately atomic. One ordinary full-board image is paired with
one location-token query and one short answer. It does not ask the model to
narrate or serialize the board. The immutable historical `qwen_sft` projection
uses opaque query and answer tokens; the semantic ms-swift projection uses only
location tokens plus ordinary language.

## Active replay_v1 contract

Source-controlled inputs:

- `replay_v1.json`: corpus sizes, source policy, classes, and query policy.
- `replay_sources_v1.json`: immutable audit of accepted raw replay payloads.
- `schemas/manifest_v2.schema.json`: one rendered board state.
- `schemas/dense_labels_v2.schema.json`: complete labels for one board.
- `schemas/query_plan_v1.schema.json`: eight scheduled queries for one image.
- `schemas/qwen_sft_v2.schema.json`: one compact upstream Qwen row.
- `schemas/qwen_audit_v2.schema.json`: provenance for one atomic row.

Generated data is under:

```text
artifacts/generated/board_recognition/replay_v1/
├── metadata.json
├── manifest.jsonl
├── contracts/<state_id>.json
├── dense_labels/<state_id>.json
├── images/<state_id>.png
├── splits/{train,validation,test}.jsonl
├── diagnostics/color_diagnostic.jsonl
├── qwen_sft/
│   ├── metadata.json
│   ├── trainable_tokens.json
│   ├── {train,validation,test,color_diagnostic}.jsonl
│   ├── query_plans/<split>.jsonl
│   └── audit/<split>.jsonl
├── ms_swift_semantic_v1/
    ├── metadata.json
    ├── semantic_contract.json
    ├── trainable_tokens.json
    ├── {train,validation,test,color_diagnostic}.jsonl
    ├── audit/<split>.jsonl
    └── curriculum/
        ├── manifest.json
        ├── train_three_stage.jsonl
        ├── audit_three_stage.jsonl
        └── index_three_stage.jsonl
└── ms_swift_bidirectional_v1/
    ├── metadata.json
    ├── inverse_grounding_contract.json
    ├── trainable_tokens.json
    ├── {train,validation,test,color_diagnostic}.jsonl
    ├── audit/<split>.jsonl
    ├── mixed/<split>.jsonl
    └── mixed_index/<split>.jsonl
```

## Sources and splits

The locked replay audit accepts 53 raw non-benchmark Colonist games and excludes
13 CatanBoardBench games. Complete replay games are assigned before states are
sampled: 43 games to train, five to validation, and five to test. A replay
contributes at most 16 board-changing states.

The final corpus contains 1,216 unique 1024×1024 images:

```text
split              replay   engine   total
train                 688      336    1024
validation             64        0      64
test                   64        0      64
color_diagnostic        0       64      64
```

Engine supplementation uses only actions advertised by
`GameEngine.playable_actions`. Action candidates are canonically ordered before
the policy RNG selects one, so Python hash iteration cannot alter a seeded
trajectory. Training and diagnostic trajectories use disjoint seeds and
balanced palettes over all 11 engine colors.

No game ID, static map hash, complete visible-board-fact hash, or rendered image
may cross splits. CatanBoardBench IDs are rejected before reconstruction.

## Dense labels and images

Every state contains exactly:

- 19 tiles: `resource`, `number`, and `robber`;
- 54 nodes: atomic `occupancy`;
- 72 edges: atomic `owner`;
- 9 ports: atomic `port_type`.

Canonical handles are `<Txx>`, `<Nxx>`, `<Exx_yy>`, and `<Pxx>`. Images are
ordinary 1024px full-board renders using `configs/sft/renderer_style.json`.
They contain no boxes, coordinates, query text, crops, or answer overlays.
Validation rerenders every image from its public board contract and requires
byte equality.

## One-epoch query schedule

Each image contributes eight rows per exported epoch: two tile, two node, two
edge, and two port queries. Scheduling is deterministic at the complete-split
level, without replacement inside an image.

For the 1,024-image training split this yields 8,192 rows per epoch. The
scheduler enforces:

- every slot for every recognition head is queried;
- node and edge targets are exactly 50% positive and 50% empty;
- all 60 answer classes receive train targets;
- the 22 positive node classes receive 46–47 targets each;
- the 11 positive edge classes receive 93–94 targets each;
- no image repeats the same `(head, slot)` query.

The fixed validation, test, and color-diagnostic projections contain 512 rows
each. The diagnostic selector first proves that every settlement, city, and
road class can be assigned with a capacity of two node or edge queries per
image.

An atomic row looks like:

```json
{
  "image": "engine_train_123_a000456.png",
  "conversations": [
    {"from": "human", "value": "<image>\n<N17><Q_NODE_OCCUPANCY>"},
    {"from": "gpt", "value": "<A_NODE_OCCUPANCY_BLUE_CITY>"}
  ]
}
```

`human` and `gpt` are the legacy LLaVA transport labels required by the pinned
community 2U1 trainer. Its loader maps them to model roles `user` and
`assistant` before chat templating. They are not provider roles and the 2U1
repository is not an official Qwen trainer.

## Semantic recognition contract

The new ms-swift projections add exactly 154 regular location tokens and train
selective rows on both the input embedding and output head. All other base
embedding and output rows remain frozen.
Queries and answers use the following controlled ordinary-language contract:

```text
<T07> resource?      -> sheep
<T07> number?        -> 8
<T07> robber here?   -> yes
<N17> building?      -> blue city
<E17_39> road?       -> red road
<P03> port?          -> ore port
```

Legal answers are closed per head:

- resource: `desert`, `wood`, `brick`, `sheep`, `wheat`, `ore`;
- number: `none`, `2`, `3`, `4`, `5`, `6`, `8`, `9`, `10`, `11`, `12`;
- robber: `no`, `yes`;
- building: `empty` or one of the 11 engine colors followed by `settlement` or
  `city`;
- road: `empty` or one of the 11 engine colors followed by `road`;
- port: `3:1 port` or `wood`, `brick`, `sheep`, `wheat`, or `ore` followed by
  `port`.

Engine `MYSTIC_BLUE` is rendered as `mystic blue`; all other enum names are
lowercased. Evaluation scores only the legal phrases for the row's head and
also reports unconstrained exact generation separately. No `<Q_...>` or
`<A_...>` token is added by this projection.

The historical `qwen_sft` inventory remains exactly 220 regular added tokens:
154 atlas slots, 6 opaque queries, and 60 opaque answers. It is preserved as
prior experimental evidence and must not be overwritten. Both projections
consume an explicit token-inventory file; unrelated global Catan tokens are not
admitted.

## Inverse grounding and mixed projection

The companion bidirectional projection teaches the reverse mapping from visible,
board-local descriptions to atlas tokens:

```text
image + Where is O10/WH5/S6?                        -> <N17>
image + Where is ore 10/wheat 5/sheep 6?           -> <N17>
image + Where is ore/wheat/sheep?                   -> <N17>  # only if unique
image + Where is the blue city at O10/WH5/S6?       -> <N17>
image + Where is the red road between ... and ...?  -> <E17_39>
```

`WO` means wood and `WH` means wheat. Compact, full-word, resource-only,
and resource-plus-number aliases are all derived from the same contract. An
alias is used only when it uniquely identifies one node on that board;
otherwise the query uses a unique tile-anchored direction.

Each image contributes one inverse tile, node, edge, and port row. Alternating
complete node and edge atlas cycles prefer occupied targets so colors,
settlements, cities, and roads occur in the inverse direction as well. The generated
`mixed/train.jsonl` interleaves eight existing forward rows with four inverse
rows per image, preserving a controlled 2:1 forward/inverse ratio. Use the
standalone inverse files for exact localization evaluation and the mixed file
for SFT.

## Optional density curriculum

The semantic projection keeps `train.jsonl` as the uniformly mixed control and
also emits an opt-in sequential curriculum containing every train row exactly
once:

```text
early   0–16 pieces   empty + setup   2,072 rows
mid    17–31 pieces   sparse          1,816 rows
late   32+ pieces     dense           4,304 rows
```

Rows are deterministically interleaved by head and class inside each stage. The
hashed index records original and curriculum positions, query identity, piece
count, and row hashes. Training must use the explicit sequential sampler when
this projection is selected; ordinary Trainer shuffling would erase the stage
boundary.

## Build and validate

Audit or rebuild the replay source lock:

```bash
uv run python -m scripts.board_recognition.audit_catan_board_recognition_sources --write
```

Build the 1,216-state corpus and rerender every image:

```bash
uv run python -m scripts.board_recognition.build_catan_board_recognition_dataset --overwrite
```

Validate without mutation:

```bash
uv run python -m scripts.board_recognition.build_catan_board_recognition_dataset --validate-only
```

Export and validate the eight-query projection:

```bash
uv run python -m scripts.board_recognition.export_catan_board_recognition_sft \
  --queries-per-state 8 \
  --overwrite

uv run python -m scripts.board_recognition.export_catan_board_recognition_sft \
  --validate-only

uv run python -m scripts.board_recognition.export_catan_board_recognition_ms_swift --overwrite
uv run python -m scripts.board_recognition.export_catan_board_recognition_ms_swift --validate-only
uv run python -m scripts.board_recognition.export_catan_inverse_grounding_ms_swift --overwrite
uv run python -m scripts.board_recognition.export_catan_inverse_grounding_ms_swift --validate-only
uv run python -m scripts.board_recognition.build_catan_board_recognition_density_curriculum --overwrite
uv run python -m scripts.board_recognition.build_catan_board_recognition_density_curriculum --validate-only
```

The semantic exporter reads the historical epoch-zero query plans as immutable
provenance and never writes under `qwen_sft/`. Validation covers hashes,
rerenders, topology, source provenance, replay and map isolation, query
reproducibility, class/polarity/slot balance, the respective 220- and 154-token
inventories, inverse alias uniqueness, complete train atlas-target coverage,
and every Draft 2020-12 schema.

## PyTorch state batches

The state loader opens one image and returns its eight scheduled targets:

```python
from data_pipeline.board_recognition import (
    BoardRecognitionStateDataset,
    make_board_recognition_dataloader,
)

states = BoardRecognitionStateDataset(
    "artifacts/generated/board_recognition/replay_v1",
    split="train",
    queries_per_state=8,
)
loader = make_board_recognition_dataloader(states, batch_size=4)
```

`set_epoch(n)` constructs an explicit deterministic plan for epoch `n`. The
current Modal corpus exports epoch zero and admits one full epoch; a future
multi-epoch run must provide explicit epoch-specific shards instead of silently
repeating or resampling the file.

## Legacy synthetic pilot

`curriculum.json` and the schemas without a version suffix describe the legacy
640-state counterfactual stage pilot under
`artifacts/generated/board_recognition/curriculum/`. Keep it for rendering and
label regression tests. It is not the active paid-training projection and must
not be overwritten by `replay_v1`.
