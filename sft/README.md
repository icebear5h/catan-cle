# Catan SFT Tooling

`Sft` contains Catan-specific dataset builders, conversion utilities, and the
native Hugging Face TRL/PEFT vision-SFT path. Generated datasets, diagnostics,
checkpoints, and reports live outside this installable package.

The first training target is board grounding: stable atlas tokens identify board
entities, while text or pixels supply their transient state. Strategy imitation
comes only after grounding and leakage checks pass. See
[`EXPERIMENT_DESIGN.md`](EXPERIMENT_DESIGN.md) for the curriculum and trainable
scope ladder.

## Safety contract

CatanBoardBench-100 is held out. Never train on its game IDs, images, contracts, QA,
or derived text. The required fail-closed ledger is:

```text
evals/catan_board_bench/datasets/catan_board_bench_100/leakage/benchmark_game_ids.json
```

The active board-recognition source lock is:

```text
data/curriculum/board_recognition/replay_sources_v1.json
```

It admits only validated payloads under `artifacts/raw/colonist/replays/`,
excludes the held-out ledger, and rejects staging or metadata-only candidates.
Do not create a bypass for production training.

## Layout

- `sft/scripts/train_trl_catan_vision.py`: active model, data, optimizer, and save contract.
- `sft/modal_catan_vision_sft.py`: active dry-by-default H200 launcher.
- `sft/scripts/`: dataset, conversion, renderer, and evaluation utilities.
- `sft/modal_qwen_series_*.py`, `sft/qwen_series_vision_sft.py`, and
  `sft/ms_swift_*.py`: legacy experiments; they are not part of the active run.
- `configs/sft/renderer_style.json`: accepted renderer calibration.
- `artifacts/generated/sft/`: ignored, regenerable SFT datasets.
- `artifacts/fixtures/sft/`: tracked smoke and renderer fixtures.
- `artifacts/diagnostics/sft/`: ignored local diagnostic images.
- `artifacts/runs/sft/`: ignored checkpoints/logs plus compact accepted evidence.
- `reports/sft/`: tracked run reports.

The active vision launcher requires separate `--train-jsonl` and
`--image-root` arguments. Image references must be relative and remain beneath
that explicit root; it never searches annotation parents or repository parents.
The exact token inventory is a third identity supplied with
`--token-inventory`.

## Build deterministic atlas data

```bash
uv run python -m sft.scripts.build_atlas_topology_dataset \
  --output artifacts/generated/sft/atlas_topology/catan_atlas_topology.jsonl
```

The canonical build contains 390 text-only rows spanning tile, node, edge, and
port topology.

## Build and render node-factor data

```bash
uv run python -m sft.scripts.build_node_factor_dataset
uv run python -m sft.scripts.render_contract_images
```

Defaults write to `artifacts/generated/sft/node_factors/` and use
`configs/sft/renderer_style.json`. The standard build creates 594 controlled
board contracts and 3,564 visual QA rows. Rendered chat rows use relative image
references; the renderer does not duplicate the answer key.

Tune renderer dimensions locally with:

```bash
uv run python -m sft.scripts.renderer_tuning_app --port 8765
```

The tuner reads tracked calibration contracts from
`artifacts/fixtures/sft/render_contracts/`. Regenerate the dense fixture with:

```bash
uv run python -m sft.scripts.build_colonist_dummy_fixture
```

## Build active board-recognition data

The engine-backed corpus, loader, query schedule, schemas, and legacy-pilot
boundary are documented in
[`data/curriculum/board_recognition/README.md`](../data/curriculum/board_recognition/README.md).
The final build contains 1,024 train, 64 validation, 64 test, and 64 separate
color-diagnostic 1024px images. Eight atomic rows per image yield 8,192 train
rows per epoch.

```bash
uv run python scripts/build_catan_board_recognition_dataset.py --overwrite
uv run python scripts/export_catan_board_recognition_sft.py \
  --queries-per-state 8 --overwrite
uv run python scripts/export_catan_board_recognition_sft.py --validate-only
```

To train atlas tokens in both directions, build the companion inverse corpus.
It contributes four localization rows per image and also emits a deterministic
12-row-per-image mixture with the existing eight forward rows:

```bash
uv run python scripts/export_catan_board_recognition_ms_swift.py --overwrite
uv run python scripts/export_catan_inverse_grounding_ms_swift.py --overwrite

# Training annotations:
artifacts/generated/board_recognition/replay_v1/ms_swift_bidirectional_v1/mixed/train.jsonl
```

Node descriptions alternate compact and full-word aliases such as
`O10/WH5/S6` and `ore 10/wheat 5/sheep 6`; resource-only aliases are admitted
only when unique on the current board. Targets are exact atlas tokens.

The older 640-state synthetic corpus under
`artifacts/generated/board_recognition/curriculum/` remains a legacy rendering
pilot and is not the active training projection.

## Build leakage-checked replay QA

Only use fresh non-benchmark QA and manifest rows:

```bash
uv run python -m sft.scripts.build_vlm_sft_dataset \
  --qa-jsonl evals/catan_board_bench/datasets/my_fresh_train/questions/qa.jsonl \
  --manifest-jsonl evals/catan_board_bench/datasets/my_fresh_train/manifest.jsonl \
  --image-root evals/catan_board_bench/datasets/my_fresh_train \
  --output artifacts/generated/sft/replay_qa/train.jsonl
```

Keep curriculum stages separate until each path trains and evaluates cleanly:

```text
Phase 0: text-only atlas topology
Phase 1: post-atlas node visual grounding
Phase 2: replay-derived image and targeted visual QA
```

Generate the curriculum-ready spatial and robber supplement without modifying
the immutable 12,288-row replay projection:

```bash
uv run python scripts/export_catan_spatial_robber_sft.py
uv run python scripts/export_catan_spatial_robber_sft.py --validate-only
```

The train supplement contains 1,032 empty-board spatial rows covering all 54
nodes and 19 tiles, with balanced yes/no hard negatives, plus 3,072 robber rows
(positive presence, negative presence, and tile-token localization for every
train state). The staged output is written to
`artifacts/generated/board_recognition/replay_v1/spatial_robber_v1/`. It is a
supplement, not yet the final composed production curriculum.

The same export also writes `curriculum_smoke_32.jsonl`: eight ordered rows per
stage, mixing the immutable source rows with the new supplement. At gradient
accumulation 8, a four-step smoke consumes exactly one stage per optimizer step.

Compose the validated production input only after the supplement exists:

```bash
uv run python scripts/build_catan_board_recognition_production_curriculum.py --overwrite
uv run python scripts/build_catan_board_recognition_production_curriculum.py --validate-only
```

This writes `production_curriculum_v1/train.jsonl`. It uses every immutable
12,288-row bidirectional example and every 4,104-row spatial/robber example once,
interleaves later-stage source and robber rows at 4:1, and adds 88 marked
spatial replay rows solely to keep all four stage boundaries aligned with the
production microbatch size of 32. The resulting epoch has 16,480 examples and
515 optimizer steps at gradient accumulation 1.

### Terrain readout (`terrain_readout_v1`)

Stage 2 of the gaussian ladder asks only "what is at each tile and port?".
`data_pipeline/board_recognition/terrain_readout.py` takes every replay
board image already on disk and emits the production forward prompts for
all 19 tiles and all 9 ports, duplicates and generic ports included
(`<T00> resource?` -> `wood`, `<T00> number?` -> `11`, desert -> `desert` /
`none`, `<P00> port?` -> `sheep port` or `3:1 port`), plus one complete
readout per image, `Read all tiles and ports.` -> `<T00> wheat 11; ...;
<P08> 3:1 port`, in fixed token order so omissions are visible. No inverse
rows (not a bijection on duplicate tiles, and they interfere with the
forward heads) and no piece rows. Splits follow the manifest, which splits
by layout: 77 train, 5 validation, 5 test replays, every state of a replay
in one split. 1,024 train images, 49,152 rows; validation 64 images, 3,072
rows. Images are the shared `replay_v1/images`.

```bash
uv run python -m data_pipeline.board_recognition.terrain_readout \
  artifacts/generated/board_recognition/replay_v1 --overwrite
```

The evaluator routes rows whose answer is a readout (task type
`terrain_readout`, or any expected answer over 48 characters) to
`--long-max-new-tokens`; everything else keeps the 16-token budget.

### Node and edge readout (`node_edge_readout_v1`)

Stage 3 of the gaussian ladder asks only "what is at each node and edge?".
`data_pipeline/board_recognition/node_edge_readout.py` takes every replay
board image and emits the production forward prompts, token as query only
(`<N17> building?` -> `red settlement`, `red city` or `empty`;
`<E17_18> road?` -> `blue road` or `empty`), plus two complete readouts per
image that walk every token in atlas order with empties explicit: 54 nodes
(`<N00> empty; <N01> red settlement; ...`) and 72 edges. Listing every
location is the stopping criterion; an occupied-only list has none. Train
images are capped at 4 occupied and 4 empty locations per family, the
empties ranked hardest first (touching a same-type piece, touching the
other type, two hops, far; quota 2 / 1 / 1 with fill from the ranked
pool). Validation, test and `color_diagnostic` are balanced: every occupied node
and edge of every image plus as many empties drawn by the same kind shares
(validation 2,622 rows), so exact accuracy cannot be earned by answering
`empty`; `--eval-coverage full` writes every location for diagnostics. The
evaluator reports per-class recall and precision, balanced accuracy and the
occupied versus empty item split of every readout; the scorecard table
leads with road, settlement and city recall and the precision of `empty`. No inverse rows, no tile,
port or robber rows. Splits follow the manifest by layout and are checked
pairwise: 77 train, 5 validation, 5 test replays, 16 colour-diagnostic
engine layouts (eval only; the only split where all eleven piece colours
appear). 1,024 train images, 17,708 rows; each eval split 64 images, 8,192
rows. Images are hard links of `replay_v1/images`.

```bash
uv run python -m data_pipeline.board_recognition.node_edge_readout \
  artifacts/generated/board_recognition/replay_v1 --overwrite
```

Readout answers run to 1,525 characters, so the trainer's row guard
(`MAX_ANSWER_CHARACTERS`) is 2,048; the evaluator routes `node_readout`
and `edge_readout` to the long budget and scores them by items. The panel
carries the two eval splits as `node-edge` and `node-edge-colors`; the
scorecard reports readouts (exact and item rates, dropped tokens, values
shifted onto the previous token) under `readouts` and `sequence_skips`.

### Mixed rungs (`mix_rung_data`)

Rungs trained on one family forget the others: the terrain rung dropped the
markers, the piece rung dropped terrain (port 1 of 576 after 256 steps).
`data_pipeline/board_recognition/mix_rung_data.py` draws one training set
from the exports already on disk by a JSON recipe: groups with row quotas,
density-bin weights, balance keys (colour, piece, empty kind) with optional
shares, and `max_repeats` to oversample a short cell before topping up from
the rest of its bin. Rows keep their source ids and metadata; images are
hard-linked into one root; `metadata.json` reports realised counts,
shortfalls, repeats and estimated completion-token shares per group. An
`eval_sample` block builds a small in-run eval file (every k-th row of the
sources' validation splits) so the trainer's eval is minutes, not the
33-minute full-coverage pass; the full sets stay on the side evaluator.

```bash
uv run python -m data_pipeline.board_recognition.node_edge_readout \
  artifacts/generated/board_recognition/replay_v1 \
  --output-dir artifacts/generated/board_recognition/replay_v1/node_edge_readout_pool_v1 \
  --train-full-coverage --overwrite        # every node and edge of every train image, the pool
uv run python -m data_pipeline.board_recognition.mix_rung_data \
  configs/sft/mix_rung3b_v1.json \
  artifacts/generated/board_recognition/replay_v1/mixed_rung3b_v1 --overwrite
```

`configs/sft/mix_rung3b_v1.json` is rung 3b: pieces plus terrain rehearsal,
14,770 rows over the 1,024 train images, density bins near-equal (setup
4,563 / sparse 4,833 / dense 5,024 / empty 350). Estimated token shares:
piece short rows 38% (7,000 occupied balanced over colour x piece with
roads at half, 4,500 empties hardest-first), terrain 25% (3,000 short
rows, 120 readouts), node and edge readouts 37% (80 + 70). Readouts are
60 to 100 times longer than a short row, so a token-balanced mix holds a
few hundred of them; the terrain rung reached 63 of 64 exact readouts with
about a thousand at half its tokens, the reweighted piece run starved them
at 81.

### O-LoRA rungs (`olora_frozen_bundle` profile)

Rungs trained on one family overwrite the others through the shared LoRA and
vision tower. The `olora_frozen_bundle` profile follows O-LoRA (Wang et al.,
2023): a finished bundle is the frozen task, the next task gets fresh
adapters, and the new adapters' input rows are held orthogonal to the frozen
task's subspaces by a penalty. Concretely, `--frozen-bundle <dir>` loads
that bundle's vision weights exactly (frozen, fp32), merges its language
LoRA and atlas rows into the base in memory, then adds rank-`--lora-rank`
adapters on every language linear layer and on the vision tower's
attention, MLP and merger projections (`VISION_LORA_SUFFIXES`), plus the
trainable atlas rows (`--token-init keep`). The loss adds
`--orthogonal-lambda` times the squared projection of every trainable
`lora_A` onto its protected basis: the frozen adapter's own `lora_A` rows
for language modules, and the `lora_A` rows of `--visual-delta-factors`
(the SVD factors of the frozen task's visual delta, see
`sft/scripts/extract_visual_delta.py`) for vision modules. Logged as
`orth_loss`, `orth_fraction` (share of the adapters' squared norm inside
the protected subspaces) and `orth_unprotected_modules`. Norms, biases,
patch embedding and position embedding stay frozen.

A bundle from this profile only reproduces its model on top of the merged
parent, so every checkpoint and the final carry `frozen_adapter/` (the
parent's `adapter_config.json` and `adapter_model.safetensors`) and
`frozen_bundle.json`; the evaluator, `--initial-bundle` and the reload
validation merge that adapter into the base before loading the bundle.

```bash
modal run --detach sft/modal_catan_vision_sft.py \
  --profile olora_frozen_bundle --token-init keep --lora-rank 16 --lora-alpha 32 \
  --frozen-bundle /runs/catan-vision-sft/<parent run>/checkpoints/checkpoint-384 \
  --visual-delta-factors artifacts/diagnostics/sft/visual_delta_gauss_s2_ck384_20260905/derived/visual_delta_rank16_linear.safetensors \
  --orthogonal-lambda 0.5 --vision-lora-learning-rate 1e-4 \
  --train-jsonl .../mixed_rung3b_v1/stage1/train.jsonl ... --no-dry-run --spawn-training
```

### Rung slices for staged real-board training

The four-stage file can be cut into standalone rungs without rebuilding:

```bash
uv run python scripts/build_catan_board_recognition_production_curriculum.py \
  --slice-stages spatial_grounding,clean_board_grounding \
  --slice-output artifacts/generated/board_recognition/replay_v1/production_curriculum_v1_rung_a --overwrite
```

A slice keeps source order, re-indexes `curriculum_index`, records the
source file's sha256, and resolves images from the shared `replay_v1/images`
directory; train it with `--no-require-curriculum` from the previous rung's
`final` bundle. The rungs follow the replay density bins, which are piece
counts in disguise:

| Rung | Stages | Pieces per board | Rows | Steps per epoch |
|---|---|---|---|---|
| a, early | spatial_grounding + clean_board_grounding | 0 to 16, median 9 | 4,960 | 155 |
| b, mid | pieces_and_colors | 17 to 31, median 24 | 3,424 | 107 |
| c, dense | real_game_distribution | 32 to 74, median 54 | 8,096 | 253 |

Planned schedules: 3 epochs (465 steps) for a, 4 epochs (428) for b, 2
epochs (506) for c, eval and save every 128, gated on the real-board panel
per density bin. No validation board has more than 50 pieces, so a 50+
endgame rung would need eval boards before it can be gated.

## Native TRL/PEFT vision SFT

### Empty-board spatial localization experiment

Build the isolated two-stage corpus from the validated replay_v1 empty boards:

```bash
uv run python -m data_pipeline.board_recognition.spatial_localization \
  artifacts/generated/board_recognition/replay_v1 --overwrite
```

The output lives under `spatial_localization_v1/` and contains:

- Stage 1: shuffled A/B/C/D marker localization in both directions, with full
  normalized target/control bboxes for all 154 atlas tokens. Nodes and edges
  receive a 2x sampling multiplier.
- Stage 2: all 1,179 canonical direction/topology facts repeated across eight
  prompt/empty-board variations, with the minority yes/no polarity per entity
  and relationship topped up by further repetitions so adjacency and
  connectivity are balanced, plus exactly 25% Stage-1 replay.
- Probes: held-out gray-dot localization for every node and edge at two dot
  sizes, sub-patch (`heldout_gray_dot_small`) and marker-sized
  (`heldout_gray_dot_large`). Neither style is used as a training marker.

Train rows in both stages are written in a deterministic shuffled order. The
trainer never shuffles, and the unshuffled Stage 1 file put only two or three
boards in every batch of 32. Validation, test, and probe files keep canonical
order.

### Single-piece localization (`spatial_localization_v2`)

The marker curriculum above proves the tower can read a position, but the cue
we need read is a game piece. `spatial_localization_v2` renders each validated
empty board with exactly one real settlement, city, or road through the
production renderer, and emits five rows per image: "Which node has the
settlement?" and "Which node has the red settlement?" -> `<N17>`, the exact
production prompt `<N17> building?` -> "red settlement", and a configurable
set of hard negatives answered "empty" (`--negatives adjacent=1,far=1` by
default): `occupancy_negative_adjacent` queries same-type locations within
three hops of the piece, touching ones first and then each ring outward, and
`occupancy_negative_far` queries locations beyond three hops. Each kind draws
without replacement in a stable hashed order, nothing repeats within an
image, and every negative row records `negative_distance` (1, 2, 3, or
"far"). At the default counts the near negative always touches the piece;
the outer rings only matter for heavier mixes, where three hops is the
smallest ring that gives coastal placements seven near candidates. The
shipped v2 and v3 builds predate the split and carry one uniformly sampled
`occupancy_negative` per image, which touched the piece only 5% of the time.
Pink is withheld from training and appears only in validation and test, so the
held-out color is the generalization probe.

```bash
uv run python -m data_pipeline.board_recognition.single_piece_localization \
  artifacts/generated/board_recognition/replay_v1 --overwrite
```

The default build renders 70 node and 70 edge placements per training board
(6,020 images, 24,080 rows) and 30 each per validation and test board (300
images, 1,200 rows) in about two minutes on a laptop with a process pool.
Train from the marker-only control through `--initial-bundle` so the learned
position-to-token mapping carries over; `spatial_localization_v1` stays as the
gray-dot multiple-choice probe and the marker control comparison.

`spatial_localization_v3` is the same build with `--tile-rows` and 40
placements per entity per board: each image also carries `<T05> resource?`,
`<T05> number?`, and `Where is the 8 wheat tile?`, so the token-to-position
direction gets a large printed target on every image. All eleven colors are
trained. Validation and test add a novel probe color that exists in no sprite
set, a seeded hue from the gaps between the real colors applied by hue-rotating
the red sprites into `<output>/assets/`; those images carry only rows whose
answer does not name the color. Build it with:

```bash
uv run python -m data_pipeline.board_recognition.single_piece_localization \
  artifacts/generated/board_recognition/replay_v1 \
  --output-dir artifacts/generated/board_recognition/replay_v1/spatial_localization_v3 \
  --tile-rows --train-images-per-board-per-entity 40 --overwrite
```

`spatial_localization_v7` is the v3 recipe rebuilt with the default one
touching and one far negative per image (eight rows per image with tiles,
27,520 train rows), so "empty" is 25% of the file, the closest integer split
to the 20% target. It exists because the v3 final adapter, scored on
single-piece validation, answered adjacent empties with the real piece 6
times in 8 while far empties failed 19 in 139: the model localizes to the
right neighborhood and blames the wrong entity. Build it with the v3 command
plus `--output-dir .../spatial_localization_v7`; continue from the v3
`final` bundle through `--initial-bundle` rather than from the marker
control. Three heavier mixes were built and launched first and abandoned
within minutes each: `v4` (the same counts under the older touching-only
sampler), `v5` (two near, one cross-type, two far), and `v6` (seven near,
seven far, which made "empty" 70% of rows). Read the v7 in-run eval by row
type: the negatives should climb from the v3 panel's 89.0%, while
`occupancy_positive` (94.2%), the tile rows, and `piece_to_token` (100%) are
the forgetting check.

### Adjacent-pair localization (`spatial_localization_pairs_v1`)

The v3 single-piece adapter scored zero-shot on the real-board production
heads (`evals/validation_v1.jsonl`) reads tiles at 100% but node occupancy
at 62.5% and edge owner at 60.2%. Of its 99 occupancy misses, 43 answer
"empty" on an occupied spot next to other pieces, 40 name a piece that sits
one or two hops away or the touching piece of the other type, and only 3
are colour or type slips. One-piece images let the model answer "describe
the piece I see"; two touching pieces do not.

`data_pipeline/board_recognition/adjacent_pair_localization.py` renders two
pieces on touching locations per image: `node_node` (edge endpoints; the
distance rule is ignored on purpose), `edge_edge` (edges sharing a node), and
`node_edge` (a building and a road that touch, same colour half the time).
Each image emits `occupancy_positive` and `colored_piece_to_token` for both
pieces, `piece_to_token` only when the piece type alone identifies one of
them, one touching `occupancy_negative_adjacent`, one `occupancy_negative_far`,
and the tile rows. Every row carries `pair_kind`, `partner_token`,
`partner_piece`, `partner_color`, `partner_distance`, and `same_color`, and
negatives carry `queried_token` and `negative_distance`, so the evaluator's
`neighbor_confusion` block can tell "named the partner" from "named the
target" without the board graph. Validation and test put the novel colour on
one piece of a fifth of the pairs and drop only the rows that name it.

```bash
uv run python -m data_pipeline.board_recognition.adjacent_pair_localization \
  artifacts/generated/board_recognition/replay_v1 \
  --output-dir artifacts/generated/board_recognition/replay_v1/spatial_localization_pairs_v1 \
  --tile-rows --train-images-per-board-per-kind 40 --eval-images-per-board-per-kind 15 --overwrite
```

`--train-images-per-board-per-kind` and `--eval-images-per-board-per-kind`
take one count or a per-kind list such as
`node_node=30,edge_edge=80,node_edge=50`; `spatial_localization_pairs_v2` is
that road-heavy mix (roads are two thirds of the occupancy positives) after
the 256-step pairs_v1 run left roads as the weakest forward head (77.5% on
single-piece roads versus 84% settlements and 91% cities, every miss
"empty"). Its in-run eval was still improving at step 256 (loss 0.176,
0.089, 0.068, 0.059 at steps 64, 128, 192, 256), so pair-stage runs use
512 steps.

The three `*_far` pair kinds are the control condition: the same two pieces
placed more than three hops apart. They default to zero training images.
`spatial_localization_pairs_control_v1` is a validation-and-test-only export
(`--train-images-per-board-per-kind 0 --eval-images-per-board-per-kind
node_node_far=10,edge_edge_far=20,node_edge_far=15`) and sits in the
regression panel beside the touching-pair set. That export predates the
parser fix that zeroes unnamed kinds, so it also carries 30 touching pairs
per board per kind (675 images, 6,461 rows); read it through `by_pair_kind`.
A per-kind list is now explicit: kinds it does not name are zero. Read the two together: a
checkpoint that answers far pairs but not touching pairs fails on neighbour
discrimination; one that fails both has a multi-piece prior problem, and
the aggregate eval loss cannot tell the two apart because the near-free
tile, localization, and far-empty rows dilute it four to one.

Continue from the v3 single-piece `final` bundle through `--initial-bundle`.
The gate before moving to sparse boards: occupancy positives and negatives
both at or above 98% on the pair validation set with no hop-1 false
positives, and no regression on the tile or localization rows.

The active trainer can combine completion NLL with a normalized post-merger
patch loss. It uses cosine similarity directly between Qwen's merged visual
patches and the requested semantic-token embedding; no learnable localization
projector is inserted. The three Stage-1 arms are selected only by flags:

```text
marker-only control:      --patch-loss-weight 0 --spatial-target-mode correct
correct-target treatment: --patch-loss-weight 1 --spatial-target-mode correct
shuffled-target control:  --patch-loss-weight 1 --spatial-target-mode shuffled
```

For each arm, provide Stage 1 `train.jsonl` and `validation.jsonl`, use the
shared `images/` directory, pass `--no-require-curriculum`, and give it a unique
`--run-name`. Modal defaults to microbatch 4 with accumulation 8, for an
effective batch size of 32. The launcher remains dry by default; only
`--no-dry-run --spawn-training` uploads data and allocates the H200.

Stage 2 starts from the winning Stage-1 `/runs/.../final` directory through
`--initial-bundle`. This loads the PEFT/token rows and full visual state but
intentionally resets optimizer, scheduler, and RNG state. It is distinct from
`--resume-latest`, which resumes an interrupted run within the same stage.

Run exact-match and causal visual controls with
`sft/modal_qwen_series_eval.py`. Its `--image-variant` accepts `original`,
`blank`, `shuffle`, `target_occlusion`, and `control_occlusion`.

Every eval also records a location-only score. For rows with a closed answer
set (atlas-token answers restrict to tokens of the requested entity type,
marker answers to A/B/C/D, polarity answers to yes/no) the evaluator ranks the
candidates by first-token log-probability and stores `candidate_score` on the
record: the restricted argmax, whether it matched, and the expected token's
rank. Summaries carry `candidate_exact_accuracy` and
`candidate_expected_rank_mean` overall and per dimension. Read this beside
`exact_accuracy`: on the step-256 marker-only control, 78 of 126 free-generation
probe misses were wrong-entity-type tokens, which the restricted score removes.
Pass `--no-candidate-scoring` to skip it.

Comma-separated `--eval-jsonl` and `--image-variant` values score every set
and variant in one container with a single model load, nesting outputs under
`<output-dir>/<set>/<variant>` with a `batch_summary.json`. The fixed
regression panel (marker validation, gray-dot probes, single-piece and tile
validation, and the replay production heads) runs against any checkpoint with:

```bash
uv run python -m sft.scripts.eval_regression_panel \
  --adapter-dir /runs/catan-vision-sft/<run>/<identity>/checkpoints/checkpoint-256 \
  --label <run>-ck256
```

The panel also supports cheap, exact-row backfills instead of rerunning every
set and visual control. `--include` takes stable set names and behavior-matrix
backfills only need the original images:

```bash
uv run python -m sft.scripts.eval_regression_panel \
  --adapter-dir /runs/catan-vision-sft/<run>/<identity>/final \
  --label <checkpoint>-targeted-backfill \
  --include single-v7,pairs-v2,pairs-control,full-board \
  --variants original \
  --dry-run
```

New summaries include `by_behavior`, `eval_set_id`, and the source JSONL hash.
Build the chronological accuracy matrix and the forgetting matrix (defined as
best-so-far accuracy minus current accuracy) from local/downloaded result
roots with:

```bash
uv run python -m sft.scripts.report_behavior_history \
  --checkpoint v3=/path/to/v3-panel \
  --checkpoint pairs-v1=/path/to/pairs-v1-panel \
  --checkpoint pairs-v2=/path/to/pairs-v2-panel \
  --output-dir artifacts/runs/sft/behavior-history
```

Repeat a checkpoint label to merge an existing panel and a targeted backfill.
The report refuses to calculate forgetting when the underlying row
fingerprints differ. `sft/modal_behavior_history.py` provides the same report
directly on `catan-sft-runs`; its semicolon-separated `--checkpoints` argument
uses container paths under `/runs/` and is dry-run-first.

The gradient-conflict probe is also dry-run-first:

```bash
uv run modal run sft/modal_gradient_conflicts.py \
  --adapter-dir /runs/catan-vision-sft/<run>/<identity>/final \
  --label <checkpoint>
```

It selects 24 training rows for each of pair positives, pair adjacent
negatives, lone-road positives, and tile anchors. Pair and tile probes use
eight examples per kind; each behavior uses unique states, avoids reuse across
behaviors until the finite board pool is exhausted, and uses deterministic
seed-42 selection. Six microbatches of four run through the same
completion-only chunked NLL as SFT, in eval mode, with backward passes but no
optimizer step. The JSON and Markdown reports separate vision, merger,
language-LoRA, combined semantic input/output token rows, and the full update;
they include raw and learning-rate-scaled norms, mean-gradient dot products
and cosine, plus minibatch cosine mean/std/min/fraction-negative. Run the same
selection against the v3 final, pairs-v1 final, and latest complete pairs-v2
bundle. Only `--no-dry-run` crosses the paid H200 boundary.

After downloading the three probe result directories, align their norms and
cosines into a checkpoint trajectory with:

```bash
uv run python -m sft.scripts.report_gradient_trajectory \
  --checkpoint v3=/path/to/v3-gradient-probe \
  --checkpoint pairs-v1=/path/to/pairs-v1-gradient-probe \
  --checkpoint pairs-v2=/path/to/pairs-v2-gradient-probe \
  --output-dir artifacts/runs/sft/gradient-trajectory
```

Compare the
resulting summaries with the fail-closed gate checker:

```bash
uv run python -m sft.scripts.check_spatial_grounding_gates stage1 --help
uv run python -m sft.scripts.check_spatial_grounding_gates stage2 --help
```

Experimental publication is disabled by default. The configured destination
is `TetraCorp/catan-qwen3.8-27b-spatial-sft`; do not pass `--publish-to-hub`
until that repository and token permission are confirmed.

There are exactly two active run files:

```text
sft/scripts/train_trl_catan_vision.py  model/data/training/save contract
sft/modal_catan_vision_sft.py          Modal image, Volumes, upload, H200 boundary
```

`--token-init` selects how the 154 atlas rows are seeded when no
`--initial-bundle` is given: `mean_noise` (default, the base-vocabulary mean
plus noise at a tenth of the vocabulary's spread, so all rows start nearly
identical), `vocab_gaussian` (each row drawn at the vocabulary's own
per-dimension mean and standard deviation, so rows start as distinct as
random real words), or `family_words`, which starts each row from the base
embedding of " node", " edge", " tile", or " port" plus the small noise so
every family carries its own shared direction from step one. Motivation: on
the pair-stage bundles the tile row `<T10>` pointed away from the tile
centroid while every other tile sat at +0.22 to +0.30, and a token's
loading on its family direction predicted its accuracy at r = -0.49.

The production-default `vision_tokens_lora` profile trains the full BF16 vision
tower and merger, the 154 atlas rows on both the input embedding and untied
output head, and rank-8 language LoRA. Original language weights stay frozen.
The optimizer keeps distinct rates: `5e-4` for token rows, `1e-4` for language
LoRA, `5e-5` for the merger, and `5e-6` for the vision tower, with a `0.1`
warmup ratio. `vision_tokens` remains the lower-capacity grounding ablation and
infrastructure smoke.

Three contracts protect the visual path and the atlas rows:

- The full visual module is promoted to fp32 master weights after wrapping,
  after an initial bundle load, and after a checkpoint resume. The base model
  still loads in bf16 and autocast keeps the matmuls in bf16. Without this, the
  vision and merger AdamW steps are below half a bf16 ulp and round to zero;
  the 2026-09-01 `spatial-sft-b32` run trained only LoRA and token rows for
  that reason. Checkpoints keep the fp32 visual state (about 1.8 GB); the
  final bundle writes bf16 so the eval and Hub contract are unchanged.
- The 154 atlas rows are initialized from the base-vocabulary mean plus small
  seeded noise on both the input embedding and the output head before PEFT
  copies them. Qwen's embedding matrix is already padded past the tokenizer, so
  `resize_token_embeddings` never touches these ids and they would otherwise
  start from untrained padding rows. `trainable_parameters.json` records the
  row norms before and after.
- Every log step reports `answer_token_accuracy` and `answer_row_exact`, which
  exclude `<|im_end|>` and the trailing newline, plus a pre-clip
  `grad_norm_<group>` per optimizer group. Read those instead of `loss` and
  `mean_token_accuracy`: a completion is three tokens and two are free, so a
  0.5 loss with 0.83 token accuracy means the answer token is near chance.

Production JSONL must already be arranged as one sequential four-stage run and
must label every row with one of these `curriculum_stage` values:

```text
spatial_grounding
clean_board_grounding
pieces_and_colors
real_game_distribution
```

All four stages must be present in that order. The Trainer disables dataset
shuffling and uses sequential sampling, so the stage order is preserved in one
run. The current 12,288-row mixed export is valid source material but has no
stage labels yet; production launch therefore fails closed on it.

Inspect the current export as an explicitly unstaged one-step smoke plan:

```bash
uv run modal run sft/modal_catan_vision_sft.py \
  --train-jsonl artifacts/generated/board_recognition/replay_v1/ms_swift_bidirectional_v1/mixed/train.jsonl \
  --image-root artifacts/generated/board_recognition/replay_v1/images \
  --token-inventory artifacts/generated/board_recognition/replay_v1/ms_swift_bidirectional_v1/trainable_tokens.json \
  --no-require-curriculum \
  --max-steps 1
```

This only prints an immutable plan. A paid H200 allocation requires the
additional `--no-dry-run --spawn-training` flags and `modal run --detach`.
Without `--detach` the ephemeral app stops when the local entrypoint exits and
Modal cancels the spawned training call before it creates a run directory. Hub upload is separately disabled unless
`--publish-to-hub` is passed. The final bundle is reloaded and validated before
upload and contains a PEFT token-row adapter, tokenizer, and
`visual_model.safetensors` for the full visual path.

The old Qwen-Series and ms-swift files remain only as historical experiments;
the active launcher does not import or execute them.

## Legacy conversion and evaluation utilities

For standalone conversion:

```bash
python -m sft.scripts.convert_to_qwen_series_sft \
  --input artifacts/fixtures/sft/modal_vlm_smoke/train.jsonl \
  --output /tmp/catan_qwen_series_train.json \
  --copy-images-to /tmp/catan_qwen_series_images
```

The older 220-token adapter format can still be inspected through its legacy
Modal evaluator:

```bash
uv run modal run sft/modal_qwen_series_eval.py \
  --eval-jsonl artifacts/generated/board_recognition/replay_v1/qwen_sft/validation.jsonl \
  --image-root artifacts/generated/board_recognition/replay_v1/images \
  --token-inventory artifacts/generated/board_recognition/replay_v1/qwen_sft/trainable_tokens.json \
  --limit 4
```

The smoke fixture validates infrastructure only; it is not a model-quality
benchmark. The quarantined historical short train/held-out files share source
images and must not be reported as an independent visual split.
