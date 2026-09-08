# Node and edge recognition rung (gaussian ladder, stage 3)

Status, 2026-09-05 14:35 PDT: the approved reweighted piece-recognition
run is training on Modal (step 9/512 observed). At the user's request the
CPU budget watchdog was cancelled without restarting training; the native
seven-hour training timeout and resource limits remain. Run identifiers and
the original budget envelope are below. The original
export and held-out sets remain unchanged. The combined 154-location readout
plus `robber <Txx>` is explicitly deferred until piece recognition improves;
it is **not** included in this run. Working checklist: `tasks/todo.md`. Numbers land in
`reports/sft/2026-09-03-qwen38-single-piece-v2.json` under
`gaussian_ladder_20260904.stage3`.

## Context

The board reader has to reconstruct real replay boards exactly. The ladder
so far, all from the base model with `vocab_gaussian` token rows:

| Rung | Bundle | Result |
|---|---|---|
| 1, entity-shaped markers, 256 steps | `gauss-s1-markers-entity` final | markers 99.1% on its own set; 0 twins |
| 2, terrain readout, stopped at step 416 | `gauss-s2-terrain` checkpoint-384 | tile number, resource, port 100% on 5 unseen layouts; readouts 64 of 64; 0 twins |

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

## Training reweighting addendum, 2026-09-05

The user requested reweighting after the balance audit. A **separate local
training export** is now built at
`artifacts/generated/board_recognition/replay_v1/node_edge_readout_reweighted_v1`.
No training was launched or remote bundle changed. The counts above describe
the original export; the following counts describe the new training input.

This is **token-budgeted row resampling**, not a custom weighted loss. It works
with the existing sequential TRL loader without changing chunked NLL, vision
precision, optimizer settings, prompts or answer labels. The starting budget
is 50% short occupied / 25% short empty / 25% complete readout **completion
tokens**, measured with the stage-2 checkpoint-384 tokenizer including its
atomic atlas tokens. Counts include the answer plus `<|im_end|>\n`, exclude
prompts, and are corpus exposure estimates rather than exact gradient shares.
Batch normalization, row difficulty and the training prefix also matter.
These are explicit, tunable starting weights, not an empirically optimal mix.

| Training exposure | Original | Reweighted |
|---|---:|---:|
| short occupied, completion tokens | 5.1% | 50.4% |
| short empty, completion tokens | 4.0% | 25.2% |
| full readouts, completion tokens | 90.9% | 24.5% |
| short occupied rows | 7,468 | 10,328 |
| short empty rows | 8,192 | 7,299 |
| full readout rows | 2,048 | 81 |
| total rows | 17,708 | 17,708 |
| empty items, short + readout answers | 70.4% | 48.9% |
| images appearing anywhere in training | 1,024 | 1,024 |
| images with a sampled complete readout | 1,024 | 80 |

**Coverage tradeoff:** downweighting long answers by resampling substantially
reduces the number of readout examples. The 81 selected readouts still contain
every node or edge, empties included. Their own empty fraction is 75.3%; this
does not rebalance items *inside* a readout. Keeping all 2,048 readouts while
reducing their contribution would require a different, per-item or per-row
loss-weighting implementation. This export should not be described as that.

For occupied queries, the token budget is road 50%, settlement 25%, city 25%,
then equal across all eleven colours within each piece type. Short positives
are drawn from all **35,292 existing labelled occupied spots**, expanding the
old four-per-family cap by extracting the same labels from full readouts.
Every old short label must agree with its board readout. No new images,
recoloured boards, inverse prompts or new answer formats were introduced.

Red and white now each have 996 short positives (498 road, 249 settlement,
249 city), versus 1,559 and 169 before. Bronze has 796 and mystic blue 568:
their names take more tokenizer tokens, so their row counts are lower while
their within-piece token exposure matches the other colours. Balance is on
**colour × piece tokens**, not an assertion of equal numbers of independent
examples. There are 17,155 unique sampled queries; the most-repeated query
appears three times. The exporter refuses requests needing more than four
copies unless explicitly configured otherwise. More repeats do not add visual
diversity.

Within node/edge families, source hard-negative-kind and readout-density row
proportions are preserved up to integer rounding. Positive reweighting shifts
the overall density distribution toward dense boards, where cities and many
rare colours occur; the metadata records that shift. The first 16,384 rows
(512 steps at effective batch 32) contain 49.7% occupied / 24.9% empty / 25.4%
readout completion tokens, with 79 readouts over 78 images.

Implementation: `data_pipeline/board_recognition/reweight_node_edge.py`;
tests: `tests/test_reweight_node_edge.py`. Reproduce to a **new** output path:

```bash
.venv/bin/python -m data_pipeline.board_recognition.reweight_node_edge \
  artifacts/generated/board_recognition/replay_v1/node_edge_readout_v1 \
  --output-dir artifacts/generated/board_recognition/replay_v1/node_edge_readout_reweighted_v1 \
  --tokenizer-json /path/to/checkpoint-384/tokenizer.json \
  --occupied-share 0.50 --empty-share 0.25 --readout-share 0.25 \
  --seed 42 --max-repeats 4
```

The command refuses existing outputs. `metadata.json` records source and
tokenizer fingerprints, bucket quotas, colour × piece counts and token totals,
before/after item counts, density shifts, repeat counts and the training-prefix
audit. Training SHA-256:
`c0be726b243fb306b4bd382d529fe97e18feac34652acc0b82c440d533ef3521`.
Validation, test and colour-diagnostic JSONLs are byte-identical to the
original export, with hash equality checked after copying. The old export is
untouched. Scarce diagnostic colour × piece support is **not** repaired by
training reweighting; nor are the scorer issues or retention gates changed.

Local verification: 44 tests passed across reweighting, the original exporter
and the trainer; Ruff and `git diff --check` passed. The existing trainer's
dataset-contract audit accepts all 17,708 rows and 1,024 images. Independently,
all 17,627 sampled short answers agree with the source engine contracts. No
GPU training or GPU evaluation was used for these checks.

## Approved launch, 2026-09-05

**Guard change, 14:35 PDT:** user requested removal of the budget guard. The
watchdog Python process was paused before cancelling its FunctionCall with
container termination, preventing the fail-closed exception handler from
cancelling training. Modal confirmed the watchdog call cancelled and the
original training call still running; no training restart or extra run was
launched. The watchdog no longer enforces the 21:36:43 absolute cancellation
deadline. The native seven-hour per-attempt timeout, resource limits and
entry-time deadline validation remain unchanged in the live training function.
The original $79 envelope below is launch history, not an active aggregate
spending guarantee; the $25 eval reserve remains a planning allocation.

Budget constraint added by the user: **under $80 for this rung's incremental
compute, training plus evaluations**. The launched plan uses $79: a bounded
training window worth at most approximately $45.27 at checked list rates,
$25 reserved for standalone evaluations, and $8.73 headroom for the CPU
watchdog, startup/cancellation overhead and billing uncertainty. The earlier
$40 training allocation was a planning estimate; this larger envelope allows
up to seven hours without authorizing another experiment.
This excludes prior spending, shared storage and the workspace subscription;
estimates are gross, before credits. The user separately authorized this launch.

At [Modal's published rates](https://modal.com/pricing), checked 2026-09-05,
the requested H200 + 16 physical CPU cores + 128 GiB RAM costs approximately
$6.32/hour (usage above the CPU/RAM reservation can add cost). Five to six
billed hours would be $32–38. Comparable 512-step runs in the
`2026-09-04-modal-spending.md` ledger cost about $34. Runtime of the new mix
is not measured; the earlier output-token reduction is not a proportional
reduction in vision compute. Allow roughly $20–30 for side/final evals for a
provisional total of **$52–68**, subject to measured throughput.

Budget policy: preserve the final node/edge, colour and retention measurements;
fit intermediate side evaluations within the remaining eval allocation. No
additional training variants or discretionary reruns without a fresh budget
check. The legacy launcher retains its 24-hour default, but this run uses the
new `--budget-usd 79` path: execution timeout 7 hours, startup timeout 10 minutes,
hard CPU and RAM limits of 16 cores / 128 GiB, one H200 container, and no
configured function-error retries. A separate CPU-only watchdog observes the
specific training call and cancels its containers at the **absolute** deadline,
including across infrastructure rescheduling. If arming fails locally, the
launcher cancels training. On watcher execution errors it also fails closed.
This is a resource/time envelope at published rates, **not a Modal account-wide
dollar cap**. No standalone evals or further training are automatically spawned.

Confirmed submission:

- App: `ap-ZwkBRDHVyLMbNl3fg4BiuH`
  ([dashboard](https://modal.com/apps/tetracorp/main/ap-ZwkBRDHVyLMbNl3fg4BiuH)).
- Training call: `fc-01M1SQEX7D43F1JGWZSMA6YKVT`.
- CPU watchdog call: `fc-01M1SQEXBHVSB8Y4ZNHCX2RF7X` (**cancelled by user request**).
- Original absolute cancellation deadline: **2026-09-05 21:36:43 PDT**;
  watchdog enforcement removed at 14:35. Native seven-hour timeout remains.
  A timed-out run is incomplete, not a successful 512-step run.
- Output:
  `/runs/catan-vision-sft/catan-qwen38-gauss-s3-nodes-edges-rw-20260905/bbbc99d8bc8c`.
- Identity: `ad6c1526635c60cc4e1c0b5116ee5b61e07bfe1840a57492890b98b73fcc8856`.
- Local receipt:
  `artifacts/runs/sft/gauss-s3-nodes-edges-rw-20260905/launch.json`.
- Watchdog status on `catan-sft-runs`:
  `catan-vision-sft/catan-qwen38-gauss-s3-nodes-edges-rw-20260905/bbbc99d8bc8c-budget_guard.json`.
- Startup verified: watchdog persisted `watching` at 14:27:12 PDT; training
  reached step 3/512 at 14:31:22 PDT. The live `trainable_parameters.json`
  reports no scope errors and FP32 for all 327 vision tensors, six merger
  tensors, 992 language-LoRA tensors and both atlas-row tensors. These are
  startup checks, not evidence of held-out accuracy or effective update size.
- Preflight: checkpoint-384 bundle present, original 154-token inventory
  matched exactly, no active Qwen run found, new output name unused. Fifty-six
  focused tests passed, including budget cancellation and existing trainer/data
  tests. Actual checkpoint metrics are pending; submission is not evidence of
  learned piece recognition.

- Baseline: pairs_v2 final on `node-edge` and `node-edge-colors`, original
  variant, label `pairs-v2-final-node-edge` (completed; recorded in `tasks/todo.md`).
- Rung: `sft/modal_catan_vision_sft.py` from
  `/runs/catan-vision-sft/catan-qwen38-gauss-s2-terrain-20260904/398f0a023ec9/checkpoints/checkpoint-384`,
  train and validation from `node_edge_readout_reweighted_v1/stage1`, with
  `node_edge_readout_reweighted_v1/images` as image root, batch 16 x 2,
  512 steps, eval and save every 128, `--no-require-curriculum`, run name
  `catan-qwen38-gauss-s3-nodes-edges-rw-20260905`.
  The original dry run and output suffix `e64dcbc2cc4c` apply only to the old
  dataset; the new dataset identity is `bbbc99d8bc8c`. Local dataset-contract
  validation passes and the reweighted bundle has been uploaded. The previous
  runtime estimate is not yet revalidated for this shorter-answer mix.
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

## Possible contributor: reusing the same LoRA across stages

Recorded 2026-09-05 at the user's request. **Untested hypothesis, not an
established cause or an approved training change.**

Verified loading behavior: stage 3 inherited the terrain stage's
checkpoint-384 through `initial_bundle`. The loader calls
`PeftModel.from_pretrained(..., is_trainable=True)` and separately restores
the saved visual state. It does not call `merge_and_unload()` and attach a
fresh adapter. The existing rank-8 language adapter was continued, with new
optimizer/scheduler state. The launch receipt and the run's persisted
`initial_bundle.json` both identify this parent.

- Evidence: [launch receipt](../../artifacts/runs/sft/gauss-s3-nodes-edges-rw-20260905/launch.json)
  and [initial-bundle loader](../../sft/scripts/train_trl_catan_vision.py).
- Hypothesis: fitting pieces by modifying the existing rank-8 adapter may
  have contributed to a compromise with previously learned terrain/port
  behavior. Merging the old language update into frozen base weights and
  training a fresh rank-8 adapter would instead provide an additional
  low-rank update around the stage-2 solution. These are different
  parameterizations; the existing factors are not confined to a fixed
  learned subspace as they train.
- Limits: unmerged does not mean unloaded or inactive. Merging alone does
  not protect behavior, and a fresh adapter can still counteract the old
  one. The vision tower, merger and atlas rows are separate possible routes
  for regression. This hypothesis does not establish rank saturation or
  rule out data-mix and optimization effects.
- Candidate comparison, **not launched**: start both arms from exactly the
  same checkpoint-384; continue the existing adapter in one, versus merge
  its language update into frozen weights and attach a fresh rank-8
  adapter in the other. Preserve learned atlas rows. Verify equivalent
  pre-training evaluation before comparing, allowing numerical tolerance.
  Match data/order, step budget, optimizer schedule, adapter scaling and
  visual/token-row trainability across arms. Judge occupied-piece gains
  against terrain/port retention, not aggregate loss alone. A benefit would
  support this intervention, but would not distinguish extra cumulative
  capacity from changed initialization/optimization without further controls.

No merge, adapter reset, configuration change or paid experiment is
authorized by this note; any test needs a fresh budget decision.

## What runs when

| When | What |
|---|---|
| now | training active; startup and FP32 scope verified; watchdog cancelled by user request, native timeout unchanged |
| every 128 steps | in-run trainer eval and checkpoint; standalone side evals require a remaining-budget check and are not auto-launched |
| final | prioritize node/edge, colour and terrain-retention evaluation within the $25 reserve; scorecard and row inspection on saved artifacts; no automatic next rung |
