# Current hosted VLMs: unified Catan board benchmark

Date: 2026-08-25

## Result

The active scorecard now contains only current hosted models. Historical Qwen
checkpoint results remain on disk for reproducibility but are excluded from this
report and from the unified machine comparison.

The two image cohorts produce different winners:

| Current hosted model | 110 perception exact | Components | Strict raw-image 60 | Valid JSON |
|---|---:|---:|---:|---:|
| Qwen3.8 Max | **48/110 (43.6%)** | **90/218 (41.3%)** | 11/60 (18.3%) | 20/60 |
| Gemma 4 31B | 22/110 (20.0%) | 43/218 (19.7%) | **28/60 (46.7%)** | **60/60** |
| GLM-4.6V | 18/110 (16.4%) | 53/218 (24.3%) | 22/60 (36.7%) | **60/60** |
| DeepSeek V4 Flash Vision Exp | 15/110 (13.6%) | 33/218 (15.1%) | 23/60 (38.3%) | **60/60** |

These numerators are deliberately not added. The 110 cohort uses the established
component-presence CatanBoardBench scorer and emphasizes direct board-object
recognition. The 60 cohort uses strict typed JSON and includes topology,
aggregation, empty-state, and multi-entity questions.

Gemma is the strongest current model on the broader strict image contract. Qwen
Max remains strongest on the narrower perception diagnostic, but its strict-60
result is dominated by output-protocol failure: it returned valid JSON on only
20 questions and emitted 10,292 completion tokens, usually explanatory prose,
despite native reasoning being disabled and the prompt requiring one JSON object.
This is a complete hosted-stack behavior, not a pure vision-backbone ranking.

## Matched current-Qwen image/text result

Qwen3.8 Max was run on both projections of the same 60 semantic questions:

| Input | Exact | Valid JSON | Prompt tokens | Median latency |
|---|---:|---:|---:|---:|
| Raw 1024px engine screenshot | 11/60 (18.3%) | 20/60 | 87,963 | 9,509.5 ms |
| Frozen `indexed_tile_rows` text | **60/60 (100%)** | 60/60 | 667,671 | 1,927.5 ms |

All eleven image-correct questions were also text-correct. Text uniquely solved
49, image uniquely solved none, and neither modality failed a question after the
text state was supplied. The paired exact difference is +81.7 percentage points;
the reference two-sided exact McNemar binomial probability is approximately
3.55e-15.

The text representation uses 7.59 times as many reported prompt tokens. That
cost buys authoritative state and explicit joins rather than better pixels. The
result therefore localizes the main gap to visual extraction, grounding, and
image-conditioned protocol behavior—not insufficient downstream reasoning over
correct state.

## How the strict 60 was projected to images

The selected source is the latest strict `text_format_optimization_probe`: 12
leakage-safe engine states from 12 games, six questions in each of ten categories,
and strict scorer `strict_typed_json/v2`.

The projection is intentionally simple:

1. Load each public engine-state contract.
2. Invert the text probe's deterministic opaque aliases to canonical engine
   tile, node, edge, and port IDs.
3. Translate each question target and strict answer through that bijection.
4. Render the ordinary engine board at 1024px with 90% board framing.
5. Supply the same fixed, state-free engine-atlas convention used by the image
   benchmark. Dynamic state remains visible only in the screenshot.

The images contain no labels, boxes, coordinate overlays, or other annotations.
JSON engine state is retained only as the rendering/scoring oracle and is not
sent to the VLM. Question IDs, board states, categories, output shapes, and
semantic targets remain paired with the text cohort.

The text run uses the frozen `indexed_tile_rows` representation and its original
board-local aliases. Thus the image/text pairing is semantic and question-ID
paired, not byte-identical: image questions use canonical IDs because an
unannotated screenshot cannot expose a per-board random naming permutation.

## Strict raw-image category results

Each category contains six questions.

| Category | Qwen Max | Gemma 4 | GLM-4.6V | DeepSeek V4 |
|---|---:|---:|---:|---:|
| Direction to tile | 2 | **4** | 3 | **5** |
| Tile to direction | 2 | **6** | 5 | **6** |
| Node state | 0 | 3 | 2 | **4** |
| Edge state | 0 | 0 | **2** | 1 |
| Nodes connected | 2 | **6** | **6** | 4 |
| Node-adjacent tiles | 0 | **2** | 0 | 1 |
| Port occupancy | 0 | **3** | 0 | 0 |
| Roll production | 0 | **1** | 0 | 0 |
| Building counts | 2 | 1 | **3** | 2 |
| Road inventory | **3** | 2 | 1 | 0 |

The topology-only categories should not be cited as pure perception evidence.
Tile rows and one state-free node/edge anchor are supplied as fixed atlas
context so canonical IDs can be grounded on an unannotated image. Dynamic
answers—occupancy, ownership, production, and counts—still require reading the
screenshot.

Dense state remains unreliable across every current family:

- edge ownership peaks at 2/6;
- port occupancy peaks at 3/6;
- nominal production peaks at 1/6;
- complete non-empty road inventories are not reliable;
- several road-inventory successes are empty-player cases.

Gemma's strict lead therefore reflects both stronger image use and perfect JSON
compliance, not completion of the board-parsing problem.

## Resolution finding retained for the current model

The controlled Qwen3.8 Max resolution comparison remains relevant and does not
use an older checkpoint:

- original 512px framing: 29/110;
- 512px with 90% board framing: 30/110;
- 1024px with the same 90% framing: 48/110.

The 512px framing change was negligible, while 1024px added 18 exact answers and
roughly quadrupled image tokens. Most gains came from tile and robber categories;
node occupancy and complete road localization remained nearly unsolved. More
pixels help a capable hosted stack recognize local tile content, but do not by
themselves solve atlas-to-pixel binding.

## Request controls

All current runs used direct Novita hosting with one request at a time,
temperature 0, `enable_thinking=false`, and zero reported reasoning tokens.
Strict runs used 256 requested completion tokens and no API response-format
constraint; JSON compliance is therefore measured rather than forced.

Reported token counts are provider/model tokenizer outputs and are not directly
normalized across model families:

| Strict image model | Prompt tokens | Completion tokens | Median latency |
|---|---:|---:|---:|
| Qwen3.8 Max | 87,963 | 10,292 | 9,509.5 ms |
| Gemma 4 31B | 36,547 | 787 | 2,533.5 ms |
| GLM-4.6V | 106,188 | 1,468 | 4,541.5 ms |
| DeepSeek V4 Flash Vision Exp | 44,808 | 1,093 | 2,098.0 ms |

## Reproduction

Render the raw strict cohort:

```bash
uv run python scripts/render_catan_strict_vision_probe.py
```

Run one current hosted model on all 60 raw-image questions:

```bash
uv run python scripts/eval_catan_strict_vision_probe.py \
  --model google/gemma-4-31b-it \
  --output-dir artifacts/runs/catan_board_bench/strict_60_unified_20260825/image/novita/gemma4_31b
```

Run current Qwen Max on the frozen text projection:

```bash
uv run python scripts/eval_catan_strict_text_probe_novita.py \
  --output-dir artifacts/runs/catan_board_bench/strict_60_unified_20260825/text/novita/qwen3_8_max_indexed_tile_rows
```

The exact unification command and every input run path are preserved by:

- `scripts/summarize_catan_unified_benchmark.py`
- `artifacts/runs/catan_board_bench/strict_60_unified_20260825/comparison.json`

## Artifacts

- Raw unannotated image cohort:
  `artifacts/generated/catan_board_bench/strict_vision_probe_60_1024_board90/`
- Current strict image runs:
  `artifacts/runs/catan_board_bench/strict_60_unified_20260825/image/novita/`
- Current matched text run:
  `artifacts/runs/catan_board_bench/strict_60_unified_20260825/text/novita/qwen3_8_max_indexed_tile_rows/`
- Unified current-model comparison:
  `artifacts/runs/catan_board_bench/strict_60_unified_20260825/comparison.json`

## Verification

- The 60 source questions regenerate exactly from the locked text dataset.
- Alias inversion is bijective across 19 tiles, 54 nodes, 72 edges, and 9 ports.
- All 12 images are 1024px ordinary renderer outputs with no annotation layer.
- Each current image model has 60 unique, zero-error Novita responses.
- The current Qwen text run has 60 unique, zero-error Novita responses.
- All 300 strict requests report zero reasoning tokens.
- Strict scores were recomputed from raw responses; stored summaries agree.
- The unified artifact rejects old Qwen checkpoints and contains no such row.
- Twenty-nine focused tests, Ruff, formatting, and artifact integrity checks pass.
- The repository-wide suite is 276 passed, one skipped, and one unrelated
  rename-receipt failure caused by pre-existing dirty UI files plus the changed
  evaluator checksum. The historical receipt was not regenerated because that
  would bless unrelated user changes.

## Limitations

- The 60-question cohort is a deterministic diagnostic, not a population-level
  estimate.
- The 110 and 60 cohorts use different scoring contracts and cannot support one
  meaningful combined accuracy numerator.
- Fixed atlas geometry helps identify canonical entities and can answer some
  topology questions without reading dynamic image state.
- Only Qwen3.8 Max was rerun on text; the other current families have image-only
  strict-60 results.
- Direct Novita identifies the host but does not expose a separate vision
  backbone or quantization label, so this remains a hosted-stack comparison.
