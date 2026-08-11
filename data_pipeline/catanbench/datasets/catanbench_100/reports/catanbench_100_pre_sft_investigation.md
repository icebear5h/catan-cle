# CatanBench-100 Pre-SFT Investigation

Purpose: define what we know before Catan-specific tuning, what failure modes matter,
and what evidence should count as real improvement after SFT.

Related artifacts:

- `data_pipeline/catanbench/datasets/catanbench_100/leakage/benchmark_game_ids.md`
- `data_pipeline/catanbench/datasets/catanbench_100/reports/catanbench_100_openrouter_baseline.md`
- `data_pipeline/catanbench/datasets/catanbench_100/openrouter_eval/`

## Leakage Boundary

CatanBench-100 is evaluation-only. The game IDs in
`data_pipeline/catanbench/datasets/catanbench_100/leakage/benchmark_game_ids.md`
must not appear in SFT, CPT, validation, prompt tuning, data filtering, or
synthetic data generation prompts.

Current local raw replay inventory is contaminated for training because every replay
in `data_pipeline/bootstrapping/data/raw_replays` overlaps the held-out benchmark.
Use `data_pipeline/bootstrapping/scrapers/4p_games_training_candidates.json` as the
candidate source for fresh non-overlapping replay pulls.

## Current Finding

Small/off-the-shelf VLMs are not failing because Catan strategy is hard. They are
failing earlier: local board grounding.

The important pre-SFT pattern is:

| Area | Current read |
| --- | --- |
| Tile resource + number | Very weak. Qwen3-VL-8B repeatedly misses `<Txx>` to visible resource/number binding. |
| Robber tile | Very weak. Often chooses a nearby or arbitrary tile instead of the robber location. |
| Edge road owner | Very weak/unstable. Occupied roads are often answered as `EMPTY` or wrong color. |
| Ports | Partial. Node IDs are sometimes right, but resource/ratio is often wrong or missing. |
| Node occupancy | Mixed. Empty nodes are easier; positive settlements/cities expose failures. |
| Global sentinel answers | Too easy. `NONE`-heavy tasks can inflate headline accuracy. |

The baseline table captures this numerically; this doc captures how to interpret it.

## Failure Modes To Inspect

1. Atlas binding failure

The model sees the board but does not reliably bind atlas IDs like `<T08>`, `<N11>`,
or edge IDs to their visual locations. This is the central failure. If the model
cannot answer "what is on this specific ID", strategy SFT is premature.

2. Visual primitive failure

The model may not distinguish Catan primitives reliably: road color, settlement vs
city, port icon, robber piece, resource icon, and dice number. This is separate from
ID binding. We need examples that isolate "can you see the object?" from "can you map
it to the atlas ID?"

3. Positive/negative imbalance

Node occupancy looks better when many queried nodes are empty. Post-SFT evaluation
must include positive settlements and cities by color, not mostly empty nodes.

4. Format and token brittleness

Some port answers are semantically close but fail exact matching because of missing
ratio text or formatting. We should track exact accuracy and component accuracy
together, but tuning should still teach the canonical answer format.

5. Shortcut learning

The model can improve on easy sentinel answers without learning the board. Improvements
on `longest_road_holder = NONE`, `EMPTY`, or default generic ports are not enough.

## What SFT Should Improve

Real SFT progress should show up first in local grounding tasks:

| Target | Desired post-SFT movement |
| --- | --- |
| `tile_resource_number` | Large exact and component gain; fewer `<DESERT> NO_NUMBER` hallucinations. |
| `robber_tile` | Correct tile ID for robber, not just nearby resource/number descriptions. |
| `edge_road_owner` | Occupied roads stop collapsing to `EMPTY`; color binding improves. |
| `port_type_nodes` | Keeps good node binding while adding correct resource/ratio. |
| `node_occupancy` | Positive settlement/city cases improve, not only empty-node cases. |

The win condition is not "the model sounds more Catan-aware." The win condition is
that it can answer constrained board-state questions from a held-out rendered board.

## What Not To Count As Success

- Better performance only on `NONE`, `EMPTY`, or other sentinel-heavy answers.
- Gains caused by benchmark game leakage.
- Gains caused by adding CatanBench images or contracts to training.
- Gains from custom tokens on the initial off-the-shelf benchmark. Custom tokens are
  valid for tuned models, but the raw baseline should use plain text/IDs.
- Free-form explanations that contain the right visual object but the wrong atlas ID.
- Strategy QA improvement before board-state extraction is reliable.

## Investigation Table To Maintain

For each model/checkpoint, record:

| Field | Meaning |
| --- | --- |
| Model/checkpoint | Base model or tuned checkpoint name. |
| Training data source | Fresh game IDs only; include exclusion check. |
| Token setup | Plain IDs, added Catan tokens, or both. |
| Eval split | Must be CatanBench-100 held-out IDs. |
| Exact accuracy | Main metric. |
| Component accuracy | Diagnostic metric for partially correct structured answers. |
| Positive-node accuracy | Settlement/city-only subset. |
| Occupied-road accuracy | Non-empty road subset. |
| Port type accuracy | Resource/ratio only. |
| Atlas binding error rate | Wrong ID despite seeing correct object. |

## Immediate Next Steps

1. Pull fresh replay JSONs from `4p_games_training_candidates.json`.
2. Build a training split manifest with explicit game IDs.
3. Add subset reports for positive nodes, occupied roads, and non-generic ports.
4. Run the same baseline table on the first SFT checkpoint.
5. Compare against this investigation before claiming improvement.
