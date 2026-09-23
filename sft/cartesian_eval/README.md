# Stock Qwen3.8: exact 2D Cartesian evaluation

## Latest: sparse h shorthand

The sparse `h=sqrt(3)/4` variant completed on stock Qwen3.8: **153/200 (76.5%)**
strictly, versus the dense radical panel's 143/200. Input tokens fell 42.5%.
Two additional correct owner names failed only uppercase formatting; the separate
case-insensitive-color diagnostic is 155/200. See the
[paired run report](../../reports/sft/2026-09-21-sparse-h-cartesian-eval.md).

Build with `uv run --no-sync python -m sft.cartesian_eval.shorthand_dataset`.
Artifacts are under `artifacts/generated/sft/cartesian_h_eval_v1/`. The builder
preserves all 200 source cases and keeps every tile/port/robber record, omitting
only empty dynamic nodes and edges. Static inventories remain complete.

Examples: `T(0,0)`, `T(4h,0)`, `T(2h,1.5)`, `N(2h,0.5)`, `E(h,0.75)`.
The dense-root implementation below remains the historical parent and oracle.

The selected experiment is **stock Qwen3.8-27B, inference only**. No Catan adapter,
new tokenizer tokens, coordinate fine-tuning, or representation sweep is involved.

## Coordinates

Regular pointy-top hexagons have side length 1. Both Cartesian axes use the same
unit length; x points right and y points up. Fractions and `sqrt(3)` remain exact.

| Entity | Address |
| --- | --- |
| Central tile | `T(0,0)` |
| Its top node | `N(0,1)` |
| Its upper-right node | `N(sqrt(3)/2,1/2)` |
| Road midpoint between those nodes | `E(sqrt(3)/4,3/4)` |
| Adjacent tile to the right | `T(sqrt(3),0)` |

Node/tile identities are their physical positions. Roads and ports use typed
midpoints; `E` and `P` distinguish a road slot and port at the same position.
Geometry is represented internally with rational coefficients of `sqrt(3)` and
rational y-values, never rounded floating-point identities.

## Panel and scoring

The panel preserves the 200 canonical cases from the earlier coordinate
comparison: direction, adjacency, incidence, and ownership. It retains 198 test
and two validation cases, with their original source/provenance. Only the
Cartesian version is emitted. Canonical atlas IDs remain private oracle metadata.

Static questions include the full entity inventory; dynamic questions include
the original complete board state. Both use the Cartesian convention and remove
the earlier premise that the model has a learned Catan atlas.

Answers are mapped back to the existing exact task oracle. Simple algebraically
equivalent radical/fraction spellings are accepted (such as `sqrt(3)/2`, `√3/2`,
or `1/2*sqrt(3)`); unknown positions, rounded approximations, duplicate aliases,
wrong entity types, and extra prose receive no credit. Parsing executes no code.

## Build and prepare

```bash
uv run --no-sync python -m sft.cartesian_eval
uv run --no-sync python -m sft.miles_eval.run prepare \
  --panel cartesian=artifacts/generated/sft/cartesian_eval_v2/eval.jsonl \
  --output artifacts/generated/sft/miles_cartesian_eval_v2
```

The builder requires the original local `coordinate_comparison_v1/paired.jsonl`;
its content and source cases are pinned. Output directories must be new. The
generated `preview.md` contains actual prompts and gold answers; `mapping.json`
is for inspection and never enters the model prompt.

The active dataset is v2: its legend distinguishes entity-coordinate sets from
participant-color answers. The earlier v1 draft was never evaluated on a model.

Run through the [Miles eval recipe](../miles_eval/README.md) using the original
Qwen3.8-27B HF weights and tokenizer. Cartesian-only preflight does not require
or add the 154 trained atlas tokens. Legacy atlas panels retain their tokenizer
requirements.

The first stock-Qwen GPU evaluation completed: **143/200 exact (71.5%)** with
thinking disabled. See the [run report](../../reports/sft/2026-09-21-stock-cartesian-eval.md).
This directory's dataset artifacts remain inputs; model outputs are stored under
`artifacts/runs/sft/miles-cartesian-stock-20260921-r01/`.
