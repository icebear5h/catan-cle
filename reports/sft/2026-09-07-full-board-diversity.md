# Diversified full-board training data

Generate new board layouts and legal piece configurations instead of only
reshuffling the first pilot's 1,024 examples. Preserve the complete-board output
objective and all fixed held-out examples.

## Dataset

- 4,096 new boards: 256 new layouts, 16 positions per layout.
- Each new layout contributes one empty, three setup, five sparse, and seven
  dense boards. Identical piece configurations are excluded even when the
  robber position differs.
- New terrain/number/port layouts, engine-generated legal roads/buildings/cities,
  varied robber locations, and rotating four-player palettes drawn from all
  supported colors. No forced resources or hand-placed illegal game pieces.
- Retain the original 1,024 training boards: 5,120 total, with 20% original data.
- Keep validation, test, and color-diagnostic exports byte-identical. New static
  layout hashes must differ from every layout in the previous corpus, including
  held-out layouts. Full visible-state hashes must be globally unique.

This diversifies game content. It retains the existing 1024px renderer and style;
it does not establish generalization to other rendering styles or real screenshots.
The legal datagen policy is synthetic, so the old replay examples remain in the mix.
Trajectories lacking enough distinct positions in a required density are skipped.

## Reproducibility and checks

Generator: `data_pipeline/board_recognition/full_board_diversify.py`.
Seed: 9072026. Up to 1,500 advertised legal actions per trajectory. Selection
uses deterministic hashes and caps each accepted layout at 16 unique piece
configurations. Generation runs locally with four worker processes.

The focused selection/schema tests passed (seven tests). A two-layout end-to-end
smoke generated 32 new boards, verified export validity and unchanged evaluation
bytes, and passed the complete build. A dense rendered sample was inspected.
Ruff passed for the new generator and tests.

Output root: `artifacts/generated/board_recognition/full_board_diverse_v1`.
The complete export is under `full_board_readout_v1/stage1/`.
`build.json` records progress, skipped trajectories, final hashes, and color,
piece, robber, and occupied-location coverage. It must say `completed` before use.

The current epoch-two baseline/training/evaluation cycle keeps its immutable
uploaded data and finishes its before/after comparison. This new shard is
prepared separately for subsequent training; it is not silently substituted
into that active run.

Status: completed and ready for a subsequent training stage. It has not been
uploaded or substituted into the current Modal run.

## Verified outcome

Generated 4,096 new boards over 256 new layouts from 257 attempted trajectories
(one lacked sufficient distinct density support). Combined training: 5,120 boards
over 333 layouts, comprising 2,330 dense, 1,507 sparse, 984 setup, and 299 empty.

The new boards cover all 11 colors, all 33 color/piece classes, all 126 occupied
node/edge locations, and all 19 robber tiles; they contain 5,982 city instances.
These are dataset coverage counts, not model accuracy or independent city samples.

Training targets are 681–797 tokens using the checkpoint tokenizer, below the
1,280-token generation cap. The trainer dataset contract passes. Validation, test,
and color-diagnostic JSONL bytes are unchanged. Generator/engine/renderer source
hashes were verified unchanged during generation.

Training SHA256:
`f3dc88bd70710e27804f1e12a2020d4eafd404e409144e5df00d29951192b4f7`.

Use `dataset_inputs.json` for the next launcher; `readiness.json` contains the
trainer contract and tokenizer audit, and `build.json` contains coverage/hashes.
