# Board-recognition curriculum smoke fixture

Deterministic infrastructure fixture generated with:

```bash
uv run python scripts/build_catan_board_recognition_curriculum.py \
  --output-dir artifacts/fixtures/board_recognition/curriculum_smoke \
  --image-size 64 \
  --pairs-per-entity-type 1 \
  --seed 381427 \
  --overwrite
```

The 64px images keep the tracked fixture small and test only generation,
rendering, dense labels, counterfactual invariants, splits, and validation. They
are not suitable for training or model-quality evaluation. Real curriculum data
defaults to 1024px and belongs under the ignored
`artifacts/generated/board_recognition/curriculum/` path.
