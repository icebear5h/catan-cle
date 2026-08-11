# SFT Data

Generated VLM SFT JSONL files go here.

Do not store CatanBench-100 QA rows here. Training files must exclude the game IDs
listed in `data_pipeline/catanbench/datasets/catanbench_100/leakage/benchmark_game_ids.json`.

Synthetic factor datasets are allowed here when they are engine/atlas generated
and not derived from held-out CatanBench-100 games. The second dataset is
post-atlas node visual grounding: it assumes stable atlas tokens already exist,
then teaches bbox/local visual readout around those tokens. The generator writes
contracts, annotations, QA rows, and chat-style message rows under:

```bash
uv run python sft/scripts/build_node_factor_dataset.py \
  --output-dir sft/data/synthetic_node_factors
```

Render contract datasets into image-backed rows:

```bash
uv run python sft/scripts/render_contract_images.py \
  --dataset-dir sft/data/synthetic_node_factors
```
