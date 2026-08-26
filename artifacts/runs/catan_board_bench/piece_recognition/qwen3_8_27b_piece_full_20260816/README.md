# Qwen3.8-27B piece-recognition evaluation

This directory preserves the tracked evaluation artifacts for the 199-row
isolated/local-patch visual run. The exact QA rows used by the accepted run are
frozen as `qa_snapshot.jsonl`.

The regenerable images and source dataset are ignored under
`artifacts/generated/catan_board_bench/piece_recognition/`. Rebuild them with:

```bash
uv run python -m scripts.build_catan_board_bench_piece_visuals
```

Create a new run rather than overwriting this accepted evidence:

```bash
uv run python scripts/eval_catan_board_bench_openrouter.py \
  --suite probe \
  --bench-dir artifacts/generated/catan_board_bench/piece_recognition \
  --question-dir artifacts/generated/catan_board_bench/piece_recognition/questions \
  --models qwen/qwen3.8-27b \
  --categories isolated_tile_resource_number,isolated_road_owner,isolated_node_occupancy,isolated_port_trade_type,isolated_robber_presence,local_patch_tile_resource_number,local_patch_edge_road_owner,local_patch_node_occupancy,local_patch_port_trade_type,local_patch_robber_presence \
  --limit-samples 0 \
  --concurrency 2 \
  --temperature 0 \
  --max-tokens 64 \
  --timeout 180 \
  --output-dir artifacts/runs/catan_board_bench/piece_recognition/<new_run_id>
```

Rebuild the semantic diagnostics in this frozen run with:

```bash
uv run python scripts/rescore_piece_semantic_aliases.py \
  --responses artifacts/runs/catan_board_bench/piece_recognition/qwen3_8_27b_piece_full_20260816/responses.jsonl \
  --qa artifacts/runs/catan_board_bench/piece_recognition/qwen3_8_27b_piece_full_20260816/qa_snapshot.jsonl \
  --output-dir artifacts/runs/catan_board_bench/piece_recognition/qwen3_8_27b_piece_full_20260816
```
