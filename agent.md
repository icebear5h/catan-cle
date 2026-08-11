# Qwen 3 vs Qwen 3.5 eval (reproducible run)

Use this note before each run so the workflow is repeatable and uses the existing code paths.

## Prereqs
- `OPENROUTER_API_KEY` is exported in your shell environment.
- Use `.venv` Python if available in this repo.
- Run from repo root: `/Users/henry/CascadeProjects/catan-learning`.

```bash
# Sanity check key
printenv OPENROUTER_API_KEY | awk 'BEGIN{FS=""} {print "OPENROUTER_API_KEY is set (" length($0) " chars)"}'
```

## Recommended reproducible command
This uses the existing script and existing dataset, not a custom scratch script.

```bash
./.venv/bin/python scripts/eval_catanbench_openrouter.py \
  --models qwen3-vl-8b,qwen3.5-9b \
  --limit-samples 5 \
  --questions-per-sample 6 \
  --categories robber_tile,tile_resource_number,node_occupancy,edge_road_owner,port_type_nodes,longest_road_holder \
  --concurrency 2 \
  --output-dir data_pipeline/catanbench/datasets/catanbench_100/openrouter_eval/$(date -u +"%Y%m%dT%H%M%SZ")_qwen3_vs_qwen3.5
```

Notes:
- `qwen3.5-9b` is sent with `reasoning: {"enabled": false}` in the API call now, which avoids empty content responses when OpenRouter returns reasoning-first payloads.
- For `qwen3.5` calls, the script automatically raises the completion budget to at least `256` tokens so reasoning-first output does not consume all tokens.
- Response parsing now checks multiple fields (`content`, `reasoning`, `reasoning_content`, `reasoning_details`) to capture Qwen's final text.

## Suggested follow-up checks (same run)
- Compare accuracy from summary:

```bash
python - <<'PY'
import json, pathlib
out_dirs = sorted(pathlib.Path('data_pipeline/catanbench/datasets/catanbench_100/openrouter_eval').glob('*qwen3_vs_qwen3.5'))
if not out_dirs:
    raise SystemExit('No matching output directories found')
out_dir = out_dirs[-1]
summary = json.loads((out_dir / 'summary.json').read_text())
print(json.dumps(summary, indent=2))
PY
```

- Keep notes of empty-response counts and reasoning-token totals:

```bash
OUT_DIR=$(ls -td data_pipeline/catanbench/datasets/catanbench_100/openrouter_eval/*qwen3_vs_qwen3.5* 2>/dev/null | head -n 1)
if [ -z "$OUT_DIR" ]; then
  echo "No matching output directory found"
  exit 1
fi
python - <<'PY'
import json, pathlib
out_dir = pathlib.Path("$OUT_DIR")
records = [json.loads(line) for line in (out_dir / 'responses.jsonl').read_text().splitlines() if line.strip()]
print("total", len(records))
for model in ("qwen3-vl-8b", "qwen3.5-9b"):
    model_records = [r for r in records if r.get('model_key') == model]
    empty = sum(1 for r in model_records if not str(r.get('response', '')).strip())
    reasoning = sum((r.get('usage', {}).get('completion_tokens_details', {}).get('reasoning_tokens') or 0) for r in model_records)
    print(model, "count=", len(model_records), "empty=", empty, "reasoning_tokens=", reasoning)
PY
```

## 2026-05-14 UTC run: 260-request no-thinking full suite (`qwen3-vl-8b` vs `qwen3.5-9b`)

- Initial in-sandbox run failed with DNS/network resolution errors (`[Errno 8] nodename nor servname provided, or not known`), producing empty responses.
- Re-ran with escalated network access and full output to:
  - `data_pipeline/catanbench/datasets/catanbench_100/openrouter_eval/20260514T231751Z_qwen3_vs_qwen3.5_full_suite_nothink9b3`

Summary:

- `qwen3-vl-8b`: `attempted=130`, `errors=0`, `exact_accuracy=16.153846153846155%`, `component_accuracy=15.546218487394958%`, `avg_latency_ms=1284.753846153846`
- `qwen3.5-9b`: `attempted=130`, `errors=0`, `exact_accuracy=15.384615384615385%`, `component_accuracy=21.008403361344537%`, `avg_latency_ms=1897.5384615384614`
- Command used: `--models qwen3-vl-8b,qwen3.5-9b --concurrency 2` on the full suite (`limit-samples=10`, `questions-per-sample=13`).
