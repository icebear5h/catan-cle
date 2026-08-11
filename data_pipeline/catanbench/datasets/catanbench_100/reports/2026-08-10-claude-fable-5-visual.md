# Claude Fable 5 CatanBench Visual Evaluation

Date: 2026-08-10

## Configuration

- Model: `anthropic/claude-fable-5`
- Architecture/parameter count: proprietary and undisclosed
- Input modalities: text, image, file
- Suite: current `visual` suite
- Boards: 10 (`sample_000` through `sample_009`)
- Questions: 110 (11 categories × 10 boards)
- Atlas prompt: enabled
- Temperature requested: 0
- Maximum completion requested: 256 tokens
- Concurrency: 2
- Benchmark prompts and labels were not modified.

OpenRouter reports that reasoning is mandatory for this model and defaults to high effort. The current benchmark runner did not override that model-specific default.

## Matched-budget result

| Metric | Result |
| --- | ---: |
| Exact accuracy | 33/110 (30.00%) |
| Component accuracy | 32.11% |
| Request errors | 0/110 |

### Category breakdown

| Category | Exact | Component |
| --- | ---: | ---: |
| `color_building_counts` | 0/10 | 20.0% |
| `color_road_count` | 2/10 | 20.0% |
| `color_road_locations` | 0/10 | 16.7% |
| `edge_road_owner` | 7/10 | 70.0% |
| `node_occupancy` | 5/10 | 60.0% |
| `port_occupancy` | 0/10 | 0.0% |
| `port_trade_type` | 1/10 | 25.0% |
| `robber_resource_number` | 1/10 | 16.7% |
| `robber_tile` | 3/10 | 30.0% |
| `tile_has_robber` | 4/10 | 40.0% |
| `tile_resource_number` | 10/10 | 100.0% |

The model read targeted tile resource/number pairs perfectly and performed substantially better than Qwen3-VL-32B on local road-owner and node-occupancy questions. It remained unreliable on global atlas translation, ports, aggregate counts, and robber localization.

## Completion-budget confound

The 30% figure is a matched-256-token operational result, not a clean estimate of Fable's visual capability:

- 61/110 responses used exactly the full 256-token completion budget.
- Only 3/61 capped responses scored exact.
- Many capped outputs contained unfinished first-person reasoning rather than a final answer.
- Six responses exposed serialized `reasoning.text`/signature content through the provider response instead of a usable final answer.
- The local scorer can sometimes recover answer tokens from prose reasoning, so truncation can both suppress valid answers and occasionally receive credit without a proper final response.
- The 49 requests that finished below the cap scored 30/49 exact (61.2%), but this is selection-biased toward easier/local questions and is not a replacement headline score.

### Higher-budget diagnostic

An 11-question, one-board diagnostic at 1,024 tokens produced:

| Metric | Result |
| --- | ---: |
| Exact accuracy | 6/11 (54.55%) |
| Component accuracy | 61.11% |
| Requests still capped at 1,024 | 4/11 |
| Cost | $0.36809 |

The four still-capped categories were `robber_resource_number`, `color_road_locations`, `port_trade_type`, and `port_occupancy`. A separate hard road-location request at low reasoning effort and 512 tokens also ended with `finish_reason=length` and no final content, so low effort alone did not remove the issue.

## Usage and latency for the 110-question run

| Metric | Result |
| --- | ---: |
| Prompt tokens | 84,724 |
| Completion tokens | 22,815 |
| Total tokens | 107,539 |
| OpenRouter-reported cost | $1.98799 |
| Mean request latency | 7.59 s |
| Median request latency | 7.83 s |
| p95 request latency | 9.95 s |
| Maximum request latency | 11.04 s |

Additional diagnostics cost $0.4154: $0.0142 for the one-question API smoke, $0.3681 for the 1,024-token 11-question smoke, and $0.0331 for the low-effort hard-question probe.

## Comparison with Qwen3-VL-32B

| Model | Exact | Component | Run cost | Median latency |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-VL-32B dense, FP8 | 15.45% | 14.68% | $0.0273 | 1.18 s |
| Claude Fable 5, proprietary | 30.00% | 32.11% | $1.9880 | 7.83 s |

Fable doubled exact accuracy under the same nominal 256-token cap and demonstrated perfect targeted tile reading, but cost about 73× as much for the run. This is not a parameter-scaling comparison because Fable's architecture and size are undisclosed. The mandatory-reasoning truncation also means the matched-cap comparison understates some capabilities while measuring a genuine operational failure mode.

## Verification

- 110 unique question IDs
- Exactly 10 responses per category
- Zero API errors
- Local artifact integrity assertions passed
- `tests/test_catan_bench.py` and `tests/test_catan_tokens.py`: 7 passed

## Artifacts

Main run:

- Plan: `data_pipeline/catanbench/datasets/catanbench_100/openrouter_eval/claude_fable_5_20260810/plan.json`
- Raw responses: `data_pipeline/catanbench/datasets/catanbench_100/openrouter_eval/claude_fable_5_20260810/responses.jsonl`
- Machine summary: `data_pipeline/catanbench/datasets/catanbench_100/openrouter_eval/claude_fable_5_20260810/summary.json`

Diagnostics:

- `data_pipeline/catanbench/datasets/catanbench_100/openrouter_eval/claude_fable_5_smoke_20260810/`
- `data_pipeline/catanbench/datasets/catanbench_100/openrouter_eval/claude_fable_5_budget1024_smoke_20260810/`
