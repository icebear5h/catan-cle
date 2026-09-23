# Self-play latency: human pace vs served Qwen3.8-27B

Date: 2026-09-21. Policy: stock Qwen3.8-27B (`cle/sandbox/factory.py:51`).
Scope is latency only.

## Human pace (measured, n=65)

Per-game `data.eventHistory.endGameState.gameDurationInMS /
totalTurnCount` over `artifacts/raw/colonist/replays/*.json`
(1 file skipped: sample fixture without `endGameState`).

| stat | per player-turn | full game | turns/game |
|---|---|---|---|
| mean | 24.9s | 29.8 min | 72 |
| median | 23.4s | 28.0 min | 72 |
| stdev / range | 7.2s / 5.9s - 56.9s | - | - |

Human bar: ~25s/turn, ~5-6s per decision (a turn averages ~4-5 model
steps; full games run ~300-450 inference calls across all 4 seats).
Includes Colonist UI/animation overhead; setup placements are excluded
from `totalTurnCount`.

## Per-decision latency inputs

Runtime default board surface is text (`indexed_tile_rows`), not image.

- Prefill: ~2-5k text tokens + up to ~1k notes echo.
- Decode (think off): ~50-100 tokens action/speech.
- Decode (think on, default `high`, uncapped `max_tokens`): native
  reasoning dominates; up to ~8.9k reasoning tokens observed in one call.
- Prefix caching across a seat's persistent session makes repeat prefills
  near-free after the first.

## Single-stream TPS (roofline, decode-bound)

27B dense: FP8 ~27GB weights, NVFP4 ~15GB. H200 4.8 TB/s, B200 8 TB/s,
derated to ~65% (typical vLLM single-stream). KV is not the binding
constraint: 4 seats x 32k ctx x 32.8 KB/tok (FP8) ~= 4GB.

| config | decode TPS | prefill ~4k in | per decision, think off |
|---|---|---|---|
| FP8, 1xH200 | ~115 tok/s | ~0.3s | ~1.0s |
| FP8, 1xB200 | ~190 tok/s | ~0.15s | ~0.55s |
| NVFP4, 1xB200 | ~300 tok/s | ~0.1s | ~0.35s |
| NVFP4, 4xB200 DP=4 (1 seat/GPU) | ~300/seat, ~1200 agg | same | same, x4 concurrent seats |

No self-hosted policy TPS has been measured yet; these are estimates to
beat against (see verification below).

## Full-game extrapolation (~350 calls, mostly sequential)

| config | think off | think on (`high`) |
|---|---|---|
| FP8 1xH200 | ~6 min/game, ~5s/turn (~5x human pace) | ~2-4 hr (20-80s/call) |
| NVFP4 4xB200 DP4 | ~2-3 min/game | ~1 hr+ (7-30s/call) |

Quantization/hardware is a ~3-4x lever; the reasoning level is a ~50x
lever. Think-off on a single H200 already plays ~5x faster than humans.
Think-on-high exceeds human pace on every config here; human-speed
self-play with reasoning needs a tight reasoning cap (hundreds of
tokens), not more GPUs.

## Verification / limitations

- Turn stats are measured from the current raw corpus; rerun after the
  staging backlog is promoted.
- TPS rows are unvalidated estimates. Validate with
  `scripts/time_live_game.py` (per-call and per-wall-time tok/s) against
  a live vLLM FP8 1xH200 run before sizing any B200 work.
- Call counts assume ~1 inference/step minus free rolls and deterministic
  continuations, plus speech reactions and up to 3x retries
  (`max_decision_attempts`, `factory.py:53`).
