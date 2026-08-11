# Whole-video discovery pass: model comparison (same prompt, same 240p video, same captions)

Task: identical to the original pilot request (`request.json`), model swapped only.
Known anchors from provisional manual audit: first settlement 6-9-3, action ~150-151s;
second settlement 6-11-12, action ~233-234s, captions wrongly say "6 5 12" (ASR trap);
blocking opponent is paladyn019 (dark/grey) near 8-3-4.

## Results

| Model | Decisions | Quote grounding | 2nd-settlement claim | Latency | Prompt tok | Cost |
|---|---|---|---|---|---|---|
| qwen/qwen3.8-max (pilot) | 14 (truncated at 12k) | 28/30 exact, 0.988 | 6-5-12 (ASR trap) | 314s | 166,724 | $0.4054 |
| google/gemini-3.6-flash (default reasoning) | truncated: 11.5k tok thinking, 1.2k chars answer | n/a | n/a | 58s | 110,630 | $0.2548 |
| google/gemini-3.6-flash (reasoning minimal) | 11, finish stop | 18/20 exact, 0.913 | 6-5-12 (ASR trap) | 36s | 110,630 | $0.2170 |
| qwen/qwen3.7-flash | video NOT ingested (6,383 prompt tok), finish error | n/a | caption regurgitation | 199s | 6,383 | $0 |
| google/gemini-3.6-flash (reasoning minimal, 480p) | 11, finish stop, but 6 records have timestamps PAST the 1480s video end (~1.66x drift); second settlement missing | not audited | n/a (record missing) | 40s | 110,630 | $0.2151 |

## Key findings

1. Cross-family confirmation of the ASR trap: both qwen3.8-max and gemini-3.6-flash
   emitted "6-5-12" for the second settlement, following captions over pixels. The
   lossy pass cannot be trusted for coordinates regardless of vendor. Role B + replay
   grounding remains mandatory.
2. Both models also mark nearly every action `visual_confirmation: confirmed`
   (gemini: 11/11). Model-claimed confirmation is not confirmation (existing lesson).
3. Gemini also misattributed the actor in d002 visual evidence ("Blue settlement...
   on 6-5-12" during the narrator RED placement).
4. gemini-3.6-flash matches the discovery-pass quality of qwen3.8-max at ~half the
   cost and ~9x the speed, with a more compact video tokenizer (110k vs 167k prompt
   tokens for the same file). New default for Role A.
5. Gemini reasoning models must be run with `reasoning: {effort: minimal}` for this
   task or thinking consumes the entire completion budget (same failure mode as the
   Fable CatanBench run).
6. qwen3.7-flash silently dropped the 4.9MB base64 video (prompt shows only caption
   text tokens), returned `finish_reason: error` after 199s, and produced fluent
   caption-only confabulation in the requested schema. Cheap-tier video support must
   be verified by checking prompt token counts before trusting any output.

## Resolution test (480p vs 240p)

Gemini accepted the 16.3MB base64 480p file and produced IDENTICAL prompt tokens
(110,630): it normalizes video internally, so higher input resolution buys no
additional pixels seen by the model via this path. The 480p run was also worse:
only one of two placement records, and impossible timestamps beyond video end
(d006 t=1495 .. d011 t=2309 on a 1480s video; both files verified 24:40.40 30fps).
Conclusions: (a) 240p remains the Role A transport default; (b) run-to-run temporal
reliability is not guaranteed — add a deterministic gate rejecting records with
timestamps outside [0, duration]; (c) resolution belongs to microclips and frames
(verification lane), not the whole-video pass.

## Timestamps vs anchors

- gemini d001 (first settlement): decision_s=137, end_s=150 — pre-action window, ends
  at the true click (~150.5s). Reasonable.
- gemini d002 (second settlement): decision_s=228, end_s=235 — brackets the true
  action (~233-234s). Reasonable.
- Wrong coordinates in both models despite good time windows: alignment good,
  perception unreliable.

## Verdict

Role A default: `google/gemini-3.6-flash` with minimal reasoning effort ($0.22/video,
36s). Premium fallback: `qwen/qwen3.8-max`. Test next when needed: `gemini-3.1-flash-lite`
(cheaper tier), `kimi-k3` (dual video+reasoning). Never trust Role A coordinates;
always verify video ingestion via prompt token count.
