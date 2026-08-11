# Qwen3.8 Max Catan Video Parsing Pilot

## Source

- Video: `https://www.youtube.com/watch?v=9wtla7jz_pw`
- Title: `Catan Pro Plays Coordinated Ore Brick Setup In Ranked`
- Channel: `DandyDrew`
- Caption duration: 1,482.36 seconds
- Caption segments: 451
- Model: `qwen/qwen3.8-max` through OpenRouter

## Experiment

The first pass sent the complete 24.7-minute video at 426x240 plus the timestamped YouTube captions. The prompt asked for at most 15 evidence-linked, high-signal Catan decisions and required separate visual facts, transcript evidence, alternatives, chosen action, reasoning, uncertainty, and confidence.

Direct YouTube transport did not work with the Alibaba endpoint:

1. The YouTube page URL failed because it did not expose a multimedia `Content-Length` to the provider.
2. An ephemeral Google video CDN URL had `Content-Length` locally but the provider could not download it.
3. A base64 854x480 full-video request exceeded the provider request-body limit.
4. A base64 426x240 full-video request succeeded.

For production, use stable object storage with a provider-fetchable URL and `Content-Length`, or use a low-resolution discovery pass followed by short high-resolution clips.

## Whole-video result

- Complete decision records salvaged: 14
- Prompt tokens: 166,724
- Completion tokens: 11,998
- Reported cost: $0.405436
- Latency: 314.0 seconds
- Finish reason: `length`
- The response began a fifteenth record but hit the 12,000-token completion limit. `coarse_decision_proposals.json` retains only the 14 complete objects, marks them training-ineligible, and records the salvage provenance.

The 14 proposals cover opening board analysis, first and second placements, robber choices, development-card timing, Road Building, Monopoly, threat reassessment, port conversion, and the winning settlement.

## Deterministic transcript checks

`validation_report.json` checks each quoted evidence span against captions overlapping the claimed timestamp.

- Internal timestamp ordering valid: 14/14 records
- Caption quotes: 30
- Exact normalized quote-in-window matches: 28/30
- Mean best local fuzzy match: 0.988
- Both non-exact matches occur in `d012`; one has a weak fuzzy score of 0.695.

These checks show that most quotations were copied from nearby captions. They do not validate the true action time, board state, or strategic correctness.

## Manual visual audit

Two initial-placement proposals were checked against full-resolution source frames.

### First settlement

The coarse model proposed RED's first settlement as 6-9-3. My frame interpretation appears consistent with that proposal, but it is not authoritative proof. Its time boundary was also too broad:

- Settlement absent at 150 seconds and present at 151 seconds.
- Adjoining road absent at 152 seconds and present at 153 seconds.
- Commentary at or after 150 seconds overlaps or follows the visible action interval and was incorrectly treated as pre-action reasoning.

Result: provisional hypothesis only, with rationale evidence conservatively cut before 150 seconds.

### Second settlement

The coarse model emitted `6-5-12`, following the noisy captions. My frame interpretation suggests the following, but this remains unproven without a replay/log or independently reviewed board mapping:

- The blocking opponent is paladyn019 (dark/grey) near 8-3-4, not ORANGE near 9-4-11.
- RED's second settlement is absent at 233 seconds and present at 234 seconds.
- The visible settlement is 6-11-12: two BRICK hexes and one WHEAT hex.
- The two BRICK and one WHEAT starting-resource cards are visible.
- The road appears by 235 seconds, heading east/southeast toward the coast/port route.

A working hypothesis is that the repeated caption phrase `6 5 12` is a Catan-number ASR error and that the visible target is 6-11-12. This is corroborated by the apparent resource-card animation, but neither my reading nor another model's agreement constitutes proof.

A separate 14-second objective microclip pass recovered the 6-11-12 brick/brick/wheat action, but returned impossible absolute timestamps. The microclip had been stream-copied at a non-keyframe boundary, so future clipping must accurately decode/re-encode or preserve actual source PTS.

Provisional observations are in `manual_audit_initial_placement.json` and `provisional_visual_hypotheses.json`; both are explicitly training-ineligible pending authoritative verification.

## Cost of follow-up diagnostics

- Whole-video discovery: $0.405436
- High-resolution proposal-verification clip: $0.145982; unusable JSON because mandatory reasoning consumed 5,465 of the 5,998 completion tokens and the visible answer truncated
- Objective second-placement microclip: $0.028306
- Total successful-call cost: $0.579724

## Conclusion

Qwen3.8 Max is promising as a **candidate discovery and temporal alignment model**. One whole-video call is not trustworthy enough to emit training labels directly. It produced highly plausible reasoning while:

- potentially following a noisy ASR number triple over board evidence,
- producing an opponent/location claim that conflicts with my provisional frame reading,
- including post-action speech in pre-action reasoning, and
- overestimating its own confidence.

The next parser should use four bounded stages:

1. Low-resolution whole-video pass for candidate intervals only.
2. Accurate high-resolution microclips for objective pre/action/post visual deltas.
3. Caption normalization conditioned on the verified visual action, preserving raw ASR separately.
4. Reasoning extraction restricted to evidence before the verified action timestamp, followed by an independent reject-capable critic.

This pilot has no transcript-only accuracy baseline and does not prove that the proposed board readings are correct. It only establishes that the video model should remain a proposal generator and that authoritative or independently reproducible verification is still required.
