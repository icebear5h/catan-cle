# Paired-game transcript-injection demo (_2n5F2DxtPI / replay 242781000)

## CORRECTION (user-caught): wrong narrator seat

The first demo assumed the narrator was bootymunchr (the search username). The user
corrected this: the narrator is FunDipDevRip (color 2). Transcript proof: verbal
commit "I think it's 8 4 10" [58.8] matches his wall-94 settlement
8-BRICK/4-SHEEP/10-WOOD; "we have a 3 1 and a brick port here" [67.7] matches that
corner's ports; "6 9 3 smart. Smart man. Maybe he's stream sniping." [186.4] is
OBSERVER commentary about bootymunchr taking 6-9-3 at wall 207.4.

Consequences:
- `demo_trace.json` / `demo_decision_packet.json` are INVALID (actor
  misattribution): real quotes, passing cutoff, wrong player's action supervised.
  See `demo_trace_status.json`. Kept as the canonical failure example.
- New mandatory gate: ACTOR ATTRIBUTION — narrator deliberation supervises only the
  narrator's seat; other-seat events get observer-commentary labels instead.
- The archived replay payload is the color-5 perspective (bootymunchr's view), NOT
  the narrator's seat. Placements are public so `demo2` is unaffected, but
  narrator-hand supervision needs a playerColor=2 refetch or public-only packets.
- Corrected artifacts: `demo2_decision_packet.json`, `demo2_trace.json` —
  FunDipDevRip's actual first placement (wall 94), window cut at 91 (guard band),
  $0.0089, gates: 15/15 quotes exact, cutoff PASS, actor PASS.

---

First end-to-end run of the paired lane: replay is the state oracle, transcript is
the reasoning voice, zero video tokens.

## Pipeline outputs

1. `transcript.json` — 659 caption segments, 39.8 min (DandyDrew narrating).
2. `replay_timeline.json` — 136 public actions with wall-clock times decoded from
   the archived replay via `replay_decoder.py`: 53 rolls, 36 roads, 22 settlements,
   21 robber moves, 4 cities.
3. Alignment: video time ≈ replay wall clock, offset ≈ 0s (video is uncut).
   Anchors: narrator's first settlement (wall 207.4 vs port talk at video 207.08),
   seven-roll panic (wall 905.5, "we said we'll roll a seven next turn"),
   endgame (last event 2366.2, outro begins ~2370).
4. `demo_decision_packet.json` — first-settlement decision: 19 authoritative hexes,
   pre-action public events, chosen action corner 26 = 9-SHEEP/6-BRICK/3-ORE.
5. `demo_transcript_window.txt` — pre-action commentary cut at 206.5s (action 207.4).
6. `demo_trace.json` — gpt-5.6-terra grounded trace: $0.0120, 16s, finish stop.

## Trace quality

- 4 real alternatives mined from speech with timestamps (8-4-10 port line,
  9-5-10/8-5-10 wood-brick, 6-3-11/6-4-11, rejected 8-4-3 "and then what?").
- Honest uncertainty: no explicit verbal commit before action; port locations and
  edge-35 direction absent from packet.
- Correct use of expert number-triple dialect ("6 9 3" = 6/9/3 corner).

## Deterministic gates

- Quote gate: 14/15 exact-in-window. The 1 miss: teacher presented a smoothed
  paraphrase as verbatim (removed stutter, elided a clause from the 8-5-10 quote).
  Faithful in meaning, but violates the quote/normalization separation — this is
  the exact failure class the critic pass must reject or relabel.
- Pre-action cutoff gate: PASS (no evidence at or after 206.5s).

## Cost profile (per decision)

~2.4k prompt + 1.5k completion tokens ≈ $0.012 at gpt-5.6-terra. A full game with
~30 supervised decisions ≈ $0.36 at mid-tier, ~$0.04 at deepseek-v4-pro.
No video tokens anywhere in the paired lane.

## Gaps before industrializing

1. Port geometry missing from packet (narrator discusses brick port/3:1 heavily;
   trace had to flag it as unavailable). Decoder has port edges; wire into packet.
2. Legal actions not enumerated (needs engine integration — the CLI decide mode).
3. Alignment here leaned on the uncut-video luck; cut videos need the align agent
   with piecewise offsets and per-segment confidence.
4. No critic pass yet: generator output must face quote-verbatim, fact-vs-packet,
   and leakage checks by a second model + deterministic validators.
5. Video pairing itself still unverified against frames (todo #6 acceptance checks).
