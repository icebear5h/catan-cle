# Catan correctness audit: 2026-09-08

## Verdict and scope

The current runtime does not pass an independent correctness gate. Inventory
conservation and the existing replay audit pass, but rule, causal-history,
parser, and mutable-state boundary checks fail.

This pass adds verification only. No production logic, game rules, prompt or
action schemas were changed. No hosted model calls, live-game actions, server
restart, or writes to the real `.cle` database were performed.

The new suite preserves **31 reproduced correctness cases and four separately
labeled custom-player robustness proposals**. Cases are not distinct bugs:
several exercise different surfaces of the same underlying defect. They are
strict expected failures, not successful correctness checks.

## Coverage and results

| Check | Result |
| --- | --- |
| Existing targeted engine/harness/replay/trace tests | 395 passed |
| Full local replay corpus, run once | 66/66 replays passed |
| Corpus actions / trade-lifecycle actions | 31,506 / 15,460 |
| New combined audit, full-game seeds 0-3 enabled | 10 passed, 35 xfailed |
| New audit without full-game sweep | 6 passed, 35 xfailed |
| Unexpected audit failures or passes | None |
| Audit Ruff and independent reproduction review | Passed |

The targeted command excludes the opt-in corpus test by default, reporting
one skip. The separate complete corpus run included it and finished in
170.34 seconds, with no fatal semantic issues or asserted hand/trade mismatch.

Before preserving the regressions, isolated exploratory sweeps covered 288
engine games and six sandbox games. The engine sweeps applied 154,257
transitions. Sandbox sweeps applied 4,522 transitions; two used `AgentPlayer`
with test-only local responses. Another check applied all 15,560 concrete
menu options encountered on six trajectories to isolated copies.

These runs reached engine-declared winners. This does not certify legitimate
Catan winners: independent checks found incorrect scoring and termination.
The exploratory policy is not a production baseline or evidence of strategy
quality. The checked-in opt-in runner preserves the inventory/continuity
checks, not an assertion that it recreates every exploratory trajectory.

Conservation covered all finite supplies: 19 of each resource, the exact
25-card development mix, and per-seat 15 roads, five settlements, four cities.
Checks also compared board occupancy with piece caches, nonnegative counts,
development-card timing, legal selection, and accepted per-seat continuity.

## Priority 1: rule and outcome correctness

### Roads and Longest Road

- Road generation and strict execution allow extending through an opponent's
  settlement. Natural seed-1 evidence: WHITE extends from `(14, 15)` through
  ORANGE's settlement at 15 to `(15, 17)`, revision 361 to 362.
- Cutting a road can award Longest Road below five pieces. Natural seed-21
  evidence: a settlement grants an unqualified three-road player the award.
- Cutting a road can transfer the award on a tie instead of preserving its
  incumbent. Natural seed-203 evidence: incumbent and challenger both have six.
- Initial-road application fails to update the observation's road-length
  cache, although the board has those roads.

Sources: `cle/game_engine/models/board.py:136-145,248-268` and
`cle/game_engine/state_functions.py:24`.

The compact tests use reduced inventory-conserving positions, not complete
reachable-game histories. The natural-game evidence above was independently
observed during exploration. Revocation of an already-held award when all
players drop below five is an additional coverage gap, not a separate verified
case in this pass.

### Victory and terminal state

`winning_color()` scans all seats rather than the turn owner. A legitimate
Longest Road transfer can therefore declare an off-turn winner. Natural
seed-247 evidence: ORANGE reaches 10 during WHITE's settlement action and the
sandbox terminates. This does not depend on the minimum/tie bugs.

After victory, the outer sandbox view is terminal, but its nested observation
and decision context still expose actions. Direct strict engine stepping can
apply another action. The normal sandbox step correctly rejects it.

Source: `cle/game_engine/game.py:337-352` and terminal observation/context
boundaries in `cle/sandbox/catan.py` and `cle/sandbox/decision.py`.

### Resource production and discarding

The sole recipient of a depleted resource receives zero instead of the
remaining bank cards. The existing supply test currently asserts the wrong
result. Multiple-recipient shortages must remain a separate case.

Discard continuation uses hardcoded `> 7` after the first discarder, ignoring
configured thresholds such as five or ten. Separately, the current action
contract only offers random discarding; standard player-selected discards
require an explicit contract/capability change, not merely a numeric fix.

Sources: `cle/game_engine/state.py:298,709`,
`cle/game_engine/models/actions.py:332`, and
`tests/test_game_supply_limits.py:346`.

Rule reference checked directly: [official CATAN base-game FAQ][rules],
questions 27-29, 33-34, 40, and 74-76. These support the Longest Road,
sole-recipient shortage, blocked-road, and own-turn victory expectations.
Nondefault discard thresholds test configured variant consistency.

## Priority 2: causal player history

- Replay undo restores material state but leaves future engine events visible
  to the next decision. Re-stepping duplicates events; RNG becomes detached
  from the restored state. Source: `cle/replay/runtime/checkpoint.py:46-51`.
- Manual replay confirmation changes hands/actions without a shared canonical
  event, so the policy misses the cause of the exchange. Source:
  `cle/replay/runtime/step_executor.py:466`.
- Backward replay goto constructs a newly randomized engine, changing identity,
  seed, RNG and hidden deck order rather than restoring the original boundary.
  Source: `cle/replay/runtime/navigation.py:127`.
- Private table talk and commitments visible to a communication call disappear
  from the immediately following action decision. Decisions only consume
  `project_game_events()`, and talk is not retained in the decision transcript.
  Sources: `cle/sandbox/decision.py:32-40`, `cle/players/agent.py:102-120`.
- Normalized SQLite `game_events` omits speech. Speech remains in step JSON and
  snapshots, so this is incomplete indexing, not complete data loss. Source:
  `cle/traces/sqlite.py`, `record_step()` event insertion.

The passing local corpus audit validates resource snapshots, trade identities,
responses and completion. It permits warning-level force paths and final-score
sync. It does not establish undo/event-history/RNG equivalence or validate the
live game rules above.

## Priority 3: parser and state boundaries

- A commented-out private message can be published instead of the actual
  `SILENCE` response. An action inside an unclosed plan can execute. Whitespace
  in action tags can hide conflicts; duplicate trade fields select the first.
  Sources: `cle/harness/communication.py` and `cle/harness/context.py`.
- Player observations expose writable live trade-window/building objects.
  Engine branches and snapshots share mutable event payloads. A frozen outer
  dataclass is not a deep isolation boundary. Sources:
  `cle/game_engine/observation.py:141-143`, `cle/game_engine/events.py:107`,
  `cle/game_engine/game.py:296-323,354-367`.
- Mutating a returned counteroffer while another responder is pending can
  bypass the funding preflight and produce a partial barrier commit.
  Context/window mutations may not increment revision. Source:
  `cle/sandbox/catan.py`, `_step_barrier()`.
- An in-process typed player can supply an invalid commitment expiry that
  poisons the next step after engine mutation. A typed counteroffer can target
  a different parent from its selected menu entry. These are extension/player
  boundary cases; they are not claimed to originate from the ordinary XML
  parser. Source: `cle/sandbox/catan.py`.
- Sequential executed trades in one turn reuse the same offer ID when the
  window is recreated. Source: `cle/game_engine/state.py:436`.
- Injecting a shared transport into `mode="random"` silently installs one
  agent and invokes it. Source: `cle/sandbox/factory.py:123-154`.
- A reasoning-off vLLM game starts but cannot resume because the loader forces
  high reasoning, which that provider rejects. Source:
  `playground/game_viewer/routes/live_game.py`, `_load_live_trace_transaction()`.

Four additional tests explicitly propose cleaner rejection or retry handling
for a custom player returning `None`, a dict choice/offer, or a wrongly typed
bundle. Those violate the `SandboxPlayer` protocol. They are robustness-policy
proposals, not four demonstrated malformed-LLM-output bugs or an established
requirement that programmer errors must be retried.

## Passing controls and limits

- Pickled engine/player snapshots reproduced a 48-step stochastic suffix:
  23 rolls, 22 end turns, three discards, with identical RNG and state.
- Fresh player/session reconstruction preserved the next assembled request.
- Offline OpenRouter/vLLM probes preserved concurrent session ownership,
  borrowed-client lifetime, cancellation cleanup, and bounded retry behavior.
- Normal sandbox terminal guarding and default discard threshold controls pass.

Provider checks use mocked HTTP transport, not real-network deadline or billing
tests. Production disk/WAL crash durability and the actual `.cle` migration were
not retested here. The fixed-frontier/limited seed sweeps are not exhaustive
state-space checks. The baseline `FirstLegalPlayer` also stalled for 1,000 steps
on two exploratory seeds; that is a baseline limitation, not proof that the
engine deadlocks.

## Reproduction

From the repository root, run the preserved failures and passing controls:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -B -m pytest -p no:cacheprovider \
  -p pytest_asyncio.plugin -q -rx tests/audits
```

All expected failures require a dedicated defect exception raised only after
ordinary precondition checks. To expose the failures as a red diagnostic gate,
add `--runxfail`; do not interpret the expected-failure summary as correctness.

Enable a bounded full-game inventory/continuity sweep:

```bash
CATAN_FULL_GAME_AUDIT=1 CATAN_AUDIT_SEEDS=4 PYTHONHASHSEED=0 \
  PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -B -m pytest -p no:cacheprovider \
  -p pytest_asyncio.plugin -q -s -rx tests/audits
```

Shared audit guards block actual network connections and non-test SQLite
access, including attempted calls caught by application code. Mock HTTP and
pytest temporary databases remain allowed. Prompt sources are explicitly
read-only. Tests live in:

- `tests/audits/test_engine_rule_audit.py`
- `tests/audits/test_replay_checkpoint_audit.py`
- `tests/audits/test_harness_boundary_audit.py`

## Recommended fix order

1. Road legality, award selection/VP maintenance, and turn-owned termination.
2. Complete replay checkpoints/events and deeply isolated player inputs.
3. Structural control-field parsing and whole-barrier admission revalidation.
4. Perspective-safe action-visible talk/commitments, with an agreed context
   contract; player-selected discard with an agreed action contract.
5. Payout/threshold edge cases, unique offer IDs, provider-safe resume, and
   complete speech indexing; decide custom-player rejection policy separately.

Do not treat engine outcomes as trustworthy training labels until the rule and
causal-history gates pass. Convert each strict xfail to a normal regression
when its corresponding fix is implemented and verified.

[rules]: https://www.catan.com/faq/basegame
