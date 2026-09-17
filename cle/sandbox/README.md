# Catan sandbox

`CatanSandbox` is a lightweight asynchronous composition root:

```text
CatanSandbox
|- game_engine: GameEngine
`- players: dict[Color, SandboxPlayer]
```

`GameEngine` owns deterministic rules, materialized state, RNG, strict action
application, canonical events, visibility projection, trade windows, and explicit
snapshots. Players own provider/session state. The sandbox only orchestrates
context creation, bounded decision retries, barrier-synchronized responses,
deterministic application order, communication rounds, and acknowledgements.

New live games use `../harness/suites/shared_v1.yaml`: shared prompt component
definitions with independently composed decision/speech requests and fresh
context plus private notes. A normal decision selects a game tool or a standalone
public `say`, with bounded addressed reactions before the pending action resumes.
Game tools use named resource counts, semantic offer terms,
and literal trained location tokens such as `<N00>`, `<E00_01>`, and `<T05>`.
The model no longer selects numeric menu indices. Internally, the harness still
resolves calls to frozen canonical indexed choices and the engine validates them.

`play_knight` includes its robber destination. The sandbox preflights both
canonical actions, then applies Knight and movement synchronously in one step
with one accepted player receipt and two events. Immediate victory after Knight
suppresses movement. A victim choice remains a subsequent decision; only the
engine chooses the stolen resource. Invalid arguments cannot consume the Knight.

### Queued deterministic builds and conversions

The shared/fresh batch contract accepts 1–4 semantic `actions` instead of the
single `tool`/`arguments` fields, with one optional notes string. Allowed tools:
`build_settlement`, `build_road`, `upgrade_city`, `maritime_trade`, and terminal
`end_turn`. The [harness guide](../harness/README.md#deterministic-action-batches)
defines syntax and the strict opt-in version boundary.

Each `step()` commits **one** action. Subsequent queued steps regenerate legality
and resolve semantic arguments against the updated engine without calling the
model, refreshing inference policy, accepting another receipt or moving delivery
cursors. Thus a road can unlock the requested settlement, or a conversion can
fund the requested city. The same Step/Auto-play loop gets one navigable checkpoint
per action and one actual model call for the plan.

`SandboxSnapshot.pending_action_batch` holds detached semantic calls, next index,
actor/phase/turn, expected revision, and exact originating call/event identity.
Consumption and engine commit are synchronous. Save/load and post-commit
cancellation preserve the next unconsumed action; historical snapshots default to
no queue. Prompt rebinding cannot rewrite an admitted plan.

Full syntax/notes validation precedes the first commit. Later illegality clears
the rest, preserves all earlier actions/costs, and emits private
`ACTION_BATCH_PAUSED` feedback before fresh inference. No fallback or replay of
the prefix occurs. Setup stops after one settlement-road pair even at the snake
reversal. New events/speech, actor/turn/phase changes, trade/speech barriers and
victory also stop continuation. All randomness and player trades remain standalone.
Existing legacy speech policies still apply in mixed-policy games; any emitted
speech invalidates the remaining queue before the next action.

Other Catan rules and trade lifecycle remain unchanged. Invalid single calls receive
corrective feedback while configured attempts remain, without a gameplay penalty
or automatic fallback. Exhaustion leaves the current decision unapplied (a batch's
already committed prefix remains); the live factory
defaults to one attempt. Saved games retain their recorded suite source/hash;
v10 keeps its original XML/indexed response contract. No historical YAML, local
overrides, or engine replay actions are migrated. See the
[`harness guide`](../harness/README.md) for the shared and historical contracts.

Normal play is:

```python
result = await sandbox.step()
```

The live viewer is only an adapter around this operation: it owns one sandbox,
renders projected state, and its live `Step` button awaits one complete call.
It does not maintain a second observation/action protocol. Each completed live
step is committed to the local SQLite trace store described in
[`../traces/README.md`](../traces/README.md) before the HTTP response returns.

A barrier step freezes one causal cutoff, prompts required players concurrently,
waits for all responses, and applies them in table order. Model latency never
changes game semantics. In a trade barrier, opponents may independently signal
willingness or decline; these responses never execute a trade. Control returns
to the turn player, who may select one exact `TradeCandidate` or ignore every
candidate. Counteroffers are addressed only to the turn player. One cooperative
`SandboxPool` can run many games while vLLM continuously batches their shared
asynchronous transport requests.

Shared offers can optionally carry a narrow one-shot proposer preauthorization
(`confirm_if_accepted_by`, ordered players or `"ANY"`). The normal offer step
admits it; the normal simultaneous response step seals its bounded window. The
next `step()` either executes an ordinary strict confirmation without inference,
or consumes the instruction with proposer-private pause feedback and resumes
model control. Any arriving counteroffer pauses even if the original was accepted.
No fallback partner is selected after the preferred willing partner fails fresh
legality/funding checks. See the full
[interface semantics](../harness/README.md#one-shot-trade-preauthorization).

`SandboxSnapshot.trade_preauthorization` persists the immutable original terms,
priority, window/round, response cutoff and originating accepted-call reference.
Historical snapshots default to no authorization. Engine-only confirmation has
empty `contexts`/`attempts` and an `automatic_action` provenance record; it neither
calls `player.accept()` nor commits notes again. It advances the normal engine
action/event sequence and gets its own normal live-step checkpoint.

The engine stores complete canonical events. Fresh requests receive newly visible
events and messages through a frozen input cutoff, current state, and accepted
notes. Action and speech delivery cursors are separate from each other and from
reaction scheduling. Historical suites retain cumulative game events and windowed
talk. Raw private overlays never enter player or viewer projections.

Notes are committed only with accepted actions/speech, including valid pass.
Standalone speech followed by a failed decision is checkpointed with its pending
decision, reaction queue and remaining budget; retry does not repeat admitted
speech. Routine actor/post-event polling is disabled for the reactive contract.
Other players get one bounded pre-robber window after a seven's discards, and
addressed public speech can request bounded replies. Historical post-action speech
failure cannot undo gameplay or notes. Historical post-action cancellation remains an
`asyncio.CancelledError` carrying the committed result; the synchronous viewer
preserves that result before its thread bridge and records an applied-step warning.

Live/headless engines do not copy full state before every action. Full staging
is limited to trade barriers and bundled Knight preflight. Snapshotting is
explicit; replay/debug engines may opt into history capture.
