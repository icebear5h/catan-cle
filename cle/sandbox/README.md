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

The engine stores complete canonical events. A player receives every complete
game event visible from its perspective; only table-talk messages use a bounded
recent window. Raw private overlays never enter player or viewer projections.

Live/headless engines do not copy full state before every action. Snapshotting is
explicit. Replay/debug engines may opt into history capture while navigation is
migrated to checkpoints and recorded events.
