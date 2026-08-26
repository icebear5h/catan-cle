# Game engine

`game_engine/` contains deterministic Catan rules only. It has no player-policy,
provider, sandbox, viewer, Flask, or Playwright dependencies.

```python
from game_engine import Color, GameEngine

engine = GameEngine(
    [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE],
    seed=7,
    shuffle_players=False,
)
transition = engine.step(engine.state.playable_actions[0])
```

`GameEngine` owns mutable `GameState`, its per-engine RNG, a canonical append-only
`GameEvent` log, visibility projection, bounded trade-window state, social
message events, and explicit snapshots. `step()` accepts one already-selected
legal action and returns an `EngineTransition`; it never invokes a player.

Live/headless engines default to no automatic undo copies. Replay/debug callers
may enable `capture_history=True` while replay navigation migrates fully to
checkpoints and recorded events.

The canonical event is projected before a player or viewer receives it:

```python
visible = engine.project_events(Color.BLUE)
observation = engine.observe(Color.BLUE)
```

Development-card identity, stolen resources, discards, private messages, and
recipient-specific overlays are filtered by game-engine rules. Raw overlays must
never enter model prompts.

Trade negotiation is represented by one turn-scoped `TradeWindow` containing
multiple root `TradeOffer` values, counteroffers, responses, and executable
candidates. `TradeLimits` bounds active offers, operations, and rounds. A root
offer may have counteroffers, but a counteroffer cannot itself be countered.
Each `TradeOffer` contains who offered, its audience, named `give`/`receive`
resources, optional parent offer, responses, and lifecycle status. Positional
10/12-value tuples are decoded only at the historical replay boundary. Only the
selected bilateral exchange mutates resources. Opponent `ACCEPT_TRADE`
responses are willingness signals, not executions: multiple opponents may be
willing, but only the turn player can select one typed `TradeCandidate`, and the
turn player may choose not to trade. Counteroffers are directed only to that
turn player.

The async player/game loop belongs to `cle.sandbox.CatanSandbox`, not this
package.

Embargoes are not part of the default engine action space. A future optional
embargo feature belongs to bounded public communication/commitment state and
must remain disabled unless explicitly configured.
