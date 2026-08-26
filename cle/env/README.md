# Observation compatibility module

`cle/env/` no longer defines the game environment. The active runtime is
`cle.sandbox.CatanSandbox`, which composes one `game_engine.GameEngine` with
one `SandboxPlayer` per seat.

The remaining `observation_formatter.py` module is a presentation adapter used
by the harness, replay tooling, evaluations, and viewer. Rules, privacy,
legal-action identity, and player decisions remain outside this module.

Players do not construct actions or submit free-form action JSON. Every
`DecisionContext` contains an ordered legal-action menu, and a player returns
the zero-based index of one exact entry. `CatanSandbox` validates that choice
against the current engine state before applying it.

See:

- [`../sandbox/README.md`](../sandbox/README.md) for orchestration and stepping
- [`../harness/README.md`](../harness/README.md) for model context and prompts
- [`../../docs/engine/ENGINE_DOCS.md`](../../docs/engine/ENGINE_DOCS.md) for
  deterministic rules and state
