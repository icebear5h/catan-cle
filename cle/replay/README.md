# Replay domain

This package contains replay mechanics that previously lived under
`playground.game_viewer`:

- `colonist/`: event decoding, resources, and coordinate translation;
- `runtime/`: action matching, transactional stepping, checkpoints, trade ledger,
  audits, undo, and goto behavior.

The old viewer modules are compatibility aliases only. Core replay modules do
not import Flask, Socket.IO, frontend code, provider clients, or viewer state.
`cle.sandbox.replay.ReplaySandbox` coordinates this runtime through a duck-typed
state seam while existing `ServerState` consumers migrate.
