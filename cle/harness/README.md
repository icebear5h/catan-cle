# Catan agent-player harness

`AgentPlayer` owns one player's model-visible continuity and uses the harness
package for prompt assembly, parsing, validated YAML suites, and shared async
provider transports.

Each player owns:

- complete accepted user/assistant trajectory;
- strategic memory;
- reaction/event cursor;
- stable game/player session ID;
- accepted choice receipts.

`CatanSandbox` supplies a perspective-safe `PlayerContext`. `AgentPlayer` renders
it with `suites/catan_v9.yaml`, calls its shared transport, and returns a typed
`PlayerAttempt`. Only an engine-accepted attempt enters durable conversation.
Malformed or out-of-menu attempts remain in the sandbox decision trace and are
retried with corrective feedback.

The v9 suite combines v8's authoritative two-round placement order with the
strategy-guided first-settlement overlay validated in the initial-placement
trace evaluation. Older versions remain unchanged for reproducible prompts.

The v7 suite introduced each formatter field as a typed `PromptComponent`. Stable
identity is the sole system component; strategic memory, visible events, phase,
board, resources, opponents, trades, legal actions, phase facts, decision
request, and response schema are separate environment components. Providers
receive their deterministic join under the portable chat `user` role. Older
suite versions remain loadable as reproducible prompt baselines.

The physical board is a separate typed `BoardPresentation` attached only to the
current user turn. New games default to the complete, lossless
`indexed_tile_rows/v3` text projection used by the strict board eval; an
ordinary unannotated PNG is the alternative image projection. Both originate
from an immutable canonical snapshot of public board facts, carry state and
content hashes plus identifier-space/renderer provenance, and never include
hidden hands. Provider adapters encode the selected projection. Image bytes and
base64 data URLs are excluded from durable traces, which retain only bounded
metadata and hashes; historical games recorded before this contract restore in
legacy semantic-board mode.

Communication uses `suites/communication_v5.yaml` with the same component model.
Every call receives complete visible game events, a bounded recent-message
window, and exact unresolved non-binding commitments. `SILENCE` is the default
and never enters the game log. Catan Lab may save validated static string-only
overrides under `.cle/prompt_suites/`; active games retain their recorded suite
sources and hashes.

OpenRouter, Groq, and vLLM transports are asynchronous and provider-independent.
The complete active context is sent every time. Session affinity and provider KV
caching are optional optimizations, never continuity storage.

The decision response contains only durable strategic memory, the selected
legal-action index, and conditional structured trade terms. It never asks the
model to author a rationale. Reasoning is stored only from a distinct provider
response channel (`reasoning`, `reasoning_content`, or `reasoning_details`) or
explicit provider token evidence. The current live factory sends configurable
native-reasoning requests through OpenRouter and rejects enabled requests on
transports that do not implement that channel instead of silently faking it.
