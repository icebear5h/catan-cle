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
it with `suites/catan_v5.yaml`, calls its shared transport, and returns a typed
`PlayerAttempt`. Only an engine-accepted attempt enters durable conversation.
Malformed or out-of-menu attempts remain in the sandbox decision trace and are
retried with corrective feedback.

The v5 suite states only the objective, exact response contract, and
current-decision facts that are not already carried by the observation or action
menu. Strategy remains model-owned; v4 is retained as a reproducible prompt
baseline.

Communication uses `suites/communication_v2.yaml`. Every call receives complete
visible game events, a bounded recent-message window, and exact unresolved
non-binding commitments. `SILENCE` is the default and never enters the game log.

OpenRouter, Groq, and vLLM transports are asynchronous and provider-independent.
The complete active context is sent every time. Session affinity and provider KV
caching are optional optimizations, never continuity storage.

The suite's `<rationale>` is ordinary model-authored explanation text. It is not
native reasoning. Provider-native reasoning is stored only from a distinct
response channel (`reasoning`, `reasoning_content`, or `reasoning_details`) or
explicit provider token evidence. The current live factory sends configurable
native-reasoning requests through OpenRouter and rejects enabled requests on
transports that do not implement that channel instead of silently faking it.
