# Siphoning DeepSeek Elastic Compute (DSec) for Catan RL

Date: 2026-09-22. Status: research synthesis; per-sandbox journal implemented
as an opt-in rollout API (see section 4). Other actions remain proposals.

DSec is DeepSeek's sandbox control plane for agentic RL, described in V4
paper §5.2.5 and in the standalone 31-page DSec paper (arXiv cs.DC,
19 Sep 2026). Three Rust components — Apiserver (gateway), Edge
(per-host agent), Watcher (cluster monitor) — over a custom RPC protocol
atop 3FS, running hundreds of thousands of concurrent sandboxes in one
cluster. Four execution substrates (function calls, containers, Firecracker
microVMs, full QEMU VMs) behind one Python SDK.

Our "sandbox" is the Catan engine: deterministic, in-process, CPU-cheap.
That makes most of DSec unnecessary for us — and makes the transferable
parts easy to see, because every one of their hard problems has a small
analog in our loop. Below: each DSec idea, what they measured, our
translation, and a concrete action with a cost estimate.

## 1. One API across four substrates

At DeepSeek scale, agentic workloads span lightweight function calls to
full software-engineering pipelines with different OS and isolation needs.
DSec's answer is a single surface — command execution, file transfer, TTY
access — where switching substrates is a parameter change, not a rewrite.

We have one substrate and will for a long time. The steal is interface
discipline, not infrastructure: every tool and environment call in the
rollout path goes through one narrow interface today, so that when games
move to remote workers (the 8-GPU fleet), no harness code rewrites. The
sandbox lesson from our own history supports this — the engine owns
legality and ground truth while callers stay dumb — and DSec is the same
principle with four backends instead of one.

Action: audit rollout-path tool calls for direct engine imports that
bypass the sandbox boundary. Cost: an afternoon of grepping plus small
refactors. Do it before the fleet exists, not after.

## 2. Layered fast image loading

Their rollouts stalled on container startup against a large, growing image
corpus. Fix: layered on-demand loading over 3FS — fetch only the layers a
sandbox touches, share the rest across the cluster.

Our "image" is a game snapshot plus the static board/atlas/suite payload.
Restore is currently fast (pickles, milliseconds) and therefore
unmonitored — which is exactly how it silently becomes slow. The DSec move
is to treat restore latency as a first-class metric: log per-game restore
milliseconds, alert on creep, and keep static payloads loaded once per
worker with per-game overlays rather than reloaded per game.

Action: add restore-time logging to game startup and a soft budget.
Cost: under an hour. Highest value-to-effort ratio on this list.

## 3. Density engineering

Hundreds of thousands of sandboxes per cluster required attacking two
bottlenecks: duplicated page-cache footprints (fixed with dedup and
memory reclamation plus safe overcommit) and spinlock contention in the
container runtime (fixed, per-sandbox CPU overhead way down, packing
density way up).

Our density problem is game workers per box, and we already did the KV
math: hundreds of concurrent sequences fit in cache. The DSec additions
are (a) share read-only state aggressively — board geometry, atlas
tables, suite text loaded once per worker, never per game; (b) pin engine
workers to explicit CPU sets so they stop contending with the inference
server's dataloader threads; (c) count per-worker overhead (RSS per game
worker) the way they counted per-sandbox overhead, and treat regressions
as bugs.

Action: CPU-set pinning plus a per-worker RSS metric, folded into the
rollout worker design. Cost: a day. This is pbox thinking (Periodic's
version of the same idea: spare CPUs on GPU nodes) applied to game
workers, and both labs converging on it independently is evidence it is
load-bearing rather than fashionable.

## 4. Trajectory log plus preemption-safe resume (the crown jewel)

Every DSec sandbox keeps a globally ordered log of each command and its
result. When training preempts, sandbox resources are retained; on resume,
cached results replay for completed commands instead of re-executing.
Recovery becomes fast-forward, and tool calls are never duplicated — which
matters because re-running a tool is not just slow, it can be wrong
(non-idempotent side effects executed twice).

The implementation audit found existing step/failure checkpoint recovery, but
no caller command IDs or durable in-flight model receipts. The original claim
of "80% already implemented" was not a measured completion estimate; the legacy
schema has no generic committed-operation flags.

Implemented in [`cle/sandbox/durable`](../../cle/sandbox/durable/README.md):

- One sandbox-local append sequence across all seats' command/call records.
- Stable step IDs return saved outcomes after lost replies, without advancing.
- Model responses are durable before admission and reusable during recovery.
- Outcome/checkpoint settlement is atomic; recovered writers fence older owners.
- Accepted notes/silence and queued continuations survive failures even when
  the engine event revision has not advanced.

Recovery may recompute **uncommitted local** transitions from the last checkpoint
using recorded provider responses and the saved RNG state. It does not duplicate
committed game history. An external invocation without a saved response remains
indeterminate; resending requires explicit consent and may duplicate provider work.
Arbitrary external callback effects need their own idempotency protocol.

This is available to rollout callers explicitly. Existing viewer traces remain
the historical checkpoint/projection API. Process-crash tests validate replay
after a worker dies between durable response acquisition and step settlement.

## 5. Lifecycle bound to the GPU job

DSec sandboxes live and die with the training allocation: no orphan
services, no stranded pool, Slurm handles queueing and preemption for
free. When the job exits, everything cleans itself up.

Our version: rollout workers, sandbox procs, telemetry scrapers, and temp
artifacts are all lifetime-bound to the run. Mostly true today; make it a
stated rule with a reaper (process group kill + temp-dir manifest per
run), because the failure mode — a dead run's workers still holding GPUs
while the next run starves — is exactly the kind of thing that costs a
weekend once.

Action: run-scoped process groups plus a startup reaper that kills
stragglers from dead runs. Cost: half a day.

## Bonus: interleaved thinking across tool calls

Not sandbox, same paper family, directly relevant to our open
preserve-vs-notes debate: V4 preserves reasoning across tool-call rounds
*and* user-message boundaries whenever tools are involved, flushing only
for tool-free chatter. Frontier evidence that deliberation continuity
across tool interactions is load-bearing — raw preserved traces for the
hot window, compacted notes for the cold trail. Our policy (structured
notes plus authoritative engine checkpoints, preserved reasoning only in
short diagnostic sessions) is consistent with this; treat V4 as
confirmation, not a directive to change.

## The pattern behind the patterns

DSec, Periodic's pbox, and our own lessons converge on one method: find
the bottleneck you actually have, measure it, delete it, upstream the fix.
DeepSeek's four DSec designs map to four observations about *their*
workload (heterogeneous substrates, big slow images, density pressure,
preemption). Ours map to four observations about ours (30% rejection
rate, batch-16 scheduler, hand promotion, restore latency nobody watches).
Copy the method, translate the findings, skip the parts whose premises
don't hold (we have no MoE routers to replay, no trillion params to
shard, no containers to boot).

## Build order

1. Restore-time logging (hour, immediate signal).
2. Resume-from-commit spec (days, loop robustness).
3. Interface seam audit (afternoon, future-proofing).
4. Worker packing: CPU sets, shared statics, RSS metric (day).
5. Lifecycle rule plus reaper (half day).

Total: roughly a week of infra work that converts the hourly loop from
automated to robust. The open ask to the DSec paper itself: its density
and replay-latency figures, which would calibrate items 3 and 4 with
measured constants instead of estimates.

## Sources

- DeepSeek-V4 paper, §5.2.5 (arXiv 2606.19348, HTML version).
- DSec standalone paper, arXiv cs.DC, 19 Sep 2026 (31 pp).
- HF blog on DeepSeek-V4 (DSec + interleaved thinking summaries).
- Community mirror with §5.2.5 excerpt (github.com/owliz/deepseek-v4-research).
- Periodic Labs infra post, 15 Sep 2026 (convergent pbox evidence).
- Project lessons: exact-once gameplay, deterministic replay executor,
  trace DB as durable record (tasks/lessons.md).
