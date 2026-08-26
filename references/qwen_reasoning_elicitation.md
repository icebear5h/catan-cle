# Qwen reasoning elicitation: Humans& Qwen3 versus Qwen3.8-27B

Checked: 2026-08-16

## Sources

- Humans& NVFP4 RL article: <https://humansand.ai/blog/nvfp4-rl.html>
- Miles/LMSYS companion: <https://www.lmsys.org/blog/2026-07-29-mxfp8-nvfp4-rl/>
- Humans& implementation merge: <https://github.com/radixark/miles/pull/1261>
- Qwen3-30B-A3B model card: <https://huggingface.co/Qwen/Qwen3-30B-A3B>
- Qwen3-30B-A3B tokenizer template: <https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/tokenizer_config.json>
- DAPO-Math-17k: <https://huggingface.co/datasets/zhuzilin/dapo-math-17k>
- Qwen3.8-27B model card: <https://huggingface.co/Qwen/Qwen3.8-27B>
- Qwen3.8-27B chat template: <https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/chat_template.jinja>
- vLLM Qwen3.8 recipe: <https://recipes.vllm.ai/Qwen/Qwen3.8-27B>

## Exact Humans& checkpoint

The articles name `Qwen3-30B-A3B` but do not give a repository or revision. The implementation merged for the article downloads:

```text
Qwen/Qwen3-30B-A3B
```

This is the post-trained hybrid-thinking checkpoint, whose model-card metadata names `Qwen/Qwen3-30B-A3B-Base` as its base. It is not the Base checkpoint and not `Qwen3-30B-A3B-Thinking-2507`. No Hugging Face revision is pinned, so the publication does not identify an immutable weight commit.

## How the Humans& run elicits reasoning

The implementation does not use a numeric reasoning-effort control. It combines five mechanisms:

1. It starts from a post-trained checkpoint already trained for thinking.
2. Miles calls `apply_chat_template` without `enable_thinking=False`. The Qwen3 template therefore leaves thinking enabled by default.
3. Every DAPO prompt explicitly begins: `Solve the following math problem step by step.` It also requires a final `Answer: \\boxed{...}` line.
4. Rollout uses eight samples per prompt, temperature 1, and an 8,192-token response cap, creating outcome-based exploration over different traces.
5. The `deepscaler` reward returns zero unless the output contains `</think>` (or a legacy response marker), discards everything before `</think>`, and grades only the extracted boxed answer with symbolic/numeric equivalence checks.

Therefore, reasoning is structurally elicited but not process-scored. The RL signal rewards terminal answer correctness, not whether intermediate claims are faithful or valid. The reasoning prior comes from Qwen post-training; GRPO selects sampled trajectories that lead to correct final answers.

Qwen3-30B-A3B itself supports:

- hard thinking switch: `enable_thinking=True|False`;
- stateful soft switches: `/think` and `/no_think`;
- separate sampling recommendations for thinking and non-thinking;
- a two-call early-stop method for enforcing a hard thinking-token budget.

The old Qwen3 card recommends removing historical thinking after a new user turn while preserving it inside an active multi-step tool interaction.

## Qwen3.8 reasoning controls

Qwen3.8-27B directly supports the desired opening-deep, later-dynamic policy:

- `enable_thinking`: thinking on/off; on by default;
- `reasoning_effort`: `xhigh`, `medium`, or `low`; `xhigh` is the default;
- `preserve_thinking`: retain prior reasoning blocks; true by default.

The model card recommends thinking-mode sampling of temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 0, and repetition penalty 1.

The controls are implemented primarily through the chat template. `reasoning_effort` is not a hard compute reservation:

- `xhigh` injects: `Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.`
- `medium` injects no extra reasoning instruction but still opens the thinking channel.
- `low` injects: `Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.`
- `enable_thinking=False` prefills an empty `<think>...</think>` block so generation begins with the answer.

Actual maximum compute remains controlled by output limits or an explicit two-call thinking-budget implementation. Setting `xhigh` alone does not guarantee a particular number of reasoning tokens.

`preserve_thinking=True` renders historical `reasoning_content` into later prompts. With it false, completed reasoning before the latest user query is stripped while final answers remain. This should be tested rather than assumed beneficial in Catan: preservation may improve continuity and exact-prefix caching, but it can also retain stale strategic assumptions.

## Proposed Catan policy

Use the following initial policy before learning a compute scheduler:

| Situation | Thinking | Effort | Initial cap |
|---|---|---|---:|
| Initial board and first placement | on | xhigh | 4K-8K |
| Second setup placement | on | xhigh | 4K-8K |
| Routine forced/mechanical action | off | — | 128-256 |
| Normal policy decision | on | low | 512-1K |
| Trade, robber, or expansion choice | on | medium | 1K-2K |
| Plan invalidated or win/loss pivot | on | xhigh | 2K-4K |
| Public event with no useful reaction | no generation | — | 0 |

These caps are hypotheses for measurement, not published optimums.

Run two memory variants:

- preserved private reasoning in the active per-seat cache, with authoritative state checkpoints on every decision;
- `preserve_thinking=False` plus a concise structured strategic memory containing goals, contingencies, commitments, threats, and confidence.

Do not copy the math reward literally. Catan RL should reward authoritative action legality and outcomes. Any rationale supervision must be separately grounded against engine facts; plausible commentary must not earn policy reward merely because the final action succeeded.

## Exact Qwen3.8 Miles status

Current Miles main includes a dense Qwen3.8-27B GRPO recipe (`scripts/run_qwen3_dense.py`). It applies the model chat template without overriding its defaults, so Qwen3.8 rollouts default to `xhigh` thinking. The published Miles model page reports one 8xH200 DAPO step at about 12 minutes, with mean response length 4,585 tokens and 33% of responses truncated at the 8,192-token cap. That workload is not representative of compact Catan actions, but it demonstrates why dynamic reasoning control is necessary.
