# Catan SFT Experiment Design

This project should not treat LoRA as the only plan. We want to learn which
trainable scope actually makes Catan atlas tokens become useful spatial IDs.

## What We Are Training

The first target is not general Catan skill. The first target is a stable
token-to-board coordinate system:

```text
<N11> -> fixed node in the board graph
<T09> -> fixed tile in the board graph
<E18_40> -> fixed edge between two nodes
<P06> -> fixed port slot
```

Then visual QA trains transient state binding:

```text
pixels over <N11> -> current owner/building
pixels over <T09> -> current resource/number/robber
pixels over <E18_40> -> current road owner/empty
pixels over <P06> -> current port type
```

The Catan special tokens are mandatory for tuned/open-weight experiments because
they give the model stable atomic handles for board positions. They are not only
answer-format sugar.

## Data Phases

| Phase | Data | Purpose |
| --- | --- | --- |
| 0 | text-only atlas topology | Teach token geometry/topology before pixels. |
| 1 | image + targeted visual QA | Bind screenshot pixels to atlas tokens. |
| 2 | image + local bundle extraction | Recover neighborhoods, not just one slot. |
| 3 | image -> full board contract | Prove complete public-state extraction. |
| 4 | board contract + rules/history -> answer/action | Reasoning and policy. |

Current focus: Phase 0 and Phase 1 only.

## Trainable Scope Ladder

| Scope | Name | What changes | Why run it |
| --- | --- | --- | --- |
| A | token embeddings only | Added Catan token input embeddings | Tests whether new spatial IDs can be learned as vocabulary anchors. |
| B | embeddings + lm head | Added token embeddings plus output head rows | Tests exact token generation without changing the transformer. |
| C | language LoRA | LoRA on language linear layers | Cheap behavior adaptation; may or may not create a deep geometry manifold. |
| D | language LoRA + projector | Language adapters plus multimodal bridge | Tests whether image-to-language binding bottleneck is projector-side. |
| E | vision/projector LoRA | Vision path or projector adapters | Tests whether roads/numbers/ports need visual adaptation. |
| F | full bf16 fine-tune | All or most weights | Most invasive; only after smaller scopes show the data signal is real. |

Do not assume C/F are necessary. Do not assume A/B are sufficient. We should test.

## Pilot Scaling Matrix

Start with Phase 0 atlas topology because it is cheap and deterministic.

| Data scale | A: emb only | B: emb+head | C: LoRA | Notes |
| --- | --- | --- | --- | --- |
| 390 canonical rows | smoke | smoke | smoke | Prove code path and obvious memorization. |
| 2k rows | yes | yes | optional | Add paraphrases/inverses. |
| 10k rows | yes | yes | yes | First useful manifold check. |
| 50k rows | optional | optional | yes | Only if probes show signal. |

Metrics:

- atlas QA exact accuracy
- held-out paraphrase accuracy
- linear probe recovery of node/tile/edge/port labels
- activation-space topology correlation
- causal patching/steering success
- transfer into Phase 1 visual QA

## Why Not Jump Straight To Full Fine-Tuning?

Full fine-tuning may be the right endpoint if the goal is deeply ingrained board
geometry, but it is the wrong first diagnostic. If a smaller trainable scope
cannot learn the deterministic atlas dataset at all, that is useful information.
If it can learn the behavior but not a probe-visible manifold, that is also useful.

The first question is:

```text
representation quality = f(data scale, trainable scope)
```

That is the scaling law we care about.

## Immediate Recommendation

1. Keep custom Catan tokens in every tuned/open-weight run.
2. Run Phase 0 smoke with embeddings + lm head first.
3. Run Phase 0 smoke with language LoRA as a comparison, not as the assumed final.
4. Build larger Phase 0 topology/paraphrase data before spending on visual QA SFT.
5. Only move to visual QA once token geometry is measurable.
