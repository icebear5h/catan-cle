# Direct Catan Vision-Tower Readout

Status: research and training design, 2026-08-25. No trainer has been
implemented by this document.

## Decision

Train the Qwen3.8 vision tower as a **query-conditioned Catan slot classifier**,
not as an autoregressive board narrator.

The inference contract is:

```text
input:  board image + (atlas slot, attribute)
output: one class distribution in the same forward pass
```

Examples:

```text
(image, <N07>, occupancy) -> BLUE_SETTLEMENT
(image, <E12_13>, owner)  -> EMPTY
(image, <T04>, resource)  -> ORE
(image, <T04>, number)    -> 8
```

There is no call to `generate()`, no chain of thought, and no earlier answer
token. The class ID can be converted deterministically to the canonical Catan
string after argmax. If a native VLM text response is eventually required, it
should be a separate one-token distillation target, not the primary perception
objective.

This is narrower than general visual question answering. That is intentional:
the Catan board has a fixed atlas of 19 tiles, 54 nodes, 72 edges, and 9 ports.
The difficult problem is binding the current pixels to the correct fixed slot,
not discovering an unknown output schema or verbally reasoning about the board.

## Why this is the right local target

The repository's results separate recognition from dense binding:

- The prior full-board Qwen3.8 image run scored 24/110 (21.82%), while supplying
  the same state as text reached at most 93/110 under the historical permissive
  scorer. Dense visual grounding is therefore the largest observed gap, though
  that comparison was not provider controlled.
- With a complete closed class vocabulary, isolated tile resource/number
  recognition reached 97--100% strict exact, including 400/400 content-level
  number reads. The tower does not need to relearn ordinary tile appearance from
  scratch.
- Node occupancy was 0 exact in both isolated and local probes. Roads, nodes,
  ports, and full-board slot binding are the high-value targets.
- A balanced clean hex-direction probe reached 91.67%. This argues against
  treating general directional reasoning as the first defect to repair.
- The 512/768 resolution sweep was nearly tied on the small Gemini slice, and
  color-road listing was 0% at every resolution. Resolution alone is not a
  training objective.

The current post-atlas node dataset already gives a useful smoke-test seed:
`sft/scripts/build_node_factor_dataset.py` creates 594 board contracts and 3,564
QA rows at one variant, covers every node under contradictory occupancy states,
adds distractors, and stores atlas bounding boxes. Its existing long JSON and
neighborhood answers should not be used for this experiment. Read the contracts
as dense labels and emit direct class targets instead.

## What the literature contributes

| Source | Result worth borrowing | Catan translation |
| --- | --- | --- |
| [DETR](https://arxiv.org/abs/2005.12872) | Learned object queries make a fixed set of predictions in parallel. | Use learned atlas queries for Catan slots. Unlike detection, the slot identities are known, so no Hungarian matching is needed. |
| [Perceiver IO](https://arxiv.org/abs/2107.14795) | Output queries support structured, differently shaped outputs from one latent representation. | Combine a slot embedding and attribute embedding to ask for exactly one board fact, or evaluate all facts in parallel. |
| [MDETR](https://arxiv.org/abs/2104.12763) | A query can be aligned directly to image regions and object predictions. | Cross-attend the requested Catan slot to visual patches instead of asking a decoder to describe the board. |
| [Learning to Localize Objects Improves Spatial Reasoning](https://arxiv.org/abs/2404.07449) | Location prediction, reverse-location prediction, and explicit negative examples improve visual grounding. | Train slot-to-state, state-to-slot, and `EMPTY`/absent hard negatives. Keep the objectives classificatory rather than generative. |
| [SpatialVLM](https://arxiv.org/abs/2401.12168) | Its reported frozen-versus-unfrozen ViT ablation improved fine-grained distance accuracy from 5.6 to 8.4 when the ViT was unfrozen; the authors argue contrastive vision features can discard fine spatial detail. | A head-only baseline is mandatory, but controlled unfreezing of late vision layers is justified if it improves small-piece binding. |
| [Ferret-v2](https://arxiv.org/abs/2404.07973) | High-resolution, multi-granularity visual processing and dense alignment target fine-grained grounding limits. | Preserve the pre-merger patch grid for roads and buildings; do not reduce the board to one pooled vector. |
| [Chess state recognition](https://arxiv.org/abs/2104.14963) | A synthetic-to-real board pipeline rectifies the board, then predicts occupancy and piece type per fixed square. | Canonicalize the board crop and predict state per known tile/node/edge/port. Catan geometry is fixed just as chess squares are fixed. |
| [Linear Mechanisms for Spatiotemporal Reasoning in VLMs](https://arxiv.org/abs/2601.12626) and its checked-in code | The accompanying spatial fine-tuning experiment selects the last six vision MLPs and adds an auxiliary spatial-ID alignment loss. | Use its selective late-vision adaptation as an ablation. Replace its language-token spatial target with direct atlas-slot classification and behavioral localization tests. |

The papers do **not** establish that full vision-tower fine-tuning is always
best. The useful common thread is more specific: fixed queries, explicit
localization, hard negatives, dense/high-resolution features, and an ablation
between frozen and selectively unfrozen visual features.

## Proposed model: Catan Slot Readout

```text
canonical 768x768 board image
              |
              v
     Qwen3.8 vision patch embed
              |
       27 vision blocks
              |
       pre-merger patch states  <---- train late vision layers only
              |
    cross-attention readout block
       ^                  |
       |                  v
slot ID + attribute   attribute-specific classifier
  embedding                   |
                              v
                    one class distribution
```

### Use the pre-merger features

The current official Qwen3.8-27B config identifies the architecture as
`Qwen3_5ForConditionalGeneration`. Its vision config has 27 blocks, hidden size
1,152, patch size 16, spatial merge size 2, and a 5,120-dimensional merger
output. The current Transformers `Qwen3_5VisionModel` returns:

- `last_hidden_state`: the final vision-block states before the merger;
- `pooler_output`: the spatially merged states normally sent toward the LLM.

At 768x768, the nominal single-image grid is 48x48 before the 2x2 merge and
24x24 after it. A road or building can occupy little more than one merged cell.
The classifier should therefore cross-attend to `last_hidden_state`, while a
merged-feature head is retained as an ablation.

This is a supported model output, not a forward hook into a private intermediate.
The precise reshaping still needs to respect `image_grid_thw` and Qwen's patch
ordering.

### Queries and heads

Represent a request as:

```text
q = slot_embedding[slot_id] + attribute_embedding[attribute]
```

Run one small cross-attention layer from `q` to the patch states. Add a soft
spatial attention bias centered on the canonical atlas bounding box for that
slot. The bias provides stable geometry but no board-state answer. Randomized
crop, scale, translation, and mild perspective augmentation prevent it from
becoming a brittle pixel lookup.

Use separate closed-set classifier heads:

| Attribute | Classes |
| --- | --- |
| tile resource | `DESERT`, `WOOD`, `BRICK`, `SHEEP`, `WHEAT`, `ORE` |
| tile number | `NONE`, `2`, `3`, `4`, `5`, `6`, `8`, `9`, `10`, `11`, `12` |
| tile robber | `NO`, `YES` |
| node occupancy | `EMPTY` plus five colors x `{SETTLEMENT, CITY}` = 11 classes |
| edge owner | `EMPTY` plus five player colors = 6 classes |
| port type | `THREE_TO_ONE` plus the five 2:1 resources = 6 classes |

The five colors match the current dataset builder (`RED`, `BLUE`, `ORANGE`,
`WHITE`, and `BLACK`). If production games are strictly four-player, preserve
the fifth class in training only when it can actually appear in the renderer.

One query returns one answer. For dense diagnostics, batch every slot and
attribute query against the same image; this is still one vision forward and
parallel classification, not autoregressive reconstruction.

## Training objective

The minimum viable loss is weighted class cross-entropy:

```text
L = sum over requested facts class_weight[y] * CE(logits, y)
```

Balance by task and class. Raw accuracy is unsafe because most node and edge
slots are empty.

Two auxiliary losses are worth testing only after the basic classifier works:

1. **Attention localization:** penalize query-attention mass outside the target
   slot's annotated bbox. This teaches where to look without supplying state.
2. **Counterfactual locality:** for paired boards differing in exactly one slot,
   require the target prediction to change and predictions for untouched slots
   to remain stable.

The second is the stronger causal test. An attractive attention map by itself
does not prove that the prediction used the marked pixels.

## Data recipe

### Unit of data

Each rendered board should store dense ground truth for all 154 atlas entities,
not one natural-language QA row. Sample `(slot, attribute)` requests from that
contract during training. This gives many supervised reads per render without
forcing a multi-token board serialization.

### Required examples

1. **Counterfactual pairs identical outside the target region:** copy a board
   and change only one target node, edge, tile robber, or port. The current
   node-factor generator provides contradictory target states, but its
   surrounding board is also randomized; add true minimal pairs.
2. **Nearby-slot hard negatives:** put a same-color road or building on an
   adjacent slot and label the queried slot `EMPTY`.
3. **Class balance:** oversample occupied nodes/edges, cities, ports, and uncommon
   colors. Report the natural board distribution separately.
4. **Style split:** train on renderer-randomized themes and reserve complete
   themes plus real replay screenshots for validation. Do not split different
   questions from the same board across train and test.
5. **Geometry augmentation:** crop/scale/translation, modest perspective jitter,
   JPEG compression, UI occlusion, color shifts, and piece-size variation. Keep
   labels and bboxes transformed exactly.

### Perceptual curriculum, not reasoning curriculum

Use all stages in a retained mixture:

1. 96--160 px local crops around a target, closed-set class output;
2. full board with a visible target box or Set-of-Mark-style identifier;
3. full board with only the atlas query and a soft internal bbox prior;
4. unmarked full board from held-out render themes and real replay captures.

Marks are training scaffolding and a diagnostic. They should not be required at
deployment unless the product already renders them.

For a first real pilot, 10,000 diverse full boards provide roughly 1.54 million
entity labels before attribute expansion. The existing 594 contracts are enough
to prove the data path and overfit a smoke subset, not enough evidence of
cross-style generalization.

## What to train

Run this ablation ladder in order:

| Run | Trainable parameters | Question answered |
| --- | --- | --- |
| A | readout block and class heads only | Are pretrained patch features already sufficient? |
| B | A + last six vision MLP sublayers | Does selective tower adaptation improve Catan binding? |
| C | A + last six complete vision blocks | Is attention adaptation needed in addition to feature rewriting? |
| D | A + all 27 vision blocks | Does full tower tuning justify its cost and greater forgetting risk? |
| E | merged-feature version of best run | Does preserving the pre-merger grid materially help small pieces? |

Start in BF16, not 4-bit or 8-bit quantization. A reasonable pilot is:

```text
readout/head learning rate: 1e-4
unfrozen vision learning rate: 1e-6
optimizer: AdamW
weight decay: 0.01
warmup: 3%
schedule: cosine
```

These are starting values, not a claimed optimum. The `1e-6` vision rate follows
the conservative scale used by Qwen's official Qwen3-VL fine-tuning example.
Sweep it against `3e-7` and `3e-6` before widening the trainable tower scope.

Keep the 27B language model frozen and preferably unloaded. The direct trainer
should instantiate/extract `Qwen3_5VisionModel`, the readout block, and the heads.
The repository's current `sft/modal_qwen_series_train.py` is not this trainer: it
uses 4-bit LoRA and explicitly freezes the LLM, vision tower, and merger. Its
pinned upstream also warns that 4/8-bit quantization should not be combined with
vision training. Build this as a separate BF16 vision experiment rather than
silently changing the existing SFT run.

## Evaluation gates

The primary evaluation must exercise the same zero-decoding contract as
training.

Report:

- macro F1 and balanced accuracy per attribute;
- accuracy for every slot and class, especially occupied nodes/edges;
- target-only counterfactual flip accuracy;
- held-out renderer-theme and real-replay performance;
- latency for one image encoding plus one or all parallel slot queries.

Add causal controls:

1. **Shuffled image:** keep the query and replace the board. Accuracy should fall
   toward class-prior chance.
2. **No image:** the head must not solve the balanced test set from slot priors.
3. **Target occlusion:** covering the queried bbox should collapse target
   confidence.
4. **Control occlusion:** covering a distant slot should have a much smaller
   effect.
5. **Minimal pair:** changing one rendered piece should flip that fact without
   broadly changing other logits.

Only conclude that the tower needed fine-tuning if B/C/D beats A on held-out real
boards, not merely on the synthetic renderer. Selective tower tuning is the
hypothesis; the frozen-tower readout is the necessary control.

## Optional native one-token bridge

If the policy model must itself accept a textual query and return a token, distill
the trained classifier into a frozen-LLM first-answer-position objective:

```text
image + "<N07> OCCUPANCY" -> BLUE_SETTLEMENT
```

Use canonical labels that are each represented by one added token, mask every
prompt position, compute loss only on the first answer position, freeze the LLM,
and train the new label-token embedding/output rows plus the vision tower and
multimodal merger at low rates. Evaluate with exactly one decoder step
(`max_new_tokens=1`).

This bridge satisfies “no prior generated tokens,” but it is second choice:
language priors and the merger can hide whether the tower truly localized the
piece. The direct slot head is cheaper, easier to falsify, and provides a clean
perception module for the later Catan policy.

## Sources and implementation references

Primary research:

- [DETR](https://arxiv.org/abs/2005.12872)
- [Perceiver IO](https://arxiv.org/abs/2107.14795)
- [MDETR](https://arxiv.org/abs/2104.12763)
- [Learning to Localize Objects Improves Spatial Reasoning](https://arxiv.org/abs/2404.07449)
- [SpatialVLM](https://arxiv.org/abs/2401.12168)
- [Ferret-v2](https://arxiv.org/abs/2404.07973)
- [Chess state recognition](https://arxiv.org/abs/2104.14963)
- [Linear Mechanisms for Spatiotemporal Reasoning in VLMs](https://arxiv.org/abs/2601.12626)

Current model and trainer references:

- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Qwen3.8-27B config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
- [Transformers Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [Official Qwen3-VL fine-tuning guide](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/README.md)
- [Repository-pinned Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune)

## Recommended first experiment

Do one bounded experiment before building a large corpus:

1. Convert the 594 node-factor contracts to direct `(<Nxx>, occupancy) -> class`
   samples and create true one-slot counterfactual pairs.
2. Train A (head only) and B (head + last six vision MLPs) on pre-merger features.
3. Use balanced synthetic boards for development and a manually checked set of
   at least 200 real replay crops/boards as the decision set.
4. Run shuffled-image, target-occlusion, and adjacent-slot controls.
5. Continue only if B improves occupied-node macro F1 and counterfactual accuracy
   on the real set without increasing distant-slot sensitivity.

That experiment directly answers the question that matters: whether changing
the Catan vision tower makes the requested fact immediately readable, before any
language reasoning can compensate.
