# Board-recognition training dataset

Date: 2026-08-26

## Result

A complete 1024px board-recognition dataset now exists at:

`artifacts/generated/board_recognition/curriculum/`

It contains:

- 640 ordinary raw full-board images;
- 320 one-label counterfactual pairs;
- 512 train, 64 validation, and 64 test states;
- 160 states in each of empty/setup, initial-placement, sparse-midgame, and
  dense-endgame complexity stages;
- complete labels for 19 tiles, 54 nodes, 72 edges, and 9 ports per image;
- 122,880 available typed slot/attribute facts;
- a 10,240-row official Qwen image/conversation SFT projection;
- a PyTorch state-batch loader that encodes each image once and supplies 16
  direct slot-classification targets.

Total generated size is 349,287,540 bytes (about 333 MiB), of which 282,443,761
bytes are images.

The tracked generation receipt is:

`artifacts/manifests/board_recognition/curriculum_1024_640_v1.json`

## Canonical dense data

The dense state payload is authoritative. Every image is paired with one engine
contract and one typed label payload:

```text
images/<state>.png
contracts/<state>.json
dense_labels/<state>.json
```

The dense labels expose only classifier targets:

- tile: resource, number, robber;
- node: occupancy;
- edge: owner;
- port: port type.

Counterfactual pair members share a split and differ in exactly one declared
dense label. Validation re-renders every engine contract and requires the saved
pixels and labels to match it.

## PyTorch state batches

`data_pipeline.board_recognition.BoardRecognitionStateDataset` returns one RGB
image plus 16 queries. The collator produces:

```text
images             [batch, channels, height, width] when transformed to tensors
entity_type_ids     [batch, 16]
attribute_ids       [batch, 16]
slot_indices        [batch, 16]
class_indices       [batch, 16]
```

Without an image transform, images remain a list of RGB PIL objects for a Qwen
processor. Entity types are exactly balanced: four tile, four node, four edge,
and four port queries per state. One query is reserved for the state's declared
counterfactual target. Remaining queries balance classes available in that
state before selecting a slot. Pair members receive the same 16 slot/attribute
queries, so exactly the declared target class changes. `set_epoch()`
deterministically changes the remaining aligned slot sample.

## Qwen SFT projection

The compatibility export follows the official Qwen VLM annotation shape:

```json
{
  "image": "images/dense_endgame_edge_p000_base.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nClassify one symbolic slot..."
    },
    {
      "from": "gpt",
      "value": "ABSENT"
    }
  ]
}
```

Files:

```text
qwen_sft/train.jsonl         8,192 rows
qwen_sft/validation.jsonl    1,024 rows
qwen_sft/test.jsonl          1,024 rows
qwen_sft/audit/*.jsonl       one provenance row per annotation
```

The Qwen files contain only `image` and `conversations`. Audit sidecars preserve
state, split, stage, counterfactual group, canonical slot, attribute, class,
source-label path, and prompt hash.

Primary format references:

- [Official Qwen3-VL fine-tuning format](https://github.com/QwenLM/Qwen3-VL/blob/50068df2/qwen-vl-finetune/README.md)
- [Hugging Face TRL VLM SFT guide](https://huggingface.co/docs/trl/main/en/training_vlm_sft)

## Why 1024px

The controlled frozen-Qwen strict evaluation did not support using 1536px for
the first training dataset:

| Resolution | Strict exact | Valid JSON | Image tokens/request | Median latency |
|---|---:|---:|---:|---:|
| 1024px | 11/60 | 20/60 | 1,026 | 9.5 s |
| 1536px | 10/60 | 16/60 | 2,306 | 14.7 s |

Eight questions were correct at both sizes, three only at 1024px, and two only
at 1536px. The 2048px hosted input was capped at 2,502 image tokens rather than
the expected 4,096 and was stopped at 23/60 when work shifted to data
construction. This does not prove that a trained direct head cannot benefit from
1536px, but it gives no present evidence that the 2.25× token cost is justified.
The generator remains resolution-configurable for a later ablation.

## Reproduction

```bash
uv run python scripts/build_catan_board_recognition_curriculum.py \
  --pairs-per-entity-type 20 \
  --image-size 1024 \
  --seed 381427 \
  --overwrite

uv run python scripts/export_catan_board_recognition_sft.py \
  --queries-per-state 16 \
  --seed 381427 \
  --overwrite
```

## Verification

- 30 focused curriculum, DataLoader, SFT, renderer, and strict-vision tests pass.
- The full repository suite is 307 passed and one skipped with no warnings.
- All 640 contracts reproduce their dense labels and 1024px rendered pixels.
- All manifest, dense-label, and 10,240 Qwen rows validate against Draft 2020-12
  schemas.
- Pair members receive aligned query sets and differ on exactly the declared
  target class.
- Train/validation/test state and SFT counts match their metadata.
- PyTorch loading was checked with worker processes and a real tensor batch of
  shape `[2, 3, 1024, 1024]` plus query tensors of shape `[2, 16]`.
- The tracked generation receipt matches dataset size and all recorded hashes.

## Limitations

- This is a first synthetic corpus, not enough evidence of real-board
  generalization.
- Complexity stages are controlled piece-density proxies; contracts are not all
  guaranteed reachable through legal game trajectories.
- The renderer/theme is fixed because the user selected raw full boards only.
- Canonical slot prompts assume the atlas-token prerequisite or a direct learned
  slot embedding.
- Qwen conversation rows train autoregressive class emission. The state-batch
  loader remains the preferred interface for the non-generative direct head.
