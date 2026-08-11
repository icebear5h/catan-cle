# Catan Added Vocabulary Tokens

Use these as regular added tokens for open-weight Catan VLM/SFT experiments.
Do not register them as chat/control special tokens.

Manifest:

- `configs/training_configs/catan_added_tokens.json`

Source code:

- `data_pipeline/catanbench/tokens.py`
- `scripts/export_catan_tokens.py`

## Counts

The manifest currently contains 197 tokens:

- 54 node tokens: `<N00>` through `<N53>`
- 72 edge tokens: canonical edge pairs like `<E03_17>`
- 19 tile tokens: `<T00>` through `<T18>`
- 9 port tokens: `<P00>` through `<P08>`
- 6 resource/desert tokens
- 1 board-object token: `<ROBBER>`
- 11 color tokens
- 3 building tokens
- 22 action tokens

## Tokenizer Usage

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

from data_pipeline.catanbench.tokens import added_tokens

processor = AutoProcessor.from_pretrained(model_name)
num_added = processor.tokenizer.add_tokens(added_tokens())

model = AutoModelForImageTextToText.from_pretrained(model_name)
model.resize_token_embeddings(len(processor.tokenizer))
```

Equivalent helper:

```python
from data_pipeline.catanbench.tokens import add_tokens_to_tokenizer

num_added = add_tokens_to_tokenizer(processor.tokenizer)
```

## Benchmark Usage

Do not require these custom tokens for the first Catan Bench 100 pass on existing
off-the-shelf VLMs. Those models have not had the Catan tokens added to their
tokenizers, so prompts like `<N42>` or `<E12_17>` would be fragmented into
ordinary sub-tokens and would partly test tokenizer mismatch rather than pure
visual perception.

For initial external-model benchmarks, use plain canonical labels and numeric
IDs in prompts and answers:

```text
node 42
edge 12-17
tile 8
port 3
```

Use the custom tokens after tokenizer extension for open-weight SFT, LoRA, GRPO,
and post-tuning evals where the model can learn stable atomic atlas symbols.

## Training Format Recommendation

Use both the token and raw numeric id in generated contracts:

```json
{
  "node": "<N22>",
  "node_id": 22,
  "building": "<SETTLEMENT>",
  "color": "<BLUE>",
  "resource": "<WHEAT>",
  "object": "<ROBBER>"
}
```

The token gives the model a stable atomic symbol. The numeric id keeps engine
validation and human debugging simple.
