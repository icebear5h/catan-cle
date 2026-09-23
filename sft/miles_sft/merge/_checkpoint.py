"""Complete HF shard inventories and inference assets, without model loading."""

from __future__ import annotations

import math
from pathlib import Path

from sft.safetensor_types import TensorHeader, open_tensors

from ._contracts import (
    DTYPE_BYTES,
    INDEX,
    ROW_MODULES,
    VOCAB_SIZE,
    Checkpoint,
    integer,
    object_map,
    read_json,
    relative_name,
    require,
    sequence,
    text,
)

TOKENIZER_ASSETS = frozenset({
    "tokenizer_config.json", "tokenizer.json", "tokenizer.model", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja", "chat_template.json",
})
BASE_ASSETS = frozenset({
    "config.json", "generation_config.json", "processor_config.json", "preprocessor_config.json",
    "video_preprocessor_config.json", "image_processor_config.json",
})


def tensor_headers(path: Path) -> dict[str, TensorHeader]:
    with open_tensors(path) as handle:
        return {key: {"shape": handle.get_slice(key).get_shape(),
                      "dtype": handle.get_slice(key).get_dtype()} for key in handle.keys()}


def tensor_bytes(header: TensorHeader) -> int:
    require(header["dtype"] in DTYPE_BYTES, f"unsupported tensor dtype: {header['dtype']}")
    require(all(width > 0 for width in header["shape"]), "empty tensor dimensions unsupported")
    return math.prod(header["shape"]) * DTYPE_BYTES[header["dtype"]]


def inspect_checkpoint(root: Path) -> Checkpoint:
    """Require exact equality of declared keys and each shard's physical keys."""
    index = read_json(root / INDEX)
    weights = {text(key): relative_name(value)
               for key, value in object_map(index.get("weight_map")).items()}
    require(bool(weights), "empty HF weight_map")
    names = set(weights.values())
    require(all(name.endswith(".safetensors") for name in names), "non-safetensors base shard")
    actual_files = {p.relative_to(root).as_posix() for p in root.rglob("*.safetensors")}
    require(names == actual_files, "base shards differ from declared complete index")
    headers: dict[str, TensorHeader] = {}
    shards: dict[str, tuple[str, ...]] = {}
    for name in sorted(names):
        found = tensor_headers(root / name)
        expected = {key for key, shard in weights.items() if shard == name}
        require(set(found) == expected, f"shard keys differ from index: {name}")
        headers.update(found)
        shards[name] = tuple(sorted(found))
    size = sum(tensor_bytes(header) for header in headers.values())
    metadata = object_map(index.get("metadata"))
    # The pinned Qwen index spells its exact byte count as 55562855904.0.
    declared_size = metadata.get("total_size")
    require(type(declared_size) in (int, float) and declared_size == size,
            "index total_size does not match tensors")
    validate_base_config(root, headers)
    return Checkpoint(weights, headers, shards, size)


def validate_base_config(root: Path, headers: dict[str, TensorHeader]) -> None:
    config = read_json(root / "config.json")
    text_config = object_map(config.get("text_config", config))
    require(integer(text_config.get("vocab_size")) == VOCAB_SIZE, "base vocab_size must be 248320")
    if "vocab_size" in config:
        require(integer(config["vocab_size"]) == VOCAB_SIZE, "inconsistent top-level vocab_size")
    ties = [c["tie_word_embeddings"] for c in (config, text_config) if "tie_word_embeddings" in c]
    require(bool(ties) and all(value is False for value in ties), "tie_word_embeddings must be false")
    require(not config.get("quantization_config") and not text_config.get("quantization_config"),
            "quantized bases are unsupported")
    widths: set[int] = set()
    for module in ROW_MODULES:
        header = headers.get(module + ".weight")
        require(header is not None, f"missing untied base weight: {module}")
        if header is None:
            raise ValueError(module)
        shape = header["shape"]
        require(len(shape) == 2 and shape[0] == VOCAB_SIZE, f"full vocabulary required: {module}")
        widths.add(shape[1])
    require(len(widths) == 1, "embedding/head hidden widths differ")


def inference_assets(base: Path, adapter: Path) -> dict[str, Path]:
    for name in ("config.json", "generation_config.json"):
        require((base / name).is_file(), f"missing base asset: {name}")
    require(any((base / name).is_file() for name in ("processor_config.json", "preprocessor_config.json")),
            "missing native base processor configuration")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        require((adapter / name).is_file(), f"missing adapter tokenizer asset: {name}")
    assets = {name: base / name for name in BASE_ASSETS if (base / name).is_file()}
    # The adapter is the authoritative tokenizer snapshot, including template selection.
    assets.update({name: adapter / name for name in TOKENIZER_ASSETS if (adapter / name).is_file()})
    assets.update({p.relative_to(adapter).as_posix(): p
                   for p in (adapter / "chat_templates").rglob("*.jinja") if p.is_file()})
    tokenizer_config = read_json(adapter / "tokenizer_config.json")
    require(bool(tokenizer_config.get("chat_template")) or any(
        name.startswith("chat_template") for name in assets), "missing adapter chat template")
    return assets


def validate_tokenizer(adapter: Path, ids: tuple[int, ...]) -> None:
    """Validate the row addresses against the real saved tokenizer JSON vocabulary."""
    tokenizer = read_json(adapter / "tokenizer.json")
    vocab = object_map(object_map(tokenizer.get("model")).get("vocab"))
    mapping = {token: integer(value) for token, value in vocab.items()}
    specials: set[int] = set()
    for raw in sequence(tokenizer.get("added_tokens", [])):
        entry = object_map(raw)
        token, token_id = text(entry.get("content")), integer(entry.get("id"))
        require(token not in mapping or mapping[token] == token_id, "inconsistent added token ID")
        mapping[token] = token_id
        require(type(entry.get("special")) is bool, "added token special must be boolean")
        if entry["special"]:
            specials.add(token_id)
    values = list(mapping.values())
    require(len(set(values)) == len(values) and all(0 <= i < VOCAB_SIZE for i in values),
            "tokenizer vocabulary has duplicate or out-of-bounds IDs")
    require(set(ids) <= set(values) and not set(ids) & specials,
            "semantic IDs must address 154 regular saved tokenizer tokens")
