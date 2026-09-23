"""Fail-closed validation of the historical standard-LoRA checkpoint bundle."""

from __future__ import annotations

import re
from pathlib import Path

from sft.safetensor_types import TensorHeader

from ._checkpoint import inference_assets, inspect_checkpoint, tensor_headers, validate_tokenizer
from ._contracts import (
    ADAPTER,
    FLOAT_DTYPES,
    INDEX,
    PREFIX,
    ROW_MODULES,
    ROW_SUFFIX,
    TOKEN_COUNT,
    VISUAL,
    VISUAL_COUNT,
    MergePlan,
    identities,
    integer,
    object_map,
    read_json,
    require,
    sequence,
    text,
    valid_token_ids,
)

LANGUAGE = re.compile(r"model\.language_model\.layers\.\d+\..+")


def validate_lora_config(config: dict[str, object]) -> None:
    require(config.get("peft_type") == "LORA" and config.get("bias") == "none"
            and integer(config.get("r")) == 16, "requires standard LORA r16 bias=none")
    for key, expected in (("lora_alpha", 32), ("lora_dropout", 0.05)):
        value = config.get(key)
        require(type(value) in (int, float) and value == expected, f"requires {key}={expected}")
    for key in ("use_rslora", "use_dora", "lora_bias", "fan_in_fan_out", "use_qalora",
                "ensure_weight_tying"):
        require(key not in config or config[key] is False, f"unsupported LoRA flag: {key}")
    for key in ("rank_pattern", "alpha_pattern", "modules_to_save", "target_parameters",
                "layer_replication", "alora_invocation_tokens", "arrow_config", "megatron_config",
                "layers_to_transform", "layers_pattern", "exclude_modules"):
        value = config.get(key)
        require(value is None or value == {} or value == [], f"unsupported LoRA option: {key}")
    require(config.get("task_type", "CAUSAL_LM") == "CAUSAL_LM", "requires causal LM adapter")


def resolve_token_ids(config: dict[str, object], adapter: Path,
                      explicit: tuple[int, ...]) -> tuple[int, ...]:
    candidates: list[tuple[int, ...]] = []
    raw = config.get("trainable_token_indices")
    if raw is not None:
        rows = object_map(raw)
        require(set(rows) == set(ROW_MODULES), "expected input embedding and lm_head token mappings")
        candidates.extend(valid_token_ids(rows[module]) for module in ROW_MODULES)
    if explicit:
        candidates.append(valid_token_ids(list(explicit)))
    inventory = adapter / "trainable_parameters.json"
    if inventory.is_file():
        semantic = object_map(read_json(inventory).get("semantic_tokens"))
        candidates.append(valid_token_ids(semantic.get("token_ids")))
    require(bool(candidates), "missing token IDs: provide config mapping, inventory, or explicit IDs")
    require(all(ids == candidates[0] for ids in candidates), "token ID order/mapping differs across sources")
    return candidates[0]


def _selected_modules(config: dict[str, object], base: dict[str, TensorHeader]) -> set[str]:
    targets = config.get("target_modules")
    # Raw HF shards can include MTP weights that the HF language module never
    # instantiated. PEFT suffixes refer to the language stack, not those frozen extras.
    modules = {name.removesuffix(".weight") for name in base if name.endswith(".weight")
               and LANGUAGE.fullmatch(name.removesuffix(".weight"))}
    if isinstance(targets, str):
        require(targets != "all-linear", "all-linear target inference is unsupported; use saved targets")
        pattern = re.compile(targets)
        selected = {name for name in modules if pattern.fullmatch(name)}
    else:
        suffixes = tuple(text(item) for item in sequence(targets))
        require(bool(suffixes) and all(suffixes), "empty target_modules")
        selected = {name for name in modules if any(
            name == suffix or name.endswith("." + suffix) for suffix in suffixes)}
        require(all(any(name == suffix or name.endswith("." + suffix) for name in selected)
                    for suffix in suffixes), "adapter declares unmatched target_modules")
    require(bool(selected) and all(LANGUAGE.fullmatch(name) for name in selected),
            "targets must select only language layer linear weights")
    return selected


def adapter_mappings(config: dict[str, object], headers: dict[str, TensorHeader],
                     base: dict[str, TensorHeader]) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    lora: dict[str, tuple[str, str]] = {}
    rows: dict[str, str] = {}
    expected: dict[str, list[int]] = {}
    for module in sorted(_selected_modules(config, base)):
        key = module + ".weight"
        shape = base[key]["shape"]
        require(len(shape) == 2 and base[key]["dtype"] in FLOAT_DTYPES, f"invalid LoRA base: {key}")
        a, b = PREFIX + module + ".lora_A.weight", PREFIX + module + ".lora_B.weight"
        lora[key] = (a, b)
        expected[a], expected[b] = [16, shape[1]], [shape[0], 16]
    for module in ROW_MODULES:
        key = module + ".weight"
        require(base[key]["dtype"] == "BF16", f"row base must already be BF16: {key}")
        rows[key] = PREFIX + module + ROW_SUFFIX
        expected[rows[key]] = [TOKEN_COUNT, base[key]["shape"][1]]
    require(set(headers) == set(expected), "unexpected or missing adapter keys (all keys must be consumed)")
    for key, shape in expected.items():
        require(headers[key]["shape"] == shape and headers[key]["dtype"] in FLOAT_DTYPES,
                f"invalid adapter shape/dtype: {key}")
    return lora, rows


def visual_mapping(headers: dict[str, TensorHeader], base: dict[str, TensorHeader]) -> dict[str, str]:
    result: dict[str, str] = {}
    for source, header in headers.items():
        # _frozen.save_visual_state saves model.state_dict() with the PEFT wrapper present.
        target = source.removeprefix(PREFIX)
        require(target.startswith("model.visual.") and target not in result,
                f"unexpected/duplicate visual key: {source}")
        require(target in base and header["shape"] == base[target]["shape"] and header["dtype"] == "F32",
                f"visual tensor must match base shape and retain FP32: {source}")
        result[target] = source
    require(len(result) == VISUAL_COUNT and set(result) == {k for k in base if k.startswith("model.visual.")},
            "requires all 333 visual tensors, exactly matching the base visual state")
    return result


def preflight(base: Path, adapter: Path, output: Path, token_ids: tuple[int, ...]) -> MergePlan:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    require(not output.resolve().is_relative_to(base) and not output.resolve().is_relative_to(adapter),
            "output must be outside both source directories")
    require(base.is_dir() and adapter.is_dir(), "base and adapter must be local directories")
    require(not (adapter / "frozen_adapter").exists() and not (adapter / "frozen_bundle.json").exists(),
            "nested frozen adapter bundles are unsupported")
    for name in (ADAPTER, VISUAL, "adapter_config.json"):
        if not (adapter / name).is_file():
            raise FileNotFoundError(adapter / name)
    config = read_json(adapter / "adapter_config.json")
    validate_lora_config(config)
    ids = resolve_token_ids(config, adapter, token_ids)
    base_checkpoint = inspect_checkpoint(base)
    lora, rows = adapter_mappings(config, tensor_headers(adapter / ADAPTER), base_checkpoint.headers)
    visual = visual_mapping(tensor_headers(adapter / VISUAL), base_checkpoint.headers)
    assets = inference_assets(base, adapter)
    validate_tokenizer(adapter, ids)
    base_names = set(base_checkpoint.shards) | {INDEX} | {
        name for name, path in assets.items() if path.parent == base}
    adapter_names = {ADAPTER, VISUAL, "adapter_config.json"} | {
        name for name, path in assets.items() if path.is_relative_to(adapter)}
    adapter_names.update(name for name in ("trainable_parameters.json", "training_config.json")
                         if (adapter / name).is_file())
    return MergePlan(base_checkpoint, lora, rows, visual, ids, assets,
                     identities(base, base_names), identities(adapter, adapter_names))
