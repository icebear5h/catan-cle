from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict

from sft.json_types import JsonDict, JsonValue, as_dict, as_int, as_list, as_str
from sft.lora_expansion import expected_adapter_shapes, standard_lora_rank
from sft.scripts.train.train_trl_catan_vision import (
    RUN_CONFIG_FILE,
    TRAINABLE_SCOPE_FILE,
    VISUAL_STATE_FILE,
    discover_components,
    language_linear_targets,
    load_checkpoint_text_tokenizer,
    sha256_file,
)

from ._config import CACHE, CONTEXT, INFERENCE_SIDECARS, REQUIRED_BUNDLE, eval_image
from ._validation import read_json

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizerBase

# Modal leaves `Image.imports()` unannotated; name the context manager it returns.
container_imports: Callable[[], AbstractContextManager[None]] = eval_image.imports
with container_imports():
    # Redundant aliases re-export these names through the package facade.
    import torch as torch
    from huggingface_hub import HfApi as HfApi
    from huggingface_hub import snapshot_download as snapshot_download
    from peft import LoraConfig as LoraConfig
    from peft.tuners.tuners_utils import check_target_module_exists as check_target_module_exists
    from safetensors import safe_open as safe_open
    from transformers import AddedToken as AddedToken
    from transformers import AutoConfig as AutoConfig
    from transformers import AutoModelForMultimodalLM, AutoTokenizer


class TensorHeader(TypedDict):
    """One safetensors entry as read from the file header, without tensor data."""

    shape: list[int]
    dtype: str


class TensorSlice(Protocol):
    """`safe_open(...).get_slice(name)`, narrowed to the header accessors used here."""

    def get_shape(self) -> list[int]: ...
    def get_dtype(self) -> str: ...


class TensorFile(Protocol):
    """An open safetensors file; the stubs ship `safe_open` without annotations."""

    def __enter__(self) -> TensorFile: ...
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...
    def keys(self) -> list[str]: ...
    def get_slice(self, name: str) -> TensorSlice: ...
    def get_tensor(self, name: str) -> torch.Tensor: ...


class TensorOpener(Protocol):
    """`safetensors.safe_open` as these launchers call it."""

    def __call__(self, filename: str | Path, framework: str, device: str = ...) -> TensorFile: ...


class TokenizerLoader(Protocol):
    """`AutoTokenizer.from_pretrained` as the base preflight calls it."""

    def __call__(self, name: str | Path, /, *, local_files_only: bool,
                 trust_remote_code: bool) -> PreTrainedTokenizerBase: ...


class ArchitectureFactory(Protocol):
    """`AutoModelForMultimodalLM.from_config` as the target preflight calls it."""

    def __call__(self, config: PretrainedConfig, /, *, dtype: torch.dtype,
                 attn_implementation: str) -> torch.nn.Module: ...


# Resolved per call: the container-only imports above are unbound locally.
def open_tensors(path: str | Path, *, framework: str, device: str) -> TensorFile:
    opener: TensorOpener = safe_open
    return opener(path, framework=framework, device=device)


def load_tokenizer(name: str | Path) -> PreTrainedTokenizerBase:
    loader: TokenizerLoader = AutoTokenizer.from_pretrained
    return loader(name, local_files_only=True, trust_remote_code=False)


def snapshot(repo: str, revision: str, *, subdir: str = "", adapter: bool) -> tuple[Path, list[str]]:
    """Download only enumerated inference files; existing immutable blobs are reused."""
    prefix = subdir + "/" if subdir else ""
    files = {
        name.removeprefix(prefix)
        for name in HfApi().list_repo_files(repo, revision=revision)
        if name.startswith(prefix)
    }
    if adapter:
        missing = REQUIRED_BUNDLE - files
        if missing or any(name.startswith("frozen_adapter/") for name in files):
            raise ValueError(
                "Expected a standalone PEFT inference bundle at the selected subdirectory; "
                f"missing={sorted(missing)}. Merged-only models and frozen-parent bundles "
                "are incompatible with this evaluator."
            )
        selected = REQUIRED_BUNDLE | (files & INFERENCE_SIDECARS)
    else:
        weights = {name for name in files if re.fullmatch(r"model(?:-\d+-of-\d+)?\.safetensors", name)}
        if "config.json" not in files or not weights:
            raise ValueError("base repository requires native config and safetensors inference weights")
        selected = weights | (files & INFERENCE_SIDECARS)
        if "model.safetensors.index.json" in files:
            selected.add("model.safetensors.index.json")
        elif weights != {"model.safetensors"}:
            raise ValueError("sharded base weights require model.safetensors.index.json")
    selected |= {name for name in files if re.fullmatch(r"chat_templates/[^/]+\.jinja", name)}
    root = Path(snapshot_download(
        repo_id=repo, revision=revision, cache_dir=CACHE,
        allow_patterns=sorted(prefix + name for name in selected),
        max_workers=8, force_download=False,
    ))
    if root.name != revision:
        raise ValueError("HF snapshot did not resolve to the requested immutable revision")
    root = root / subdir if subdir else root
    for name in selected:
        if not (root / name).is_file() or (root / name).stat().st_size == 0:
            raise FileNotFoundError(f"incomplete inference file: {root / name}")
    return root, sorted(selected)


def tensor_headers(path: Path) -> dict[str, TensorHeader]:
    with open_tensors(path, framework="pt", device="cpu") as handle:
        return {key: {"shape": handle.get_slice(key).get_shape(),
                      "dtype": handle.get_slice(key).get_dtype()} for key in handle.keys()}


def volume_bundle(adapter_dir: str) -> tuple[Path, list[str]]:
    """Inspect the exact saved bundle in place; never select or modify a checkpoint."""
    bundle = Path(adapter_dir)
    if not bundle.is_dir():
        raise FileNotFoundError(f"saved adapter directory does not exist: {bundle}")
    files = {name for name in REQUIRED_BUNDLE | INFERENCE_SIDECARS if (bundle / name).is_file()}
    missing = REQUIRED_BUNDLE - files
    if missing or (bundle / "frozen_adapter").exists():
        raise ValueError(f"incompatible saved PEFT bundle: missing={sorted(missing)}; frozen parents unsupported")
    files |= {str(path.relative_to(bundle)) for path in (bundle / "chat_templates").glob("*.jinja")
              if path.is_file()}
    for name in files:
        if (bundle / name).stat().st_size == 0:
            raise ValueError(f"empty inference file: {bundle / name}")
    return bundle, sorted(files)


def adapter_preflight(bundle: Path, inventory: JsonDict, model_id: str,
                      model_revision: str) -> tuple[PreTrainedTokenizerBase, JsonDict]:
    """Check the evaluator's sidefile contract before downloading the 27B base."""
    saved = read_json(bundle / "adapter_config.json")
    parent = read_json(bundle / RUN_CONFIG_FILE)
    rank = standard_lora_rank(saved, parent)
    pinned_snapshot = str(Path(CACHE) / ("models--" + model_id.replace("/", "--")) / "snapshots" / model_revision)
    scope = read_json(bundle / TRAINABLE_SCOPE_FILE)
    tokenizer = load_checkpoint_text_tokenizer(
        bundle, [as_str(token) for token in as_list(inventory["tokens"])])
    semantic = as_dict(scope["semantic_tokens"])
    indices = saved.get("trainable_token_indices")
    if (
        not isinstance(saved.get("target_modules"), (list, str))
        or not isinstance(indices, dict) or len(indices) != 2
        or any(ids != semantic["token_ids"] for ids in indices.values())
        or parent.get("model_id") not in (model_id, pinned_snapshot) or scope.get("errors")
        or (bundle / "frozen_adapter").exists() or (bundle / "frozen_bundle.json").exists()
    ):
        raise ValueError(f"incompatible language rank-{rank} LoRA / atlas rows / base-model sidefiles")
    headers = tensor_headers(bundle / "adapter_model.safetensors")
    rows = [value["shape"] for key, value in headers.items() if "trainable_tokens_delta" in key]
    if len(rows) != 2 or any(len(shape) != 2 or shape[0] != 154 for shape in rows):
        raise ValueError("adapter weights must contain both 154-row atlas tensors")
    a = {name.removesuffix(".lora_A.weight"): value["shape"]
         for name, value in headers.items() if name.endswith(".lora_A.weight")}
    b = {name.removesuffix(".lora_B.weight"): value["shape"]
         for name, value in headers.items() if name.endswith(".lora_B.weight")}
    if (not a or a.keys() != b.keys()
            or any(len(shape) != 2 or shape[0] != rank for shape in a.values())
            or any(len(shape) != 2 or shape[1] != rank for shape in b.values())):
        raise ValueError(f"adapter weights must contain paired rank-{rank} LoRA tensors")
    return tokenizer, {"adapter_config": saved, "semantic_tokens": semantic,
                       "lora_rank": rank, "lora_alpha": saved["lora_alpha"],
                       "saved_model_id": parent["model_id"],
                       "training_profile": parent["profile"], "adapter_tensors": len(headers),
                       "tokenizer_size": len(tokenizer)}


def base_preflight(base: Path, files: list[str], bundle: Path, inventory: JsonDict,
                   checkpoint: JsonDict) -> JsonDict:
    """Inspect tensor headers and configuration only; never instantiate the base model."""
    config = AutoConfig.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    text_config = getattr(config, "text_config", config)
    if getattr(text_config, "max_position_embeddings", CONTEXT) < CONTEXT:
        raise ValueError("base model context is smaller than 4096")
    tokenizer = load_tokenizer(base)
    tokens = [as_str(token) for token in as_list(inventory["tokens"])]
    semantic = as_dict(checkpoint["semantic_tokens"])
    tokenizer.add_tokens([AddedToken(token, normalized=False, special=False) for token in tokens])
    if [tokenizer.encode(token, add_special_tokens=False) for token in tokens] != [
        [token_id] for token_id in as_list(semantic["token_ids"])
    ]:
        raise ValueError("pinned base tokenizer and checkpoint atlas token IDs differ")
    headers: dict[str, TensorHeader] = {}
    locations: dict[str, str] = {}
    for name in files:
        if name.endswith(".safetensors"):
            shard = tensor_headers(base / name)
            if headers.keys() & shard.keys():
                raise ValueError("duplicate tensors across base weight files")
            headers.update(shard)
            locations.update(dict.fromkeys(shard, name))
    index_path = base / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = read_json(index_path)["weight_map"]
        if weight_map != locations:
            raise ValueError("base weight index is incomplete")
    adapter_config = LoraConfig.from_pretrained(str(bundle), local_files_only=True)
    # Check the actual inference architecture without allocating model weights.
    # The HF files also contain training-only MTP tensors that are not modules
    # in AutoModelForMultimodalLM and must not become false-positive LoRA targets.
    with torch.device("meta"):
        build: ArchitectureFactory = AutoModelForMultimodalLM.from_config
        architecture = build(config, dtype=torch.bfloat16, attn_implementation="sdpa")
    intended = set(language_linear_targets(architecture, discover_components(architecture)))
    modules = {name for name, _ in architecture.named_modules() if name}
    matched = {name for name in modules if check_target_module_exists(adapter_config, name)}
    del architecture
    if not intended or intended != matched:
        raise ValueError("LoRA targets must match exactly the pinned base's language linear layers")
    saved = as_dict(checkpoint["adapter_config"])
    rank = standard_lora_rank(saved, read_json(bundle / RUN_CONFIG_FILE))
    row_modules = as_dict(saved["trainable_token_indices"])
    if set(row_modules) != {"model.language_model.embed_tokens", "lm_head"}:
        raise ValueError("expected both Qwen language input and output atlas rows")
    token_widths: dict[str, int] = {}
    for module in row_modules:
        header = headers.get(module + ".weight")
        shape = header["shape"] if header is not None else []
        if len(shape) != 2:
            raise ValueError("unsupported atlas embedding/head module")
        token_widths[module] = shape[1]
    expected = expected_adapter_shapes(
        {module: headers[module + ".weight"]["shape"] for module in intended}, token_widths, rank,
    )
    # Match prepare_semantic_tokens: resize to the saved tokenizer, padded to 128 rows.
    current_vocab = headers["model.language_model.embed_tokens.weight"]["shape"][0]
    runtime_vocab = ((max(as_int(checkpoint["tokenizer_size"]), current_vocab) + 127) // 128) * 128
    ids = [as_int(token_id) for token_id in as_list(semantic["token_ids"])]
    if min(ids) < 0 or max(ids) >= runtime_vocab:
        raise ValueError("atlas token IDs exceed the evaluator's resized embedding/head slots")
    actual = {name: value["shape"] for name, value in tensor_headers(bundle / "adapter_model.safetensors").items()}
    if actual != expected:
        raise ValueError("adapter tensor names/shapes do not match complete language LoRA plus atlas rows")
    visual = tensor_headers(bundle / VISUAL_STATE_FILE)
    expected_visual = {f"base_model.model.{name}": value["shape"] for name, value in headers.items()
                       if name.startswith("model.visual.")}
    if not expected_visual or {name: value["shape"] for name, value in visual.items()} != expected_visual:
        raise ValueError("saved visual sidefile is incomplete or incompatible with the pinned base")
    return {"model_type": config.model_type, "base_tensors": len(headers),
            "runtime_vocab_size": runtime_vocab,
            "language_lora_modules": len(intended), "visual_tensors": len(visual),
            "visual_dtypes": dict[str, JsonValue](Counter(value["dtype"] for value in visual.values()))}


def file_manifest(root: Path, names: list[str]) -> JsonDict:
    return {name: {"bytes": (root / name).stat().st_size, "sha256": sha256_file(root / name)}
            for name in names}
