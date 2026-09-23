"""CPU-only, function-preserving expansion of a standard Catan LoRA bundle."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import torch
from peft import LoraConfig
from peft.tuners.tuners_utils import check_target_module_exists
from safetensors.torch import save_file

from sft.json_types import (
    JsonDict,
    JsonLike,
    as_dict,
    as_float,
    as_int,
    as_list,
    as_str,
    json_dict,
    json_list,
    load_json_dict,
)
from sft.safetensor_types import TensorHeader, open_tensors

from ._constants import (
    ADAPTER_FILE,
    CONFIG_FILE,
    EXPANSION_REPORT_FILE,
    INFERENCE_ASSETS,
    SCOPE_FILE,
    VISUAL_FILE,
)
from ._shapes import expected_adapter_shapes, standard_lora_rank, tensor_headers


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(root: Path, names: list[str]) -> JsonDict:
    return {name: {"bytes": (root / name).stat().st_size, "sha256": _sha256(root / name)}
            for name in sorted(names)}


def _json(path: Path) -> JsonDict:
    return load_json_dict(path)


def _write_json(path: Path, value: Mapping[str, JsonLike]) -> None:
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _saved_key(name: str) -> str:
    return (name.replace(".lora_A.default.weight", ".lora_A.weight")
            .replace(".lora_B.default.weight", ".lora_B.weight")
            .removesuffix(".default"))


def _expand_pair(
    a: torch.Tensor, b: torch.Tensor, *, rank: int, generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        a.device.type != "cpu" or b.device.type != "cpu"
        or a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]
        or rank <= a.shape[0] or min(*a.shape, *b.shape) <= 0
        or not a.is_floating_point() or a.dtype != b.dtype
        or not torch.isfinite(a).all() or not torch.isfinite(b).all()
    ):
        raise ValueError("expected finite CPU LoRA A/B factors with a strictly larger target rank")
    old_rank = a.shape[0]
    new_a = torch.empty((rank, a.shape[1]), dtype=a.dtype, device="cpu")
    new_a[:old_rank].copy_(a)
    # PEFT's ordinary Linear LoRA initialization: U(-1/sqrt(fan_in), +1/sqrt(fan_in)).
    torch.nn.init.kaiming_uniform_(new_a[old_rank:], a=math.sqrt(5), generator=generator)
    new_b = torch.zeros((b.shape[0], rank), dtype=b.dtype, device="cpu")
    new_b[:, :old_rank].copy_(b)
    return new_a, new_b


def _verify_pair(a: torch.Tensor, b: torch.Tensor, new_a: torch.Tensor,
                 new_b: torch.Tensor, *, scale: float) -> JsonDict:
    old_rank = a.shape[0]
    # Compare bytes, including signed zero, rather than approximate floating values.
    retained = (
        torch.equal(a.contiguous().view(torch.uint8), new_a[:old_rank].contiguous().view(torch.uint8))
        and torch.equal(b.contiguous().view(torch.uint8), new_b[:, :old_rank].contiguous().view(torch.uint8))
    )
    tail_a, tail_b = new_a[old_rank:], new_b[:, old_rank:]
    live_a = bool(torch.isfinite(tail_a).all() and torch.all(torch.any(tail_a != 0, dim=1)))
    zero_b = bool(torch.count_nonzero(tail_b) == 0)
    if not retained or not live_a or not zero_b:
        raise RuntimeError("LoRA expansion failed retained-block / nonzero-A / zero-B verification")
    # Two deterministic mathematical probes, never a dense out_features x in_features BA.
    x = torch.stack((torch.ones(a.shape[1], dtype=torch.float64),
                     torch.linspace(-1, 1, a.shape[1], dtype=torch.float64)))
    original = ((x @ a.double().T) @ b.double().T) * scale
    expanded = ((x @ new_a.double().T) @ new_b.double().T) * scale
    torch.testing.assert_close(expanded, original, rtol=1e-11, atol=1e-11)
    return {"retained_blocks_bitwise_equal": retained, "new_A_rows_nonzero": live_a,
            "new_B_zero": zero_b, "probe_max_abs_error": float((expanded - original).abs().max())}


def _shape(parameter: JsonDict) -> list[int]:
    return [as_int(width) for width in as_list(parameter["shape"])]


def _expanded_scope(
    scope: JsonDict, headers: dict[str, TensorHeader], rank: int, alpha: int,
) -> JsonDict:
    result = copy.deepcopy(scope)
    parameters = [as_dict(parameter) for parameter in as_list(result["parameters"])]
    for parameter in parameters:
        key = _saved_key(as_str(parameter["name"]))
        if parameter["category"] == "language_lora":
            parameter["shape"] = json_list(headers[key]["shape"])
    for category, group in as_dict(result["groups"]).items():
        entries = [p for p in parameters if p["category"] == category]
        as_dict(group).update(
            tensors=len(entries), parameters=sum(math.prod(_shape(p)) for p in entries),
            names=json_list(p["name"] for p in entries),
        )
    result["lora"] = {"rank": rank, "alpha": alpha, "scaling": alpha / rank, "use_rslora": False}
    return result


def expand_lora_bundle(
    source: str | Path, destination: str | Path, *, rank: int = 16, alpha: int = 32, seed: int = 44,
) -> JsonDict:
    """Expand the approved r8/a16 bundle to r16/a32, returning its durable receipt.

    ``destination`` must not exist. A successful bundle contains
    ``lora_expansion.json`` with hashes of every input/output inference file,
    per-pair numerical checks and the algebraic proof. A failed conversion raises
    and may leave an incomplete destination without that success receipt.
    Source training/optimizer receipts are not asserted to describe a new run.
    """
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination = destination.resolve()
    if destination.is_relative_to(source):
        raise ValueError("destination must be outside the source bundle")
    if type(rank) is not int or type(alpha) is not int or (rank, alpha) != (16, 32):
        raise ValueError("only the approved rank-16 alpha-32 expansion is supported")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    required = {ADAPTER_FILE, "adapter_config.json", VISUAL_FILE, CONFIG_FILE, SCOPE_FILE,
                "tokenizer_config.json", "tokenizer.json"}
    if missing := sorted(name for name in required if not (source / name).is_file()):
        raise FileNotFoundError(f"incomplete source bundle: {missing}")
    if (source / "frozen_adapter").exists() or (source / "frozen_bundle.json").exists():
        raise ValueError("frozen adapter parents are unsupported")
    saved, parent, scope = (_json(source / name) for name in ("adapter_config.json", CONFIG_FILE, SCOPE_FILE))
    old_rank = standard_lora_rank(saved, parent)
    source_scaling = as_float(saved["lora_alpha"]) / old_rank
    if (old_rank, saved["lora_alpha"]) != (8, 16) or alpha / rank != source_scaling:
        raise ValueError("source must be rank-8 alpha-16 with unchanged standard-LoRA scaling")
    if scope.get("errors") or scope.get("profile") != "vision_tokens_lora":
        raise ValueError("invalid source trainable scope")
    components = as_dict(scope["components"])
    semantic = as_dict(scope["semantic_tokens"])
    ids, tokens = as_list(semantic["token_ids"]), as_list(semantic["tokens"])
    if (len(ids) != 154 or len(set(ids)) != 154 or len(tokens) != 154 or len(set(tokens)) != 154
            or any(type(i) is not int or i < 0 for i in ids)):
        raise ValueError("source must retain exactly 154 unique atlas tokens and IDs")
    rows = {as_str(components["input_embedding"]): ids, as_str(components["output_head"]): ids}
    if len(rows) != 2 or saved.get("trainable_token_indices") != rows:
        raise ValueError("source must contain the exact input/output replacement row mapping")
    headers = tensor_headers(source / ADAPTER_FILE)
    parameters = [as_dict(parameter) for parameter in as_list(scope["parameters"])]
    declared = {_saved_key(as_str(p["name"])): p["shape"] for p in parameters
                if p["category"] in ("language_lora", "atlas_input_rows", "atlas_output_rows")}
    if declared != {key: value["shape"] for key, value in headers.items()}:
        raise ValueError("source adapter tensor names/shapes differ from its trainable scope")
    pairs = sorted(key.removesuffix(".lora_A.weight") for key in headers if key.endswith(".lora_A.weight"))
    if len(pairs) != 496:
        raise ValueError(f"expected all 496 language Linear LoRA pairs; found {len(pairs)}")
    peft_config = LoraConfig.from_pretrained(str(source), local_files_only=True)
    linear_shapes: dict[str, list[int]] = {}
    for key in pairs:
        module = key.removeprefix("base_model.model.")
        b_header = headers.get(key + ".lora_B.weight")
        a, b = headers[key + ".lora_A.weight"]["shape"], b_header["shape"] if b_header else []
        if (not module.startswith(as_str(components["language"]) + ".")
                or not check_target_module_exists(peft_config, module)
                or len(a) != 2 or len(b) != 2 or a[0] != old_rank or b[1] != old_rank):
            raise ValueError(f"invalid language LoRA pair: {key}")
        linear_shapes[module] = [b[0], a[1]]
    hidden_size = as_int(components["hidden_size"])
    expected = expected_adapter_shapes(linear_shapes, dict.fromkeys(rows, hidden_size), old_rank)
    if {key: value["shape"] for key, value in headers.items()} != expected:
        raise ValueError("source must contain only complete language LoRA plus 154 input/output rows")
    visual = tensor_headers(source / VISUAL_FILE)
    if len(visual) != 333 or Counter(value["dtype"] for value in visual.values()) != {"F32": 333}:
        raise ValueError("source must contain all 333 FP32 visual tensors")
    assets = sorted(name for name in INFERENCE_ASSETS if (source / name).is_file())
    assets += sorted(str(p.relative_to(source)) for p in (source / "chat_templates").glob("*.jinja") if p.is_file())
    if not {"processor_config.json", "preprocessor_config.json"}.intersection(assets):
        raise ValueError("source must include its native processor configuration")
    source_names = sorted(required | set(assets))
    source_manifest = _manifest(source, source_names)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    output: dict[str, torch.Tensor] = {}
    checks: JsonDict = {}
    with open_tensors(source / ADAPTER_FILE, framework="pt", device="cpu") as handle:
        adapter_metadata = handle.metadata()
        for key in pairs:
            factor_a = handle.get_tensor(key + ".lora_A.weight")
            factor_b = handle.get_tensor(key + ".lora_B.weight")
            new_a, new_b = _expand_pair(factor_a, factor_b, rank=rank, generator=generator)
            output[key + ".lora_A.weight"], output[key + ".lora_B.weight"] = new_a, new_b
        for key in headers:
            if key not in output:
                output[key] = handle.get_tensor(key).clone()
    # Create once, after all source validation and factor allocation have succeeded.
    destination.mkdir(parents=True, exist_ok=False)
    save_file(output, destination / ADAPTER_FILE, metadata=adapter_metadata)
    del output
    with open_tensors(source / ADAPTER_FILE, framework="pt", device="cpu") as original, open_tensors(
        destination / ADAPTER_FILE, framework="pt", device="cpu",
    ) as expanded:
        for key in pairs:
            checks[key] = _verify_pair(
                original.get_tensor(key + ".lora_A.weight"), original.get_tensor(key + ".lora_B.weight"),
                expanded.get_tensor(key + ".lora_A.weight"), expanded.get_tensor(key + ".lora_B.weight"),
                scale=alpha / rank,
            )
        for key in headers:
            if ".lora_A." not in key and ".lora_B." not in key:
                if not torch.equal(original.get_tensor(key).view(torch.uint8), expanded.get_tensor(key).view(torch.uint8)):
                    raise RuntimeError(f"non-LoRA tensor changed: {key}")
    copied = sorted(set(assets) | {VISUAL_FILE})
    for name in copied:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    output_headers = tensor_headers(destination / ADAPTER_FILE)
    updated_scope = _expanded_scope(scope, output_headers, rank, alpha)
    _write_json(destination / SCOPE_FILE, updated_scope)
    _write_json(destination / "adapter_config.json", {**saved, "r": rank, "lora_alpha": alpha})
    _write_json(destination / CONFIG_FILE, {**parent, "lora_rank": rank, "lora_alpha": alpha,
                "output_dir": str(destination), "initial_bundle": str(source), "resume_from_checkpoint": None})
    output_manifest = _manifest(destination, source_names)
    if any(output_manifest[name] != source_manifest[name] for name in copied):
        raise RuntimeError("copied visual/tokenizer/processor bytes changed")
    if _manifest(source, source_names) != source_manifest:
        raise RuntimeError("source bundle changed during expansion")
    receipt: JsonDict = {
        "schema": "catan_lora_expansion/v1", "status": "completed",
        "source": str(source), "destination": str(destination), "seed": seed,
        "method": "retain_A_rows_B_columns_append_kaiming_uniform_A_zero_B",
        "source_rank": old_rank, "source_alpha": saved["lora_alpha"], "rank": rank, "alpha": alpha,
        "scaling": alpha / rank, "use_rslora": False, "language_lora_modules": len(pairs),
        "source_files": source_manifest, "output_files": output_manifest,
        "copied_files_bitwise_equal": json_list(copied), "adapter_tensors": len(output_headers),
        "source_adapter_parameters": sum(math.prod(h["shape"]) for h in headers.values()),
        "adapter_parameters": sum(math.prod(h["shape"]) for h in output_headers.values()),
        "visual_tensors": len(visual),
        "visual_dtypes": json_dict(Counter(h["dtype"] for h in visual.values())),
        "algebraic_parity": {"exact": True, "source_scaling": source_scaling,
            "output_scaling": alpha / rank,
            "identity": "(32/16) [B8, 0] [A8; A_new] = (16/8) B8 A8",
            "scope": "adapter function in exact arithmetic; no model/logit evaluation",
            "probe_dtype": "float64", "probe_rtol": 1e-11, "probe_atol": 1e-11,
            "pairs_verified": len(checks), "pairs": checks},
        "initialization": {"distribution": "kaiming_uniform_(a=sqrt(5), mode=fan_in)",
            "rng": "local torch.Generator(cpu), sorted saved module names", "torch_version": torch.__version__},
        "metadata": {"training_state_restored": False,
            "scope": "source trainable scope with expanded factor shapes/counts; text training re-audits scope",
            "omitted_source_files": json_list(sorted(
                str(p.relative_to(source)) for p in source.rglob("*")
                if p.is_file() and str(p.relative_to(source)) not in source_names)),
            "receipt_self_hash": "excluded from output_files; hash the receipt separately if required"},
    }
    _write_json(destination / EXPANSION_REPORT_FILE, receipt)
    return receipt
