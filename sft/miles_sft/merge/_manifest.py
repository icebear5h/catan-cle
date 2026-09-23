"""Content-addressed provenance and standalone validation of completed exports."""

from __future__ import annotations

from pathlib import Path

from sft.safetensor_types import TensorHeader

from ._checkpoint import (
    BASE_ASSETS,
    inspect_checkpoint,
    tensor_bytes,
    tensor_headers,
    validate_tokenizer,
)
from ._contracts import (
    ADAPTER,
    COMPLETE,
    DTYPE_BYTES,
    INDEX,
    MANIFEST,
    ROW_MODULES,
    SCHEMA,
    VISUAL,
    VISUAL_COUNT,
    FileIdentity,
    MergePlan,
    Rounding,
    file_identity,
    integer,
    object_map,
    read_json,
    relative_name,
    require,
    sequence,
    sha256,
    text,
    valid_token_ids,
)
from ._preflight import adapter_mappings, validate_lora_config, visual_mapping


def build_manifest(base: Path, adapter: Path, plan: MergePlan, base_revision: str | None,
                   weights: dict[str, str], headers: dict[str, TensorHeader],
                   rounding: dict[str, Rounding], files: dict[str, FileIdentity]) -> dict[str, object]:
    tensors: dict[str, object] = {}
    for key, header in headers.items():
        operation = "preserved"
        adapter_keys: list[str] = []
        if key in plan.lora:
            operation, adapter_keys = "lora_fp32_then_bf16", list(plan.lora[key])
        elif key in plan.rows:
            operation, adapter_keys = "replacement_rows_bf16", [plan.rows[key]]
        elif key in plan.visual:
            operation, adapter_keys = "visual_fp32_exact", [plan.visual[key]]
        tensors[key] = {**header, "shard": weights[key], "operation": operation,
                        "base_shard": plan.base.weight_map[key],
                        "base_dtype": plan.base.headers[key]["dtype"], "adapter_keys": adapter_keys}
    return {
        "schema": SCHEMA, "status": "completed", "base": str(base), "adapter": str(adapter),
        "declared_base_revision": base_revision, "base_files": plan.base_files,
        "adapter_files": plan.adapter_files, "output_files": files, "tensors": tensors,
        "adapter_config": read_json(adapter / "adapter_config.json"),
        "adapter_tensor_headers": tensor_headers(adapter / ADAPTER),
        "visual_tensor_headers": tensor_headers(adapter / VISUAL),
        "token_ids": list(plan.token_ids), "vocab_size": 248320, "tie_word_embeddings": False,
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "scaling": 2.0,
                 "modules": len(plan.lora), "independent_factors": True},
        "visual": {"tensors": len(plan.visual), "dtype": "F32", "bitwise_preserved": True},
        "assets": {name: str(path) for name, path in sorted(plan.assets.items())},
        "rounding": rounding,
        "semantics": {
            "initialization": "functional warmstart for fresh Megatron LoRA over a frozen merged base",
            "exact_factor_continuation": False, "bitwise_adapter_arithmetic": False,
            "training_state_restored": False, "token_rows": "replacement, not additive; frozen in base",
            "mtp": "all base MTP tensors retained unchanged; freeze when loading for training",
            "arithmetic": "CPU FP32 W + (32/16)*(B@A); final merged weights and rows rounded to BF16",
            "limits": "BF16 rounding may erase small updates. FP32 accumulation and reassociated "
                      "matmuls differ from live adapter arithmetic; no logit-equivalence claim. "
                      "Untouched tensors retain source dtype; loaders must preserve visual FP32.",
            "bf16_epsilon": 0.0078125, "bf16_max_finite": 3.3895313892515355e38,
            "bf16_smallest_normal": 1.1754943508222875e-38,
            "fp32_epsilon": 1.1920928955078125e-7,
            "rounding_metrics": "elementwise final FP32-to-BF16 cast only; not matmul/logit error",
        },
    }


def _headers(value: object) -> dict[str, TensorHeader]:
    result: dict[str, TensorHeader] = {}
    for key, raw in object_map(value).items():
        header = object_map(raw)
        result[key] = {"shape": [integer(v) for v in sequence(header.get("shape"))],
                       "dtype": text(header.get("dtype"))}
        tensor_bytes(result[key])
    return result


def _validate_provenance(manifest: dict[str, object], tensors: dict[str, object],
                         files: dict[str, object], base_headers: dict[str, TensorHeader]) -> None:
    base_files, adapter_files = (object_map(manifest.get(key)) for key in ("base_files", "adapter_files"))
    require({INDEX, "config.json", "generation_config.json"} <= base_files.keys(), "missing base identities")
    require({ADAPTER, VISUAL, "adapter_config.json", "tokenizer.json", "tokenizer_config.json"}
            <= adapter_files.keys(), "missing adapter identities")
    for source in (base_files, adapter_files):
        for name, raw in source.items():
            relative_name(name)
            entry = object_map(raw)
            digest = text(entry.get("sha256"))
            require(integer(entry.get("bytes")) > 0 and len(digest) == 64
                    and all(c in "0123456789abcdef" for c in digest), "invalid source identity")
    config = object_map(manifest.get("adapter_config"))
    validate_lora_config(config)
    lora, rows = adapter_mappings(config, _headers(manifest.get("adapter_tensor_headers")), base_headers)
    visual = visual_mapping(_headers(manifest.get("visual_tensor_headers")), base_headers)
    for key, raw in tensors.items():
        entry = object_map(raw)
        require(relative_name(entry.get("base_shard")) in base_files, f"missing source shard identity: {key}")
        require(entry.get("base_dtype") in DTYPE_BYTES, f"unsupported base dtype: {key}")
        expected_keys: list[str] = []
        operation = "preserved"
        if key in lora:
            expected_keys, operation = list(lora[key]), "lora_fp32_then_bf16"
        elif key in rows:
            expected_keys, operation = [rows[key]], "replacement_rows_bf16"
        elif key in visual:
            expected_keys, operation = [visual[key]], "visual_fp32_exact"
        require(entry.get("adapter_keys") == expected_keys and entry.get("operation") == operation,
                f"tensor source/operation mismatch: {key}")
    lora_report = object_map(manifest.get("lora"))
    require(lora_report == {"r": 16, "alpha": 32, "dropout": 0.05, "scaling": 2.0,
                            "modules": len(lora), "independent_factors": True}, "invalid LoRA report")
    assets = object_map(manifest.get("assets"))
    shards = {text(object_map(raw).get("shard")) for raw in tensors.values()}
    require(set(assets) == set(files) - shards - {INDEX}, "incomplete asset provenance")
    for name, path in assets.items():
        owner = "base" if name in BASE_ASSETS else "adapter"
        source_files = base_files if owner == "base" else adapter_files
        require(path == str(Path(text(manifest.get(owner))) / name)
                and name in source_files and files[name] == source_files[name],
                f"asset source identity mismatch: {name}")


def validate_payload(output: Path, manifest: dict[str, object]) -> None:
    require(manifest.get("schema") == SCHEMA and manifest.get("status") == "completed",
            "unsupported or incomplete merge manifest")
    files = object_map(manifest.get("output_files"))
    require({INDEX, "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"}
            <= files.keys(), "manifest missing required output files")
    require(bool({"processor_config.json", "preprocessor_config.json"} & files.keys()),
            "manifest missing native processor configuration")
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    require(actual - {MANIFEST, COMPLETE} == set(files), "export file inventory differs from manifest")
    for name, raw in files.items():
        path = output / relative_name(name)
        require(not path.is_symlink() and path.resolve().is_relative_to(output.resolve()),
                f"output must contain regular local files: {name}")
        identity = object_map(raw)
        require(set(identity) == {"bytes", "sha256"} and identity == file_identity(path),
                f"output hash/size mismatch: {name}")
    checkpoint = inspect_checkpoint(output)
    tensors = object_map(manifest.get("tensors"))
    require(set(tensors) == set(checkpoint.headers), "manifest tensor inventory differs from HF index")
    visual: set[str] = set()
    rounded: set[str] = set()
    base_headers: dict[str, TensorHeader] = {}
    for key, header in checkpoint.headers.items():
        declared = object_map(tensors[key])
        base_headers[key] = {"shape": header["shape"], "dtype": text(declared.get("base_dtype"))}
        require(declared.get("shape") == header["shape"] and declared.get("dtype") == header["dtype"]
                and declared.get("shard") == checkpoint.weight_map[key], f"tensor metadata mismatch: {key}")
        operation = declared.get("operation")
        if key.startswith("model.visual."):
            visual.add(key)
            require(header["dtype"] == "F32" and operation == "visual_fp32_exact", "visual FP32 invariant violated")
        elif operation in ("lora_fp32_then_bf16", "replacement_rows_bf16"):
            rounded.add(key)
            require(header["dtype"] == "BF16", f"merged weight must be BF16: {key}")
        else:
            require(operation == "preserved" and header["dtype"] == declared.get("base_dtype"),
                    f"invalid preserved tensor: {key}")
    require(len(visual) == VISUAL_COUNT, "export must retain 333 FP32 visual tensors")
    for module in ROW_MODULES:
        require(object_map(tensors[module + ".weight"]).get("operation") == "replacement_rows_bf16",
                "both untied token matrices must have replacement rows")
    require(set(object_map(manifest.get("rounding"))) == rounded, "incomplete rounding report")
    ids = valid_token_ids(manifest.get("token_ids"))
    validate_tokenizer(output, ids)
    require(manifest.get("vocab_size") == 248320 and manifest.get("tie_word_embeddings") is False,
            "invalid vocabulary/tying report")
    _validate_provenance(manifest, tensors, files, base_headers)


def load_manifest(output: Path, *, expected_sha256: str | None = None) -> dict[str, object]:
    """Read a completed manifest, optionally pinned to an externally trusted digest.

    This verifies the completion seal; use validate_merged_export for all file hashes.
    Hashes establish content identity, not authenticity without an external digest.
    """
    marker = read_json(output / COMPLETE)
    digest = sha256(output / MANIFEST)
    require(marker.get("schema") == SCHEMA and marker.get("manifest_sha256") == digest,
            "completion marker does not seal this manifest")
    require(expected_sha256 is None or digest == expected_sha256, "pinned manifest digest differs")
    manifest = read_json(output / MANIFEST)
    require(manifest.get("schema") == SCHEMA and manifest.get("status") == "completed",
            "unsupported or incomplete merge manifest")
    return manifest


def validate_merged_export(output: Path, *, expected_sha256: str | None = None) -> dict[str, object]:
    """Verify completion, all SHA256s, index completeness, shapes/dtypes and tokenizer IDs.

    Original source directories need not exist; their immutable identities remain in
    the manifest. No torch model, GPU, PEFT, network access or weight download is used.
    """
    manifest = load_manifest(output, expected_sha256=expected_sha256)
    validate_payload(output, manifest)
    return manifest


def output_size(headers: dict[str, TensorHeader]) -> int:
    return sum(tensor_bytes(header) for header in headers.values())
