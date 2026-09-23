from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import TypedDict

import torch
from transformers import PreTrainedTokenizerBase

from evals.catan_board_bench.tokens import atlas_tokens
from sft.json_types import JsonDict, JsonLikeDict, as_dict, as_int, as_list, as_str
from sft.launchers.board_fluency import modal_board_fluency_eval as shared
from sft.launchers.modal_catan_vision_sft import sft_runs
from sft.scripts.eval import eval_qwen_vl_adapter as evaluator
from sft.scripts.train import train_trl_catan_vision as trainer
from sft.scripts.train.train_trl_catan_vision import (
    _message_pair,
    encode_text_pair,
    pad_text_inputs,
)

from ._config import ATOL, BASE, RTOL
from ._data import progress, rows_at


class ProbeResult(TypedDict):
    """Active-vocabulary logits plus the token evidence two probes must share."""

    logits: torch.Tensor
    active_ids: list[int]
    vocab_sha256: str
    greedy_ids: list[int]
    atlas_ids: int | list[int]
    prompt_ids: list[list[int]]


def probe_metadata(result: ProbeResult) -> JsonLikeDict:
    """Every probe field except the logits tensor, in probe order."""
    return {"active_ids": result["active_ids"], "vocab_sha256": result["vocab_sha256"],
            "greedy_ids": result["greedy_ids"], "atlas_ids": result["atlas_ids"],
            "prompt_ids": result["prompt_ids"]}


def visual_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    visual = trainer.resolve_wrapped_module(model, "model.visual")
    for name, tensor in sorted(visual.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def visual_file_digest(path: Path) -> str:
    """Hash every visual tensor, not nondeterministic safetensors JSON ordering."""
    digest = hashlib.sha256()
    with shared.open_tensors(path, framework="pt", device="cpu") as handle:
        names = sorted(handle.keys())
        if len(names) != 333:
            raise ValueError("expected the full 333-tensor visual sidefile")
        for name in names:
            tensor = handle.get_tensor(name)
            if tensor.dtype != torch.float32:
                raise ValueError(f"frozen visual precision changed: {name}")
            digest.update(name.encode())
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def probe(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    rows: list[JsonDict],
) -> ProbeResult:
    """Full active-vocabulary logits at fixed real native prompt boundaries."""
    features: list[dict[str, list[int]]] = []
    for row in rows:
        prompt, answer = _message_pair(row, line_number=0, input_mode="text")
        pair = encode_text_pair(tokenizer, prompt, answer, max_sequence_length=4096)
        prefix = as_list(pair["labels"]).count(-100)
        features.append({"input_ids": [as_int(i) for i in as_list(pair["input_ids"])[:prefix]]})
    active = sorted(set(tokenizer.get_vocab().values()))
    was_training = model.training
    tf32 = torch.backends.cuda.matmul.allow_tf32
    wrapped_forward = model.forward
    try:
        model.eval()
        torch.backends.cuda.matmul.allow_tf32 = False
        # Accelerate installs an inner autocast wrapper during training. Compare
        # the same inference arithmetic without altering the training wrapper.
        if "_original_forward" in model.__dict__:
            model.forward = model.__dict__["_original_forward"]
        device = model.device
        if not isinstance(device, torch.device):
            raise TypeError("probe model must expose a torch.device")
        inputs = {k: v.to(device) for k, v in pad_text_inputs(tokenizer, features, left=True).items()}
        # Match text inference: BF16 base with PEFT's FP32 adapter arithmetic.
        # Autocasting the LoRA matmuls changes numerical behavior with rank.
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            output = model(**inputs, use_cache=False, logits_to_keep=1)
        logits = output.logits[:, -1].detach().float().cpu()
        if not torch.isfinite(logits).all():
            raise RuntimeError("gate produced nonfinite logits")
        return {"logits": logits[:, active], "active_ids": active,
                "vocab_sha256": shared.digest(tokenizer.get_vocab()),
                "greedy_ids": logits.argmax(-1).tolist(),
                "atlas_ids": tokenizer.convert_tokens_to_ids(atlas_tokens()),
                "prompt_ids": [f["input_ids"] for f in features]}
    finally:
        model.forward = wrapped_forward
        model.train(was_training)
        torch.backends.cuda.matmul.allow_tf32 = tf32


def compare_probes(before: ProbeResult, after: ProbeResult) -> JsonLikeDict:
    a, b = before["logits"], after["logits"]
    maximum = float((a - b).abs().max()) if a.shape == b.shape else None
    progress(f"parity: max_abs={maximum}, before_ids={before['greedy_ids']}, after_ids={after['greedy_ids']}")
    before_metadata, after_metadata = probe_metadata(before), probe_metadata(after)
    for key in ("active_ids", "vocab_sha256", "greedy_ids", "atlas_ids", "prompt_ids"):
        if before_metadata[key] != after_metadata[key]:
            raise ValueError(f"gate equivalence failed: {key}; max_abs={maximum}; before={before_metadata[key] if key == 'greedy_ids' else 'metadata'}; after={after_metadata[key] if key == 'greedy_ids' else 'metadata'}")
    if a.shape != b.shape or not torch.allclose(a, b, atol=ATOL, rtol=RTOL):
        maximum = float((a - b).abs().max()) if a.shape == b.shape else None
        raise ValueError(f"active-vocabulary logits differ: max_abs={maximum}, atol={ATOL}, rtol={RTOL}")
    return {"atol": ATOL, "rtol": RTOL, "max_abs": float((a - b).abs().max()),
            "active_vocab_size": len(before["active_ids"]), "vocab_sha256": before["vocab_sha256"],
            "greedy_ids": before["greedy_ids"], "atlas_ids": before["atlas_ids"],
            "prompt_ids_sha256": shared.digest(before["prompt_ids"])}


def load_eval(
    plan: JsonDict,
    checkpoint: str,
) -> tuple[torch.nn.Module, PreTrainedTokenizerBase, JsonDict]:
    progress(f"loading {checkpoint}")
    torch.manual_seed(44)
    model, tokenizer, evidence = evaluator.load_model(model_id=BASE, adapter_dir=checkpoint, bits=16,
                                disable_flash_attn2=True,
                                token_inventory=as_str(as_dict(plan["config"])["token_inventory"]),
                                preserve_visual_fp32=True, input_mode="text",
                                model_revision=shared.MODEL_REVISION)
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError("text-mode load_model must return a tokenizer")
    return model, tokenizer, evidence


def eval_panel(plan: JsonDict, model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase,
               evidence: JsonDict, checkpoint: str, panel: str, output: Path) -> JsonDict:
    args = argparse.Namespace(
        model_id=BASE, model_revision=shared.MODEL_REVISION, adapter_dir=checkpoint,
        input_mode="text", max_sequence_length=4096, bits=16, batch_size=16,
        long_batch_size=16, max_new_tokens=512, long_max_new_tokens=512,
        candidate_scoring=False, enable_thinking=False, do_sample=False,
        preserve_visual_fp32=True, disable_flash_attn2=True, image_root=None, limit=None,
        token_inventory=as_dict(plan["config"])["token_inventory"], occlusion_margin=0.03,
    )
    progress(f"greedy512/batch16 evaluation: {panel}")
    evaluator.run_eval_job(model=model, processor=tokenizer, adapter_evidence=evidence, args=args,
                           eval_jsonl=as_str(plan["data_dir"]) + f"/{panel}.jsonl",
                           image_variant="original", output_dir=output)
    records = rows_at(output / "records.jsonl")
    expected = as_dict(as_dict(as_dict(plan["inputs"])["panels"])[panel])
    ids = [r.get("id") for r in records]
    if len(ids) != expected["rows"] or Counter(ids) != Counter(as_list(expected["ids"])):
        raise ValueError(f"{panel}: incomplete/duplicate/wrong evaluation IDs")
    by_id = {as_str(r.get("id") or r["row_id"]): r
             for r in rows_at(Path(as_str(plan["data_dir"])) / f"{panel}.jsonl")}
    for record in records:
        row = by_id[as_str(record["id"])]
        metadata = evaluator.evaluation_metadata(row, image_variant="original")
        metadata["input_mode"] = "text"
        if (record["expected"] != evaluator.expected_text(row) or record["metadata"] != metadata
                or not isinstance(record["response"], str) or record.get("candidate_score") is not None
                or record["score"] != evaluator.score_response(
                    as_str(record["expected"]), record["response"], metadata=metadata)
                or type(as_dict(record["score"]).get("correct")) is not bool):
            raise ValueError(f"{panel}: incomplete or incorrectly dispatched score")
    summary = shared.read_json(output / "summary.json")
    for key, value in {"model_id": BASE, "model_revision": shared.MODEL_REVISION,
                       "adapter_dir": checkpoint, "input_mode": "text", "bits": 16,
                       "max_sequence_length": 4096, "batch_size": 16, "max_new_tokens": 512,
                       "long_max_new_tokens": 512, "candidate_scoring": False,
                       "reasoning_enabled": False, "rows": expected["rows"], "attempted": expected["rows"]}.items():
        if summary.get(key) != value:
            raise ValueError(f"{panel}: inference condition differs: {key}")
    if summary["correct"] != sum(as_int(as_dict(r["score"])["correct"]) for r in records):
        raise ValueError("eval summary disagrees with verified records")
    sft_runs.commit()
    return {"correct": summary["correct"], "rows": summary["rows"],
            "exact_accuracy": summary["exact_accuracy"], "output_dir": str(output),
            "files": shared.file_manifest(output, ["records.jsonl", "summary.json"])}
