"""Fail-closed configuration and prepared-sample admission; no tokenization."""

from __future__ import annotations

import re
from pathlib import Path

from sft.json_types import as_dict, as_list, as_str
from sft.miles_sft.data.contracts import STATIC_OPERATIONS, EncodedPair
from sft.miles_sft.data.encoding import message_pair, reject_media

from .contracts import REQUIRED_TARGETS, RolloutArgs, RuntimeArgs, SampleView, require


def validate_rollout_args(args: RolloutArgs) -> None:
    require(args.rollout_global_dataset, "SFT requires the global dataset")
    require(args.n_samples_per_prompt == 1, "SFT requires one sample per prompt")
    require(type(args.rollout_batch_size) is int and args.rollout_batch_size > 0,
            "invalid rollout batch size")
    require(not args.apply_chat_template and args.input_key == "messages"
            and args.metadata_key == "metadata", "prepared messages/metadata must be unmodified")
    require(args.rollout_max_prompt_len is None, "disable upstream prompt filtering/tokenization")


def validate_args(args: RuntimeArgs) -> None:
    validate_rollout_args(args)
    require(args.train_backend == "megatron" and args.megatron_to_hf_mode == "bridge",
            "fresh SFT requires Megatron Bridge")
    require(args.debug_train_only and args.lora_train_only and not args.debug_rollout_only
            and not args.debug_disable_optimizer, "a real train-only LoRA optimizer is required")
    require(args.lora_type == "lora" and args.lora_rank == 16 and args.lora_alpha == 32,
            "requires standard fused LoRA rank16 alpha32")
    require(args.lora_adapter_path is None and not getattr(args, "adapter_path", None),
            "old adapter import is unsupported; merge into the full HF base first")
    require(set(args.target_modules) == set(REQUIRED_TARGETS) and not args.exclude_modules,
            "require exact language decoder targets including GDN in_proj/out_proj")
    require(args.loss_type == "sft_loss" and args.calculate_per_token_loss
            and not args.compute_advantages_and_returns, "requires supervised token loss only")
    require(not args.enable_mtp_training and not args.use_critic, "MTP/critic training forbidden")
    require(args.rollout_num_gpus == args.eval_num_gpus == 0, "no rollout/eval GPU fleet allowed")
    require(args.actor_num_nodes == 1 and args.actor_num_gpus_per_node == args.world_size
            == args.tensor_model_parallel_size, "only one tensor-parallel trainer group is supported")
    require(args.pipeline_model_parallel_size == args.context_parallel_size
            == args.expert_model_parallel_size == 1
            and args.expert_tensor_parallel_size in (None, 1)
            and args.virtual_pipeline_model_parallel_size is None,
            "pipeline/context/expert parallelism is unsupported for this audit")
    require(not getattr(args, "indep_dp", False) and not getattr(args, "fully_async", False),
            "requires synchronous single-replica SFT")
    require(not getattr(args, "overlap_param_gather", False),
            "disable overlap-param-gather so post-step witnesses see updated parameters")
    require(getattr(args, "lora_B_init_method", "zero") == "zero", "fresh LoRA B must start at zero")
    require(args.load is not None and Path(args.load).resolve() == Path(args.hf_checkpoint).resolve(),
            "--load and --hf-checkpoint must point to the same merged full HF checkpoint")
    require(not (Path(args.hf_checkpoint) / "adapter_config.json").exists(),
            "HF initialization must be a merged full checkpoint")
    require(bool(args.save) and bool(args.save_hf) and not args.no_save_optim,
            "native optimizer state and merged --save-hf export are mandatory")
    require(args.num_rollout > 0 and args.start_rollout_id == 0, "fresh training must start at rollout zero")
    require(args.global_batch_size > 0 and args.rollout_batch_size % args.global_batch_size == 0,
            "rollout batch must contain whole optimizer steps")


def _integer(value: object, minimum: int, label: str) -> int:
    require(type(value) is int, f"{label} must be an integer")
    if not isinstance(value, int) or value < minimum:
        raise ValueError(f"invalid {label}")
    return value


def encoded_sample(sample: SampleView, max_tokens: int) -> tuple[EncodedPair, str]:
    """Validate the data agent's encoded suffix contract, preserving exact IDs."""
    metadata = as_dict(sample.metadata)
    reject_media(metadata)
    require(metadata.get("split") == "train" and metadata.get("operation") in STATIC_OPERATIONS,
            "only train-split topology primitives are admitted")
    source = as_dict(metadata.get("source_metadata"))
    require(source.get("split") == "train" and source.get("task_role", "train") == "train"
            and source.get("review_only", False) is False
            and source.get("admitted_for_training", True) is True,
            "source metadata is not training data")
    require(as_dict(source.get("target", {})).get("state") is None,
            "state-conditioned targets are forbidden")
    for key in ("operation", "task_type", "training_family"):
        require(source.get(key, metadata["operation"]) == metadata["operation"],
                "source operation declarations disagree")
    require(as_dict(source.get("provenance", {})).get("split", "train") == "train",
            "non-train provenance")
    for key in ("source_sha256", "source_row_sha256"):
        require(re.fullmatch(r"[0-9a-f]{64}", as_str(metadata.get(key))) is not None,
                f"invalid {key}")
    require(bool(as_str(metadata.get("source_file"))), "missing source file")
    _integer(metadata.get("source_line"), 1, "source line")
    messages = message_pair(as_list(sample.prompt))
    for media in (sample.multimodal_inputs, sample.multimodal_train_inputs):
        # Upstream processors can return an all-None mapping for text-only messages.
        require(media is None or (isinstance(media, dict)
                                  and all(v is None for v in media.values())),
                "multimodal tensors are forbidden")
    tokens = [_integer(t, 0, "token ID") for t in as_list(metadata.get("tokens"))]
    prefix = _integer(metadata.get("prefix_length"), 1, "prefix length")
    response = _integer(metadata.get("response_length"), 1, "response length")
    mask = [_integer(t, 0, "loss mask") for t in as_list(metadata.get("loss_mask"))]
    require(len(tokens) == prefix + response and len(tokens) <= max_tokens,
            "invalid encoded boundary or sequence overflow")
    require(len(mask) == response and all(t == 1 for t in mask),
            "loss mask must supervise the complete answer/end-of-turn suffix only")
    return {"tokens": tokens, "prefix_length": prefix, "response_length": response,
            "loss_mask": mask}, messages[1]["content"]
