"""Pinned single-H200 smoke configuration; no arbitrary trainer overrides."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .runtime import MILES_COMMIT, REQUIRED_TARGETS

BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
# Paired with Core 73b54618 in Miles PR #3336; Bridge 2e09c234 needs newer Core.
BRIDGE_REVISION = "40b930897717941cfe2bd9806f417e28bf1bfa65"
MEGATRON_REVISION = "73b54618f7e58e0f25f619bcfecbe2640765475a"
IMAGE_REF = "radixark/miles@sha256:6628bff749ffd32e6a62b479a1128daee25a8c0e3c28eb86301a0d620f5dd598"


@dataclass(frozen=True)
class TrainPlan:
    checkpoint: Path
    data: Path
    output: Path
    steps: int = 2
    batch_size: int = 2
    max_tokens: int = 4096
    learning_rate: float = 5e-5
    timeout_seconds: int = 1740

    def __post_init__(self) -> None:
        for name in ("steps", "batch_size", "max_tokens", "timeout_seconds"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_tokens > 4096 or self.timeout_seconds > 1740:
            raise ValueError("initial port is bounded to 4096 tokens and 1740 seconds")
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1e-3:
            raise ValueError("invalid learning rate")
        paths = [p.expanduser().resolve() for p in (self.checkpoint, self.data, self.output)]
        if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
               for i, a in enumerate(paths) for b in paths[i + 1:]):
            raise ValueError("checkpoint, prepared data and output must be separate directories")


def miles_arguments(plan: TrainPlan) -> list[str]:
    """One fresh optimizer update per rollout, no generation or implicit continuation."""
    checkpoint, data, output = (p.expanduser().resolve() for p in
                                (plan.checkpoint, plan.data, plan.output))
    return [
        "--train-backend", "megatron", "--megatron-to-hf-mode", "bridge",
        "--debug-train-only", "--lora-train-only", "--hf-checkpoint", str(checkpoint),
        "--load", str(checkpoint), "--start-rollout-id", "0",
        "--num-gpus-per-node", "1", "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "1",
        "--rollout-num-gpus", "0", "--eval-num-gpus", "0",
        "--tensor-model-parallel-size", "1", "--pipeline-model-parallel-size", "1",
        "--context-parallel-size", "1", "--expert-model-parallel-size", "1",
        "--expert-tensor-parallel-size", "1", "--micro-batch-size", "1",
        "--lora-type", "lora", "--lora-rank", "16", "--lora-alpha", "32",
        "--lora-dropout", "0.0", "--target-modules", ",".join(REQUIRED_TARGETS),
        "--no-gradient-accumulation-fusion", "--no-load-optim", "--no-load-rng",
        "--prompt-data", str(data / "input.jsonl"), "--input-key", "messages",
        "--metadata-key", "metadata", "--n-samples-per-prompt", "1",
        "--rollout-batch-size", str(plan.batch_size), "--global-batch-size", str(plan.batch_size),
        "--num-rollout", str(plan.steps), "--seq-length", str(plan.max_tokens),
        "--max-position-embeddings", "262144", "--qkv-format", "bshd",
        "--loss-type", "sft_loss", "--calculate-per-token-loss",
        "--disable-compute-advantages-and-returns",
        "--rollout-function-path", "sft.miles_sft.runtime.rollout.generate_rollout",
        "--custom-megatron-init-path", "sft.miles_sft.runtime.hooks.initialize",
        "--custom-megatron-before-train-step-hook-path", "sft.miles_sft.runtime.hooks.before_train_step",
        "--custom-megatron-post-save-hook-path", "sft.miles_sft.runtime.hooks.post_save",
        "--save", str(output / "checkpoints"), "--save-interval", str(plan.steps),
        "--save-hf", str(output / "exports" / "rollout-{rollout_id}" / "bridge"),
        "--optimizer", "adam", "--lr", str(plan.learning_rate), "--lr-decay-style", "constant",
        "--lr-warmup-iters", "0", "--weight-decay", "0.01", "--adam-beta1", "0.9",
        "--adam-beta2", "0.999", "--adam-eps", "1e-8", "--clip-grad", "1.0", "--seed", "44",
        "--bf16", "--attention-dropout", "0.0", "--hidden-dropout", "0.0",
        "--recompute-granularity", "full", "--recompute-method", "uniform", "--recompute-num-layers", "1",
        "--attention-backend", "flash", "--disable-bias-linear", "--qk-layernorm",
        "--group-query-attention", "--num-attention-heads", "24", "--num-query-groups", "4",
        "--kv-channels", "256", "--num-layers", "64", "--hidden-size", "5120",
        "--ffn-hidden-size", "17408", "--normalization", "RMSNorm", "--apply-layernorm-1p",
        "--position-embedding-type", "rope", "--norm-epsilon", "1e-6", "--rotary-percent", "0.25",
        "--swiglu", "--untie-embeddings-and-output-weights", "--vocab-size", "248320",
        "--rotary-base", "10000000", "--attention-output-gate",
    ]


__all__ = ["BASE_REVISION", "BRIDGE_REVISION", "IMAGE_REF", "MEGATRON_REVISION",
           "MILES_COMMIT", "TrainPlan", "miles_arguments"]
