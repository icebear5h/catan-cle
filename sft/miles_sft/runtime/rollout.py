"""Miles-loaded SFT hook. Imports real Miles types; there is no local fallback."""

from __future__ import annotations

import importlib
from typing import cast

from .admission import encoded_sample, validate_rollout_args
from .contracts import (
    DataSource,
    MilesRollout,
    MilesTypes,
    RolloutArgs,
    RolloutOutput,
    require,
)

MILES_TYPES = cast(MilesTypes, importlib.import_module("miles.utils.types"))
MILES_ROLLOUT = cast(MilesRollout, importlib.import_module("miles.rollout.base_types"))


def generate_rollout(
    args: RolloutArgs, rollout_id: int, data_source: DataSource, evaluation: bool = False,
) -> RolloutOutput:
    """Return the actual Miles RolloutFnTrainOutput, containing data-source Samples."""
    require(not evaluation, "this hook admits training data only")
    require(type(rollout_id) is int and rollout_id >= 0, "invalid rollout ID")
    validate_rollout_args(args)
    groups = data_source.get_samples(args.rollout_batch_size)
    require(len(groups) == args.rollout_batch_size, "data source returned a short/oversized batch")
    # Validate every sample before mutating any sample in the batch.
    prepared = []
    for group in groups:
        require(len(group) == 1, "expected singleton sample groups")
        sample = group[0]
        require(isinstance(sample, MILES_TYPES.Sample), "data source must return real Miles Samples")
        prepared.append(encoded_sample(sample, args.seq_length))
    for group, (encoded, answer) in zip(groups, prepared, strict=True):
        sample = group[0]
        sample.tokens = encoded["tokens"]
        sample.response_length = encoded["response_length"]
        sample.loss_mask = encoded["loss_mask"]
        sample.response = answer
        sample.reward = 0.0  # Miles transports rewards, but sft_loss does not consume them.
        sample.status = MILES_TYPES.Sample.Status["COMPLETED"]
        sample.validate()
    return MILES_ROLLOUT.RolloutFnTrainOutput(samples=groups)
