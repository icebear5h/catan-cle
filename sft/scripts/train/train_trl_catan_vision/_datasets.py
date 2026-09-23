"""Training/text/vision dataset loaders."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from sft.scripts.train.train_trl_catan_vision._common import CURRICULUM_STAGES, JsonDict, iter_jsonl
from sft.scripts.train.train_trl_catan_vision._config import TrainConfig
from sft.scripts.train.train_trl_catan_vision._text_data import _message_pair
from sft.scripts.train.train_trl_catan_vision._vision_data import (
    inspect_jsonl_contract,
    resolve_image_path,
    validate_spatial_targets,
)

if TYPE_CHECKING:  # Heavy; transformers loads lazily on the runtime path.
    from transformers import PreTrainedTokenizerBase


class TrainingDataset(Protocol):
    """The part of `datasets.Dataset` the trainer path relies on.

    `datasets` ships no type information, and the loader imports it
    dynamically, so the row container is described structurally here.
    """

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> JsonDict: ...


def load_training_dataset(config: TrainConfig, *,
                          tokenizer: PreTrainedTokenizerBase | None = None,
                          ) -> tuple[TrainingDataset, JsonDict]:
    if config.text_only:
        return load_text_dataset(
            config.train_jsonl, tokenizer=tokenizer,
            max_sequence_length=config.max_sequence_length,
            require_curriculum=config.require_curriculum,
        )
    if config.image_root is None:
        raise ValueError("image_root is required in vision mode")
    return load_vision_dataset(
        config.train_jsonl,
        config.image_root,
        require_curriculum=config.require_curriculum,
    )


def load_text_dataset(
    jsonl_path: str | Path, *, tokenizer: PreTrainedTokenizerBase | None,
    max_sequence_length: int | None,
    require_curriculum: bool,
) -> tuple[TrainingDataset, JsonDict]:
    datasets = importlib.import_module("datasets")
    report = inspect_jsonl_contract(
        jsonl_path, require_curriculum=require_curriculum, input_mode="text",
        tokenizer=tokenizer, max_sequence_length=max_sequence_length,
    )
    examples: list[JsonDict] = []
    for line_number, row in iter_jsonl(Path(jsonl_path).expanduser().resolve()):
        prompt, answer = _message_pair(row, line_number=line_number, input_mode="text")
        examples.append({"messages": [
            {"role": "user", "content": prompt}, {"role": "assistant", "content": answer},
        ]})
    return datasets.Dataset.from_list(examples), report


def load_vision_dataset(
    jsonl_path: str | Path,
    image_root: str | Path,
    *,
    require_curriculum: bool,
) -> tuple[TrainingDataset, JsonDict]:

    datasets = importlib.import_module("datasets")

    report = inspect_jsonl_contract(
        jsonl_path,
        image_root,
        require_curriculum=require_curriculum,
    )
    root = Path(image_root).expanduser().resolve()
    examples: list[JsonDict] = []
    for line_number, row in iter_jsonl(Path(jsonl_path).expanduser().resolve()):
        prompt, answer = _message_pair(row, line_number=line_number)
        image_path = resolve_image_path(root, row, line_number=line_number)
        stage = row.get("curriculum_stage", CURRICULUM_STAGES[0])
        examples.append(
            {
                "image": str(image_path),
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "completion": [{"role": "assistant", "content": answer}],
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "preserve_thinking": False,
                },
                "curriculum_stage": stage,
                "curriculum_stage_index": CURRICULUM_STAGES.index(stage),
                "spatial_targets": list(validate_spatial_targets(row, line_number=line_number)),
            }
        )
    dataset = datasets.Dataset.from_list(examples).cast_column("image", datasets.Image(decode=True))
    return dataset, report
