"""Typed call boundaries for the unannotated `safetensors` / `transformers` loaders.

`safe_open` and `AutoTokenizer.from_pretrained` ship without annotations, so
strict mypy rejects calling them from typed code. These wrappers bind each one
to a Protocol that states how the launchers use it. Both are looked up at call
time, so a test that monkeypatches `AutoTokenizer.from_pretrained` is still hit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, PreTrainedTokenizerBase

__all__ = ["TensorFile", "TensorSlice", "load_tokenizer", "open_tensors"]


class TensorSlice(Protocol):
    """`safe_open(...).get_slice(name)`, narrowed to the header accessors used."""

    def get_shape(self) -> list[int]: ...
    def get_dtype(self) -> str: ...


class TensorFile(Protocol):
    """An open safetensors file."""

    def __enter__(self) -> TensorFile: ...
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...
    def keys(self) -> list[str]: ...
    def get_slice(self, name: str) -> TensorSlice: ...
    def get_tensor(self, name: str) -> torch.Tensor: ...


class _TensorOpener(Protocol):
    def __call__(self, filename: str | Path, framework: str, device: str = ...) -> TensorFile: ...


class _TokenizerLoader(Protocol):
    def __call__(self, name: str | Path, /, *, local_files_only: bool) -> PreTrainedTokenizerBase: ...


def open_tensors(path: str | Path, framework: str, device: str | None = None) -> TensorFile:
    """`safe_open(path, framework=..., device=...)`, omitting `device` when not given."""
    opener: _TensorOpener = safe_open
    if device is None:
        return opener(path, framework=framework)
    return opener(path, framework=framework, device=device)


def load_tokenizer(name: str | Path) -> PreTrainedTokenizerBase:
    """`AutoTokenizer.from_pretrained(name, local_files_only=True)`."""
    loader: _TokenizerLoader = AutoTokenizer.from_pretrained
    return loader(name, local_files_only=True)
