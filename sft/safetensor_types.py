"""Typed access to `safetensors.safe_open`, whose stubs ship without annotations.

Assigning `safe_open` to a Protocol-typed name narrows the untyped class to the
accessors SFT code actually calls, without changing what runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypedDict

import torch
from safetensors import safe_open


class TensorHeader(TypedDict):
    """One safetensors entry as read from the file header, without tensor data."""

    shape: list[int]
    dtype: str


class TensorSlice(Protocol):
    """`safe_open(...).get_slice(name)`, narrowed to its header accessors."""

    def get_shape(self) -> list[int]: ...
    def get_dtype(self) -> str: ...


class TensorFile(Protocol):
    """An open safetensors file opened with `framework="pt"`."""

    def __enter__(self) -> TensorFile: ...
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...
    def keys(self) -> list[str]: ...
    def metadata(self) -> dict[str, str] | None: ...
    def get_slice(self, name: str) -> TensorSlice: ...
    def get_tensor(self, name: str) -> torch.Tensor: ...


class TensorOpener(Protocol):
    """`safetensors.safe_open` as SFT code calls it."""

    def __call__(self, filename: str | Path, framework: str, device: str = ...) -> TensorFile: ...


def open_tensors(path: str | Path, *, framework: str = "pt", device: str = "cpu") -> TensorFile:
    """Open a safetensors file lazily; tensors load only when asked for."""
    opener: TensorOpener = safe_open
    return opener(path, framework=framework, device=device)
