"""Typed handles for the unannotated Modal and transformers APIs used by the probes.

Deliberately stdlib-only: the Modal benchmark images carry torch/httpx, not this
repository's dependencies, and the container imports whatever the app module imports.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from typing import Protocol, TypedDict

#: ``Image.imports()``, which modal does not annotate.
ContainerImports = Callable[[], AbstractContextManager[None]]


class VLMResult(TypedDict, total=False):
    """One screenshot/model reply, or the reason there is none."""

    response: str
    latency_ms: int
    model: str
    usage: dict[str, object]
    error: str


class SpawnedCall(Protocol):
    """A ``Function.spawn`` handle for one queued call."""
    def get(self) -> VLMResult: ...


class Shaped(Protocol):
    """A tokenized tensor, of which only the shape is read."""
    @property
    def shape(self) -> tuple[int, ...]: ...


class ChatInputs(Protocol):
    """The processor's tokenized batch, passed on to ``model.generate``."""
    def to(self, device: str) -> ChatInputs: ...
    def keys(self) -> Iterable[str]: ...
    def __getitem__(self, key: str) -> Shaped: ...


class Processor(Protocol):
    """A transformers processor, narrowed to what these scripts call."""
    def apply_chat_template(
        self, conversation: list[dict[str, object]], **kwargs: object
    ) -> ChatInputs: ...
    def decode(self, token_ids: object, *, skip_special_tokens: bool) -> str: ...


class ProcessorLoader(Protocol):
    """``AutoProcessor.from_pretrained`` as these scripts call it."""
    def __call__(
        self, model_name: str, /, *, trust_remote_code: bool, cache_dir: str = ...
    ) -> Processor: ...
