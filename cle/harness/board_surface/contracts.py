"""Provider-independent public-board presentation contracts."""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeAlias

from PIL import Image

from cle.game_engine.models.player import Color
from cle.harness import board_surface

if TYPE_CHECKING:
    from cle.players.contracts import PlayerContext


BOARD_PRESENTATION_SCHEMA: Final = "catan_board_presentation/v1"
BoardIdentitySpace: TypeAlias = Literal[
    "canonical_engine_ids",
    "opaque_board_local_ids",
]
BoardMediaType: TypeAlias = Literal["image/png", "image/jpeg", "image/webp"]
MAX_BOARD_IMAGE_BYTES = 5 * 1024 * 1024
MAX_BOARD_IMAGE_DIMENSION = 2048
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: str, field_name: str) -> None:
    if not board_surface._SHA256.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class BoardPresentationProvenance:
    """Identity and rendering provenance shared by every board modality."""

    source_id: str
    perspective: Color | None
    board_schema: str
    board_sha256: str
    identity_space: BoardIdentitySpace
    renderer_id: str
    renderer_version: str
    canonical_id_map: tuple[tuple[str, str], ...] = ()
    canonical_id_map_sha256: str | None = None
    schema: Literal["catan_board_presentation/v1"] = field(
        default=BOARD_PRESENTATION_SCHEMA,
        init=False,
    )

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("Board presentation source_id cannot be empty")
        if not self.board_schema.strip():
            raise ValueError("Board presentation board_schema cannot be empty")
        if not self.renderer_id.strip() or not self.renderer_version.strip():
            raise ValueError("Board presentation renderer provenance cannot be empty")
        board_surface._require_sha256(self.board_sha256, "board_sha256")
        if self.canonical_id_map_sha256 is not None:
            board_surface._require_sha256(
                self.canonical_id_map_sha256,
                "canonical_id_map_sha256",
            )
        aliases = [alias for alias, _canonical in self.canonical_id_map]
        if len(aliases) != len(set(aliases)):
            raise ValueError("Board presentation aliases must be unique")
        if self.identity_space == "opaque_board_local_ids" and not (
            self.canonical_id_map or self.canonical_id_map_sha256
        ):
            raise ValueError(
                "Opaque board identities require a canonical ID map or digest"
            )


@dataclass(frozen=True, slots=True)
class TextBoardPresentation:
    """Exact text projection of one immutable public-board state."""

    provenance: BoardPresentationProvenance
    format: str
    content: str = field(repr=False)
    content_sha256: str
    byte_length: int
    kind: Literal["text"] = field(default="text", init=False)
    media_type: Literal["text/plain"] = field(default="text/plain", init=False)

    def __post_init__(self) -> None:
        if not self.format.strip():
            raise ValueError("Text board presentation format cannot be empty")
        if not self.content:
            raise ValueError("Text board presentation content cannot be empty")
        encoded = self.content.encode("utf-8")
        if self.byte_length != len(encoded):
            raise ValueError("Text board presentation byte length is inconsistent")
        board_surface._require_sha256(self.content_sha256, "content_sha256")
        if board_surface._sha256_bytes(encoded) != self.content_sha256:
            raise ValueError("Text board presentation digest is inconsistent")

    @classmethod
    def create(
        cls,
        *,
        provenance: BoardPresentationProvenance,
        format: str,
        content: str,
    ) -> "TextBoardPresentation":
        encoded = content.encode("utf-8")
        return cls(
            provenance=provenance,
            format=format,
            content=content,
            content_sha256=board_surface._sha256_bytes(encoded),
            byte_length=len(encoded),
        )


@dataclass(frozen=True, slots=True)
class ImageBoardPresentation:
    """Bounded local image projection; raw bytes are never trace metadata."""

    provenance: BoardPresentationProvenance
    format: str
    media_type: BoardMediaType
    data: bytes = field(repr=False)
    content_sha256: str
    byte_length: int
    width: int
    height: int
    contains_entity_labels: bool
    kind: Literal["image"] = field(default="image", init=False)

    def __post_init__(self) -> None:
        if not self.format.strip():
            raise ValueError("Image board presentation format cannot be empty")
        if not self.data:
            raise ValueError("Image board presentation data cannot be empty")
        if self.byte_length != len(self.data):
            raise ValueError("Image board presentation byte length is inconsistent")
        if self.byte_length > board_surface.MAX_BOARD_IMAGE_BYTES:
            raise ValueError(
                f"Board image exceeds {board_surface.MAX_BOARD_IMAGE_BYTES} bytes"
            )
        if not (
            1 <= self.width <= board_surface.MAX_BOARD_IMAGE_DIMENSION
            and 1 <= self.height <= board_surface.MAX_BOARD_IMAGE_DIMENSION
        ):
            raise ValueError(
                "Board image dimensions must be within "
                f"1..{board_surface.MAX_BOARD_IMAGE_DIMENSION}"
            )
        board_surface._require_sha256(self.content_sha256, "content_sha256")
        if board_surface._sha256_bytes(self.data) != self.content_sha256:
            raise ValueError("Image board presentation digest is inconsistent")
        signatures = {
            "image/png": self.data.startswith(b"\x89PNG\r\n\x1a\n"),
            "image/jpeg": self.data.startswith(b"\xff\xd8"),
            "image/webp": (
                self.data.startswith(b"RIFF") and self.data[8:12] == b"WEBP"
            ),
        }
        if self.media_type not in signatures:
            raise ValueError(f"Unsupported board image media type: {self.media_type}")
        if not signatures[self.media_type]:
            raise ValueError(
                f"Board image bytes do not match {self.media_type}"
            )
        with Image.open(io.BytesIO(self.data)) as image:
            actual_dimensions = image.size
            actual_format = image.format
        expected_format = {
            "image/png": "PNG",
            "image/jpeg": "JPEG",
            "image/webp": "WEBP",
        }[self.media_type]
        if actual_format != expected_format:
            raise ValueError(
                f"Decoded board image is {actual_format}, not {expected_format}"
            )
        if actual_dimensions != (self.width, self.height):
            raise ValueError(
                "Board image dimensions do not match the decoded image"
            )

    @classmethod
    def create(
        cls,
        *,
        provenance: BoardPresentationProvenance,
        format: str,
        media_type: BoardMediaType,
        data: bytes,
        width: int,
        height: int,
        contains_entity_labels: bool,
    ) -> "ImageBoardPresentation":
        return cls(
            provenance=provenance,
            format=format,
            media_type=media_type,
            data=data,
            content_sha256=board_surface._sha256_bytes(data),
            byte_length=len(data),
            width=width,
            height=height,
            contains_entity_labels=contains_entity_labels,
        )


BoardPresentation: TypeAlias = TextBoardPresentation | ImageBoardPresentation


class BoardPresenter(Protocol):
    """Build one immutable board presentation for an exact decision context."""

    def present(
        self,
        context: "PlayerContext",
    ) -> BoardPresentation | None:
        """Return one public-board projection for this decision, if enabled."""


# Preserve both historical snapshot loading and the identity of newly written pickles.
BoardPresentationProvenance.__module__ = "cle.harness.board_surface"
TextBoardPresentation.__module__ = "cle.harness.board_surface"
ImageBoardPresentation.__module__ = "cle.harness.board_surface"
BoardPresenter.__module__ = "cle.harness.board_surface"
