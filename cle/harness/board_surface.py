"""Provider-independent public-board presentation contracts."""

from __future__ import annotations

import base64
import hashlib
import io
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias

from PIL import Image

from cle.game_engine.models.player import Color

if TYPE_CHECKING:
    from cle.players.contracts import PlayerContext


BOARD_PRESENTATION_SCHEMA = "catan_board_presentation/v1"
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
    if not _SHA256.fullmatch(value):
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
        _require_sha256(self.board_sha256, "board_sha256")
        if self.canonical_id_map_sha256 is not None:
            _require_sha256(
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
        _require_sha256(self.content_sha256, "content_sha256")
        if _sha256_bytes(encoded) != self.content_sha256:
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
            content_sha256=_sha256_bytes(encoded),
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
        if self.byte_length > MAX_BOARD_IMAGE_BYTES:
            raise ValueError(
                f"Board image exceeds {MAX_BOARD_IMAGE_BYTES} bytes"
            )
        if not (
            1 <= self.width <= MAX_BOARD_IMAGE_DIMENSION
            and 1 <= self.height <= MAX_BOARD_IMAGE_DIMENSION
        ):
            raise ValueError(
                "Board image dimensions must be within "
                f"1..{MAX_BOARD_IMAGE_DIMENSION}"
            )
        _require_sha256(self.content_sha256, "content_sha256")
        if _sha256_bytes(self.data) != self.content_sha256:
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
            content_sha256=_sha256_bytes(data),
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


def board_presentation_payload(
    presentation: BoardPresentation | None,
    *,
    include_text_content: bool,
) -> dict[str, Any] | None:
    """Serialize presentation provenance without ever serializing image bytes."""

    if presentation is None:
        return None
    provenance = presentation.provenance
    payload: dict[str, Any] = {
        "schema": provenance.schema,
        "kind": presentation.kind,
        "format": presentation.format,
        "media_type": presentation.media_type,
        "source_id": provenance.source_id,
        "perspective": (
            provenance.perspective.value if provenance.perspective else None
        ),
        "board_schema": provenance.board_schema,
        "board_sha256": provenance.board_sha256,
        "identity_space": provenance.identity_space,
        "renderer_id": provenance.renderer_id,
        "renderer_version": provenance.renderer_version,
        "canonical_id_map": dict(provenance.canonical_id_map),
        "canonical_id_map_sha256": provenance.canonical_id_map_sha256,
        "content_sha256": presentation.content_sha256,
        "byte_length": presentation.byte_length,
    }
    if isinstance(presentation, TextBoardPresentation):
        if include_text_content:
            payload["content"] = presentation.content
    else:
        payload.update(
            {
                "width": presentation.width,
                "height": presentation.height,
                "contains_entity_labels": presentation.contains_entity_labels,
                "data": None,
            }
        )
    return payload


def openai_messages_with_board(
    messages: tuple[Any, ...],
    presentation: BoardPresentation | None,
    *,
    allow_image_input: bool,
) -> list[dict[str, Any]]:
    """Encode one board presentation onto only the latest user message."""

    payload = [
        {"role": message.role, "content": message.content}
        for message in messages
    ]
    if presentation is None:
        return payload
    user_indexes = [
        index for index, message in enumerate(payload) if message["role"] == "user"
    ]
    if not user_indexes:
        raise ValueError("A board presentation requires a user message")
    current_index = user_indexes[-1]
    current_content = payload[current_index]["content"]
    if not isinstance(current_content, str):
        raise TypeError("Board presentation encoding requires text model messages")

    if isinstance(presentation, TextBoardPresentation):
        payload[current_index]["content"] = (
            "PUBLIC BOARD:\n"
            f"{presentation.content}\n\n"
            f"{current_content}"
        )
        return payload

    if not allow_image_input:
        raise ValueError(
            "This transport has not explicitly enabled board image input"
        )
    encoded = base64.b64encode(presentation.data).decode("ascii")
    payload[current_index]["content"] = [
        {
            "type": "image_url",
            "image_url": {
                "url": (
                    f"data:{presentation.media_type};base64,{encoded}"
                )
            },
        },
        {"type": "text", "text": current_content},
    ]
    return payload


def sanitize_provider_payload(
    value: Any,
    presentation: BoardPresentation | None = None,
) -> Any:
    """Remove inline image data before provider payloads enter durable traces."""

    if isinstance(value, str) and value.startswith("data:image/"):
        if isinstance(presentation, ImageBoardPresentation):
            return (
                "local-board-image://sha256/"
                f"{presentation.content_sha256}"
            )
        return "local-board-image://redacted"
    if isinstance(value, dict):
        return {
            key: sanitize_provider_payload(item, presentation)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_provider_payload(item, presentation) for item in value]
    if isinstance(value, tuple):
        return tuple(
            sanitize_provider_payload(item, presentation) for item in value
        )
    return value
