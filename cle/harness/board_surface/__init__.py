"""Provider-independent public-board presentation contracts and serialization."""

from cle.game_engine.models.player import Color

from .contracts import (
    _SHA256,
    BOARD_PRESENTATION_SCHEMA,
    MAX_BOARD_IMAGE_BYTES,
    MAX_BOARD_IMAGE_DIMENSION,
    BoardIdentitySpace,
    BoardMediaType,
    BoardPresentation,
    BoardPresentationProvenance,
    BoardPresenter,
    ImageBoardPresentation,
    TextBoardPresentation,
    _require_sha256,
    _sha256_bytes,
)
from .provider import (
    ImageContent,
    ImageURL,
    OpenAIMessage,
    TextContent,
    openai_messages_with_board,
)
from .redaction import sanitize_provider_payload
from .serialization import board_presentation_payload

__all__ = [
    "BOARD_PRESENTATION_SCHEMA", "MAX_BOARD_IMAGE_BYTES", "MAX_BOARD_IMAGE_DIMENSION",
    "BoardIdentitySpace", "BoardMediaType", "BoardPresentation", "BoardPresentationProvenance",
    "BoardPresenter", "ImageBoardPresentation", "TextBoardPresentation", "Color",
    "ImageContent", "ImageURL", "OpenAIMessage", "TextContent", "board_presentation_payload",
    "openai_messages_with_board", "sanitize_provider_payload", "_SHA256", "_require_sha256",
    "_sha256_bytes",
]
