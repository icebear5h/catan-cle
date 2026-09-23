"""Exact OpenAI-compatible message and inline image shapes."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Literal, TypedDict

from .contracts import BoardPresentation, TextBoardPresentation

if TYPE_CHECKING:
    from cle.harness.models import ModelMessage


class ImageURL(TypedDict):
    url: str


class ImageContent(TypedDict):
    type: Literal["image_url"]
    image_url: ImageURL


class TextContent(TypedDict):
    type: Literal["text"]
    text: str


class OpenAIMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str | list[ImageContent | TextContent]


def openai_messages_with_board(
    messages: tuple[ModelMessage, ...],
    presentation: BoardPresentation | None,
    *,
    allow_image_input: bool,
) -> list[OpenAIMessage]:
    """Encode one board presentation onto only the latest user message."""

    payload: list[OpenAIMessage] = [
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
