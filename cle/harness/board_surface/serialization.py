"""Trace-safe board presentation metadata."""

from cle.players.data import JsonValue

from .contracts import BoardPresentation, TextBoardPresentation


def board_presentation_payload(
    presentation: BoardPresentation | None,
    *,
    include_text_content: bool,
) -> dict[str, JsonValue] | None:
    """Serialize presentation provenance without ever serializing image bytes."""

    if presentation is None:
        return None
    provenance = presentation.provenance
    payload: dict[str, JsonValue] = {
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
