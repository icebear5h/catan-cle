"""Validated communication prompt suite and response parsing."""

from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from cle.harness.context import ContextAssembler
from cle.harness.models import ModelMessage, ModelRequest, ModelResponse
from cle.players.contracts import (
    CommitmentProposal,
    CommunicationChoice,
    CommunicationMode,
    TalkContext,
)
from game_engine.models.player import Color


class CommunicationSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    version: int
    system_template: str
    user_template: str


def default_communication_suite_path() -> Path:
    return Path(str(files("cle.harness.suites").joinpath("communication_v2.yaml")))


def load_communication_suite(path: str | Path | None = None) -> CommunicationSuite:
    target = Path(path) if path is not None else default_communication_suite_path()
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    return CommunicationSuite.model_validate(raw)


def build_communication_request(
    context: TalkContext,
    session_id: str,
    suite: CommunicationSuite,
) -> ModelRequest:
    values = {
        "color": context.player.value,
        "cause": ContextAssembler._format_events((context.cause,)),
        "game_events": ContextAssembler._format_events(context.game_events),
        "recent_messages": ContextAssembler._format_events(context.recent_messages)
        or "None",
        "commitments": "\n".join(
            f"{item.id}: {item.condition} -> {item.promise} "
            f"(expires turn {item.expires_turn})"
            for item in context.active_commitments
        )
        or "None",
    }
    return ModelRequest(
        decision_id=context.context_id,
        session_id=session_id,
        messages=(
            ModelMessage("system", _render(suite.system_template, values)),
            ModelMessage("user", _render(suite.user_template, values)),
        ),
    )


def parse_communication_response(
    response: ModelResponse,
    *,
    speaker: Color,
    participants: tuple[Color, ...],
) -> CommunicationChoice:
    text = _tag(response.content, "message")
    if not text or text.strip().upper() == "SILENCE":
        return CommunicationChoice()

    audience_text = _tag(response.content, "audience").upper()
    if not audience_text or audience_text == "PUBLIC":
        audience = tuple(color for color in participants if color != speaker)
    else:
        names = {name.strip() for name in audience_text.split(",") if name.strip()}
        audience = tuple(
            color
            for color in participants
            if color != speaker and color.value in names
        )
        if not audience:
            audience = tuple(color for color in participants if color != speaker)

    commitment = None
    condition = _tag(response.content, "commitment_condition")
    promise = _tag(response.content, "commitment_promise")
    expires = _tag(response.content, "commitment_expires_turn")
    if condition and promise and expires.isdigit():
        commitment = CommitmentProposal(condition, promise, int(expires))

    return CommunicationChoice(
        mode=CommunicationMode.SAY,
        text=text,
        audience=audience,
        intent=_tag(response.content, "intent") or None,
        commitment=commitment,
    )


def _render(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    unresolved = re.findall(r"{{[A-Za-z_][A-Za-z0-9_]*}}", rendered)
    if unresolved:
        raise ValueError(f"Unresolved communication template fields: {unresolved}")
    return rendered.strip()


def _tag(text: str, tag: str) -> str:
    match = re.search(
        rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>",
        text or "",
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""
