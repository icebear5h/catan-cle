"""Render the communication request and its typed prompt components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from cle.game_engine.models.player import Color
from cle.game_engine.observation import PlayerObservation
from cle.harness.board_surface import BoardPresenter
from cle.harness.catan_board_surface import IndexedTileRowsBoardPresenter
from cle.harness.communication.suite import CommunicationSuite
from cle.harness.components import (
    TEMPLATE_VARIABLE,
    ComponentInputs,
    observation_component_values,
    render_component_definitions,
)
from cle.harness.context import ContextAssembler
from cle.harness.models import ModelMessage, ModelRequest, PromptComponent
from cle.harness.reasoning import native_reasoning_request
from cle.players.contracts import PlayerContext, TalkContext


def _render(template: str, values: dict[str, str]) -> str:
    return ContextAssembler._render_template(template, values)


def build_communication_request(
    context: TalkContext,
    session_id: str,
    suite: CommunicationSuite,
    *,
    notes: str = "",
    board_presenter: BoardPresenter | None = None,
) -> ModelRequest:
    components = render_communication_components(context, suite, notes=notes)
    board_presentation = None
    if suite.components and context.observation is not None:
        presenter = board_presenter or IndexedTileRowsBoardPresenter()
        # Presenters consume only these three facts, not a fabricated legal menu.
        board_context = _TalkBoardContext(context.context_id, context.player, context.observation)
        board_presentation = presenter.present(cast(PlayerContext, board_context))
    return ModelRequest(
        decision_id=context.context_id,
        session_id=session_id,
        messages=(
            ModelMessage("system", "\n\n".join(
                component.rendered for component in components
                if component.channel == "system" and component.rendered
            )),
            ModelMessage(
                "user",
                "\n\n".join(
                    component.rendered
                    for component in components
                    if component.channel == "environment"
                ),
            ),
        ),
        components=components,
        board_presentation=board_presentation,
        reasoning_request=tuple(native_reasoning_request("low").items()),
        max_tokens=None,
        trigger_reason=context.trigger_reason,
    )


@dataclass(frozen=True, slots=True)
class _TalkBoardContext:
    context_id: str
    actor: Color
    observation: PlayerObservation


def render_communication_components(
    context: TalkContext,
    suite: CommunicationSuite,
    *,
    notes: str = "",
) -> tuple[PromptComponent, ...]:
    if suite.components:
        if context.observation is not None and context.observation.my_color != context.player:
            raise ValueError("Observation perspective does not match talk player")
        return render_component_definitions(
            suite.components,
            suite.order,
            ComponentInputs(
                color=context.player.value,
                notes=notes,
                max_notes_chars=str(suite.max_notes_chars),
                trigger=(
                    ("Seven: discards are complete; speak before the robber destination is chosen.\n"
                     if context.trigger_reason == "pre_robber" else "")
                    + ContextAssembler._format_events((context.cause,), shared=True)
                ),
                visible_events=ContextAssembler._format_events(context.game_events, shared=True),
                recent_table_talk=ContextAssembler._format_events(context.recent_messages),
                commitments=ContextAssembler._format_commitments(context.active_commitments),
                **observation_component_values(
                    context.observation,
                    include_initial_placement_order=suite.initial_placement_order == "both_rounds",
                ),
            ),
        )
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
    # Historical communication suites always carry a system template; the suite
    # validator rejects the alternative before any request is rendered.
    system_template = cast("str", suite.system_template)
    components = [
        PromptComponent(
            id="system.identity",
            channel="system",
            template=system_template,
            value=context.player.value,
            rendered=_render(system_template, values),
            variables=(("color", context.player.value),),
        )
    ]
    if suite.user_template is not None:
        referenced = tuple(
            (name, values[name])
            for name in sorted(set(TEMPLATE_VARIABLE.findall(suite.user_template)))
        )
        components.append(
            PromptComponent(
                id="environment.communication",
                channel="environment",
                template=suite.user_template,
                value="",
                rendered=_render(suite.user_template, values),
                variables=referenced,
            )
        )
        return tuple(components)

    component_values = {
        "communication_policy": "",
        "trigger": values["cause"],
        "visible_events": values["game_events"] or "None",
        "recent_table_talk": values["recent_messages"],
        "commitments": values["commitments"],
        "response_schema": "",
    }
    for name in suite.order:
        template = suite.sections[name].template
        value = component_values[name]
        section_variables = set(TEMPLATE_VARIABLE.findall(template))
        variables = (("value", value),) if "value" in section_variables else ()
        components.append(
            PromptComponent(
                id=f"environment.{name}",
                channel="environment",
                template=template,
                value=value,
                rendered=_render(template, {"value": value}),
                variables=variables,
            )
        )
    return tuple(components)
