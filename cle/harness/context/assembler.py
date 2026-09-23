"""Deterministic context assembly from typed authoritative inputs."""

from __future__ import annotations

from typing import cast

from cle.harness.action_tools import (
    render_action_tools,
    render_shared_legal_actions,
    trade_responder_note,
)
from cle.harness.board_surface import BoardPresenter
from cle.harness.catan_board_surface import IndexedTileRowsBoardPresenter
from cle.harness.components import (
    ComponentInputs,
    observation_component_values,
    render_component_definitions,
)
from cle.harness.context.environment import environment_values
from cle.harness.context.formatting import (
    TEMPLATE_VARIABLE,
    color_name,
    format_commitments,
    format_detail,
    format_events,
    format_value,
    historical_event_detail,
    legacy_section_template,
    render_template,
    shared_event_detail,
)
from cle.harness.models import (
    ModelMessage,
    ModelRequest,
    PlayerSession,
    PromptComponent,
)
from cle.harness.reasoning import (
    reasoning_budget_for_attempt,
    reasoning_budget_instruction,
    validate_native_reasoning_request,
)
from cle.harness.suite import ContextSuite, SystemConfig
from cle.players.contracts import PlayerContext


class ContextAssembler:
    """Render one complete active model context from typed authoritative inputs."""

    def __init__(
        self,
        suite: ContextSuite,
        *,
        board_presenter: BoardPresenter | None = None,
    ) -> None:
        self.suite = suite
        self.board_presenter = board_presenter or IndexedTileRowsBoardPresenter()

    def assemble(
        self,
        context: PlayerContext,
        session: PlayerSession,
        feedback: str | None = None,
    ) -> ModelRequest:
        components = self.render_components(context, session, feedback)
        board_presentation = self.board_presenter.present(context)
        budget = reasoning_budget_for_attempt(context, feedback)
        system = "\n\n".join(
            component.rendered for component in components
            if component.channel == "system" and component.rendered
        )
        environment = "\n\n".join(
            component.rendered
            for component in components
            if component.channel == "environment" and component.rendered
        )
        return ModelRequest(
            decision_id=context.context_id,
            session_id=session.session_id,
            messages=(
                ModelMessage(role="system", content=system),
                *self._history(session),
                ModelMessage(role="user", content=environment),
            ),
            components=components,
            board_presentation=board_presentation,
            reasoning_request=tuple(
                validate_native_reasoning_request({"effort": budget.effort}).items()
            ),
            max_tokens=budget.max_tokens,
        )

    def render_components(
        self,
        context: PlayerContext,
        session: PlayerSession,
        feedback: str | None = None,
    ) -> tuple[PromptComponent, ...]:
        """Render typed system/environment strings from one private perspective."""
        if context.actor != session.color:
            raise ValueError(
                f"Context actor {context.actor} does not match player {session.color}"
            )

        guidance = self.suite.phase_guidance.get(context.prompt_key)
        # Historical suites used one shared setup-road key. Restored games
        # retain their recorded source while current routing uses road ordinals.
        if guidance is None and context.prompt_key in {
            "initial_road_1",
            "initial_road_2",
        }:
            guidance = self.suite.phase_guidance.get("initial_road")
        if guidance is None:
            raise ValueError(
                f"Suite {self.suite.id}@{self.suite.version} has no guidance "
                f"for {context.prompt_key!r}"
            )

        if self.suite.context.mode == "shared":
            return self._shared_components(context, session, guidance, feedback)
        return self._historical_components(context, session, guidance, feedback)

    def _shared_components(
        self,
        context: PlayerContext,
        session: PlayerSession,
        guidance: str,
        feedback: str | None,
    ) -> tuple[PromptComponent, ...]:
        if context.observation.my_color != context.actor:
            raise ValueError("Observation perspective does not match context actor")
        request = "Choose one action using its tool and named arguments."
        if self.suite.context.reactive_speech:
            request = (
                "Choose one game action OR standalone say. Say keeps this game decision pending; "
                "after bounded replies you must act."
                if context.speech_allowed else
                "Choose one game action now. Standalone say is unavailable for this decision."
            )
        if self.suite.context.deterministic_batches:
            request += " You may replace a single game action with a bounded deterministic actions batch, as defined in the response schema."
        if feedback:
            request += f"\n\nCORRECTION FROM THE SANDBOX:\n{feedback}"
        request += (
            f"\n\n{reasoning_budget_instruction(reasoning_budget_for_attempt(context, feedback))}"
        )
        if self.suite.context.deterministic_batches:
            legal_actions = render_action_tools(context, shared=True)
            responder_note = trade_responder_note(context)
            if responder_note:
                legal_actions = f"{legal_actions}\n\n{responder_note}"
        else:
            legal_actions = render_shared_legal_actions(context)
        return render_component_definitions(
            self.suite.components,
            self.suite.context.order,
            ComponentInputs(
                color=color_name(session.color),
                notes=session.strategic_memory,
                max_notes_chars=str(self.suite.context.max_notes_chars),
                visible_events=format_events(context.events, shared=True),
                recent_table_talk=format_events(context.recent_messages),
                commitments=format_commitments(context.active_commitments),
                phase_guidance=guidance,
                legal_actions=legal_actions,
                decision_request=request,
                **observation_component_values(
                    context.observation,
                    include_initial_placement_order=(
                        self.suite.context.initial_placement_order == "both_rounds"
                    ),
                ),
            ),
        )

    def _historical_components(
        self,
        context: PlayerContext,
        session: PlayerSession,
        guidance: str,
        feedback: str | None,
    ) -> tuple[PromptComponent, ...]:
        system_values = {
            "color": color_name(session.color),
            "phase_guidance": guidance,
            "response_instruction": self.suite.response.instruction,
        }
        system_template = cast("SystemConfig", self.suite.system).template
        referenced_system_values = tuple(
            (name, system_values[name])
            for name in sorted(set(TEMPLATE_VARIABLE.findall(system_template)))
        )
        components = [
            PromptComponent(
                id="system.identity",
                channel="system",
                template=system_template,
                value=system_values["color"],
                rendered=render_template(
                    system_template,
                    system_values,
                ),
                variables=referenced_system_values,
            )
        ]
        content = self._environment_values(context, session, guidance, feedback)
        for section_name in self.suite.context.order:
            if section_name == "trajectory":
                continue
            section = self.suite.sections[section_name]
            value = content.get(section_name, "")
            if not value:
                if section.empty == "omit":
                    continue
                value = section.empty_text
            template = section.template or legacy_section_template(section.heading)
            referenced = set(TEMPLATE_VARIABLE.findall(template))
            variables = (("value", value),) if "value" in referenced else ()
            rendered = render_template(template, {"value": value})
            components.append(
                PromptComponent(
                    id=f"environment.{section_name}",
                    channel="environment",
                    template=template,
                    value=value,
                    rendered=rendered,
                    variables=variables,
                )
            )

        if feedback and "decision_request" not in self.suite.context.order:
            correction_template = "CORRECTION FROM THE SANDBOX:\n{{ value }}"
            components.append(
                PromptComponent(
                    id="environment.correction",
                    channel="environment",
                    template=correction_template,
                    value=feedback,
                    rendered=render_template(
                        correction_template,
                        {"value": feedback},
                    ),
                    variables=(("value", feedback),),
                )
            )
        return tuple(components)

    def _history(self, session: PlayerSession) -> tuple[ModelMessage, ...]:
        if self.suite.context.memory_mode == "fresh_notes":
            return ()
        messages = tuple(session.messages)
        maximum = self.suite.context.trajectory.max_messages
        if maximum is None or len(messages) <= maximum:
            return messages

        messages = messages[-maximum:]
        if messages and messages[0].role == "assistant":
            messages = messages[1:]
        return messages

    def _environment_values(
        self,
        context: PlayerContext,
        session: PlayerSession,
        guidance: str,
        feedback: str | None,
    ) -> dict[str, str]:
        return environment_values(self.suite, context, session, guidance, feedback)

    # The historical private formatting surface stays callable on the class:
    # communication rendering and prompt contracts reach it by these names.
    _color_name = staticmethod(color_name)
    _format_value = staticmethod(format_value)
    _format_detail = staticmethod(format_detail)
    _format_commitments = staticmethod(format_commitments)
    _legacy_section_template = staticmethod(legacy_section_template)
    _historical_event_detail = staticmethod(historical_event_detail)
    _shared_event_detail = staticmethod(shared_event_detail)
    _format_events = staticmethod(format_events)
    _render_template = staticmethod(render_template)
