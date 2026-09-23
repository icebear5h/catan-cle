"""A provider-free Inspect model API that replays recorded outputs."""

from __future__ import annotations

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    GenerateConfig,
    Model,
    ModelAPI,
    ModelOutput,
    modelapi,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolChoice, ToolInfo

from evals.inspect_archives.config import ARCHIVE_IMPORT_SCHEMA
from evals.inspect_archives.support import _archive_metadata, _model_usage, _seconds, _stop_reason


@modelapi("archive")
class ArchivedModelAPI(ModelAPI):
    """Identity-only model API that fails if an importer attempts inference."""

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        del input, tools, tool_choice, config
        raise RuntimeError(
            "Archived Inspect tasks must replay retained outputs and never call a provider"
        )


def archived_model(model_id: str) -> Model:
    """Return an Inspect model carrying the archived model's exact identity."""

    return Model(ArchivedModelAPI(model_id), GenerateConfig())


@solver
def replay_archived_output() -> Solver:
    """Append one retained assistant response without invoking ``generate``."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        archive = _archive_metadata(state)
        response = str(archive.get("response") or "")
        error = archive.get("error")
        model_id = str(archive["model_id"])
        output = ModelOutput.from_content(
            model=model_id,
            content=response,
            stop_reason=_stop_reason(archive.get("finish_reason")),
            error=str(error) if error else None,
        )
        output.usage = _model_usage(archive.get("usage"))
        output.time = _seconds(archive.get("latency_ms"))
        output.metadata = {
            "archive_import_schema": ARCHIVE_IMPORT_SCHEMA,
            "provider": archive.get("provider"),
            "served_model": archive.get("served_model"),
            "adapter_source": archive.get("adapter_source"),
            "provider_response_id": archive.get("provider_response_id"),
            "provider_request_id": archive.get("provider_request_id"),
            "provider_native_finish_reason": archive.get(
                "provider_native_finish_reason"
            ),
            "recorded_at": archive.get("recorded_at"),
            "source_path": archive.get("source_path"),
        }
        state.messages.append(
            ChatMessageAssistant(
                content=response,
                source="generate",
                model=model_id,
                metadata={
                    "archived": True,
                    "recorded_at": archive.get("recorded_at"),
                },
            )
        )
        state.output = output
        state.completed = True
        return state

    return solve

