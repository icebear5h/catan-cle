"""The authored view of a suite: what the Studio editor renders and edits."""

from cle.harness.models import PromptComponent
from cle.harness.prompt_store import ActivePromptSuites, PromptSuiteDocument
from cle.harness.shared_suite import parse_shared_prompt_suite

__all__ = ["component_payload", "editor_payload", "suite_metadata"]


def component_payload(component: PromptComponent) -> dict[str, object]:
    return {
        "id": component.id,
        "channel": component.channel,
        "template": component.template,
        "value": component.value,
        "rendered": component.rendered,
        "variables": dict(component.variables),
    }


def suite_metadata(document: PromptSuiteDocument) -> dict[str, object]:
    return {
        "id": document.id,
        "version": document.version,
        "status": document.status,
        "sha256": document.sha256,
        "overridden": document.overridden,
    }


def editor_payload(active: ActivePromptSuites) -> dict[str, object]:
    if active.shared is None:
        # A pinned legacy pair still runs and previews, but the Studio never edits it.
        return {
            "mode": "legacy",
            "read_only": True,
            **{
                kind: suite_metadata(document)
                for kind, document in (
                    ("decision", active.decision), ("communication", active.communication),
                )
                if document is not None
            },
        }
    shared = active.shared
    bundle = shared.bundle or parse_shared_prompt_suite(shared.source)
    return {
        "mode": "shared",
        "shared": {**suite_metadata(shared), "document": bundle.model_dump(mode="json")},
    }
