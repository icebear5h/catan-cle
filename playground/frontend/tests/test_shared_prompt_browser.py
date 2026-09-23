"""Real Prompt Studio browser flows, with isolated routes/storage and no inference.

Run with uv run --no-sync python -m pytest playground/frontend/tests/test_shared_prompt_browser.py.
Requires installed frontend dependencies and Playwright Chromium; see conftest.py.
"""

from collections.abc import Iterator
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import playwright.sync_api as pw
import pytest
from playwright.sync_api import expect

from cle.harness.yaml_source import BUILTIN_SUITES_DIR
from playground.frontend.tests.browser_support import (
    fixture_server,
    network_sandbox,
    open_prompt_studio,
)

Studio = tuple[pw.Page, dict[str, object]]
LEGACY_PINS = {
    "CATAN_CONTEXT_SUITE": str(BUILTIN_SUITES_DIR / "catan_v11.yaml"),
    "CATAN_COMMUNICATION_SUITE": str(BUILTIN_SUITES_DIR / "communication_v5.yaml"),
}


@pytest.fixture
def mounted_studio(request: pytest.FixtureRequest, browser_build: tuple[pw.Browser, Path],
                   viewport: pw.ViewportSize, tmp_path: Path) -> Iterator[Studio]:
    """Studio over an `empty`, `loaded` (seeded game) or `legacy` (pinned pair) server."""
    browser, build = browser_build
    mode = getattr(request, "param", "empty")
    pins = LEGACY_PINS if mode == "legacy" else None
    server_mode = "loaded" if mode == "loaded" else "empty"
    with (fixture_server(build, server_mode, tmp_path, pins) as origin,
          network_sandbox(browser, origin, viewport, tmp_path) as sandbox):
        page = sandbox.open_page()
        payload = open_prompt_studio(page)
        assert payload["mode"] == ("legacy" if mode == "legacy" else "shared")
        yield page, payload


def select_component(page: pw.Page, name: str) -> pw.Locator:
    page.get_by_role("navigation", name="Prompt components").get_by_role(
        "button", name=name, exact=True,
    ).click()
    return page.get_by_role("textbox", name="Selected prompt component string")


def validate(page: pw.Page) -> dict[str, object]:
    with page.expect_response(lambda response: response.url.endswith("/api/prompt-suite/validate")) as result:
        page.get_by_role("button", name="Validate", exact=True).click()
    assert result.value.status == 200, result.value.text()
    expect(page.get_by_role("status").filter(
        has_text="All component strings are valid. Nothing was written.",
    )).to_be_visible()
    return result.value.json()["candidate"]


@pytest.mark.parametrize("mounted_studio", ["loaded"], indirect=True)
def test_shared_notes_candidate_and_live_game_save(mounted_studio: Studio, tmp_path: Path) -> None:
    page, original = mounted_studio
    expect(page.get_by_role("navigation", name="Prompt components").get_by_role("heading")).to_have_text([
        "Shared Definitions", "Compositions", "Phase Guidance",
    ])
    definitions = page.get_by_role("navigation", name="Prompt components").locator("section").filter(
        has=page.get_by_role("heading", name="Shared Definitions", exact=True),
    )
    components = original["shared"]["document"]["components"]
    expect(definitions.locator("button code")).to_have_text([
        f"{definition['channel']}.{name}" for name, definition in components.items()
    ])
    editor = select_component(page, "notes environment.notes")
    expect(page.get_by_text("Referenced by: decision, speech", exact=True)).to_be_visible()
    editor.fill("BROWSER NOTES:\n{{ notes }}")
    expect(page.get_by_role("status")).to_contain_text("Saves apply to the next decision or speech batch")
    expect(page.get_by_role("button", name="Save active prompts", exact=True)).to_be_enabled()
    expect(page.get_by_role("button", name="Reset built-in", exact=True)).to_be_enabled()
    candidate = validate(page)
    assert candidate["saving_locked"] is False
    preview = page.locator(".prompt-component-preview")
    for consumer, key in (("decision", "decision"), ("speech", "communication")):
        group = candidate["preview"][key]
        assert group["status"] == "rendered"
        assert group["provenance"] == "current_typed_context"
        notes = f"Accepted {group['actor']} notes: reserve wheat."
        component = next(item for item in group["components"] if item["id"] == "environment.notes")
        assert component["variables"]["notes"] == notes
        assert component["rendered"] == f"BROWSER NOTES:\n{notes}"
        article = preview.locator("article").filter(
            has=page.get_by_role("heading", name=f"{consumer} / rendered", exact=True),
        )
        expect(article.locator("pre").first).to_have_text(component["rendered"])
        article.locator("pre").first.scroll_into_view_if_needed()
        expect(article.locator("pre").first).to_be_in_viewport()
    assert not (tmp_path / "prompts/shared.yaml").exists()
    with page.expect_response(lambda response: response.request.method == "PUT") as saved:
        page.get_by_role("button", name="Save active prompts", exact=True).click()
    assert saved.value.status == 200
    assert (tmp_path / "prompts/shared.yaml").exists()
    expect(page.get_by_role("button", name="Save active prompts", exact=True)).to_be_disabled()
    expect(page.locator(".prompt-studio-notice").filter(has_text="Active prompts saved")).to_be_visible()


@pytest.mark.parametrize("mounted_studio", ["loaded"], indirect=True)
def test_composition_reorders_are_independent(mounted_studio: Studio) -> None:
    page, original = mounted_studio
    document = deepcopy(original["shared"]["document"])
    for consumer, direction, offset in (("decision", "up", -1), ("speech", "down", 1)):
        editor = select_component(page, f"{consumer} reference order compositions.{consumer}")
        order = document["compositions"][consumer]["order"]
        expect(editor).to_have_value("\n".join(order))
        page.get_by_role("button", name=f"Move notes {direction}", exact=True).click()
        index = order.index("notes")
        order[index], order[index + offset] = order[index + offset], order[index]
        expect(editor).to_have_value("\n".join(order))
        candidate = validate(page)
        assert candidate["shared"]["document"] == document
        key = "decision" if consumer == "decision" else "communication"
        expected_ids = [f"{document['components'][name]['channel']}.{name}" for name in order]
        assert [item["id"] for item in candidate["preview"][key]["components"]] == expected_ids
        expect(page.locator(".prompt-component-preview article > div > h3:first-child")).to_have_text(expected_ids)
    for consumer in ("decision", "speech"):
        editor = select_component(page, f"{consumer} reference order compositions.{consumer}")
        expect(editor).to_have_value("\n".join(document["compositions"][consumer]["order"]))


def test_save_uses_current_source_hash_and_refresh_persists(
    mounted_studio: Studio, tmp_path: Path,
) -> None:
    page, original = mounted_studio
    assert original["saving_locked"] is False
    expected_hash = original["shared"]["sha256"]
    editor = select_component(page, "notes environment.notes")
    for number in (1, 2):
        template = f"SAVED NOTES {number}:\n{{{{ notes }}}}"
        editor.fill(template)
        save = page.get_by_role("button", name="Save active prompts", exact=True)
        expect(save).to_be_enabled()
        with page.expect_response(lambda response: response.request.method == "PUT") as result:
            save.click()
        assert result.value.request.post_data_json["expected"] == {"shared": expected_hash}
        assert result.value.status == 200, result.value.text()
        saved = result.value.json()["shared"]
        assert saved["document"]["components"]["notes"]["template"] == template
        assert saved["overridden"] is True
        assert saved["sha256"] != expected_hash
        expected_hash = saved["sha256"]
        assert sha256((tmp_path / "prompts/shared.yaml").read_bytes()).hexdigest() == expected_hash
        expect(save).to_be_disabled()
        expect(page.locator(".prompt-studio-dirty")).to_have_count(0)
        expect(page.locator(".prompt-studio-footer code")).to_have_attribute("title", expected_hash)

    editor.fill("DISCARD THIS UNSAVED EDIT:\n{{ notes }}")
    page.once("dialog", lambda dialog: dialog.accept())
    with page.expect_response(lambda response: response.url.endswith("/api/prompt-suite")) as refreshed:
        page.get_by_role("button", name="Refresh", exact=True).click()
    assert refreshed.value.json()["shared"] == saved
    expect(editor).to_have_value(template)
    expect(page.locator(".prompt-studio-dirty")).to_have_count(0)
    page.reload()
    page.get_by_role("button", name="Prompt Suite", exact=True).click()
    expect(select_component(page, "notes environment.notes")).to_have_value(template)
    expect(page.locator(".prompt-studio-footer")).to_contain_text("local override")


def test_reset_restores_built_in_and_removes_override(mounted_studio: Studio, tmp_path: Path) -> None:
    page, original = mounted_studio
    built_in = original["shared"]["document"]["components"]["notes"]["template"]
    override = tmp_path / "prompts/shared.yaml"
    footer = page.locator(".prompt-studio-footer")
    reset = page.get_by_role("button", name="Reset built-in", exact=True)
    editor = select_component(page, "notes environment.notes")
    editor.fill("OVERRIDE NOTES:\n{{ notes }}")
    with page.expect_response(lambda response: response.request.method == "PUT") as saved:
        page.get_by_role("button", name="Save active prompts", exact=True).click()
    assert saved.value.status == 200, saved.value.text()
    saved_hash = saved.value.json()["shared"]["sha256"]
    expect(footer).to_contain_text("local override")

    page.once("dialog", lambda dialog: dialog.dismiss())
    reset.click()
    expect(footer).to_contain_text("local override")
    assert override.exists()

    page.once("dialog", lambda dialog: dialog.accept())
    with page.expect_response(lambda response: response.request.method == "DELETE") as result:
        reset.click()
    assert result.value.request.post_data_json == {"expected": {"shared": saved_hash}}
    assert result.value.status == 200, result.value.text()
    restored = result.value.json()["shared"]
    assert restored["overridden"] is False
    assert restored["sha256"] == original["shared"]["sha256"]
    assert restored["document"] == original["shared"]["document"]
    assert not override.exists()
    expect(editor).to_have_value(built_in)
    expect(footer).to_contain_text("source default")
    expect(footer.locator("code")).to_have_attribute("title", original["shared"]["sha256"])
    expect(page.locator(".prompt-studio-notice").filter(
        has_text="Built-in prompts selected for the next inference boundary",
    )).to_be_visible()
    expect(page.locator(".prompt-studio-dirty")).to_have_count(0)


@pytest.mark.parametrize("mounted_studio", ["legacy"], indirect=True)
def test_pinned_legacy_pair_is_read_only(mounted_studio: Studio, tmp_path: Path) -> None:
    page, payload = mounted_studio
    assert payload["read_only"] is True
    assert "shared" not in payload
    for kind, suite_id, version in (
        ("decision", "catan-agent", "11.0.0"), ("communication", "catan-communication", "5"),
    ):
        metadata = payload[kind]
        assert (metadata["id"], metadata["version"]) == (suite_id, version)
        assert (metadata["status"], metadata["overridden"]) == ("legacy", False)
    expect(page.get_by_text("Pinned legacy decision and table talk suites; read-only.")).to_be_visible()
    expect(page.locator(".prompt-studio-lock").filter(
        has_text="A legacy prompt pair is pinned by environment or live config.",
    )).to_be_visible()
    expect(page.get_by_role("button", name="Save active prompts", exact=True)).to_be_disabled()
    expect(page.get_by_role("button", name="Reset built-in", exact=True)).to_be_disabled()
    expect(page.get_by_role("button", name="Refresh", exact=True)).to_be_enabled()
    expect(page.get_by_role("button", name="Validate", exact=True)).to_have_count(0)
    expect(page.get_by_role("navigation", name="Prompt components")).to_have_count(0)
    expect(page.get_by_role("textbox")).to_have_count(0)
    footer = page.locator(".prompt-studio-footer")
    expect(footer).to_contain_text("Decision catan-agent@11.0.0 · legacy · built-in")
    expect(footer).to_contain_text("Table talk catan-communication@5 · legacy · built-in")
    for label, key in (("Decision", "decision"), ("Table talk", "communication")):
        group = payload["preview"][key]
        heading = f"{label} {payload[key]['id']}@{payload[key]['version']} / {group['status']}"
        expect(page.get_by_role("heading", name=heading, exact=True)).to_be_visible()
    with page.expect_response(lambda response: response.url.endswith("/api/prompt-suite")) as refreshed:
        page.get_by_role("button", name="Refresh", exact=True).click()
    assert refreshed.value.json()["mode"] == "legacy"
    assert not (tmp_path / "prompts").exists()
