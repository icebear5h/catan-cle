"""Real Prompt Studio browser flows, with isolated routes/storage and no inference.

Run with .venv/bin/python -m pytest playground/frontend/tests/test_shared_prompt_browser.py.
Requires installed frontend dependencies and Playwright Chromium. CATAN_BROWSER_TMPDIR
can select an approved existing temp parent; the Vite build never touches dist.
"""

import json
import os
import select
import subprocess
import sys
from collections.abc import Iterator
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from urllib.parse import urlsplit

import playwright.sync_api as pw
import pytest
from playwright.sync_api import expect, sync_playwright

FRONTEND = Path(__file__).resolve().parents[1]
ROOT = FRONTEND.parents[1]
Studio = tuple[pw.Page, dict[str, object]]


@pytest.fixture(scope="module")
def browser_build() -> Iterator[tuple[pw.Browser, Path]]:
    parent = Path(os.environ.get("CATAN_BROWSER_TMPDIR", gettempdir()))
    subprocess.run(["ls", "-d", str(parent)], check=True, capture_output=True)
    with TemporaryDirectory(prefix="shared-prompt-browser-", dir=parent) as directory:
        build = Path(directory) / "build"
        result = subprocess.run(
            [str(FRONTEND / "node_modules/.bin/vite"), "build", "--outDir", str(build)],
            cwd=FRONTEND, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        print(f"Temporary Vite build (removed after suite): {build}")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                yield browser, build
            finally:
                browser.close()


@pytest.fixture(params=[(1600, 1000), (390, 844)], ids=["desktop", "mobile"])
def viewport(request: pytest.FixtureRequest) -> pw.ViewportSize:
    width, height = request.param
    return {"width": width, "height": height}


@pytest.fixture
def mounted_studio(request: pytest.FixtureRequest, browser_build: tuple[pw.Browser, Path],
                   viewport: pw.ViewportSize, tmp_path: Path) -> Iterator[Studio]:
    browser, build = browser_build
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTHON_DOTENV_DISABLED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "CATAN_LIVE_TRACE_DB": str(tmp_path / "traces.sqlite3"),
        "CATAN_PROMPT_SUITE_DIR": str(tmp_path / "prompts"),
    }
    for name in ("CATAN_SHARED_SUITE", "CATAN_CONTEXT_SUITE", "CATAN_COMMUNICATION_SUITE"):
        env.pop(name, None)
    mode = getattr(request, "param", "empty")
    with (tmp_path / "server.log").open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "playground.frontend.tests.shared_prompt_browser_fixture",
             str(build), mode],
            cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=log, text=True,
        )
        context = browser.new_context(viewport=viewport, service_workers="block")
        context.set_default_timeout(5000)
        blocked, errors = [], []
        try:
            assert select.select([process.stdout], [], [], 30)[0], "Fixture startup timed out"
            origin = process.stdout.readline().strip()
            log.seek(0)
            assert origin.startswith("http://127.0.0.1:"), log.read()

            def intercept(route: pw.Route) -> None:
                url = urlsplit(route.request.url)
                if url.netloc == "127.0.0.1:5001" and (
                    url.path in ("/api/prompt-suite", "/api/prompt-suite/validate", "/api/reset", "/api/start-game", "/api/state")
                    or url.path.startswith("/api/live-traces")
                    or url.path.startswith("/socket.io/")
                ):
                    response = route.fetch(url=f"{origin}{url.path}?{url.query}", max_redirects=0)
                    route.fulfill(response=response, headers={
                        **response.headers, "Access-Control-Allow-Origin": "*",
                    })
                elif route.request.url.startswith(f"{origin}/"):
                    route.continue_()
                else:
                    blocked.append(route.request.url)
                    route.abort()

            def intercept_socket(route: pw.WebSocketRoute) -> None:
                if route.url.startswith(origin.replace("http://", "ws://") + "/socket.io/"):
                    route.connect_to_server()
                else:
                    blocked.append(route.url)
                    route.close()

            context.route("**/*", intercept)
            context.route_web_socket("**/*", intercept_socket)
            context.add_init_script(f"""
                const NativeWebSocket = window.WebSocket;
                window.WebSocket = class extends NativeWebSocket {{
                    constructor(url, protocols) {{
                        const target = new URL(url);
                        if (target.host === '127.0.0.1:5001') {{
                            target.host = new URL({json.dumps(origin)}).host;
                        }}
                        super(target.href, protocols);
                    }}
                }};
            """)
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
            page.goto(origin)
            with page.expect_response(lambda response: response.url.endswith("/api/prompt-suite")) as initial:
                page.get_by_role("button", name="Prompt Suite", exact=True).click()
            assert initial.value.status == 200
            payload = initial.value.json()
            assert payload["mode"] == "shared"
            expect(page.get_by_role("heading", name="Shared Definitions", exact=True)).to_be_visible()
            yield page, payload
        finally:
            if context.pages:
                screenshot = tmp_path / "browser.png"
                context.pages[0].screenshot(path=str(screenshot))
                print(f"Browser screenshot: {screenshot}")
            context.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            process.stdout.close()
            assert not blocked, f"Unexpected network requests: {blocked}"
            assert not errors, f"Browser errors: {errors}"


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


@pytest.mark.parametrize("mounted_studio", ["session"], indirect=True)
def test_new_game_returns_to_empty_setup_and_keeps_saved_session(mounted_studio: Studio) -> None:
    page, _ = mounted_studio
    origin = page.url
    assert urlsplit(origin).port != 5001
    before = page.request.get(f"{origin}api/live-traces").json()["games"]
    assert len(before) == 1 and before[0]["step_count"] == 1
    old_id = before[0]["game_id"]
    page.get_by_role("button", name="Game", exact=True).click()
    with page.expect_response(lambda response: response.url.endswith("/api/reset")) as reset:
        page.get_by_role("button", name="New Game", exact=True).click()
    assert reset.value.status == 200
    expect(page.get_by_text("Start a game to view the board", exact=True)).to_be_visible()
    start = page.get_by_role("button", name="Start Game (Random)", exact=True)
    expect(start).to_be_enabled()
    assert page.request.get(f"{origin}api/live-traces/{old_id}").json()["step_count"] == 1
    with page.expect_response(lambda response: response.url.endswith("/api/start-game")) as created:
        start.click()
    assert created.value.status == 200
    assert created.value.json()["trace_game_id"] != old_id
    expect(page.get_by_role("button", name="New Game", exact=True)).to_be_enabled()
    expect(start).to_have_count(0)
    games = page.request.get(f"{origin}api/live-traces").json()["games"]
    assert len(games) == 2
    assert next(game for game in games if game["game_id"] == old_id)["step_count"] == 1
