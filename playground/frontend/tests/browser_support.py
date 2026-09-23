"""Isolated browser plumbing: fixture server process, network sandbox, page opening."""

import json
import os
import select
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import playwright.sync_api as pw
from playwright.sync_api import expect

FRONTEND = Path(__file__).resolve().parents[1]
ROOT = FRONTEND.parents[1]
# The built app talks to the dev API origin; the sandbox reroutes exactly these paths.
DEV_API_HOST = "127.0.0.1:5001"
PROXIED_PATHS = ("/api/prompt-suite", "/api/prompt-suite/validate", "/api/reset", "/api/start-game", "/api/state")
PROXIED_PREFIXES = ("/api/live-traces", "/socket.io/")
SUITE_PIN_ENVS = ("CATAN_SHARED_SUITE", "CATAN_CONTEXT_SUITE", "CATAN_COMMUNICATION_SUITE")


@contextmanager
def fixture_server(
    build: Path, mode: str, tmp_path: Path, pins: Mapping[str, str] | None = None,
) -> Iterator[str]:
    """Run browser_fixture_server in `mode` with private storage; yield its origin."""
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTHON_DOTENV_DISABLED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "CATAN_LIVE_TRACE_DB": str(tmp_path / "traces.sqlite3"),
        "CATAN_PROMPT_SUITE_DIR": str(tmp_path / "prompts"),
    }
    for name in SUITE_PIN_ENVS:
        env.pop(name, None)
    env.update(pins or {})
    with (tmp_path / "server.log").open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "playground.frontend.tests.browser_fixture_server",
             str(build), mode],
            cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=log, text=True,
        )
        assert process.stdout is not None
        try:
            assert select.select([process.stdout], [], [], 30)[0], "Fixture startup timed out"
            origin = process.stdout.readline().strip()
            log.seek(0)
            assert origin.startswith("http://127.0.0.1:"), log.read()
            yield origin
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            process.stdout.close()


@dataclass
class Sandbox:
    """A browser context whose only reachable host is the fixture server."""

    context: pw.BrowserContext
    origin: str
    blocked: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def intercept(self, route: pw.Route) -> None:
        url = urlsplit(route.request.url)
        if url.netloc == DEV_API_HOST and (
            url.path in PROXIED_PATHS or url.path.startswith(PROXIED_PREFIXES)
        ):
            response = route.fetch(url=f"{self.origin}{url.path}?{url.query}", max_redirects=0)
            route.fulfill(response=response, headers={
                **response.headers, "Access-Control-Allow-Origin": "*",
            })
        elif route.request.url.startswith(f"{self.origin}/"):
            route.continue_()
        else:
            self.blocked.append(route.request.url)
            route.abort()

    def intercept_socket(self, route: pw.WebSocketRoute) -> None:
        if route.url.startswith(self.origin.replace("http://", "ws://") + "/socket.io/"):
            route.connect_to_server()
        else:
            self.blocked.append(route.url)
            route.close()

    def open_page(self) -> pw.Page:
        page = self.context.new_page()
        page.on("pageerror", lambda error: self.errors.append(str(error)))
        page.on("console", lambda message: (
            self.errors.append(message.text) if message.type == "error" else None
        ))
        page.goto(self.origin)
        return page


@contextmanager
def network_sandbox(
    browser: pw.Browser, origin: str, viewport: pw.ViewportSize, tmp_path: Path,
) -> Iterator[Sandbox]:
    """Route the dev API and sockets to `origin`, block everything else, fail on errors."""
    context = browser.new_context(viewport=viewport, service_workers="block")
    context.set_default_timeout(5000)
    sandbox = Sandbox(context, origin)
    try:
        context.route("**/*", sandbox.intercept)
        context.route_web_socket("**/*", sandbox.intercept_socket)
        context.add_init_script(f"""
            const NativeWebSocket = window.WebSocket;
            window.WebSocket = class extends NativeWebSocket {{
                constructor(url, protocols) {{
                    const target = new URL(url);
                    if (target.host === {json.dumps(DEV_API_HOST)}) {{
                        target.host = new URL({json.dumps(origin)}).host;
                    }}
                    super(target.href, protocols);
                }}
            }};
        """)
        yield sandbox
    finally:
        if context.pages:
            screenshot = tmp_path / "browser.png"
            context.pages[0].screenshot(path=str(screenshot))
            print(f"Browser screenshot: {screenshot}")
        context.close()
    assert not sandbox.blocked, f"Unexpected network requests: {sandbox.blocked}"
    assert not sandbox.errors, f"Browser errors: {sandbox.errors}"


def open_prompt_studio(page: pw.Page) -> dict[str, object]:
    """Switch to the Prompt Suite tab and return the Studio's initial payload."""
    with page.expect_response(lambda response: response.url.endswith("/api/prompt-suite")) as initial:
        page.get_by_role("button", name="Prompt Suite", exact=True).click()
    assert initial.value.status == 200
    payload: dict[str, object] = initial.value.json()
    if payload["mode"] == "shared":
        expect(page.get_by_role("heading", name="Shared Definitions", exact=True)).to_be_visible()
    return payload
