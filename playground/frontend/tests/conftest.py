"""Shared browser-test fixtures: one Chromium, a temporary Vite build, a viewport matrix.

CATAN_BROWSER_TMPDIR can select an approved existing temp parent; the build never
touches dist.
"""

import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir

import playwright.sync_api as pw
import pytest
from playwright.sync_api import sync_playwright

from playground.frontend.tests.browser_support import FRONTEND


@pytest.fixture(scope="session")
def chromium() -> Iterator[pw.Browser]:
    """The one Playwright instance per session: sync Playwright cannot nest."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture(scope="session")
def browser_build(chromium: pw.Browser) -> Iterator[tuple[pw.Browser, Path]]:
    parent = Path(os.environ.get("CATAN_BROWSER_TMPDIR", gettempdir()))
    subprocess.run(["ls", "-d", str(parent)], check=True, capture_output=True)
    with TemporaryDirectory(prefix="frontend-browser-", dir=parent) as directory:
        build = Path(directory) / "build"
        result = subprocess.run(
            [str(FRONTEND / "node_modules/.bin/vite"), "build", "--outDir", str(build)],
            cwd=FRONTEND, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        print(f"Temporary Vite build (removed after suite): {build}")
        yield chromium, build


@pytest.fixture(params=[(1600, 1000), (390, 844)], ids=["desktop", "mobile"])
def viewport(request: pytest.FixtureRequest) -> pw.ViewportSize:
    width, height = request.param
    return {"width": width, "height": height}
