"""Optional real-browser text rendering check using an existing Chromium binary.

No browser/dependency download. Core API/replay fixtures always run separately.
"""

import html
import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from test_demo import ATTACK, demo_fixture

from atlasrank.artifacts.dense import file_hash
from atlasrank.serving.app import create_app

pytestmark = pytest.mark.skipif(
    os.environ.get("ATLASRANK_BROWSER_TESTS") != "1",
    reason="Set ATLASRANK_BROWSER_TESTS=1 to opt into the optional headless browser check",
)


def browser_binary() -> str | None:
    brave = Path("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser")
    return (
        str(brave)
        if brave.is_file()
        else next(
            (
                path
                for name in ["chromium", "chromium-browser", "google-chrome"]
                if (path := shutil.which(name))
            ),
            None,
        )
    )


@pytest.mark.parametrize("with_fixture", [True, False])
def test_browser_renders_untrusted_text_without_executing_markup(
    tmp_path: Path, with_fixture: bool
) -> None:
    binary = browser_binary()
    if binary is None:
        pytest.skip("No existing Chromium browser; do not download one in routine tests")
    index, fixture, _ = demo_fixture(tmp_path)
    app = create_app(
        index, fixture if with_fixture else None, file_hash(fixture) if with_fixture else None
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.started
        node = shutil.which("node")
        if (
            node is None
            or subprocess.run(
                [node, "-e", 'process.exit(typeof WebSocket === "function" ? 0 : 1)'],
                check=False,
                capture_output=True,
                timeout=5,
            ).returncode
            != 0
        ):
            pytest.skip("Optional browser check requires an existing Node with built-in WebSocket")
        log = (tmp_path / "browser.log").open("w")
        browser = subprocess.Popen(
            [
                binary,
                "--headless",
                "--disable-gpu",
                "--no-first-run",
                "--disable-extensions",
                "--disable-background-networking",
                "--disable-sync",
                "--disable-component-update",
                "--disable-features=MediaRouter,OptimizationHints,Translate",
                "--remote-debugging-port=0",
                "--user-data-dir=" + str(tmp_path / "browser-profile"),
                "about:blank",
            ],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            active = tmp_path / "browser-profile/DevToolsActivePort"
            deadline = time.monotonic() + 10
            while not active.exists() and browser.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            assert active.exists(), (tmp_path / "browser.log").read_text()
            output = tmp_path / "rendered.json"
            subprocess.run(
                [
                    node,
                    str(Path(__file__).with_name("browser_demo.mjs")),
                    active.read_text().splitlines()[0],
                    f"http://127.0.0.1:{port}/demo#saved",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            dom = json.loads(output.read_text())
            assert dom["systemDefault"] == "dark"
            assert dom["themePreserved"] is True
            assert dom["pendingRaceSafe"] is True
            assert dom["completedRaceSafe"] is True
            assert dom["reducedMotion"] is True
            assert {(v["theme"], v["width"], v["mode"]) for v in dom["views"]} == {
                (theme, width, mode)
                for theme in ["light", "dark"]
                for width in [320, 375, 414, 768, 1280]
                for mode in ["saved", "live"]
            }
            for view in dom["views"]:
                assert view["scroll"] <= view["width"], view
                assert not view["wrapped"], view
                assert min(view["contrast"]) >= 4.5, view
            assert dom["views"][0]["paper"] != dom["views"][-1]["paper"]
            assert "data-pwned=" not in dom["saved"]
            assert "<img src=x" not in dom["saved"]
            if with_fixture:
                assert html.escape(ATTACK, quote=False) in dom["saved"]
                assert "saved development replay" in dom["saved"]
                assert "Development judgment · unknown" in dom["saved"]
            else:
                assert (
                    "Saved comparison unavailable: load a verified five-example fixture."
                    in dom["saved"]
                )
            assert "products returned · live BM25" in dom["live"]
            assert "Measured search time" in dom["live"]
        finally:
            try:
                os.killpg(browser.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                browser.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(browser.pid, signal.SIGKILL)
                browser.wait(timeout=5)
            log.close()

    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        assert not thread.is_alive()
