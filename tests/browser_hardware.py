"""Run the desktop studio hardware workflow in Chromium with simulated ComfyUI."""

import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


class ComfyStub(BaseHTTPRequestHandler):
    stats = None

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/system_stats" and self.stats is None:
            self.send_error(503)
            return
        contents = json.dumps(self.stats if self.path == "/system_stats" else {}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(contents)))
        self.end_headers()
        self.wfile.write(contents)


def browser_checks(url):
    ComfyStub.stats = {"system": {"ram_total": 8 * GIB}, "devices": [{"type": "cpu", "name": "Test CPU"}]}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=os.environ.get("NOVA_TEST_CHROMIUM"))
        page = browser.new_page(viewport={"width": 375, "height": 820})
        page.goto(url)
        status = page.locator("#hardware-status")
        width = page.get_by_label("Width", exact=True)
        expect(status).to_contain_text("CPU mode")
        expect(width).to_have_value("384")
        expect(page.get_by_text("CPU generation can take", exact=False)).to_be_visible()
        width.fill("1024")
        expect(page.locator("#hardware-warning")).to_be_visible()

        ComfyStub.stats = {"system": {"ram_total": 32 * GIB}, "devices": [
            {"type": "cuda", "name": "Test GPU", "vram_total": 16 * GIB, "vram_free": 12 * GIB},
        ]}
        page.get_by_role("button", name="Scan again").click()
        expect(status).to_have_text("Accelerated")
        expect(page.get_by_text("Test GPU · CUDA", exact=True)).to_be_visible()
        expect(width).to_have_value("1024")
        expect(page.get_by_label("Steps", exact=True)).to_have_value("12")
        page.get_by_role("button", name="Use recommended settings").click()
        expect(page.get_by_label("Steps", exact=True)).to_have_value("28")
        expect(page.locator("#hardware-warning")).to_be_hidden()
        expect(page.locator("#hardware-budget")).to_have_text(re.compile("batch 2"))
        assert page.evaluate("document.documentElement.scrollWidth === innerWidth")

        ComfyStub.stats = None
        page.get_by_role("button", name="Scan again").click()
        expect(status).to_contain_text("Hardware limits unavailable")
        expect(width).to_have_value("1024")
        browser.close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as private, ThreadingHTTPServer(("127.0.0.1", 0), ComfyStub) as comfy:
        threading.Thread(target=comfy.serve_forever, daemon=True).start()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        env = {**os.environ, "NOVA_DATA_DIR": private,
               "NOVA_COMFY_URL": f"http://127.0.0.1:{comfy.server_port}", "NOVA_STARTER_MODELS": "0"}
        python = os.environ.get("NOVA_TEST_PYTHON", sys.executable)
        app = subprocess.Popen([python, "-m", "uvicorn", "nova.app:app", "--host", "127.0.0.1",
                                "--port", str(port), "--no-access-log"], cwd=ROOT, env=env)
        try:
            for _ in range(100):
                try:
                    with urlopen(url, timeout=0.2):
                        break
                except OSError:
                    if app.poll() is not None:
                        raise RuntimeError("Nova stopped during startup.")
                    time.sleep(0.1)
            else:
                raise TimeoutError("Nova did not start.")
            browser_checks(url)
            print("Hardware browser workflow passed.")
        finally:
            app.terminate()
            app.wait(timeout=10)
            comfy.shutdown()
