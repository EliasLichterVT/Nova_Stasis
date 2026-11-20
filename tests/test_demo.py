import asyncio
import os
import unittest
from unittest.mock import patch

import httpx

from nova.app import app, lifespan


class DemoTests(unittest.IsolatedAsyncioTestCase):
    async def test_browser_demo_runs_sample_graph_and_keeps_storage_temporary(self):
        self.enterContext(patch.dict(os.environ, {"NOVA_BROWSER_DEMO": "1"}))
        async with lifespan(app):
            directory = app.state.assets.parent
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://localhost",
                headers={"X-Nova-Request": "1"},
            ) as browser:
                demo = (await browser.get("/api/demo")).json()
                self.assertTrue(demo["enabled"])
                first, second = demo["references"]
                reference = await browser.get(f"/api/uploads/{first}")
                self.assertEqual(reference.headers["content-type"], "image/png")
                self.assertFalse((await browser.get("/api/imports")).json()["enabled"])
                models = (await browser.get("/api/models")).json()
                job = await browser.post("/api/generations", json={
                    "prompt": "Two adults hold a lantern on a bridge.",
                    "ref_a": first, "ref_b": second, "batch": 2, "authorized": True,
                    **{key: models[key][0] for key in ("checkpoint", "adapter", "vision")},
                })
                self.assertEqual(job.status_code, 202, job.text)
                path = f"/api/generations/{job.json()['id']}"
                await asyncio.sleep(2.05)
                result = (await browser.get(path)).json()
                self.assertEqual(result["status"], "complete")
                self.assertEqual(len(result["images"]), 2)
                preview = await browser.get(f"{path}/images/1")
                self.assertTrue(preview.content.startswith(b"\x89PNG"))
        self.assertFalse(directory.exists())
