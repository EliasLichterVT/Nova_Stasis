import os
import tempfile
import unittest
from unittest.mock import patch

import httpx

from nova.app import app, lifespan


class LoraTests(unittest.IsolatedAsyncioTestCase):
    async def test_custom_keywords_match_scene_and_keep_imported_metadata(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch.dict(os.environ, {"NOVA_DATA_DIR": directory}))
        await self.enterAsyncContext(lifespan(app))
        browser = await self.enterAsyncContext(httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost",
            headers={"X-Nova-Request": "1"},
        ))
        installed = ["fog.safetensors", "oilpaint.safetensors", "city.safetensors"]
        app.state.db.put("lora_source", "fog.safetensors", {
            "title": "Mist Study", "base": "SD 1.5", "triggers": ["fog_lora"],
        })
        app.state.db.put("lora_source", "missing.safetensors", {
            "title": "River", "triggers": ["river"],
        })
        with patch("nova.app.node_choices", return_value=installed):
            saved = await browser.put("/api/lora-keywords", json={
                "name": "fog.safetensors", "triggers": [" fog_v2 ", "FOG_V2"], "keywords": ["river"],
            })
            self.assertEqual(saved.status_code, 200, saved.text)
            self.assertEqual(saved.json()["triggers"], ["fog_v2"])
            self.assertEqual(saved.json()["source_triggers"], ["fog_lora"])
            catalog = (await browser.get("/api/lora-keywords")).json()
            self.assertEqual(len(catalog), 3)

            response = await browser.post("/api/lora-keywords/suggest", json={
                "prompt": "Mara opens a lantern beside the river.", "selected": ["oilpaint.safetensors"],
            })
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual([entry["name"] for entry in response.json()], [
                "oilpaint.safetensors", "fog.safetensors",
            ])
            self.assertEqual(response.json()[1]["matches"], ["river"])
            self.assertEqual(response.json()[1]["base"], "SD 1.5")

            missing = await browser.put("/api/lora-keywords", json={
                "name": "not-installed.safetensors", "triggers": ["test"], "keywords": [],
            })
            self.assertEqual(missing.status_code, 404)
