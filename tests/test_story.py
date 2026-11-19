import os
import tempfile
import unittest
from unittest.mock import patch

import httpx

from nova.app import app, lifespan
from nova.story import StoryRequest, translate_story


class StoryTests(unittest.TestCase):
    def test_scenes_follow_source_order_and_layout(self):
        request = StoryRequest(
            story='Mara raised a lamp on the bridge. "Over here," Ivo said. Fog hung over the river.\n'
                  '***\n'
                  'Ivo opened the red door as Mara held the lantern.',
            name_a="Mara", name_b="Ivo", layout="vertical",
        )
        result = translate_story(request)
        first, second = [scene["prompt"] for scene in result["scenes"]]
        self.assertIn("Mara matching Reference A in the upper part", first)
        self.assertIn("Ivo matching Reference B in the lower part", second)
        self.assertIn("Fog hung over the river.", first)
        self.assertNotIn("Over here", first)
        self.assertNotIn("Ivo said", first)
        self.assertIn("Ivo opened the red door", second)
        self.assertNotIn("bridge", second)

    def test_most_visual_details_take_priority(self):
        result = translate_story(StoryRequest(
            story="Mara wished things had turned out differently. Later, Ivo stood beside a blue door. "
                  "Mara held up a gold key. Fog covered the river. She wondered what would happen next.",
        ))
        prompt = result["scenes"][0]["prompt"]
        self.assertIn("blue door", prompt)
        self.assertIn("gold key", prompt)
        self.assertIn("Fog covered", prompt)
        self.assertNotIn("wondered", prompt)
        self.assertNotIn("wished", prompt)

    def test_scene_limit_and_duplicate_names(self):
        result = translate_story(StoryRequest(story="\n\n".join(f"Story action {i}." for i in range(15))))
        self.assertEqual(len(result["scenes"]), 12)
        self.assertTrue(result["truncated"])
        with self.assertRaises(ValueError):
            StoryRequest(story="A person crossed the street.", name_a="Ivo", name_b="ivo")


class StoryApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_translation_reaches_normal_generation_gate(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch.dict(os.environ, {"NOVA_DATA_DIR": directory}))
        await self.enterAsyncContext(lifespan(app))
        client = await self.enterAsyncContext(httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost",
            headers={"X-Nova-Request": "1"},
        ))
        dialogue = await client.post("/api/story-prompts", json={"story": '"Go," said Mara.'})
        self.assertEqual(dialogue.status_code, 422)
        translated = await client.post("/api/story-prompts", json={"story": "Two adults walked in the rain."})
        self.assertEqual(translated.status_code, 200)
        prompt = translated.json()["scenes"][0]["prompt"]
        self.assertIn("walked in the rain.", prompt)
        submitted = await client.post("/api/generations", json={
            "prompt": prompt, "checkpoint": "demo", "adapter": "demo", "vision": "demo",
            "ref_a": "00000000-0000-0000-0000-000000000001",
            "ref_b": "00000000-0000-0000-0000-000000000002",
        })
        self.assertEqual(submitted.status_code, 422)
        self.assertIn("authorized", submitted.text)
