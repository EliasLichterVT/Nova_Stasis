import io
import os
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx
from PIL import Image

from nova.app import app, lifespan
from nova.media import region_mask
from nova.workflow import ASSET_CHOICES, Settings, build_workflow


def sample_png(size=(64, 64)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "orange").save(buffer, "PNG")
    return buffer.getvalue()


class GraphTests(unittest.TestCase):
    def test_separate_adapters_and_masks(self):
        settings = Settings(prompt="Two explorers", checkpoint="sdxl.safetensors",
                            adapter="adapter.bin", vision="vit.safetensors",
                            ref_a=uuid4(), ref_b=uuid4(), strength_a=0.55, strength_b=0.9)
        images = {name: f"{name}.png" for name in ("ref_a", "ref_b", "region_a", "region_b")}
        graph = build_workflow(settings, images, 0, "job")
        adapters = [(key, node["inputs"]) for key, node in graph.items()
                    if node["class_type"] == "IPAdapterAdvanced"]
        (first_id, first), (second_id, second) = adapters
        self.assertEqual(second["model"], [first_id, 0])
        self.assertEqual([first["weight"], second["weight"]], [0.55, 0.9])
        for entry, slot in zip([first, second], "ab"):
            self.assertEqual(graph[entry["image"][0]]["inputs"]["image"], images[f"ref_{slot}"])
            self.assertEqual(graph[entry["attn_mask"][0]]["inputs"]["image"], images[f"region_{slot}"])
        sampler = next(node["inputs"] for node in graph.values() if node["class_type"] == "KSampler")
        self.assertEqual(sampler["model"], [second_id, 0])
        self.assertEqual(sampler["seed"], 0)

    def test_layout_masks(self):
        for layout, start, end in (("horizontal", (0, 2), (31, 2)), ("vertical", (2, 0), (2, 31))):
            a, b = (region_mask((32, 32), layout, first) for first in (True, False))
            self.assertEqual((a.getpixel(start), a.getpixel(end)), (255, 0))
            self.assertEqual((b.getpixel(start), b.getpixel(end)), (0, 255))


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.data = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch.dict(os.environ, {"NOVA_DATA_DIR": self.data}))
        await self.enterAsyncContext(lifespan(app))
        self.client = await self.enterAsyncContext(httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost",
        ))
        self.stub = await self.enterAsyncContext(httpx.AsyncClient(
            transport=httpx.MockTransport(self.comfy), base_url="http://localhost:8188",
        ))
        self.enterContext(patch.object(app.state.comfy.client, "request", self.stub.request))
        self.graph = None
        self.folder = {}

    def comfy(self, request):
        path = request.url.path
        if path.startswith("/object_info/"):
            node = path.rsplit("/", 1)[1]
            name = next(field for kind, field in ASSET_CHOICES.values() if kind == node)
            return httpx.Response(200, json={node: {"input": {"required": {name: [["demo"]]}}}})
        if path == "/upload/image":
            data = request.content
            name = data.split(b'filename="')[1].split(b'"')[0].decode()
            self.folder[name] = data
            return httpx.Response(200, json={"name": name, "subfolder": "nova_stasis"})
        if path == "/prompt":
            import json
            self.graph = json.loads(request.content)["prompt"]
            return httpx.Response(200, json={"prompt_id": "p1"})
        if path == "/history/p1":
            save_id, save = next((key, node) for key, node in self.graph.items()
                                 if node["class_type"] == "SaveImage")
            folder, basename = save["inputs"]["filename_prefix"].split("/")
            output = {"filename": f"{basename}_00001_.png", "subfolder": folder, "type": "output"}
            return httpx.Response(200, json={"p1": {
                "status": {"completed": True, "status_str": "success"},
                "outputs": {save_id: {"images": [output]}},
            }})
        if path == "/view":
            return httpx.Response(200, content=sample_png())
        raise AssertionError(path)

    async def post(self, path, **kwargs):
        return await self.client.post(path, headers={"X-Nova-Request": "1"}, **kwargs)

    async def upload(self, size=(64, 64)):
        response = await self.post("/api/uploads", files={"image": ("reference.png", sample_png(size), "image/png")})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["id"]

    async def test_inpaint_pose_batch_and_library(self):
        self.assertEqual((await self.client.get("/")).status_code, 200)
        self.assertEqual((await self.client.get("/api/models")).json()["adapter"], ["demo"])
        a, b, source, mask, pose = [await self.upload() for _ in range(5)]
        self.assertEqual((await self.client.get(f"/api/uploads/{a}")).content, sample_png())
        profile = await self.post("/api/library/profile",
                            json={"name": "Explorer", "reference": a, "strength": 0.7})
        self.assertEqual(profile.status_code, 201)
        self.assertEqual(len((await self.client.get("/api/library/profile")).json()), 1)
        profile_id = profile.json()["id"]
        deleted = await self.client.delete(f"/api/library/profile/{profile_id}", headers={"X-Nova-Request": "1"})
        self.assertEqual(deleted.status_code, 204)

        settings = {"prompt": "Two adult explorers walking down a trail", "checkpoint": "demo",
                    "adapter": "demo", "vision": "demo", "mode": "inpaint", "source": source,
                    "mask": mask, "pose": pose, "controlnet": "demo", "batch": 2, "seed": 0,
                    "width": 512, "height": 512, "ref_a": a, "ref_b": b, "lora_b": "demo"}
        self.assertEqual((await self.post("/api/library/preset", json={"name": "Trail", "settings": settings})).status_code, 201)
        response = await self.post("/api/generations", json={**settings, "authorized": True})
        self.assertEqual(response.status_code, 202, response.text)
        job = response.json()
        self.assertEqual(job["seed"], 0)
        self.assertEqual(len(self.folder), 7)
        types = [node["class_type"] for node in self.graph.values()]
        self.assertIn("VAEEncodeForInpaint", types)
        self.assertIn("ControlNetApplyAdvanced", types)
        self.assertIn("RepeatLatentBatch", types)
        self.assertIn("LoraLoader", types)
        result = (await self.client.get(f"/api/generations/{job['id']}")).json()
        self.assertEqual(result["status"], "complete")
        output = await self.client.get(f"/api/generations/{job['id']}/images/0")
        self.assertEqual(output.content, sample_png())
        self.assertEqual((await self.client.get(f"/api/generations/{job['id']}/images/-1")).status_code, 404)
        self.assertEqual((await self.client.get("/api/generations")).json()[0]["status"], "complete")

    async def test_safeguards(self):
        self.assertEqual((await self.client.post("/api/uploads")).status_code, 403)
        foreign = await self.client.post("/api/uploads", headers={
            "X-Nova-Request": "1", "Origin": "https://other.example",
        })
        self.assertEqual(foreign.status_code, 403)
        self.assertEqual((await self.client.get("/", headers={"Host": "other.example"})).status_code, 400)
        self.assertEqual((await self.post("/api/uploads", files={"image": ("bad.png", b"not an image")})).status_code, 422)
        settings = {"prompt": "Two adults", "checkpoint": "x", "adapter": "x", "vision": "x",
                    "ref_a": str(uuid4()), "ref_b": str(uuid4())}
        self.assertEqual((await self.post("/api/generations", json=settings)).status_code, 422)
        for prompt in ("Two teens", "An unauthorized sexual deepfake", "Nude adults"):
            response = await self.post("/api/generations", json={**settings, "authorized": True, "prompt": prompt})
            self.assertEqual(response.status_code, 422)
        self.assertIsNone(self.graph)


if __name__ == "__main__":
    unittest.main()
