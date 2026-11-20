import asyncio
import hashlib
import io
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import httpx

from nova.app import app, lifespan
from nova.civitai import ImportErrorDetail, is_delivery_target, link_target, model_filename


CDN = "civitai-delivery-worker-prod.5ac0637cfd0766c97916cefa3764fbdf.r2.cloudflarestorage.com"
LINK = "https://civitai.red/models/41/example"
HEADER = json.dumps({"test": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}).encode()
WEIGHTS = struct.pack("<Q", len(HEADER)) + HEADER + b"\x00"


class LinkTests(unittest.TestCase):
    def test_supported_links(self):
        self.assertEqual(link_target(LINK), ("civitai.red", "model", 41, ""))
        self.assertEqual(link_target(f"{LINK}?modelVersionId=75"), ("civitai.red", "version", 75, ""))
        self.assertEqual(link_target("https://www.civitai.com/api/download/models/75?fileId=15"),
                         ("civitai.com", "version", 75, "15"))

    def test_unsafe_urls_and_names(self):
        for url in ("http://civitai.com/models/1", "https://civitai.red.example.org/models/1",
                    "https://user@civitai.com/models/1", "https://civitai.red:8000/models/1"):
            with self.subTest(url=url), self.assertRaises(ImportErrorDetail):
                link_target(url)
        for name in ("../evil.safetensors", "C:evil.safetensors", "aux.safetensors", "model.ckpt"):
            self.assertIsNone(model_filename(name))


class ImportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.models = self.directory / "models"
        self.requests = []
        self.deliver = WEIGHTS
        self.file_name = "example.safetensors"
        self.expected = WEIGHTS
        self.destination = f"https://{CDN}/signed"
        self.enterContext(patch.dict(os.environ, {
            "NOVA_DATA_DIR": str(self.directory), "NOVA_MODELS_DIR": str(self.models),
        }))
        await self.enterAsyncContext(lifespan(app))
        self.browser = await self.enterAsyncContext(httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost",
            headers={"X-Nova-Request": "1"},
        ))
        remote = await self.enterAsyncContext(httpx.AsyncClient(transport=httpx.MockTransport(self.remote)))
        self.enterContext(patch.object(app.state.importer, "client", remote))

    def remote(self, request):
        self.requests.append((request.url.host, request.headers.get("authorization")))
        if request.url.path == "/api/v1/models/41":
            return httpx.Response(200, json={"name": "Example", "type": "LORA", "modelVersions": [
                {"id": 75, "name": "v1", "baseModel": "SD 1.5",
                 "trainedWords": ["blue lantern", "BLUE LANTERN", "fog_style"], "files": [
                    {"id": 15, "name": self.file_name, "sizeKB": len(self.expected) / 1024,
                     "hashes": {"SHA256": hashlib.sha256(self.expected).hexdigest()}},
                    {"id": 16, "name": "example.ckpt"},
                ]},
            ]})
        if request.url.path == "/api/download/models/75":
            self.assertEqual(request.url.params["fileId"], "15")
            return httpx.Response(307, headers={"Location": self.destination})
        if is_delivery_target(request.url.host, request.url.path):
            return httpx.Response(200, content=self.deliver)
        self.fail(f"Unexpected outgoing request to {request.url.host}")

    async def start(self, **overrides):
        payload = {"url": LINK, "api_key": "testing", "version": 75, "file": 15, "category": "loras"}
        return await self.browser.post("/api/imports", json={**payload, **overrides})

    async def wait(self):
        await asyncio.wait_for(app.state.importer.task, timeout=5)
        return (await self.browser.get("/api/imports")).json()["job"]

    async def test_verified_import_and_no_overwrite(self):
        inspection = await self.browser.post("/api/imports/inspect", json={"url": LINK})
        self.assertEqual(inspection.status_code, 200)
        files = inspection.json()["versions"][0]["files"]
        self.assertEqual([(file["id"], file["category"]) for file in files], [(15, "loras")])
        response = await self.start()
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual((await self.wait())["state"], "complete")
        target = self.models / "loras" / "example.safetensors"
        self.assertEqual(target.read_bytes(), WEIGHTS)
        metadata = app.state.db.get("lora_source", target.name)
        self.assertEqual(metadata["triggers"], ["blue lantern", "fog_style"])
        self.assertEqual(metadata["base"], "SD 1.5")
        self.assertIn(("civitai.red", "Bearer testing"), self.requests)
        self.assertIn((CDN, None), self.requests)
        self.assertEqual((await self.start()).status_code, 202)
        self.assertIn("already", (await self.wait())["error"])
        self.assertEqual(target.read_bytes(), WEIGHTS)

    async def test_bad_digest_and_private_redirect_leave_no_file(self):
        self.deliver = WEIGHTS[:-1] + b"\xff"
        self.assertEqual((await self.start()).status_code, 202)
        self.assertIn("checksum", (await self.wait())["error"])
        self.assertEqual(list((self.models / "loras").iterdir()), [])
        self.destination = "https://127.0.0.1:443/private"
        self.assertEqual((await self.start()).status_code, 202)
        self.assertIn("unrecognized", (await self.wait())["error"])
        self.assertNotIn("127.0.0.1", [host for host, _ in self.requests])

    async def test_legacy_delivery_host_receives_no_api_key(self):
        self.destination = "https://b2.civitai.com/signed"
        self.assertEqual((await self.start()).status_code, 202)
        self.assertEqual((await self.wait())["state"], "complete")
        self.assertIn(("b2.civitai.com", None), self.requests)

    async def test_current_and_historical_r2_hosts_receive_no_api_key(self):
        account = "5ac0637cfd0766c97916cefa3764fbdf.r2.cloudflarestorage.com"
        for url in (
            f"https://civitai-prod-settled.{account}/signed",
            f"https://civitai-delivery-worker-prod-2023-10-01.{account}/signed",
            f"https://{account}/civitai-prod/signed",
            "https://civitai-modelfiles.s3.us-west-004.backblazeb2.com/signed",
        ):
            self.destination = url
            self.assertEqual((await self.start()).status_code, 202)
            self.assertEqual((await self.wait())["state"], "complete", self.destination)
            host = httpx.URL(url).host
            self.assertIn((host, None), self.requests)
            self.assertNotIn((host, "Bearer testing"), self.requests)
            (self.models / "loras" / "example.safetensors").unlink()

    async def test_sign_in_redirect_and_unrelated_bucket_are_not_followed(self):
        for url, message in (
            ("https://auth.civitai.com/login?token=secret", "requires sign-in"),
            ("https://civitai.red/model-versions/75", "Early Access"),
            ("https://civitai-prod-settled.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.r2.cloudflarestorage.com/signed?token=secret",
             "unrecognized download host"),
        ):
            self.destination = url
            self.assertEqual((await self.start()).status_code, 202)
            error = (await self.wait())["error"]
            self.assertIn(message, error)
            self.assertNotIn("secret", error)
            if httpx.URL(url).host != "civitai.red":
                self.assertNotIn(httpx.URL(url).host, [host for host, _ in self.requests])

    def prepare_zip(self, entries):
        data = io.BytesIO()
        with ZipFile(data, "w") as archive:
            for path, content in entries.items():
                archive.writestr(path, content)
        self.file_name = "bundle.zip"
        self.deliver = self.expected = data.getvalue()

    async def test_verified_zip_routes_mixed_weights_and_skips_unrelated_files(self):
        self.prepare_zip({
            "models/checkpoints/scene.safetensors": WEIGHTS,
            "models/loras/character.safetensors": WEIGHTS,
            "color.vae.safetensors": WEIGHTS,
            "fallback.safetensors": WEIGHTS,
            "readme.txt": b"example",
        })
        lookup = await self.browser.post("/api/imports/inspect", json={"url": LINK})
        self.assertEqual(lookup.json()["versions"][0]["files"][0]["name"], "bundle.zip")
        self.assertEqual((await self.start(category="controlnet")).status_code, 202)
        job = await self.wait()
        self.assertEqual(job["state"], "complete", job)
        self.assertEqual(job["skipped"], 1)
        self.assertEqual(job["installed"], [
            "checkpoints/scene.safetensors", "loras/character.safetensors",
            "vae/color.vae.safetensors", "controlnet/fallback.safetensors",
        ])
        for path in job["installed"]:
            self.assertEqual((self.models / path).read_bytes(), WEIGHTS)

    async def test_zip_with_unsafe_paths_or_invalid_weights_publishes_nothing(self):
        for path, contents, problem in (
            ("../escape.safetensors", WEIGHTS, "unsafe"),
            ("loras/broken.safetensors", b"invalid", "not valid"),
        ):
            self.prepare_zip({"loras/valid.safetensors": WEIGHTS, path: contents})
            self.assertEqual((await self.start()).status_code, 202)
            self.assertIn(problem, (await self.wait())["error"])
            self.assertEqual(list(self.models.rglob("*.safetensors")), [])

    async def test_folder_and_file_selection_are_validated(self):
        self.assertEqual((await self.start(category="../../outside")).status_code, 422)
        self.assertEqual((await self.start(file=16)).status_code, 422)
        self.assertIsNone(app.state.importer.job)
