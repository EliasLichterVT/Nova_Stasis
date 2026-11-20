"""Inspect Civitai links and install verified model weights for ComfyUI."""

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import stat
import tempfile
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from zipfile import BadZipFile, ZipFile
from urllib.parse import parse_qs, urljoin, urlsplit

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, SecretStr
from safetensors import SafetensorError, safe_open

from .lora import clean_terms


HOSTS = {"civitai.com", "civitai.red", "www.civitai.com", "www.civitai.red"}
DELIVERY = re.compile(r"civitai-delivery-worker-prod\.[a-f0-9]{32}\.r2\.cloudflarestorage\.com")
MAX_BYTES = 32 * 1024**3
CATEGORIES = {
    "checkpoints": "Checkpoint",
    "loras": "LoRA / LyCORIS",
    "vae": "VAE",
    "controlnet": "ControlNet",
    "ipadapter": "IP-Adapter",
    "clip_vision": "CLIP Vision",
    "text_encoders": "Text encoder (CLIP / T5)",
    "embeddings": "Embedding / textual inversion",
    "upscale_models": "Upscaler",
    "diffusion_models": "Diffusion model (UNet)",
}
MODEL_TYPES = {
    "checkpoint": "checkpoints", "lora": "loras", "locon": "loras", "lycoris": "loras",
    "dora": "loras", "lokr": "loras", "vae": "vae", "controlnet": "controlnet",
    "textualinversion": "embeddings", "embedding": "embeddings", "upscaler": "upscale_models",
    "ipadapter": "ipadapter", "clipvision": "clip_vision", "textencoder": "text_encoders",
    "unet": "diffusion_models", "diffusionmodel": "diffusion_models",
}
router = APIRouter(prefix="/api/imports")


class ImportErrorDetail(Exception):
    pass


class Link(BaseModel):
    url: str = Field(min_length=1, max_length=2000)
    api_key: SecretStr | None = Field(default=None, max_length=500, exclude=True)


class Selection(Link):
    version: int = Field(gt=0)
    file: int = Field(gt=0)
    category: str = Field(max_length=40)


def link_target(url):
    try:
        parsed = urlsplit(url.strip())
        if (parsed.scheme != "https" or parsed.hostname not in HOSTS
                or parsed.port not in (None, 443) or parsed.username is not None or parsed.password is not None):
            raise ValueError
        host = parsed.hostname.removeprefix("www.")
        match = re.fullmatch(r"/api/(?:v1/model-versions|download/models)/([1-9][0-9]{0,15})/?", parsed.path)
        if match:
            file_id = parse_qs(parsed.query).get("fileId", [""])[0]
            return host, "version", int(match[1]), file_id
        match = re.fullmatch(r"/models/([1-9][0-9]{0,15})(?:/[^/]*)?/?", parsed.path)
        if match:
            version = parse_qs(parsed.query).get("modelVersionId", [""])[0]
            if version:
                if not re.fullmatch(r"[1-9][0-9]{0,15}", version):
                    raise ValueError
                return host, "version", int(version), ""
            return host, "model", int(match[1]), ""
    except ValueError:
        pass
    raise ImportErrorDetail("Paste an HTTPS Civitai model page or model download link.")


def model_filename(name, extensions=(".safetensors",)):
    if (not isinstance(name, str) or len(name.encode("utf-8")) > 220
            or not name.lower().endswith(extensions) or name.lower() in extensions
            or re.search(r'[/\\<>:"|?*\x00-\x1f]', name)
            or name.startswith((".", " "))
            or re.match(r"(?i)(?:con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])(?:[ .]|$)", name)):
        return None
    return name


def file_options(version, requested_file, model_type):
    options = []
    for entry in version.get("files") or []:
        if not isinstance(entry, dict):
            continue
        name = model_filename(entry.get("name"), (".safetensors", ".zip"))
        file_id = entry.get("id")
        if not name or type(file_id) is not int or file_id <= 0:
            continue
        hashes = entry.get("hashes")
        digest = hashes.get("SHA256") if isinstance(hashes, dict) else None
        if not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
            continue
        category = "vae" if entry.get("type") == "VAE" else MODEL_TYPES.get(
            re.sub(r"[^a-z]", "", model_type.lower()),
        )
        size_kib = entry.get("sizeKB")
        size = round(size_kib * 1024) if isinstance(size_kib, (float, int)) and (
            math.isfinite(size_kib) and 0 < size_kib <= MAX_BYTES / 1024
        ) else None
        options.append({"id": file_id, "name": name, "bytes": size, "category": category,
                        "primary": bool(entry.get("primary")), "sha256": digest.lower()})
    options.sort(key=lambda entry: (str(entry["id"]) != requested_file, not entry["primary"]))
    return options


def validate_weights(path):
    try:
        with safe_open(path, framework="np") as weights:
            if not weights.keys():
                raise ImportErrorDetail("The file contains no tensors.")
    except SafetensorError as exc:
        raise ImportErrorDetail("The downloaded file is not valid Safetensors weights.") from exc


ARCHIVE_FOLDERS = {
    **{name: name for name in CATEGORIES},
    "checkpoint": "checkpoints", "lora": "loras", "lycoris": "loras", "unet": "diffusion_models",
    "clip": "text_encoders", "controlnets": "controlnet", "ip-adapter": "ipadapter",
    "upscalers": "upscale_models",
}


def target_file(directory, category, name):
    target = directory / category / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if directory not in target.parent.resolve().parents:
        raise ImportErrorDetail("The model folder must resolve inside the configured model directory.")
    if target.exists() or target.is_symlink():
        raise ImportErrorDetail(f"A file named {name} is already in the {category} folder.")
    return target


def extract_models(archive, directory, default_category):
    installed = []
    planned = {}
    try:
        with ZipFile(archive) as bundle, tempfile.TemporaryDirectory(dir=directory) as staging:
            entries = bundle.infolist()
            if len(entries) > 1000:
                raise ImportErrorDetail("This ZIP exceeds the 1,000-entry limit.")
            skipped = 0
            total = 0
            for entry in entries:
                parts = entry.filename.split("/")
                if (entry.filename.startswith("/") or "\\" in entry.filename
                        or any(part in {".", ".."} or ":" in part for part in parts)
                        or stat.S_ISLNK(entry.external_attr >> 16)):
                    raise ImportErrorDetail("This ZIP contains an unsafe path or symbolic link.")
                if entry.is_dir():
                    continue
                name = model_filename(parts[-1])
                if not name or "__MACOSX" in parts:
                    skipped += 1
                    continue
                category = next((ARCHIVE_FOLDERS[part.lower()] for part in reversed(parts[:-1])
                                 if part.lower() in ARCHIVE_FOLDERS), default_category)
                if name.lower().endswith(".vae.safetensors"):
                    category = "vae"
                target = target_file(directory, category, name)
                key = (category, name.casefold())
                if key in planned:
                    raise ImportErrorDetail("This ZIP contains duplicate names in the same model folder.")
                total += entry.file_size
                planned[key] = (entry, target, category)
            if not planned:
                raise ImportErrorDetail("No supported Safetensors files were found in the ZIP.")
            if total > min(MAX_BYTES, shutil.disk_usage(directory).free):
                raise ImportErrorDetail("The extracted files exceed available space or the 32 GiB limit.")
            verified = []
            remaining = total
            for index, (entry, target, category) in enumerate(planned.values()):
                partial = Path(staging) / f"{index}.safetensors"
                with bundle.open(entry) as source, partial.open("wb") as output:
                    while chunk := source.read(2 * 1024**2):
                        remaining -= len(chunk)
                        if remaining < 0:
                            raise ImportErrorDetail("The ZIP exceeded its advertised extraction size.")
                        output.write(chunk)
                validate_weights(partial)
                verified.append((partial, target, category))
            for partial, target, category in verified:
                os.link(partial, target)
                installed.append((target, category))
            return installed, skipped
    except (BadZipFile, NotImplementedError, RuntimeError) as exc:
        raise ImportErrorDetail("This ZIP is damaged, encrypted, or uses unsupported compression.") from exc
    finally:
        # A failed package import leaves existing models and the model lists intact.
        if len(installed) and len(installed) != len(planned):
            for target, _ in installed:
                with suppress(OSError):
                    target.unlink()


class Importer:
    def __init__(self, directory, storage=None):
        self.directory = directory
        self.storage = storage
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(45, connect=15), follow_redirects=False)
        self.task = None
        self.job = None

    async def close(self):
        if self.task:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
        await self.client.aclose()

    @asynccontextmanager
    async def response(self, url, token, *, delivery=False):
        original_host = urlsplit(url).hostname
        for _ in range(6):
            try:
                target = urlsplit(url)
                trusted = (target.scheme == "https" and target.port in (None, 443)
                           and target.username is None and target.password is None)
            except ValueError as exc:
                raise ImportErrorDetail("Civitai returned an invalid redirect.") from exc
            is_api = target.hostname in HOSTS and target.path.startswith(
                "/api/download/models/" if delivery else "/api/v1/",
            )
            is_cdn = delivery and (
                target.hostname == "b2.civitai.com" or DELIVERY.fullmatch(target.hostname or "")
            )
            if not trusted or not (is_api or is_cdn):
                raise ImportErrorDetail("Civitai redirected to an unrecognized download host.")
            headers = {"Accept-Encoding": "identity"}
            if is_api and token and target.hostname.removeprefix("www.") == original_host:
                headers["Authorization"] = f"Bearer {token}"
            try:
                async with self.client.stream("GET", url, headers=headers) as response:
                    if response.is_redirect and "location" in response.headers:
                        try:
                            url = urljoin(url, response.headers["location"])
                        except ValueError as exc:
                            raise ImportErrorDetail("Civitai returned an invalid redirect.") from exc
                        continue
                    if response.status_code in (401, 403):
                        raise ImportErrorDetail("Civitai denied access. Check your API key and access to this model.")
                    if response.status_code == 404:
                        raise ImportErrorDetail("Civitai could not find this model or file on the selected site.")
                    if response.status_code == 429:
                        raise ImportErrorDetail("Civitai is rate limiting requests. Try again shortly.")
                    if response.status_code != 200:
                        raise ImportErrorDetail(f"Civitai returned HTTP {response.status_code}.")
                    yield response
                    return
            except httpx.HTTPError as exc:
                raise ImportErrorDetail("The Civitai connection failed. Try again.") from exc
        raise ImportErrorDetail("Civitai returned too many redirects.")

    async def metadata(self, host, kind, identifier, token):
        resource = "model-versions" if kind == "version" else "models"
        async with self.response(f"https://{host}/api/v1/{resource}/{identifier}", token) as response:
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) > 24 * 1024**2:
                    raise ImportErrorDetail("Civitai returned more metadata than Nova can process.")
        try:
            result = json.loads(data)
            if isinstance(result, dict):
                return result
        except (UnicodeDecodeError, ValueError):
            pass
        raise ImportErrorDetail("Civitai did not return model metadata.")

    async def inspect(self, url, token):
        host, kind, identifier, requested_file = link_target(url)
        info = await self.metadata(host, kind, identifier, token)
        if kind == "version":
            model = info.get("model") or {}
            versions = [info]
        else:
            model = info
            versions = info.get("modelVersions") or []
        model_type = model.get("type") or ""
        choices = [
            {"id": item["id"], "name": item.get("name") or "Unnamed version",
             "base": item.get("baseModel") or "", "triggers": clean_terms(item.get("trainedWords")),
             "files": file_options(item, requested_file, model_type)}
            for item in versions if isinstance(item, dict) and type(item.get("id")) is int
        ]
        if not any(version["files"] for version in choices):
            raise ImportErrorDetail("No Safetensors or ZIP files with a SHA-256 checksum were found for this link.")
        return {"host": host, "model": model.get("name") or "Civitai model",
                "versions": [item for item in choices if item["files"]]}

    async def install(self, host, version, selected, category, token, lora_metadata):
        archive = selected["name"].lower().endswith(".zip")
        partial = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            target = None if archive else target_file(self.directory, category, selected["name"])
            download_dir = self.directory if archive else target.parent
            url = f"https://{host}/api/download/models/{version}?fileId={selected['id']}"
            async with self.response(url, token, delivery=True) as response:
                advertised = response.headers.get("content-length", "")
                total = int(advertised) if advertised.isdecimal() else selected["bytes"]
                if total is not None:
                    if not 0 < total <= MAX_BYTES or total > shutil.disk_usage(download_dir).free:
                        raise ImportErrorDetail("The file is too large or the model folder is out of space.")
                    self.job["total"] = total
                digest = hashlib.sha256()
                with tempfile.NamedTemporaryFile(dir=download_dir, suffix=".part", delete=False) as output:
                    partial = Path(output.name)
                    async for chunk in response.aiter_bytes(2 * 1024**2):
                        self.job["received"] += len(chunk)
                        if self.job["received"] > MAX_BYTES:
                            raise ImportErrorDetail("The download exceeded the 32 GiB limit.")
                        digest.update(chunk)
                        output.write(chunk)
                if digest.hexdigest() != selected["sha256"]:
                    raise ImportErrorDetail("The downloaded file failed Civitai’s SHA-256 checksum.")
                if archive:
                    self.job["phase"] = "extracting"
                    extraction = asyncio.create_task(asyncio.to_thread(
                        extract_models, partial, self.directory, category,
                    ))
                    try:
                        installed, skipped = await asyncio.shield(extraction)
                    except asyncio.CancelledError:
                        with suppress(Exception):
                            await extraction
                        raise
                else:
                    await asyncio.to_thread(validate_weights, partial)
                    await asyncio.to_thread(os.link, partial, target)
                    installed, skipped = [(target, category)], 0
            if lora_metadata is not None and self.storage is not None:
                try:
                    for path, folder in installed:
                        if folder == "loras":
                            await asyncio.to_thread(self.storage.put, "lora_source", path.name, lora_metadata)
                except (OSError, ValueError, sqlite3.Error):
                    self.job["warning"] = "Imported the files. Their LoRA keywords could not be saved."
            self.job["installed"] = [f"{folder}/{path.name}" for path, folder in installed]
            self.job["skipped"] = skipped
            self.job.update(state="complete")
        except FileExistsError:
            self.job.update(state="failed", error="A file with that name is already in this model folder.")
        except ImportErrorDetail as exc:
            self.job.update(state="failed", error=str(exc))
        except (OSError, ValueError):
            self.job.update(state="failed", error="The import could not be saved. Check disk space and folder permissions.")
        finally:
            if partial:
                with suppress(OSError):
                    partial.unlink()

    def start(self, host, version, selected, category, token, lora_metadata=None):
        if self.task and not self.task.done():
            raise ImportErrorDetail("Wait for the current import to finish.")
        self.job = {"state": "downloading", "name": selected["name"], "category": category,
                    "received": 0, "total": selected["bytes"]}
        self.task = asyncio.create_task(self.install(host, version, selected, category, token, lora_metadata))
        return self.job


def access(request, link):
    importer = request.app.state.importer
    if importer.directory is None:
        raise HTTPException(409, "Set NOVA_MODELS_DIR to ComfyUI’s models directory or use the Nova launcher.")
    token = link.api_key.get_secret_value().strip() if link.api_key else ""
    token = token or os.environ.get("NOVA_CIVITAI_API_KEY", "")
    if token and (not token.isascii() or not token.isprintable() or any(char.isspace() for char in token)):
        raise HTTPException(422, "Enter a valid Civitai API key.")
    return importer, token


@router.post("/inspect")
async def inspect_link(link: Link, request: Request):
    importer, token = access(request, link)
    try:
        return await importer.inspect(link.url, token)
    except ImportErrorDetail as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("", status_code=202)
async def start_import(selection: Selection, request: Request):
    importer, token = access(request, selection)
    try:
        if selection.category not in CATEGORIES:
            raise ImportErrorDetail("Choose a supported model folder.")
        inspected = await importer.inspect(selection.url, token)
        for version in inspected["versions"]:
            if version["id"] != selection.version:
                continue
            for selected in version["files"]:
                if selected["id"] == selection.file:
                    metadata = None
                    if selection.category == "loras":
                        metadata = {"title": inspected["model"], "base": version["base"],
                                    "triggers": version["triggers"]}
                    return importer.start(inspected["host"], selection.version, selected,
                                          selection.category, token, metadata)
        raise ImportErrorDetail("This file is no longer available for the selected model version.")
    except ImportErrorDetail as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("")
async def import_status(request: Request):
    importer = request.app.state.importer
    return {"enabled": importer.directory is not None, "categories": CATEGORIES, "job": importer.job}
