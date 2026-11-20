"""Local dashboard API and static web app."""

import asyncio
import os
import re
import secrets
import tempfile
import unicodedata
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .anatomy import AnatomyPrompt, effective_negative
from .civitai import Importer, router as import_router
from .comfy import Comfy, ComfyError
from .demo import DemoComfy, REFERENCES, STARTER_STORY, seed_references
from .hardware import hardware_report
from .lora import KeywordEdit, SuggestionRequest, catalog_entry, clean_terms, suggest_loras
from .media import MAX_UPLOAD, image_path, png_bytes, read_upload, region_mask
from .storage import Storage
from .story import StoryRequest, translate_story
from .workflow import ASSET_CHOICES, MAX_SEED, Settings, build_workflow


WEB = Path(__file__).parent / "web"
# Defense in depth alongside per-run attestation. Keep the dashboard non-explicit.
RESTRICTED = re.compile(
    r"\b(?:minors?|under[ -]?age|children|child|kid(?:s)?|preteens?|pre[ -]?teen|teens?|"
    r"teenagers?|loli|shota|(?:[0-9]|1[0-7])\s*[- ]?\s*year[ -]?olds?|"
    r"nsfw|nudes?|nudity|naked|porn\w*|sexual\w*|erotic\w*|nsfl|bare breasts?|"
    r"deep[ -]?fakes?|non[ -]?consensual|rape)\b", re.IGNORECASE,
)


class Generation(Settings):
    authorized: Literal[True]


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=80)
    reference: UUID
    strength: float = Field(ge=0.05, le=1.5)
    lora: str | None = Field(default=None, max_length=300)
    lora_strength: float = Field(default=0.65, ge=0, le=1.5)


class Preset(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=80)
    settings: Settings


@asynccontextmanager
async def lifespan(app):
    app.state.demo = os.environ.get("NOVA_BROWSER_DEMO") == "1"
    temporary = tempfile.TemporaryDirectory(prefix="nova-browser-demo-") if app.state.demo else None
    try:
        directory = Path(temporary.name if temporary else os.environ.get(
            "NOVA_DATA_DIR", "~/.nova-stasis",
        )).expanduser().resolve()
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        app.state.assets = directory / "assets"
        app.state.assets.mkdir(mode=0o700, exist_ok=True)
        app.state.db = Storage(directory / "nova.sqlite3")
        app.state.comfy = DemoComfy() if app.state.demo else Comfy()
        models = None if app.state.demo else os.environ.get("NOVA_MODELS_DIR")
        app.state.importer = Importer(Path(models).expanduser().resolve() if models else None, app.state.db)
        try:
            if app.state.demo:
                seed_references(app.state.assets)
                app.state.db.put("lora_source", "Lantern Study", {
                    "title": "Lantern Study", "base": "SD 1.5 (demo)", "triggers": ["lantern_study"],
                })
            yield
        finally:
            await app.state.importer.close()
            await app.state.comfy.close()
    finally:
        if temporary:
            temporary.cleanup()


app = FastAPI(title="Nova Stasis", lifespan=lifespan, docs_url=None, redoc_url=None,
              openapi_url=None)
app.include_router(import_router)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])
app.mount("/studio", StaticFiles(directory=WEB), name="studio")


@app.middleware("http")
async def local_requests(request, call_next):
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        origin = request.headers.get("origin")
        if (request.headers.get("x-nova-request") != "1"
                or (origin and origin != f"http://{request.headers.get('host')}")):
            return JSONResponse({"detail": "Use the local Nova Stasis dashboard."}, status_code=403)
        limit = MAX_UPLOAD + 65536 if request.url.path == "/api/uploads" else 65536
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            return JSONResponse({"detail": "Invalid request length."}, status_code=400)
        if request.headers.get("transfer-encoding") or length < 0 or length > limit:
            return JSONResponse({"detail": "Request exceeds the upload limit."}, status_code=413)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' blob:; frame-ancestors 'none'; object-src 'none'"
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(ComfyError)
async def handle_comfy_error(_request, error):
    return JSONResponse({"detail": str(error)}, status_code=502)


@app.get("/")
def dashboard():
    return FileResponse(WEB / "index.html")


def require_image(image_id):
    path = image_path(app.state.assets, image_id)
    if not path.is_file():
        raise HTTPException(404, "A selected image is missing. Upload it again.")
    return path


@app.post("/api/uploads", status_code=201)
async def upload(image: UploadFile = File()):
    raw = await image.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "Upload a PNG, JPEG, or WebP under 16 MiB.")
    try:
        normalized = read_upload(raw)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, "Upload a still PNG, JPEG, or WebP under 20 megapixels.") from exc
    image_id = uuid4()
    normalized.save(image_path(app.state.assets, image_id), format="PNG")
    return {"id": str(image_id)}


@app.get("/api/uploads/{image_id}")
def uploaded_image(image_id: UUID):
    return FileResponse(require_image(image_id), media_type="image/png")


async def node_choices(node, field):
    response = await app.state.comfy.request(f"/object_info/{node}")
    spec = response.json().get(node, {}).get("input", {}).get("required", {}).get(field, [])
    return spec[0] if spec and isinstance(spec[0], list) else []


@app.get("/api/demo")
def demo_info():
    if not app.state.demo:
        return {"enabled": False}
    return {"enabled": True, "references": REFERENCES, "story": STARTER_STORY}


@app.get("/api/hardware")
async def hardware():
    response = await app.state.comfy.request("/system_stats")
    return hardware_report(response.json(), starter=app.state.demo or os.environ.get("NOVA_STARTER_MODELS") == "1")


@app.get("/api/models")
async def model_choices():
    choices = await asyncio.gather(*(node_choices(*entry) for entry in ASSET_CHOICES.values()))
    result = dict(zip(ASSET_CHOICES, choices))
    if os.environ.get("NOVA_STARTER_MODELS") == "1":
        result["starter_checkpoint"] = "v1-5-pruned-emaonly.safetensors"
    return result


@app.get("/api/lora-keywords")
async def lora_keywords():
    installed = await node_choices(*ASSET_CHOICES["lora_a"])
    return [catalog_entry(app.state.db, name) for name in installed]


@app.put("/api/lora-keywords")
async def save_lora_keywords(request: KeywordEdit):
    installed = await node_choices(*ASSET_CHOICES["lora_a"])
    if request.name not in installed:
        raise HTTPException(404, "This LoRA is no longer listed by ComfyUI.")
    app.state.db.put("lora_keywords", request.name, {
        "triggers": clean_terms(request.triggers), "keywords": clean_terms(request.keywords),
    })
    return catalog_entry(app.state.db, request.name)


@app.post("/api/lora-keywords/suggest")
async def lora_suggestions(request: SuggestionRequest):
    installed = await node_choices(*ASSET_CHOICES["lora_a"])
    return suggest_loras(app.state.db, installed, request.prompt, request.selected)


def check_prompt(prompt):
    normalized = unicodedata.normalize("NFKC", prompt)
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Cf")
    if RESTRICTED.search(normalized):
        raise HTTPException(422, "Use non-explicit scenes with authorized adult subjects.")


@app.post("/api/negative-prompt")
def negative_prompt(request: AnatomyPrompt):
    return {"negative": effective_negative(request.prompt, request.negative, request.anatomy)}


@app.post("/api/story-prompts")
def story_prompts(request: StoryRequest):
    result = translate_story(request)
    if not result["scenes"]:
        raise HTTPException(422, "Add narration that describes what the characters are doing or where they are.")
    return result


@app.post("/api/generations", status_code=202)
async def generate(request: Generation):
    check_prompt(request.prompt)
    settings = Settings.model_validate(request.model_dump(exclude={"authorized"}))
    job_id = str(uuid4())
    seed = settings.seed if settings.seed is not None else secrets.randbelow(MAX_SEED + 1)
    names = {}
    content = {}
    # Validate local inputs before forwarding anything to ComfyUI.
    fields = ["ref_a", "ref_b", "region_a", "region_b", "pose"]
    if settings.mode != "text":
        fields.append("source")
    if settings.mode == "inpaint":
        fields.append("mask")
    paths = {field: require_image(getattr(settings, field))
             for field in fields if getattr(settings, field)}
    size = (settings.width, settings.height)
    if "source" in paths:
        with Image.open(paths.pop("source")) as image:
            source_size = image.size
            content["source"] = png_bytes(ImageOps.fit(image, size, Image.Resampling.LANCZOS))
        if "mask" in paths:
            with Image.open(paths.pop("mask")) as image:
                if image.size != source_size:
                    raise HTTPException(422, "The inpaint mask must match the source image dimensions.")
                content["mask"] = png_bytes(ImageOps.fit(image, size, Image.Resampling.NEAREST))
    for field, path in paths.items():
        # Reuse UUID filenames for original uploads.
        names[field] = await app.state.comfy.upload(path.stem, path.read_bytes())
    for field, first in (("region_a", True), ("region_b", False)):
        if field not in names:
            content[field] = png_bytes(region_mask(size, settings.layout, first))
    for field, png in content.items():
        names[field] = await app.state.comfy.upload(f"{job_id}_{field}", png)

    graph = build_workflow(settings, names, seed, job_id)
    response = await app.state.comfy.request("/prompt", json={"prompt": graph})
    prompt_id = response.json().get("prompt_id")
    if not prompt_id:
        raise ComfyError("ComfyUI did not return a queue ID.")
    job = {"id": job_id, "prompt_id": prompt_id, "status": "queued", "seed": seed,
           "created": datetime.now(timezone.utc).isoformat(),
           "settings": settings.model_dump(mode="json"), "images": []}
    app.state.db.put("job", job_id, job)
    return job


def find_job(job_id):
    job = app.state.db.get("job", job_id)
    if not job:
        raise HTTPException(404, "Generation not found.")
    return job


@app.get("/api/generations")
def generations():
    return app.state.db.list("job", 100)


@app.get("/api/generations/{job_id}")
async def generation(job_id: UUID):
    job = find_job(job_id)
    if job["status"] != "queued":
        return job
    prompt_id = job["prompt_id"]
    response = await app.state.comfy.request(f"/history/{prompt_id}")
    item = response.json().get(prompt_id)
    if not item:
        if datetime.now(timezone.utc) - datetime.fromisoformat(job["created"]) < timedelta(hours=24):
            return job
        job.update(status="failed", error="ComfyUI returned no result within 24 hours.")
    else:
        status = item.get("status", {})
        if not status.get("completed") and status.get("status_str") != "error":
            return job
        images = [
            image for node in item.get("outputs", {}).values()
            for image in node.get("images", [])
            if image.get("type") == "output" and image.get("subfolder") == "nova_stasis"
            and image.get("filename", "").startswith(f"{job_id}_")
        ]
        if images and status.get("status_str") != "error":
            job.update(status="complete", images=images)
        else:
            job.update(status="failed", error="ComfyUI did not produce images. Check its terminal for details.")
    app.state.db.put("job", job_id, job)
    return job


@app.get("/api/generations/{job_id}/images/{index}")
async def generated_image(job_id: UUID, index: int):
    images = find_job(job_id)["images"]
    if not 0 <= index < len(images):
        raise HTTPException(404, "Image not found.")
    response = await app.state.comfy.request("/view", params=images[index])
    return Response(response.content, media_type="image/png")


@app.get("/api/library/{kind}")
def library(kind: Literal["profile", "preset"]):
    return app.state.db.list(kind, 100)


@app.post("/api/library/{kind}", status_code=201)
def save_library(kind: Literal["profile", "preset"], payload: dict):
    try:
        entry = (Profile if kind == "profile" else Preset).model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(422, "Check the name and saved settings.") from exc
    if kind == "profile":
        require_image(entry.reference)
    item_id = str(uuid4())
    record = {"id": item_id, **entry.model_dump(mode="json")}
    app.state.db.put(kind, item_id, record)
    return record


@app.delete("/api/library/{kind}/{item_id}", status_code=204)
def delete_library(kind: Literal["profile", "preset"], item_id: UUID):
    app.state.db.delete(kind, item_id)
