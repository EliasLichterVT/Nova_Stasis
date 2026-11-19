"""Build ComfyUI API prompts from validated dashboard settings."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


MAX_SEED = 2**53 - 1


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prompt: str = Field(min_length=1, max_length=3000)
    negative: str = Field(default="", max_length=2000)
    checkpoint: str = Field(min_length=1, max_length=300)
    adapter: str = Field(min_length=1, max_length=300)
    vision: str = Field(min_length=1, max_length=300)
    ref_a: UUID
    ref_b: UUID
    region_a: UUID | None = None
    region_b: UUID | None = None
    layout: Literal["horizontal", "vertical"] = "horizontal"
    strength_a: float = Field(default=0.75, ge=0.05, le=1.5)
    strength_b: float = Field(default=0.75, ge=0.05, le=1.5)
    mode: Literal["text", "image", "inpaint"] = "text"
    source: UUID | None = None
    mask: UUID | None = None
    pose: UUID | None = None
    controlnet: str | None = Field(default=None, max_length=300)
    pose_strength: float = Field(default=0.75, ge=0.05, le=2)
    lora_a: str | None = Field(default=None, max_length=300)
    lora_b: str | None = Field(default=None, max_length=300)
    lora_strength_a: float = Field(default=0.65, ge=0, le=1.5)
    lora_strength_b: float = Field(default=0.65, ge=0, le=1.5)
    width: int = Field(default=1024, ge=256, le=1536, multiple_of=64)
    height: int = Field(default=768, ge=256, le=1536, multiple_of=64)
    denoise: float = Field(default=0.65, gt=0, le=1)
    steps: int = Field(default=28, ge=1, le=80)
    cfg: float = Field(default=6, ge=1, le=20)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    batch: int = Field(default=1, ge=1, le=4)

    @model_validator(mode="after")
    def required_images(self):
        if self.mode != "text" and not self.source:
            raise ValueError("Upload a source image for this mode.")
        if self.mode == "inpaint" and not self.mask:
            raise ValueError("Upload a white-on-black inpaint mask.")
        if bool(self.pose) != bool(self.controlnet):
            raise ValueError("Select both an OpenPose map and an OpenPose ControlNet model.")
        return self


ASSET_CHOICES = {
    "checkpoint": ("CheckpointLoaderSimple", "ckpt_name"),
    "adapter": ("IPAdapterModelLoader", "ipadapter_file"),
    "vision": ("CLIPVisionLoader", "clip_name"),
    "controlnet": ("ControlNetLoader", "control_net_name"),
    "lora_a": ("LoraLoader", "lora_name"),
}


def build_workflow(settings: Settings, images: dict[str, str], seed: int, job_id: str):
    nodes = {}

    def add(kind, **inputs):
        node_id = str(len(nodes) + 1)
        nodes[node_id] = {"class_type": kind, "inputs": inputs}
        return [node_id, 0]

    def load(field):
        return add("LoadImage", image=images[field])

    def mask(field):
        return add("LoadImageMask", image=images[field], channel="red")

    model = add("CheckpointLoaderSimple", ckpt_name=settings.checkpoint)
    clip, vae = [model[0], 1], [model[0], 2]
    for name, strength in ((settings.lora_a, settings.lora_strength_a),
                           (settings.lora_b, settings.lora_strength_b)):
        if name and strength:
            model = add("LoraLoader", model=model, clip=clip, lora_name=name,
                        strength_model=strength, strength_clip=strength)
            clip = [model[0], 1]

    adapter = add("IPAdapterModelLoader", ipadapter_file=settings.adapter)
    vision = add("CLIPVisionLoader", clip_name=settings.vision)
    for slot, weight in (("a", settings.strength_a), ("b", settings.strength_b)):
        model = add(
            "IPAdapterAdvanced", model=model, ipadapter=adapter, clip_vision=vision,
            image=load(f"ref_{slot}"), attn_mask=mask(f"region_{slot}"),
            weight=weight, weight_type="linear", combine_embeds="concat",
            start_at=0.0, end_at=1.0, embeds_scaling="V only",
        )

    positive = add("CLIPTextEncode", clip=clip, text=settings.prompt)
    negative = add("CLIPTextEncode", clip=clip, text=settings.negative)
    if settings.pose:
        control = add("ControlNetLoader", control_net_name=settings.controlnet)
        positive = add("ControlNetApplyAdvanced", positive=positive, negative=negative,
                       control_net=control, image=load("pose"), strength=settings.pose_strength,
                       start_percent=0.0, end_percent=1.0, vae=vae)
        negative = [positive[0], 1]

    if settings.mode == "text":
        latent = add("EmptyLatentImage", width=settings.width, height=settings.height,
                     batch_size=settings.batch)
    else:
        if settings.mode == "inpaint":
            latent = add("VAEEncodeForInpaint", pixels=load("source"), vae=vae,
                         mask=mask("mask"), grow_mask_by=6)
        else:
            latent = add("VAEEncode", pixels=load("source"), vae=vae)
        if settings.batch > 1:
            latent = add("RepeatLatentBatch", samples=latent, amount=settings.batch)

    samples = add("KSampler", model=model, positive=positive, negative=negative,
                  latent_image=latent, seed=seed, steps=settings.steps, cfg=settings.cfg,
                  sampler_name="dpmpp_2m", scheduler="karras",
                  denoise=1.0 if settings.mode == "text" else settings.denoise)
    decoded = add("VAEDecode", samples=samples, vae=vae)
    add("SaveImage", images=decoded, filename_prefix=f"nova_stasis/{job_id}")
    return nodes
