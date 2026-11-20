"""Run Nova's browser workflow using local fixtures and simulated generation."""

import argparse
import os
import re
import time
from uuid import UUID, uuid4

import httpx
from PIL import Image, ImageDraw, ImageFont

from .comfy import ComfyError
from .media import image_path, png_bytes
from .workflow import ASSET_CHOICES


REFERENCES = [UUID(int=1), UUID(int=2)]
STARTER_STORY = (
    "Mara raised a lantern as Ivo stepped onto the stone bridge. "
    "Blue mist hung above the river and the lantern cast a golden light.\n\n"
    "Ivo opened the gate as Mara held the lantern beside him."
)
DEMO_MODELS = {
    "checkpoint": ["Demo SD 1.5"], "adapter": ["Demo IP-Adapter"], "vision": ["Demo CLIP Vision"],
    "controlnet": ["Demo OpenPose"], "lora_a": ["Lantern Study"],
}
MODEL_NODES = {node: (field, DEMO_MODELS[kind]) for kind, (node, field) in ASSET_CHOICES.items()}


def draw_character(draw, x, y, coat):
    draw.ellipse((x - 17, y, x + 17, y + 34), fill="#e6bc9d")
    draw.polygon([(x - 42, y + 123), (x - 20, y + 42), (x + 23, y + 42),
                  (x + 42, y + 123)], fill=coat)
    draw.rectangle((x - 22, y + 123, x - 6, y + 160), fill="#182033")
    draw.rectangle((x + 6, y + 123, x + 22, y + 160), fill="#182033")


def seed_references(directory):
    for reference, coat in zip(REFERENCES, ("#d58b45", "#648eaf")):
        image = Image.new("RGB", (256, 256), "#31334a")
        draw_character(ImageDraw.Draw(image), 128, 47, coat)
        image.save(image_path(directory, reference), "PNG")


def preview_image(index):
    image = Image.new("RGB", (768, 512), "#202846")
    draw = ImageDraw.Draw(image)
    draw.ellipse((544, 54, 630, 140), fill="#efc285")
    draw.polygon([(0, 327), (190, 220), (384, 302), (545, 205), (768, 326), (768, 420), (0, 420)],
                 fill="#374260")
    draw.rectangle((0, 370, 768, 511), fill="#25334e")
    draw.rounded_rectangle((92, 324, 675, 378), radius=10, fill="#646471")
    draw_character(draw, 294 + index * 8, 165, "#d58b45")
    draw_character(draw, 469 - index * 8, 170, "#648eaf")
    draw.rounded_rectangle((26, 28, 456, 76), radius=8, fill="#171c30")
    draw.text((42, 41), "SIMULATED PREVIEW", fill="#f4e7f5", font=ImageFont.load_default(size=21))
    return png_bytes(image)


class DemoComfy:
    def __init__(self):
        self.jobs = {}

    async def close(self):
        pass

    async def upload(self, name, content):
        return f"nova_stasis/{name}.png"

    async def request(self, path, **kwargs):
        if path == "/system_stats":
            gib = 1024**3
            return httpx.Response(200, json={
                "system": {"ram_total": 16 * gib, "ram_free": 10 * gib},
                "devices": [{"type": "cpu", "name": "Simulated CPU"}],
            })
        if path.startswith("/object_info/"):
            node = path.rsplit("/", 1)[-1]
            field, models = MODEL_NODES[node]
            return httpx.Response(200, json={node: {"input": {"required": {field: [models]}}}})
        if path == "/prompt":
            graph = kwargs["json"]["prompt"]
            output_node, output = next((key, node["inputs"]) for key, node in graph.items()
                                       if node["class_type"] == "SaveImage")
            batch = next((node["inputs"].get("batch_size", node["inputs"].get("amount", 1))
                          for node in graph.values()
                          if node["class_type"] in {"EmptyLatentImage", "RepeatLatentBatch"}), 1)
            prefix = output["filename_prefix"].split("/")[-1]
            prompt_id = str(uuid4())
            self.jobs[prompt_id] = (time.monotonic() + 2, {
                "status": {"completed": True, "status_str": "success"},
                "outputs": {output_node: {"images": [
                    {"filename": f"{prefix}_{index}.png", "subfolder": "nova_stasis", "type": "output"}
                    for index in range(batch)
                ]}},
            })
            if len(self.jobs) > 200:
                del self.jobs[next(iter(self.jobs))]
            return httpx.Response(200, json={"prompt_id": prompt_id})
        if path.startswith("/history/"):
            prompt_id = path.rsplit("/", 1)[-1]
            job = self.jobs.get(prompt_id)
            result = {prompt_id: job[1]} if job and time.monotonic() >= job[0] else {}
            return httpx.Response(200, json=result)
        if path == "/view":
            filename = kwargs["params"]["filename"]
            match = re.fullmatch(r"[a-f0-9-]{36}_([0-3])\.png", filename)
            if match:
                return httpx.Response(200, content=preview_image(int(match[1])))
        raise ComfyError("This operation is unavailable in the browser test session.")


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    options = parser.parse_args()
    os.environ["NOVA_BROWSER_DEMO"] = "1"
    uvicorn.run("nova.app:app", host="127.0.0.1", port=options.port)


if __name__ == "__main__":
    main()
