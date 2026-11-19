from io import BytesIO
from pathlib import Path
from uuid import UUID

from PIL import Image, ImageOps


MAX_UPLOAD = 16 * 1024 * 1024
MAX_PIXELS = 20_000_000


def read_upload(raw: bytes) -> Image.Image:
    with Image.open(BytesIO(raw), formats=["PNG", "JPEG", "WEBP"]) as image:
        if getattr(image, "is_animated", False) or image.width * image.height > MAX_PIXELS:
            raise ValueError("Use a still image under 20 megapixels.")
        clean = ImageOps.exif_transpose(image).convert("RGB")
        clean.thumbnail((4096, 4096))
        clean.info.clear()
        return clean


def png_bytes(image: Image.Image) -> bytes:
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def image_path(directory: Path, image_id: UUID) -> Path:
    return directory / f"{image_id.hex}.png"


def region_mask(size, layout, first):
    width, height = size
    length = width if layout == "horizontal" else height
    # Blend across a narrow seam while keeping the two subjects spatially distinct.
    seam = max(1, round(length * 0.04))
    values = [
        round(255 * min(1, max(0, (length / 2 + seam - pixel) / (2 * seam))))
        for pixel in range(length)
    ]
    line = Image.new("L", (length, 1))
    line.putdata(values if first else [255 - value for value in values])
    if layout == "vertical":
        line = line.transpose(Image.Transpose.TRANSPOSE)
    return line.resize(size, Image.Resampling.NEAREST)
