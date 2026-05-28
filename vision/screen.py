from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


ImageInput = bytes | bytearray | str | Path | Image.Image | np.ndarray
Region = Sequence[float]


def load_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        if image.shape[2] == 3:
            return Image.fromarray(image[:, :, ::-1]).convert("RGB")
        return Image.fromarray(image).convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(BytesIO(image)).convert("RGB")
    return Image.open(Path(image)).convert("RGB")


def ensure_portrait_image(image: ImageInput) -> Image.Image:
    portrait = load_image(image)
    if portrait.width > portrait.height:
        portrait = portrait.transpose(Image.Transpose.ROTATE_90)
    return portrait


def to_cv2(image: ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    pil_image = load_image(image)
    return np.array(pil_image)[:, :, ::-1].copy()


def crop_region(image: ImageInput, region: Region | None) -> Image.Image:
    pil_image = load_image(image)
    if region is None:
        return pil_image
    if len(region) != 4:
        raise ValueError("region must be [x, y, w, h]")
    x, y, w, h = region
    width, height = pil_image.size
    left = int(max(0.0, min(1.0, x)) * width)
    top = int(max(0.0, min(1.0, y)) * height)
    right = int(max(0.0, min(1.0, x + w)) * width)
    bottom = int(max(0.0, min(1.0, y + h)) * height)
    return pil_image.crop((left, top, right, bottom))


def normalize_box(x: float, y: float, w: float, h: float, image_size: tuple[int, int]) -> dict[str, float]:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    return {
        "x": x / width,
        "y": y / height,
        "w": w / width,
        "h": h / height,
    }


def denormalize_point(x: float, y: float, image_size: tuple[int, int]) -> tuple[int, int]:
    width, height = image_size
    return int(x * width), int(y * height)


def screen_matches(
    image: ImageInput,
    template: ImageInput | None,
    threshold: float = 0.85,
    region: Region | None = None,
) -> bool:
    if template is None:
        return False
    from .matcher import match

    target: Any = crop_region(image, region) if region is not None else image
    return match(target, template, threshold=threshold) is not None
