from __future__ import annotations

from datetime import datetime
from hashlib import blake2b
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


ImageInput = bytes | bytearray | str | Path | Image.Image | np.ndarray
Region = Sequence[float]


class ImageBytes(bytes):
    def __new__(cls, value: bytes | bytearray, source: str = "") -> "ImageBytes":
        obj = super().__new__(cls, bytes(value))
        obj.image_source = str(source or "")
        return obj


def annotate_image_bytes(image: bytes | bytearray, source: str) -> ImageBytes:
    return ImageBytes(image, source)


def describe_image_input(image: ImageInput) -> str:
    if isinstance(image, Image.Image):
        filename = str(getattr(image, "filename", "") or "")
        return f"PIL图片 {filename}" if filename else f"PIL图片 {image.width}x{image.height}"
    if isinstance(image, np.ndarray):
        return f"内存数组 shape={tuple(image.shape)} dtype={image.dtype}"
    if isinstance(image, (bytes, bytearray)):
        source = str(getattr(image, "image_source", "") or "")
        size = len(image)
        return f"{source}（bytes={size}）" if source else f"内存图片（bytes={size}）"
    try:
        return f"文件 {Path(image).resolve()}"
    except Exception:
        return f"文件 {image}"


def _safe_source_segment(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_")[:80] or "image"


def _save_failed_image_bytes(image: bytes | bytearray) -> str:
    error_dir = Path("photo") / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    source = str(getattr(image, "image_source", "") or "memory_image")
    digest = blake2b(bytes(image), digest_size=6).hexdigest()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = (error_dir / f"image_load_failed_{timestamp}_{_safe_source_segment(source)}_{digest}.jpg").resolve()
    path.write_bytes(bytes(image))
    return str(path)


def load_image(image: ImageInput) -> Image.Image:
    try:
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
    except Exception as exc:
        source = describe_image_input(image)
        if isinstance(image, (bytes, bytearray)):
            try:
                source = f"{source}；问题图片已保存={_save_failed_image_bytes(image)}"
            except Exception as save_exc:
                source = f"{source}；问题图片保存失败={save_exc}"
        raise RuntimeError(f"图片读取失败：{source}：{exc}") from exc


def image_size(image: ImageInput) -> tuple[int, int]:
    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]
        return int(width), int(height)
    if isinstance(image, Image.Image):
        return image.size
    return load_image(image).size


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
