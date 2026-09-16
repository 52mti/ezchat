from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
import pytest
from PIL import Image

from services.vision_dialogue_client import VisionDialogueClient


def _make_client() -> VisionDialogueClient:
    return VisionDialogueClient(workspace_root=".")


def _make_portrait_image() -> Image.Image:
    gradient = np.tile(np.arange(256, dtype=np.uint8), 48).reshape(48, 256)[:, :48]
    rgb = np.stack([gradient, gradient // 2, 255 - gradient], axis=-1)
    return Image.fromarray(rgb)


def test_image_to_data_uri_cv2_path_matches_jpeg_contract():
    client = _make_client()
    image = _make_portrait_image()
    data_uri = client._image_to_data_uri(image)

    assert data_uri.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(data_uri.split(",", 1)[1])
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (image.height, image.width)


def test_image_to_data_uri_pil_fallback_equivalent(monkeypatch):
    """强制 PIL 回退路径时输出同样是合法 JPEG 且尺寸一致（编码器级等价）。"""
    import sys

    client = _make_client()
    image = _make_portrait_image()

    # sys.modules["cv2"]=None 使函数内 `import cv2` 抛 ImportError → 回退 PIL。
    monkeypatch.setitem(sys.modules, "cv2", None)
    data_uri = client._image_to_data_uri(image)

    assert data_uri.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(data_uri.split(",", 1)[1])
    restored = Image.open(BytesIO(raw))
    assert restored.format == "JPEG"
    assert restored.size == image.size
