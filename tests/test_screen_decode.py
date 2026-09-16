"""GIL 友好的图片编解码路径测试（cv2 替换 PIL 的行为等价性）。"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from services.photo_capture import _portrait_bytes
from vision.screen import _decode_bytes_to_bgr, ensure_portrait_image, load_image, to_cv2


def _gradient_jpeg(width: int, height: int, *, value_format: str = "JPEG") -> bytes:
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    array[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    array[:, :, 2] = 128
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format=value_format)
    return buffer.getvalue()


def test_decode_bytes_to_bgr_matches_pil_pixels() -> None:
    data = _gradient_jpeg(64, 48)
    decoded = _decode_bytes_to_bgr(data)
    assert decoded is not None
    assert decoded.shape == (48, 64, 3)
    pil_reference = np.array(Image.open(io.BytesIO(data)).convert("RGB"))[:, :, ::-1]
    # 同一张 JPEG，两种解码器的像素差异应在个位数灰阶内。
    assert np.abs(decoded.astype(int) - pil_reference.astype(int)).max() <= 3


def test_load_image_bytes_returns_rgb_matching_pil() -> None:
    data = _gradient_jpeg(32, 24)
    loaded = load_image(data)
    assert loaded.mode == "RGB"
    reference = Image.open(io.BytesIO(data)).convert("RGB")
    assert np.abs(np.asarray(loaded).astype(int) - np.asarray(reference).astype(int)).max() <= 3


def test_to_cv2_bytes_returns_bgr_array() -> None:
    image = Image.new("RGB", (10, 10), (255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    array = to_cv2(buffer.getvalue())
    assert array.shape == (10, 10, 3)
    # 红色像素在 BGR 顺序下蓝通道为 0、红通道接近 255。
    assert array[5, 5, 0] <= 16
    assert array[5, 5, 2] >= 239


def test_to_cv2_ndarray_passthrough_and_corrupt_bytes_raises() -> None:
    array = np.zeros((4, 6, 3), dtype=np.uint8)
    assert to_cv2(array) is array

    import pytest

    with pytest.raises(RuntimeError):
        load_image(b"not-an-image-at-all")


def test_portrait_bytes_rotates_counter_clockwise_like_pil() -> None:
    # 左半红、右半蓝的横图：逆时针 90° 后红色应转到下半。
    array = np.zeros((40, 80, 3), dtype=np.uint8)
    array[:, :40] = (255, 0, 0)
    array[:, 40:] = (0, 0, 255)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=95)
    data = buffer.getvalue()

    result = _portrait_bytes(data)
    decoded = _decode_bytes_to_bgr(result)
    assert decoded is not None
    assert decoded.shape[0] > decoded.shape[1]  # 已转竖屏
    # 与 PIL 旧路径的方向严格一致。
    pil_portrait = ensure_portrait_image(data)
    assert (decoded.shape[0], decoded.shape[1]) == (pil_portrait.height, pil_portrait.width)
    red_ratio_new = float((decoded[decoded.shape[0] // 2 :, :, 2] > 200).mean())
    pil_array = np.asarray(pil_portrait)
    red_ratio_pil = float((pil_array[pil_array.shape[0] // 2 :, :, 0] > 200).mean())
    assert red_ratio_new > 0.9
    assert abs(red_ratio_new - red_ratio_pil) < 0.05


def test_file_decode_cache_hits_and_invalidates(tmp_path):
    """模板文件按 (路径,mtime,size) 缓存：重复加载零解码、文件替换自动失效。"""
    import numpy as np

    from vision.screen import _decode_file_cached, load_image, to_cv2

    template = tmp_path / "tpl.jpg"
    template.write_bytes(
        __import__("cv2").imencode(
            ".jpg", np.full((40, 60, 3), 200, dtype=np.uint8)
        )[1].tobytes()
    )

    first = _decode_file_cached(str(template))
    assert first is not None
    # 命中缓存：同一只读对象，无重复解码。
    assert _decode_file_cached(str(template)) is first
    assert not first.flags.writeable
    # to_cv2 / load_image 的文件路径同样吃缓存。
    assert to_cv2(template) is first
    assert load_image(template).size == (60, 40)

    # 文件被替换（mtime/size 变化）后缓存失效，重新解码新内容。
    template.write_bytes(
        __import__("cv2").imencode(
            ".jpg", np.full((20, 30, 3), 10, dtype=np.uint8)
        )[1].tobytes()
    )
    second = _decode_file_cached(str(template))
    assert second is not first
    assert second.shape[:2] == (20, 30)
