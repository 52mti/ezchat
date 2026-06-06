from __future__ import annotations

from datetime import datetime
from hashlib import blake2b
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any

import cv2
import numpy as np
from apps.cancellation import raise_if_cancelled
from apps.step_timing import exclude_ocr_wait_from_step_timeout
from vision.screen import ensure_portrait_image


PHOTO_ROOT = Path("photo")
DEFAULT_STATUS_CACHE_TTL_SECONDS = float("inf")
DEFAULT_IMAGE_CHANGE_MEAN_THRESHOLD = 0.8
DEFAULT_IMAGE_CHANGE_PIXEL_THRESHOLD = 5
DEFAULT_IMAGE_CHANGE_RATIO_THRESHOLD = 0.001
_STATUS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_STATUS_CACHE_LOCK = Lock()
_PROCESSED_IMAGE_CACHE: dict[str, tuple[str, np.ndarray | None, bytes]] = {}
_PROCESSED_IMAGE_CACHE_LOCK = Lock()


def _safe_segment(value: str) -> str:
    return value.replace(".", "_").replace(":", "_").replace("\\", "_").replace("/", "_") or "unknown"


def _portrait_bytes(image_bytes: bytes) -> bytes:
    image = ensure_portrait_image(image_bytes)
    output = BytesIO()
    image.save(output, format="JPEG", quality=95)
    return output.getvalue()


def _status_data(status: Any) -> dict[str, Any]:
    if not isinstance(status, dict):
        raise RuntimeError("设备状态不是有效 JSON 对象")
    data = status.get("data", status)
    if not isinstance(data, dict):
        raise RuntimeError("设备状态缺少 data 对象")
    return data


def _robot_cache_key(robot: Any) -> str:
    host = str(getattr(robot, "host", "") or "")
    port = str(getattr(robot, "port", "") or "")
    base_url = str(getattr(robot, "base_url", "") or "")
    if host and (port or base_url):
        return f"{host}:{port}:{base_url}"
    return f"robot:{id(robot)}"


def _get_cached_status(robot: Any, *, ttl_seconds: float) -> dict[str, Any]:
    key = _robot_cache_key(robot)
    now = monotonic()
    if ttl_seconds > 0:
        with _STATUS_CACHE_LOCK:
            cached = _STATUS_CACHE.get(key)
            if cached is not None and now - cached[0] <= ttl_seconds:
                return cached[1]

    status = robot.get_status()
    data = _status_data(status)
    if ttl_seconds > 0:
        with _STATUS_CACHE_LOCK:
            _STATUS_CACHE[key] = (now, data)
    return data


def build_local_perspective_image(raw_bytes: bytes, status: dict[str, Any]) -> bytes:
    data = _status_data(status)
    try:
        width = int(data["transform_width"])
        height = int(data["transform_height"])
        points = data["points"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"设备状态缺少透视参数：{exc}") from exc
    if width <= 0 or height <= 0:
        raise RuntimeError("设备透视宽高必须大于 0")
    if not isinstance(points, list) or len(points) != 4:
        raise RuntimeError("设备透视点位必须是 4 个点")

    try:
        src = np.array([[float(point["x"]), float(point["y"])] for point in points], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"设备透视点位格式错误：{exc}") from exc
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    raw_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    raw_image = cv2.imdecode(raw_array, cv2.IMREAD_COLOR)
    if raw_image is None:
        raise RuntimeError("原始截图解码失败")

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(raw_image, matrix, (width, height))
    if warped is None or warped.size == 0:
        raise RuntimeError("本地透视转换失败")
    ok, encoded = cv2.imencode(".jpg", warped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("本地透视图编码失败")
    return encoded.tobytes()


def _raw_image_digest(raw_bytes: bytes) -> str:
    return blake2b(raw_bytes, digest_size=16).hexdigest()


def _raw_image_signature(raw_bytes: bytes) -> np.ndarray | None:
    raw_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(raw_array, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)


def _same_visual_signature(
    current: np.ndarray | None,
    previous: np.ndarray | None,
    *,
    mean_threshold: float,
    pixel_threshold: int,
    ratio_threshold: float,
) -> bool:
    if current is None or previous is None or current.shape != previous.shape:
        return False
    diff = np.abs(current.astype(np.int16) - previous.astype(np.int16))
    if float(diff.mean()) > mean_threshold:
        return False
    changed_ratio = float((diff > int(pixel_threshold)).mean())
    return changed_ratio <= ratio_threshold


def _capture_local_perspective_image(
    robot: Any,
    *,
    pre_move: bool,
    status_cache_ttl_seconds: float,
    image_change_mean_threshold: float,
    image_change_pixel_threshold: int,
    image_change_ratio_threshold: float,
) -> bytes:
    if (not hasattr(robot, "get_status") or not hasattr(robot, "get_capture_image")) and hasattr(robot, "get_transform_image"):
        transformed = robot.get_transform_image(pre_move=pre_move)
        if isinstance(transformed, (bytes, bytearray)):
            return _portrait_bytes(transformed)
        if isinstance(transformed, (str, Path)) and Path(transformed).exists():
            return _portrait_bytes(transformed)
        return transformed
    if not hasattr(robot, "get_status"):
        raise RuntimeError("机械臂接口缺少 get_status，无法获取透视参数")
    if not hasattr(robot, "get_capture_image"):
        raise RuntimeError("机械臂接口缺少 get_capture_image，无法获取原始截图")
    raw_image = robot.get_capture_image(pre_move=pre_move)
    if not isinstance(raw_image, (bytes, bytearray)):
        raise RuntimeError("原始截图不是 bytes")
    raw_bytes = bytes(raw_image)
    digest = _raw_image_digest(raw_bytes)
    signature = _raw_image_signature(raw_bytes)
    key = _robot_cache_key(robot)
    with _PROCESSED_IMAGE_CACHE_LOCK:
        cached = _PROCESSED_IMAGE_CACHE.get(key)
        if cached is not None:
            cached_digest, cached_signature, cached_portrait = cached
            if cached_digest == digest or _same_visual_signature(
                signature,
                cached_signature,
                mean_threshold=image_change_mean_threshold,
                pixel_threshold=image_change_pixel_threshold,
                ratio_threshold=image_change_ratio_threshold,
            ):
                return cached_portrait

    status = _get_cached_status(robot, ttl_seconds=status_cache_ttl_seconds)
    perspective = build_local_perspective_image(raw_bytes, status)
    portrait = _portrait_bytes(perspective)
    with _PROCESSED_IMAGE_CACHE_LOCK:
        _PROCESSED_IMAGE_CACHE[key] = (digest, signature, portrait)
    return portrait


def capture_transform_image(
    robot: Any,
    config: dict[str, Any] | None = None,
    *,
    pre_move: bool = True,
    label: str = "capture",
) -> Any:
    raise_if_cancelled(robot, config)
    runtime = config.get("_runtime", {}) if isinstance(config, dict) else {}
    capture_config = config.get("capture", {}) if isinstance(config, dict) else {}
    if not isinstance(capture_config, dict):
        capture_config = {}
    raw_status_cache_ttl = capture_config.get(
        "perspective_status_cache_ttl_seconds",
        DEFAULT_STATUS_CACHE_TTL_SECONDS,
    )
    status_cache_ttl_seconds = (
        DEFAULT_STATUS_CACHE_TTL_SECONDS
        if raw_status_cache_ttl in (None, "")
        else float(raw_status_cache_ttl)
    )
    image_change_mean_threshold = float(
        capture_config.get("image_change_mean_threshold", DEFAULT_IMAGE_CHANGE_MEAN_THRESHOLD)
    )
    image_change_pixel_threshold = int(
        capture_config.get("image_change_pixel_threshold", DEFAULT_IMAGE_CHANGE_PIXEL_THRESHOLD)
    )
    image_change_ratio_threshold = float(
        capture_config.get("image_change_ratio_threshold", DEFAULT_IMAGE_CHANGE_RATIO_THRESHOLD)
    )
    host = str(runtime.get("device_ip") or getattr(robot, "host", "unknown"))
    app_name = str(runtime.get("app_name") or "unknown_app")
    save_captures = bool(runtime.get("save_captures", False) or (isinstance(config, dict) and config.get("save_captures", False)))
    path = None
    if save_captures:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = (
            PHOTO_ROOT
            / "captures"
            / _safe_segment(host)
            / _safe_segment(app_name)
            / f"{timestamp}_{_safe_segment(label)}.jpg"
        ).resolve()
    with exclude_ocr_wait_from_step_timeout(config):
        raise_if_cancelled(robot, config)
        image = _capture_local_perspective_image(
            robot,
            pre_move=pre_move,
            status_cache_ttl_seconds=status_cache_ttl_seconds,
            image_change_mean_threshold=image_change_mean_threshold,
            image_change_pixel_threshold=image_change_pixel_threshold,
            image_change_ratio_threshold=image_change_ratio_threshold,
        )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
    raise_if_cancelled(robot, config)
    if path is not None:
        try:
            path.write_bytes(bytes(image))
        except OSError:
            return image
    return image
