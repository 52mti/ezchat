from __future__ import annotations

from datetime import datetime
from hashlib import blake2b
from io import BytesIO
import json
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any

from services.coordinate_cache import load_into_runtime

import cv2
import numpy as np
from apps.cancellation import OperationCancelled, cancellable_sleep, raise_if_cancelled
from apps.step_timing import exclude_ocr_wait_from_step_timeout
from vision.screen import annotate_image_bytes, ensure_portrait_image


PHOTO_ROOT = Path("photo")
DEFAULT_STATUS_CACHE_TTL_SECONDS = 0.0
DEFAULT_IMAGE_CHANGE_MEAN_THRESHOLD = 0.8
DEFAULT_IMAGE_CHANGE_PIXEL_THRESHOLD = 5
DEFAULT_IMAGE_CHANGE_RATIO_THRESHOLD = 0.001
DEFAULT_CAPTURE_RETRY_ATTEMPTS = 2
DEFAULT_CAPTURE_RETRY_DELAY_SECONDS = 0.25
DEFAULT_PHONE_HEALTH_CONFIRM_FRAMES = 2
_STATUS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_STATUS_CACHE_LOCK = Lock()
_PROCESSED_IMAGE_CACHE: dict[str, tuple[str, np.ndarray | None, str, bytes]] = {}
_PROCESSED_IMAGE_CACHE_LOCK = Lock()


def _safe_segment(value: str) -> str:
    return value.replace(".", "_").replace(":", "_").replace("\\", "_").replace("/", "_") or "unknown"


def _runtime_context(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    runtime = config.get("_runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        config["_runtime"] = runtime
    return runtime


def _append_runtime_path(runtime: dict[str, Any], key: str, path: str, *, limit: int = 10) -> None:
    values = runtime.get(key)
    if not isinstance(values, list):
        values = []
    path_text = str(path).strip()
    if not path_text:
        return
    values = [item for item in values if str(item) != path_text]
    values.append(path_text)
    runtime[key] = values[-limit:]


def _portrait_bytes(image_bytes: bytes) -> bytes:
    image = ensure_portrait_image(image_bytes)
    output = BytesIO()
    image.save(output, format="JPEG", quality=95)
    return output.getvalue()


def _update_phone_screen_health(image_bytes: bytes, config: dict[str, Any] | None) -> None:
    """根据已取得的竖屏图判断手机画面亮度；只上报状态，不中断自动化。"""
    runtime = _runtime_context(config)
    health_config = config.get("phone_screen_health", {}) if isinstance(config, dict) else {}
    if not isinstance(health_config, dict) or health_config.get("enabled", True) is False:
        return
    color = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if color is None or color.size == 0:
        return
    array = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    mean_luma = float(array.mean())
    std_luma = float(array.std())
    median_luma, p90_luma = (float(value) for value in np.percentile(array, [50, 90]))
    dark_ratio = float((array <= int(health_config.get("dark_luma_threshold", 45))).mean())
    bright_ratio = float((array >= int(health_config.get("bright_luma_threshold", 235))).mean())
    channel_clip_ratio = (color >= int(health_config.get("channel_clip_luma_threshold", 245))).mean(axis=(0, 1))
    clipped_channel_count = int((channel_clip_ratio >= float(health_config.get("channel_clip_ratio_threshold", 0.60))).sum())
    black_ratio = float(health_config.get("black_ratio_threshold", 0.88))
    dark_ratio_threshold = float(health_config.get("dark_ratio_threshold", 0.70))
    bright_ratio_threshold = float(health_config.get("bright_ratio_threshold", 0.78))
    if dark_ratio >= black_ratio:
        candidate = "black_screen"
    elif mean_luma <= float(health_config.get("dark_mean_threshold", 65)) and dark_ratio >= dark_ratio_threshold:
        candidate = "too_dark"
    elif (
        (
            mean_luma >= float(health_config.get("bright_mean_threshold", 205))
            and median_luma >= float(health_config.get("bright_median_threshold", 215))
            and p90_luma >= float(health_config.get("bright_p90_threshold", 230))
            and std_luma <= float(health_config.get("overexposed_max_stddev", 42))
        )
        or clipped_channel_count >= int(health_config.get("overexposed_min_clipped_channels", 2))
    ):
        candidate = "overexposed"
    else:
        candidate = "normal"

    previous = str(runtime.get("phone_screen_health_candidate", "") or "")
    count = int(runtime.get("phone_screen_health_candidate_count", 0) or 0)
    count = count + 1 if candidate == previous else 1
    runtime["phone_screen_health_candidate"] = candidate
    runtime["phone_screen_health_candidate_count"] = count
    confirmed = max(1, int(health_config.get("confirm_frames", DEFAULT_PHONE_HEALTH_CONFIRM_FRAMES) or 1))
    # 状态切换统一需要连续确认。单帧亮度波动不覆盖当前设备状态，避免 UI 闪烁。
    if count >= confirmed:
        runtime["phone_screen_health"] = candidate
    elif "phone_screen_health" not in runtime:
        runtime["phone_screen_health"] = "unknown"
    runtime["phone_screen_health_metrics"] = {
        "mean_luma": round(mean_luma, 1),
        "std_luma": round(std_luma, 1),
        "median_luma": round(median_luma, 1),
        "p90_luma": round(p90_luma, 1),
        "dark_ratio": round(dark_ratio, 4),
        "bright_ratio": round(bright_ratio, 4),
        "clipped_channel_count": clipped_channel_count,
    }


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


def _perspective_status_signature(status: dict[str, Any]) -> str:
    """只比较影响本地透视图的设备参数，用于运行中修改四点后立即失效图片缓存。"""
    data = _status_data(status)
    payload = {
        "transform_width": data.get("transform_width"),
        "transform_height": data.get("transform_height"),
        "points": data.get("points"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sync_coordinate_cache(config: dict[str, Any] | None, signature: str) -> None:
    """透视坐标系未变化时保留坐标缓存；变化时只清理坐标相关缓存。"""
    runtime = _runtime_context(config)
    previous = str(runtime.get("perspective_signature", "") or "")
    if previous and previous != signature:
        cache = config.get("_runtime_cache") if isinstance(config, dict) else None
        if isinstance(cache, dict):
            for section in (
                "template_points",
                "screen_anchors",
                "current_frame_matches",
                "ocr_regions",
                "llm_vision_usage",
            ):
                cache.pop(section, None)
        runtime["perspective_cache_reset"] = True
    else:
        runtime["perspective_cache_reset"] = False
    runtime["perspective_signature"] = signature
    if not runtime.get("perspective_cache_reset"):
        load_into_runtime(config or {})


def _sync_photo_result_cache(config: dict[str, Any] | None, image_bytes: bytes) -> None:
    """照片变化时清理识别结果；坐标缓存不受照片内容变化影响。"""
    runtime = _runtime_context(config)
    signature = blake2b(bytes(image_bytes), digest_size=16).hexdigest()
    previous = str(runtime.get("processed_photo_signature", "") or "")
    if previous and previous != signature:
        cache = config.get("_runtime_cache") if isinstance(config, dict) else None
        if isinstance(cache, dict):
            for section in ("current_frame_matches", "detection_result_reuse", "ocr_result_reuse"):
                cache.pop(section, None)
        runtime["photo_cache_reset"] = True
    else:
        runtime["photo_cache_reset"] = False
    runtime["processed_photo_signature"] = signature


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
    # 截图复用直接比较原始灰度图，避免局部消息变化在缩略图中消失。
    return image


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
    config: dict[str, Any] | None,
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
    # 每次截图读取当前设备透视参数。用户修改机械臂四点或输出尺寸后无需重启程序。
    status = _get_cached_status(robot, ttl_seconds=status_cache_ttl_seconds)
    status_signature = _perspective_status_signature(status)
    _sync_coordinate_cache(config, status_signature)
    key = _robot_cache_key(robot)
    with _PROCESSED_IMAGE_CACHE_LOCK:
        cached = _PROCESSED_IMAGE_CACHE.get(key)
        if cached is not None:
            cached_digest, cached_signature, cached_status_signature, cached_portrait = cached
            same_raw_image = cached_digest == digest or _same_visual_signature(
                signature,
                cached_signature,
                mean_threshold=image_change_mean_threshold,
                pixel_threshold=image_change_pixel_threshold,
                ratio_threshold=image_change_ratio_threshold,
            )
            if same_raw_image and cached_status_signature == status_signature:
                return cached_portrait

    perspective = build_local_perspective_image(raw_bytes, status)
    portrait = _portrait_bytes(perspective)
    _sync_photo_result_cache(config, portrait)
    with _PROCESSED_IMAGE_CACHE_LOCK:
        _PROCESSED_IMAGE_CACHE[key] = (digest, signature, status_signature, portrait)
    return portrait


def capture_transform_image(
    robot: Any,
    config: dict[str, Any] | None = None,
    *,
    pre_move: bool = True,
    label: str = "capture",
) -> Any:
    raise_if_cancelled(robot, config)
    runtime = _runtime_context(config)
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
    retry_attempts = max(0, int(capture_config.get("retry_attempts", DEFAULT_CAPTURE_RETRY_ATTEMPTS) or 0))
    retry_delay_seconds = max(
        0.0,
        float(capture_config.get("retry_delay_seconds", DEFAULT_CAPTURE_RETRY_DELAY_SECONDS) or 0.0),
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
    last_error: Exception | None = None
    for attempt in range(retry_attempts + 1):
        try:
            with exclude_ocr_wait_from_step_timeout(config):
                raise_if_cancelled(robot, config)
                image = _capture_local_perspective_image(
                    robot,
                    config=config,
                    pre_move=pre_move,
                    status_cache_ttl_seconds=status_cache_ttl_seconds,
                    image_change_mean_threshold=image_change_mean_threshold,
                    image_change_pixel_threshold=image_change_pixel_threshold,
                    image_change_ratio_threshold=image_change_ratio_threshold,
                )
            break
        except OperationCancelled:
            raise
        except Exception as exc:
            last_error = exc
            runtime["last_capture_error"] = str(exc)
            runtime["last_capture_error_label"] = str(label)
            runtime["capture_retry_count"] = int(runtime.get("capture_retry_count", 0) or 0) + 1
            if attempt >= retry_attempts:
                raise
            cancellable_sleep(retry_delay_seconds, robot, config)
    else:
        raise RuntimeError(f"截图失败：{last_error}") from last_error
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
    raise_if_cancelled(robot, config)
    if not isinstance(image, (bytes, bytearray)):
        return image
    _update_phone_screen_health(bytes(image), config)
    if path is not None:
        try:
            path.write_bytes(bytes(image))
            runtime["last_capture_path"] = str(path)
            runtime["current_capture_path"] = str(path)
            runtime["last_capture_label"] = str(label)
            _append_runtime_path(runtime, "related_capture_paths", str(path))
        except OSError:
            pass
    source = f"机械臂截图 device={host} app={app_name} label={label}"
    if path is not None:
        source = f"{source} path={path}"
    return annotate_image_bytes(bytes(image), source)
