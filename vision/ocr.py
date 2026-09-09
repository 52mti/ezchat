from __future__ import annotations

import contextlib
import json
import threading
from functools import lru_cache
from typing import Any, Iterable, TypedDict

import numpy as np

from .screen import ImageInput, crop_region, load_image, normalize_box

_OCR_NATIVE_CALL_CONDITION = threading.Condition()
_OCR_NATIVE_ACTIVE_CALLS = 0
_OCR_NATIVE_WAITING_CALLS = 0
_OCR_ENGINE_INIT_LOCK = threading.RLock()


class OcrItem(TypedDict):
    text: str
    x: float
    y: float
    w: float
    h: float
    confidence: float | None


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _box_from_points(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left = min(xs)
    top = min(ys)
    return left, top, max(xs) - left, max(ys) - top


def _points_from_raw(points: Any) -> list[list[float]]:
    if isinstance(points, np.ndarray):
        return points.astype(float).tolist()
    normalized: list[list[float]] = []
    if not isinstance(points, Iterable):
        return normalized
    for point in points:
        if isinstance(point, np.ndarray):
            values = point.astype(float).tolist()
        elif isinstance(point, Iterable):
            values = list(point)
        else:
            continue
        if len(values) < 2:
            continue
        x = _to_float(values[0])
        y = _to_float(values[1])
        if x is None or y is None:
            continue
        normalized.append([x, y])
    return normalized


def _ocr_coordinate_size(image_size: tuple[int, int], transform: dict[str, Any] | None) -> tuple[int, int]:
    if transform is None:
        return image_size
    rotation = transform.get("rotation")
    width, height = image_size
    if rotation == "ccw90":
        width, height = height, width
    return max(1, int(round(width))), max(1, int(round(height)))


def _prepare_predict_image(target: Any, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    prepared = target.convert("L")
    scale = max(1.0, float(config.get("scale", 1.0) or 1.0))
    if scale != 1.0:
        from PIL import Image

        prepared = prepared.resize(
            (max(1, int(round(prepared.width * scale))), max(1, int(round(prepared.height * scale)))),
            Image.Resampling.LANCZOS,
        )
    image = np.array(prepared.convert("RGB"))
    transform: dict[str, Any] = {
        "rotation": None,
        "scale": scale,
    }
    if config.get("rotate_to_portrait", True) and target.width > target.height:
        transform["rotation"] = "ccw90"
        return np.rot90(image, 1).copy(), transform
    return image, transform


def _normalize_legacy_result(raw: Any, image_size: tuple[int, int]) -> list[OcrItem]:
    items: list[OcrItem] = []
    lines = raw[0] if raw and isinstance(raw, list) and raw and isinstance(raw[0], list) else raw
    if not lines:
        return items

    for line in lines:
        try:
            points = _points_from_raw(line[0])
            text = str(line[1][0])
            confidence = float(line[1][1]) if line[1][1] is not None else None
            x, y, w, h = _box_from_points(points)
        except (IndexError, TypeError, ValueError):
            continue
        box = normalize_box(x, y, w, h, image_size)
        items.append({"text": text, **box, "confidence": confidence})
    return items


def _normalize_predict_result(
    raw: Any,
    image_size: tuple[int, int],
    transform: dict[str, Any] | None = None,
) -> list[OcrItem]:
    items: list[OcrItem] = []
    results = raw if isinstance(raw, list) else [raw]
    for result in results:
        if not isinstance(result, dict):
            continue
        texts = result.get("rec_texts") or []
        scores = result.get("rec_scores") or []
        polygons = result.get("rec_polys") or result.get("dt_polys") or []
        total = min(len(texts), len(polygons))
        for index in range(total):
            points = _points_from_raw(polygons[index])
            if len(points) < 2:
                continue
            if transform is not None:
                scale = float(transform.get("scale", 1.0) or 1.0)
                if scale != 1.0:
                    points = [[point[0] / scale, point[1] / scale] for point in points]
            x, y, w, h = _box_from_points(points)
            box = normalize_box(x, y, w, h, _ocr_coordinate_size(image_size, transform))
            score = scores[index] if index < len(scores) else None
            items.append(
                {
                    "text": str(texts[index]),
                    **box,
                    "confidence": _to_float(score),
                }
            )
    return items


def _normalize_engine_result(raw: Any, image_size: tuple[int, int], transform: dict[str, Any] | None = None) -> list[OcrItem]:
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return _normalize_predict_result(raw, image_size, transform=transform)
    if isinstance(raw, dict):
        return _normalize_predict_result(raw, image_size, transform=transform)
    return _normalize_legacy_result(raw, image_size)


def _runtime_init_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "lang": config.get("lang", "ch"),
        "use_textline_orientation": bool(config.get("use_angle_cls", True)),
        "use_doc_orientation_classify": bool(config.get("use_doc_orientation_classify", False)),
        "use_doc_unwarping": bool(config.get("use_doc_unwarping", False)),
        "device": "gpu:0" if config.get("use_gpu", False) else "cpu",
    }
    if config.get("text_detection_model_name"):
        kwargs["text_detection_model_name"] = config["text_detection_model_name"]
    if config.get("text_recognition_model_name"):
        kwargs["text_recognition_model_name"] = config["text_recognition_model_name"]
    if config.get("text_detection_model_dir"):
        kwargs["text_detection_model_dir"] = config["text_detection_model_dir"]
    if config.get("text_recognition_model_dir"):
        kwargs["text_recognition_model_dir"] = config["text_recognition_model_dir"]
    det_thresh = _to_float(config.get("det_db_thresh"))
    rec_thresh = _to_float(config.get("rec_thresh"))
    if det_thresh is not None:
        kwargs["text_det_thresh"] = det_thresh
    if rec_thresh is not None:
        kwargs["text_rec_score_thresh"] = rec_thresh
    return kwargs


def _runtime_predict_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "use_doc_orientation_classify": bool(config.get("use_doc_orientation_classify", False)),
        "use_doc_unwarping": bool(config.get("use_doc_unwarping", False)),
        "use_textline_orientation": bool(config.get("use_angle_cls", True)),
    }
    det_thresh = _to_float(config.get("det_db_thresh"))
    rec_thresh = _to_float(config.get("rec_thresh"))
    if det_thresh is not None:
        kwargs["text_det_thresh"] = det_thresh
    if rec_thresh is not None:
        kwargs["text_rec_score_thresh"] = rec_thresh
    return kwargs


def _ocr_max_concurrency(config: dict[str, Any]) -> int:
    for key in ("max_concurrency", "ocr_max_concurrency"):
        if key not in config:
            continue
        try:
            value = int(config.get(key) or 1)
        except (TypeError, ValueError):
            value = 1
        return min(16, max(1, value))
    return 1


@contextlib.contextmanager
def _ocr_native_call_slot(max_concurrency: int):
    global _OCR_NATIVE_ACTIVE_CALLS, _OCR_NATIVE_WAITING_CALLS
    limit = min(16, max(1, int(max_concurrency or 1)))
    with _OCR_NATIVE_CALL_CONDITION:
        _OCR_NATIVE_WAITING_CALLS += 1
        while _OCR_NATIVE_ACTIVE_CALLS >= limit:
            _OCR_NATIVE_CALL_CONDITION.wait()
        _OCR_NATIVE_WAITING_CALLS -= 1
        _OCR_NATIVE_ACTIVE_CALLS += 1
    try:
        yield
    finally:
        with _OCR_NATIVE_CALL_CONDITION:
            _OCR_NATIVE_ACTIVE_CALLS -= 1
            _OCR_NATIVE_CALL_CONDITION.notify_all()


def get_ocr_queue_status() -> dict[str, int]:
    """返回当前进程中 OCR 并发槽位的等待数和执行数。"""
    with _OCR_NATIVE_CALL_CONDITION:
        return {
            "waiting": max(0, int(_OCR_NATIVE_WAITING_CALLS)),
            "active": max(0, int(_OCR_NATIVE_ACTIVE_CALLS)),
        }


@lru_cache(maxsize=8)
def _cached_paddle_engine(cache_key: str) -> Any:
    from paddleocr import PaddleOCR  # type: ignore

    init_kwargs = json.loads(cache_key)
    try:
        return PaddleOCR(**init_kwargs)
    except (TypeError, ValueError):
        return PaddleOCR(lang=init_kwargs.get("lang", "ch"))


def _get_paddle_engine(config: dict[str, Any]) -> Any:
    with _OCR_ENGINE_INIT_LOCK:
        if config.get("backend") or config.get("ocr_backend") or config.get("model_tier") or config.get("modelTier"):
            from .ocr_runtime import get_ocr_runtime_engine

            return get_ocr_runtime_engine(config)
        cache_key = json.dumps(_runtime_init_kwargs(config), sort_keys=True)
        return _cached_paddle_engine(cache_key)


def _predict_with_lock(engine: Any, image: np.ndarray, config: dict[str, Any]) -> Any:
    with _ocr_native_call_slot(_ocr_max_concurrency(config)):
        return engine.predict(image, **_runtime_predict_kwargs(config))


def _legacy_ocr_with_lock(engine: Any, image: Any, config: dict[str, Any]) -> Any:
    with _ocr_native_call_slot(_ocr_max_concurrency(config)):
        return engine.ocr(image, cls=config.get("cls", True))


def ocr(image: ImageInput, config: dict[str, Any] | None = None) -> list[OcrItem]:
    config = config or {}
    region = config.get("region")
    target = crop_region(image, region) if region else load_image(image)

    engine = config.get("engine")
    if engine is not None:
        if hasattr(engine, "predict"):
            predict_image, transform = _prepare_predict_image(target, config)
            raw = _predict_with_lock(engine, predict_image, config)
            return _normalize_engine_result(raw, target.size, transform=transform)
        else:
            raw = _legacy_ocr_with_lock(engine, target, config)
        return _normalize_engine_result(raw, target.size)

    if not config.get("runtime_enabled", False):
        return []

    try:
        paddle = _get_paddle_engine(config)
    except ImportError:
        return []
    predict_image, transform = _prepare_predict_image(target, config)
    raw = _predict_with_lock(paddle, predict_image, config)
    return _normalize_engine_result(raw, target.size, transform=transform)
