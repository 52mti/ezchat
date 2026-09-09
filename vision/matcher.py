from __future__ import annotations

from typing import Any, Sequence, TypedDict

import cv2
import numpy as np

from .screen import ImageInput, image_size, normalize_box, to_cv2


class MatchResult(TypedDict, total=False):
    x: float
    y: float
    w: float
    h: float
    score: float
    channel: str
    scale: float


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _normalize_gray(image: np.ndarray) -> np.ndarray:
    gray = _gray(image)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _edge(gray: np.ndarray) -> np.ndarray:
    return cv2.Canny(gray, 50, 150)


def _match_gray_arrays(
    source_gray: np.ndarray,
    needle_gray: np.ndarray,
    *,
    threshold: float,
    image_size: tuple[int, int],
    channel: str,
    scale: float = 1.0,
) -> MatchResult | None:
    source_height, source_width = source_gray.shape[:2]
    template_height, template_width = needle_gray.shape[:2]
    if template_width > source_width or template_height > source_height:
        return None

    if float(np.std(needle_gray)) < 1e-6:
        result = cv2.matchTemplate(source_gray, needle_gray, cv2.TM_SQDIFF_NORMED)
        min_val, _, min_loc, _ = cv2.minMaxLoc(result)
        score = 1.0 - float(min_val)
        location = min_loc
    else:
        result = cv2.matchTemplate(source_gray, needle_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        score = float(max_val)
        location = max_loc

    if score < threshold:
        return None

    box = normalize_box(
        float(location[0]),
        float(location[1]),
        float(template_width),
        float(template_height),
        image_size,
    )
    return {**box, "score": score, "channel": channel, "scale": scale}


def match(image: ImageInput, template: ImageInput, threshold: float = 0.85) -> MatchResult | None:
    source = to_cv2(image)
    needle = to_cv2(template)
    return _match_gray_arrays(
        _gray(source),
        _gray(needle),
        threshold=threshold,
        image_size=image_size(image),
        channel="gray",
        scale=1.0,
    )


def _coerce_scales(scales: Sequence[Any] | None) -> tuple[float, ...]:
    if not scales:
        return (1.0,)
    values: list[float] = []
    for value in scales:
        try:
            scale = float(value)
        except (TypeError, ValueError):
            continue
        if 0.2 <= scale <= 3.0 and scale not in values:
            values.append(scale)
    return tuple(values) or (1.0,)


def match_enhanced(
    image: ImageInput,
    template: ImageInput,
    threshold: float = 0.85,
    *,
    scales: Sequence[Any] | None = None,
    channels: Sequence[str] | None = None,
) -> MatchResult | None:
    source = to_cv2(image)
    needle = to_cv2(template)
    image_size_value = image_size(image)
    selected_channels = tuple(channels or ("gray_norm", "gray", "edge"))
    selected_scales = _coerce_scales(scales or (0.92, 0.96, 1.0, 1.04, 1.08))
    source_gray = _gray(source)
    needle_gray = _gray(needle)
    source_norm: np.ndarray | None = None
    needle_norm: np.ndarray | None = None
    source_edge: np.ndarray | None = None
    needle_edge: np.ndarray | None = None
    best: MatchResult | None = None

    for channel in selected_channels:
        if channel == "bgr":
            # Preserve icon color for UI controls whose shape alone is shared
            # by unrelated navigation icons.
            channel_source = source
            channel_needle = needle
        elif channel == "gray":
            channel_source = source_gray
            channel_needle = needle_gray
        elif channel == "gray_norm":
            if source_norm is None:
                source_norm = _normalize_gray(source)
                needle_norm = _normalize_gray(needle)
            channel_source = source_norm
            channel_needle = needle_norm
        elif channel == "edge":
            if source_edge is None:
                base_source = source_norm if source_norm is not None else _normalize_gray(source)
                base_needle = needle_norm if needle_norm is not None else _normalize_gray(needle)
                source_edge = _edge(base_source)
                needle_edge = _edge(base_needle)
            channel_source = source_edge
            channel_needle = needle_edge
            if float(np.std(channel_needle)) < 1e-6:
                continue
        else:
            continue
        if channel_needle is None:
            continue
        original_height, original_width = channel_needle.shape[:2]
        for scale in selected_scales:
            if abs(scale - 1.0) < 1e-6:
                scaled_needle = channel_needle
            else:
                width = max(1, int(round(original_width * scale)))
                height = max(1, int(round(original_height * scale)))
                scaled_needle = cv2.resize(channel_needle, (width, height), interpolation=cv2.INTER_AREA)
            result = _match_gray_arrays(
                channel_source,
                scaled_needle,
                threshold=threshold,
                image_size=image_size_value,
                channel=channel,
                scale=scale,
            )
            if result is None:
                continue
            if best is None or float(result["score"]) > float(best["score"]):
                best = result
    return best
