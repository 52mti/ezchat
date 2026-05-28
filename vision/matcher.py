from __future__ import annotations

from typing import TypedDict

import cv2
import numpy as np

from .screen import ImageInput, load_image, normalize_box, to_cv2


class MatchResult(TypedDict):
    x: float
    y: float
    w: float
    h: float
    score: float


def match(image: ImageInput, template: ImageInput, threshold: float = 0.85) -> MatchResult | None:
    source = to_cv2(image)
    needle = to_cv2(template)
    if source.ndim == 3:
        source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    else:
        source_gray = source
    if needle.ndim == 3:
        needle_gray = cv2.cvtColor(needle, cv2.COLOR_BGR2GRAY)
    else:
        needle_gray = needle

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
        load_image(image).size,
    )
    return {**box, "score": score}
