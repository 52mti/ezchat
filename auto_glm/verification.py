"""Screenshot-level evidence and action transition verification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VisualTransition:
    changed: bool
    difference_ratio: float
    before_signature: str
    after_signature: str
    message: str


def image_signature(image: Any) -> str:
    """Return a stable content signature for PIL images and NumPy screenshots."""
    array = _image_array(image)
    # Small grayscale representation intentionally ignores minor camera noise.
    reduced = np.asarray(Image.fromarray(array).convert("L").resize((48, 80), Image.Resampling.BILINEAR))
    return hashlib.sha256(reduced.tobytes()).hexdigest()


def compare_images(before: Any, after: Any, *, change_threshold: float = 0.015) -> VisualTransition:
    before_array = _image_array(before)
    after_array = _image_array(after)
    before_signature = image_signature(before_array)
    after_signature = image_signature(after_array)
    if before_signature == after_signature:
        return VisualTransition(False, 0.0, before_signature, after_signature, "截图内容未发生可观察变化")
    if before_array.shape != after_array.shape:
        return VisualTransition(True, 1.0, before_signature, after_signature, "截图尺寸发生变化")
    difference = np.mean(np.abs(before_array.astype(np.int16) - after_array.astype(np.int16))) / 255
    changed = difference >= max(0.0, float(change_threshold))
    return VisualTransition(changed, float(difference), before_signature, after_signature, "截图已发生可观察变化" if changed else "截图仅有轻微变化，暂不认为动作生效")


def verify_transition(
    before: Any,
    capture: Callable[[], Any],
    *,
    change_threshold: float = 0.015,
    stable_samples: int = 2,
) -> VisualTransition:
    """Capture after an action and require the transition to stay observable."""
    required = max(1, int(stable_samples))
    first: VisualTransition | None = None
    for _ in range(required):
        transition = compare_images(before, capture(), change_threshold=change_threshold)
        first = first or transition
        if not transition.changed:
            return transition
    assert first is not None
    return first


def _image_array(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 2:
        return np.stack([array, array, array], axis=-1)
    if array.ndim == 3 and array.shape[2] >= 3:
        return array[:, :, :3]
    raise ValueError("截图必须是 PIL 图片或 RGB/灰度 NumPy 数组")
