from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from apps.cancellation import raise_if_cancelled
from vision.screen import ensure_portrait_image


PHOTO_ROOT = Path("photo")


def _safe_segment(value: str) -> str:
    return value.replace(".", "_").replace(":", "_").replace("\\", "_").replace("/", "_") or "unknown"


def _portrait_bytes(image_bytes: bytes) -> bytes:
    image = ensure_portrait_image(image_bytes)
    output = BytesIO()
    image.save(output, format="JPEG", quality=95)
    return output.getvalue()


def capture_transform_image(
    robot: Any,
    config: dict[str, Any] | None = None,
    *,
    pre_move: bool = True,
    label: str = "capture",
) -> Any:
    raise_if_cancelled(robot, config)
    runtime = config.get("_runtime", {}) if isinstance(config, dict) else {}
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
    try:
        raise_if_cancelled(robot, config)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            image = robot.get_transform_image(pre_move=pre_move, save_path=path)
        else:
            image = robot.get_transform_image(pre_move=pre_move)
    except TypeError:
        raise_if_cancelled(robot, config)
        image = robot.get_transform_image(pre_move=pre_move)
    raise_if_cancelled(robot, config)
    if not isinstance(image, (bytes, bytearray)):
        return image

    portrait = _portrait_bytes(bytes(image))
    if path is not None:
        try:
            path.write_bytes(portrait)
        except OSError:
            return image
    return portrait
