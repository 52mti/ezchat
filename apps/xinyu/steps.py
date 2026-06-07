from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from apps.cancellation import OperationCancelled
from apps.coordinates import click_image_point
from apps.photo_capture import capture_transform_image
from apps.reporting import Reporter, format_point, report_step
from apps.step_timing import exclude_ocr_wait_from_step_timeout, step_deadline
from apps.utils import tts_input
from apps.visual_assets import (
    BACK_BUTTON,
    CHAT_LIST_SCREEN,
    CHAT_WINDOW_SCREEN,
    all_screen_states,
    back_button_template_paths,
    popup_template_region_key,
    popup_template_paths,
    match_template_with_region,
    region_rule,
    screen_state_path,
    TEMPLATE_MATCH_REGIONS_CONFIG_KEY,
    template_match_region,
    template_variant_paths,
    template_path,
    template_root,
)
from vision import match, run_ocr_mode
from vision.screen import load_image

from .config import DEFAULT_CONFIG


class StepUnavailable(RuntimeError):
    pass


class BlackScreenError(RuntimeError):
    pass


class NoPhoneScreenError(RuntimeError):
    pass


BLACK_SCREEN = "screen.black_screen"
NO_PHONE_SCREEN = "screen.no_phone"
MY_SCREEN = "screen.my"
POINTS_REGION = "ocr.regions.points"
MESSAGE_LIST_REGION = "ocr.regions.message_list"
CHAT_AREA_REGION = "ocr.regions.chat_area"
INPUT_BOX_BUTTON = "button.input_box"
MICROPHONE_BUTTON = "button.microphone"
UNREAD_BUTTON = "button.unread"
UNREAD_BUTTON_NO_RED = "button.unread.no_red"
SEND_BUTTON_KEY = "button.send"
NEW_GREETING_BUTTON = "button.new_greeting"
BACK_BUTTON_2 = "button.back.secondary"
TTS_POST_DELAY_SECONDS = 3.0
POPUP_MAX_X = 0.9
EMPTY_LIST_TARGET_NAMES = {
    "\u6682\u65e0\u6d88\u606f",
    "\u6682\u65e0\u4fe1\u606f",
    "\u6ca1\u6709\u6d88\u606f",
    "\u65e0\u6d88\u606f",
}

SCREEN_TEMPLATE_ALIASES: dict[str, tuple[str, ...]] = {
    CHAT_LIST_SCREEN: ("screen_聊天列表.jpg", "chat_list.jpg", "screen_message_list.jpg"),
    CHAT_WINDOW_SCREEN: ("screen_聊天界面.jpg", "chat_window.jpg"),
    BLACK_SCREEN: ("screen_黑屏异常.jpg",),
    NO_PHONE_SCREEN: ("screen_无手机界面.jpg",),
    MY_SCREEN: ("screen_我的界面.jpg", "my_screen.jpg"),
}

BUTTON_TEMPLATE_ALIASES: dict[str, tuple[str, ...]] = {
    BACK_BUTTON: ("btn_返回按钮.jpg", "btn_返回按钮_2.jpg", "btn_back.jpg"),
    UNREAD_BUTTON: ("btn_未读消息按钮.jpg", "btn_next_msg.jpg"),
    UNREAD_BUTTON_NO_RED: ("btn_未读消息-2.jpg",),
    INPUT_BOX_BUTTON: ("btn_输入框.jpg", "btn_input_box.jpg"),
    MICROPHONE_BUTTON: ("btn_麦克风按钮.jpg", "btn_voice_input.jpg"),
    SEND_BUTTON_KEY: ("btn_发送按钮.jpg", "btn_send.jpg"),
    NEW_GREETING_BUTTON: ("btn_新招呼按钮.jpg",),
    BACK_BUTTON_2: ("btn_返回按钮_2.jpg",),
}


def _template_hint(key: str, *, group: str) -> str:
    aliases = SCREEN_TEMPLATE_ALIASES.get(key, ()) if group == "screens" else BUTTON_TEMPLATE_ALIASES.get(key, ())
    if aliases:
        return " / ".join(aliases)
    return key


def _config(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = _deep_merge(DEFAULT_CONFIG, config or {})
    if isinstance(config, dict):
        cache = config.setdefault("_runtime_cache", {})
        if not isinstance(cache, dict):
            cache = {}
            config["_runtime_cache"] = cache
        cfg["_runtime_cache"] = cache
    _disable_dialogue_region_filter(cfg)
    return cfg


def _optimization_enabled(config: dict[str, Any], key: str) -> bool:
    return bool(config.get(key, True))


def _runtime_cache(config: dict[str, Any]) -> dict[str, Any]:
    cache = config.setdefault("_runtime_cache", {})
    if not isinstance(cache, dict):
        cache = {}
        config["_runtime_cache"] = cache
    for section in ("template_points", "screen_anchors", "ocr_regions", "metrics"):
        value = cache.setdefault(section, {})
        if not isinstance(value, dict):
            cache[section] = {}
    return cache


def _cache_get(config: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    if not _optimization_enabled(config, "enable_runtime_coordinate_cache"):
        return default
    value = _runtime_cache(config).get(section, {})
    if not isinstance(value, dict):
        return default
    return value.get(key, default)


def _cache_set(config: dict[str, Any], section: str, key: str, value: Any) -> None:
    if not _optimization_enabled(config, "enable_runtime_coordinate_cache"):
        return
    target = _runtime_cache(config).setdefault(section, {})
    if isinstance(target, dict):
        target[key] = value


def _cache_point_once(config: dict[str, Any], section: str, key: str, value: tuple[float, float]) -> None:
    if _cached_point(config, section, key) is None:
        _cache_set(config, section, key, value)


def _cache_pop(config: dict[str, Any], section: str, key: str) -> None:
    value = _runtime_cache(config).get(section, {})
    if isinstance(value, dict):
        value.pop(key, None)


def _region(config: dict[str, Any], key: str) -> Any:
    current: Any = config
    try:
        path = region_rule(key).config_path
    except KeyError:
        path = tuple(key.split("."))
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _disable_dialogue_region_filter(config: dict[str, Any]) -> None:
    ocr_modes = config.get("ocr_modes")
    if not isinstance(ocr_modes, dict):
        return
    dialogue = ocr_modes.get("dialogue")
    if not isinstance(dialogue, dict):
        return
    rules = dialogue.get("rules")
    if not isinstance(rules, dict):
        return
    basic_filter = rules.get("basic_filter")
    if not isinstance(basic_filter, dict):
        return
    basic_filter["enable_center_y_region_filter"] = False


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _threshold(config: dict[str, Any], key: str | None = None) -> float:
    thresholds = config.get("template_match_thresholds")
    if key and isinstance(thresholds, dict) and key in thresholds:
        return float(thresholds[key])
    return float(config.get("match_threshold", 0.7))


def _popup_threshold(config: dict[str, Any], key: str | None = None) -> float:
    thresholds = config.get("template_match_thresholds")
    if key and isinstance(thresholds, dict) and key in thresholds:
        return float(thresholds[key])
    return float(config.get("popup_match_threshold", 0.8))


def _popup_match(image: Any, popup_template: Path, config: dict[str, Any]) -> dict[str, float] | None:
    result = match_template_with_region(
        image,
        popup_template,
        config,
        popup_template_region_key(popup_template),
        threshold=_popup_threshold(config, popup_template_region_key(popup_template)),
        match_func=match,
    )
    if not result:
        return None
    max_x = float(config.get("popup_max_x", POPUP_MAX_X) or POPUP_MAX_X)
    if float(result.get("x", 0.0)) > max_x:
        return None
    return result


def _step_timeout(config: dict[str, Any]) -> float:
    return float(config.get("step_timeout", 10))


def _run_ocr_mode(image: Any, config: dict[str, Any], mode: str, region: Any = None) -> dict[str, Any]:
    started_at = time.monotonic()
    with exclude_ocr_wait_from_step_timeout(config):
        result = run_ocr_mode(image, config, mode, region=region)
    metrics = _runtime_cache(config).get("metrics", {})
    if isinstance(metrics, dict):
        metrics["ocr_count"] = int(metrics.get("ocr_count", 0) or 0) + 1
        metrics["last_ocr"] = {
            "mode": mode,
            "region": region,
            "seconds": round(time.monotonic() - started_at, 4),
        }
    return result


def _increment_metric(config: dict[str, Any], key: str, amount: int = 1) -> None:
    metrics = _runtime_cache(config).get("metrics", {})
    if isinstance(metrics, dict):
        metrics[key] = int(metrics.get(key, 0) or 0) + amount


def _set_metric(config: dict[str, Any], key: str, value: Any) -> None:
    metrics = _runtime_cache(config).get("metrics", {})
    if isinstance(metrics, dict):
        metrics[key] = value


def _report(reporter: Reporter | None, step: str, label: str, log_label: str | None = None) -> None:
    report_step(reporter, step, label, log_label)


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def _template_candidates(config: dict[str, Any], key: str, *, group: str) -> list[Path]:
    candidates: list[Path] = []
    if group == "buttons" and key in {BACK_BUTTON, BACK_BUTTON_2}:
        candidates.extend(back_button_template_paths(config))
    else:
        try:
            existing = screen_state_path(config, key) if group == "screens" else template_path(config, key)
            if existing is not None:
                candidates.append(existing)
        except KeyError:
            pass
        aliases = SCREEN_TEMPLATE_ALIASES.get(key, ()) if group == "screens" else BUTTON_TEMPLATE_ALIASES.get(key, ())
        root = template_root(config, "screens" if group == "screens" else "buttons")
        for filename in aliases:
            candidates.extend(template_variant_paths(root, filename))
    return _unique_paths(candidates)


def _match_region_key(config: dict[str, Any], key: str) -> str:
    if key != BACK_BUTTON_2:
        return key
    regions = config.get(TEMPLATE_MATCH_REGIONS_CONFIG_KEY)
    if not isinstance(regions, dict):
        return key
    if key in regions:
        return key
    if BACK_BUTTON in regions:
        return BACK_BUTTON
    return key


def _match_any(image: Any, paths: list[Path], config: dict[str, Any], key: str) -> dict[str, Any] | None:
    region_key = _match_region_key(config, key)
    for path in paths:
        result = match_template_with_region(
            image,
            path,
            config,
            region_key,
            threshold=_threshold(config, key),
            match_func=match,
        )
        if result:
            result = dict(result)
            result["_template_path"] = str(path)
            result["_template_name"] = path.name
            return result
    return None


def _template_match(image: Any, config: dict[str, Any], key: str, *, group: str) -> dict[str, Any] | None:
    return _match_any(image, _template_candidates(config, key, group=group), config, key)


def _template_center(image: Any, config: dict[str, Any], key: str, *, group: str) -> tuple[float, float] | None:
    result = _template_match(image, config, key, group=group)
    if not result:
        return None
    return result["x"] + result["w"] / 2, result["y"] + result["h"] / 2


def _cached_point(config: dict[str, Any], section: str, key: str) -> tuple[float, float] | None:
    value = _cache_get(config, section, key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return float(value[0]), float(value[1])


def _capture_image(robot: Any, config: dict[str, Any], *, pre_move: bool = True, label: str = "screen") -> Any:
    image = capture_transform_image(robot, config, pre_move=pre_move, label=label)
    metrics = _runtime_cache(config).get("metrics", {})
    if isinstance(metrics, dict):
        metrics["capture_count"] = int(metrics.get("capture_count", 0) or 0) + 1
        metrics["last_capture_label"] = label
    return image


def _matches_screen(image: Any, config: dict[str, Any], key: str) -> bool:
    return _template_match(image, config, key, group="screens") is not None


def _clean_list_text(value: Any) -> str:
    return str(value or "").strip()


def _is_empty_list_row(row: dict[str, Any]) -> bool:
    name = _clean_list_text(row.get("name"))
    content = _clean_list_text(row.get("content"))
    return not content and name in EMPTY_LIST_TARGET_NAMES


def _list_candidate_score(index: int, row: dict[str, Any]) -> tuple[int, int, dict[str, Any]]:
    favorability = row.get("favorability")
    try:
        favorability_value = int(favorability) if favorability is not None else -1
    except (TypeError, ValueError):
        favorability_value = -1
    return favorability_value, -index, row


def _looks_like_black_screen(image: Any, config: dict[str, Any]) -> bool:
    try:
        pixels = np.array(load_image(image).convert("RGB")).astype(np.float32)
    except Exception:
        return False
    if pixels.size == 0:
        return False
    luma = pixels[:, :, 0] * 0.299 + pixels[:, :, 1] * 0.587 + pixels[:, :, 2] * 0.114
    mean_luma = float(luma.mean())
    dark_ratio = float((luma <= float(config.get("black_screen_dark_luma_threshold", 45))).mean())
    return (
        mean_luma <= float(config.get("black_screen_mean_luma_threshold", 38))
        and dark_ratio >= float(config.get("black_screen_dark_ratio_threshold", 0.88))
    )


def _raise_if_black_screen(image: Any, config: dict[str, Any]) -> None:
    if _matches_screen(image, config, NO_PHONE_SCREEN):
        raise NoPhoneScreenError("检测到无手机界面，请先把手机放回支架")
    if _matches_screen(image, config, BLACK_SCREEN) or _looks_like_black_screen(image, config):
        raise BlackScreenError("检测到黑屏界面，手机可能已熄屏或关机")


def _center_from_bounds(bounds: list[float]) -> tuple[float, float]:
    left, top, right, bottom = bounds
    return (left + right) / 2, (top + bottom) / 2


def _bottom_nav_center(
    items: list[dict[str, Any]],
    keywords: tuple[str, ...],
    *,
    min_center_y: float = 0.0,
) -> tuple[float, float] | None:
    match_item = _bottom_nav_item(items, keywords, min_center_y=min_center_y)
    if match_item is None:
        return None
    _item, center = match_item
    return center


def _bottom_nav_item(
    items: list[dict[str, Any]],
    keywords: tuple[str, ...],
    *,
    min_center_y: float = 0.0,
) -> tuple[dict[str, Any], tuple[float, float]] | None:
    for item in items:
        name = str(item.get("name", "") or "")
        if not any(keyword in name for keyword in keywords):
            continue
        bounds = item.get("bounds")
        if isinstance(bounds, list) and len(bounds) == 4:
            center = _center_from_bounds(bounds)
            if center[1] >= min_center_y:
                return item, center
    return None


def _cache_bottom_nav_from_ocr_result(config: dict[str, Any], ocr_result: dict[str, Any]) -> None:
    if not _optimization_enabled(config, "enable_bottom_nav_anchor_reuse"):
        return
    result = ocr_result.get("result", {})
    bottom_nav = result.get("bottom_nav", []) if isinstance(result, dict) else []
    if not isinstance(bottom_nav, list) or not bottom_nav:
        return
    message = _bottom_nav_item(bottom_nav, ("消息",), min_center_y=0.5)
    if message is not None:
        _cache_point_once(config, "screen_anchors", "bottom_message", message[1])
    my_item = _bottom_nav_item(bottom_nav, ("我的",), min_center_y=0.5)
    if my_item is not None:
        _cache_point_once(config, "screen_anchors", "bottom_my", my_item[1])
    bounds_list = [
        item.get("bounds")
        for item in bottom_nav
        if isinstance(item, dict)
        and isinstance(item.get("bounds"), list)
        and len(item.get("bounds")) == 4
        and _center_from_bounds(item.get("bounds"))[1] >= 0.5
    ]
    if bounds_list:
        top = max(0.0, min(float(bounds[1]) for bounds in bounds_list) - 0.02)
        bottom = min(1.0, max(float(bounds[3]) for bounds in bounds_list) + 0.02)
        if bottom > top:
            _cache_set(config, "ocr_regions", "bottom_nav", [0.0, top, 1.0, bottom - top])


def _valid_region(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    left, top, width, height = (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    if width <= 0 or height <= 0:
        return None
    if left >= 1.0 or top >= 1.0:
        return None
    left = max(0.0, left)
    top = max(0.0, top)
    width = min(1.0 - left, width)
    height = min(1.0 - top, height)
    if width <= 0 or height <= 0:
        return None
    return [left, top, width, height]


def _cache_list_compact_region_from_ocr_result(config: dict[str, Any], ocr_result: dict[str, Any]) -> None:
    if not _optimization_enabled(config, "enable_compact_list_ocr_region"):
        return
    result = ocr_result.get("result", {})
    rows = result.get("message_list", []) if isinstance(result, dict) else []
    if not isinstance(rows, list) or not rows:
        return
    full_region = _valid_region(_region(config, MESSAGE_LIST_REGION)) or [0.0, 0.1, 1.0, 0.9]
    row_bounds = [
        row.get("bounds")
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("bounds"), list) and len(row.get("bounds")) == 4
    ]
    if not row_bounds:
        return
    top_margin = float(config.get("list_compact_top_margin", 0.02) or 0.02)
    bottom_margin = float(config.get("list_compact_bottom_margin", 0.02) or 0.02)
    first_row_top = min(float(bounds[1]) for bounds in row_bounds)
    list_top = max(float(full_region[1]), first_row_top - top_margin)
    bottom_nav_region = _valid_region(_cache_get(config, "ocr_regions", "bottom_nav"))
    full_bottom = float(full_region[1]) + float(full_region[3])
    bottom_nav_top = float(bottom_nav_region[1]) if bottom_nav_region is not None else full_bottom
    list_bottom = min(full_bottom, bottom_nav_top - bottom_margin)
    if list_bottom - list_top < float(config.get("list_compact_min_height", 0.2) or 0.2):
        return
    _cache_set(config, "ocr_regions", "list_compact", [0.0, list_top, 1.0, list_bottom - list_top])


def _list_compact_region(config: dict[str, Any]) -> list[float] | None:
    if not _optimization_enabled(config, "enable_compact_list_ocr_region"):
        return None
    return _valid_region(_cache_get(config, "ocr_regions", "list_compact"))


def _run_list_ocr(
    image: Any,
    config: dict[str, Any],
    *,
    reporter: Reporter | None = None,
    prefer_compact: bool = False,
) -> dict[str, Any]:
    if prefer_compact:
        compact_region = _list_compact_region(config)
        if compact_region is not None:
            _report(reporter, "list_ocr_region", "使用列表主体小范围 OCR")
            compact_result = _run_ocr_mode(image, config, "list", region=compact_region)
            rows = compact_result.get("result", {}).get("message_list", [])
            if isinstance(rows, list) and rows:
                return compact_result
            _report(reporter, "list_ocr_region", "列表主体无有效行，回退完整 OCR")
    _report(reporter, "list_ocr_scan", "列表 OCR 扫描")
    ocr_result = _run_ocr_mode(image, config, "list", region=None)
    _cache_bottom_nav_from_ocr_result(config, ocr_result)
    _cache_list_compact_region_from_ocr_result(config, ocr_result)
    return ocr_result


def _match_result_from_bounds(bounds: list[float]) -> dict[str, float]:
    left, top, right, bottom = bounds
    return {
        "x": float(left),
        "y": float(top),
        "w": max(0.0, float(right) - float(left)),
        "h": max(0.0, float(bottom) - float(top)),
    }


def _red_pixel_stats(image: Any, match_result: dict[str, float]) -> dict[str, Any]:
    try:
        pil_image = load_image(image)
    except Exception:
        return {"has_red": False, "red_pixels": 0, "total_pixels": 0, "region": None}
    width, height = pil_image.size
    pad_x = max(6, int(float(match_result["w"]) * width * 0.35))
    pad_y = max(6, int(float(match_result["h"]) * height * 0.35))
    left = max(0, int(float(match_result["x"]) * width) - pad_x)
    top = max(0, int(float(match_result["y"]) * height) - pad_y)
    right = min(width, int((float(match_result["x"]) + float(match_result["w"])) * width) + pad_x)
    bottom = min(height, int((float(match_result["y"]) + float(match_result["h"])) * height) + pad_y)
    if right <= left or bottom <= top:
        return {"has_red": False, "red_pixels": 0, "total_pixels": 0, "region": [left, top, right, bottom]}
    pixels = np.array(pil_image.crop((left, top, right, bottom)).convert("RGB"))
    red = pixels[:, :, 0].astype(np.int16)
    green = pixels[:, :, 1].astype(np.int16)
    blue = pixels[:, :, 2].astype(np.int16)
    strong_red = (red >= 150) & (red - green >= 55) & (red - blue >= 55)
    pink_red = (red >= 80) & (red - green >= 25) & (blue - green >= 10) & (red >= blue - 25)
    muted_red = (red >= 80) & (red - green >= 8) & (red - blue >= 4)
    red_mask = strong_red | pink_red | muted_red
    red_pixels = int(red_mask.sum())
    total_pixels = int(red_mask.size)
    threshold = max(3, int(total_pixels * 0.002))
    return {
        "has_red": red_pixels >= threshold,
        "red_pixels": red_pixels,
        "total_pixels": total_pixels,
        "threshold": threshold,
        "region": [left, top, right, bottom],
    }


def _has_red_pixels(image: Any, match_result: dict[str, float]) -> bool:
    return bool(_red_pixel_stats(image, match_result).get("has_red"))


def _cache_unread_tab_band(config: dict[str, Any], bounds: list[float]) -> None:
    if not _optimization_enabled(config, "enable_unread_tab_band_cache"):
        return
    if not isinstance(bounds, list) or len(bounds) != 4:
        return
    top = float(bounds[1])
    bottom = float(bounds[3])
    height = max(0.0, bottom - top)
    if height <= 0:
        return
    pad = max(0.01, height * float(config.get("unread_tab_band_vertical_scale", 0.6) or 0.6))
    band_top = max(0.0, top - pad)
    band_bottom = min(1.0, bottom + pad)
    if band_bottom > band_top:
        _cache_set(config, "ocr_regions", "unread_tab_band", [0.0, band_top, 1.0, band_bottom - band_top])


def _chat_list_title_region(config: dict[str, Any]) -> tuple[float, float, float, float] | None:
    region = template_match_region(config, CHAT_LIST_SCREEN)
    if region is None:
        return None
    if not isinstance(region, (list, tuple)) or len(region) != 4:
        return None
    return float(region[0]), float(region[1]), float(region[2]), float(region[3])


def _chat_list_title_ocr_matches(ocr_result: dict[str, Any]) -> bool:
    texts: list[str] = []
    for item in ocr_result.get("ocr_items", []):
        if not isinstance(item, dict):
            continue
        text = re.sub(r"\s+", "", str(item.get("text") or item.get("raw_text") or "")).strip()
        if text:
            texts.append(text)
    if not texts:
        return False
    if any("消息通话" in text for text in texts):
        return True
    combined = "".join(texts)
    return "消息通话" in combined or ("消息" in combined and "通话" in combined)


def _matches_chat_list_screen(
    image: Any,
    config: dict[str, Any],
    reporter: Reporter | None = None,
) -> bool:
    if _matches_screen(image, config, CHAT_LIST_SCREEN):
        return True
    region = _chat_list_title_region(config)
    if region is None:
        return False
    _report(reporter, "chat_list_title_ocr", "聊天列表标题 OCR")
    ocr_result = _run_ocr_mode(image, config, "list", region=region)
    if _chat_list_title_ocr_matches(ocr_result):
        _report(reporter, "chat_list_title_ocr", "识别到消息通话，判定聊天列表")
        return True
    return False


def current_screen(image: Any, config: dict[str, Any]) -> str:
    for state in all_screen_states(config):
        if match_template_with_region(
            image,
            state.path,
            config,
            state.name,
            threshold=_threshold(config, state.name),
            match_func=match,
        ):
            return state.name
    if _matches_chat_list_screen(image, config):
        return CHAT_LIST_SCREEN
    if _matches_screen(image, config, CHAT_WINDOW_SCREEN):
        return CHAT_WINDOW_SCREEN
    return "unknown"


def _is_chat_window_image(image: Any, config: dict[str, Any]) -> bool:
    if _matches_screen(image, config, CHAT_WINDOW_SCREEN):
        return True
    return False


def _popup_close_point(image: Any, config: dict[str, Any]) -> tuple[float, float] | None:
    for popup_template in popup_template_paths(config):
        result = _popup_match(image, popup_template, config)
        if not result:
            continue
        return min(0.98, result["x"] + result["w"] * 0.9), max(0.02, result["y"] + result["h"] * 0.1)
    return None


def is_popup_visible(image: Any, config: dict[str, Any]) -> bool:
    if _is_chat_window_image(image, config):
        return False
    return _popup_close_point(image, config) is not None


def dismiss_popups(
    robot: Any,
    config: dict[str, Any],
    image: Any | None = None,
    reporter: Reporter | None = None,
) -> bool:
    current_image = image if image is not None else _capture_image(
        robot, config, pre_move=True, label="popup_check"
    )
    if _is_chat_window_image(current_image, config):
        return False
    close_point = _popup_close_point(current_image, config)
    if close_point is None:
        return False
    _report(reporter, "popup_check", "关闭弹窗", log_label=f"关闭弹窗：{format_point(*close_point)}")
    try:
        click_image_point(robot, *close_point)
    except Exception:
        _report(reporter, "popup_check", "弹窗点击失败，忽略")
        return False
    return True


def capture_after_popups(
    robot: Any,
    config: dict[str, Any],
    *,
    label: str = "screen",
    max_dismiss: int = 2,
    image: Any | None = None,
    reporter: Reporter | None = None,
) -> Any:
    image = image if image is not None else _capture_image(robot, config, pre_move=True, label=label)
    for _ in range(max_dismiss):
        if _is_chat_window_image(image, config):
            break
        close_point = _popup_close_point(image, config)
        if close_point is None:
            break
        _report(reporter, "popup_check", "关闭弹窗", log_label=f"关闭弹窗：{format_point(*close_point)}")
        try:
            click_image_point(robot, *close_point)
        except Exception:
            _report(reporter, "popup_check", "弹窗点击失败，忽略")
            break
        image = _capture_image(robot, config, pre_move=True, label=f"{label}_after_popup")
    return image


def _click_back_once(
    robot: Any,
    config: dict[str, Any],
    image: Any | None = None,
    reporter: Reporter | None = None,
) -> bool:
    current_image = image if image is not None else capture_after_popups(robot, config, label="back_check", reporter=reporter)
    cached = _cached_point(config, "template_points", "back")
    if cached is not None:
        _report(reporter, "return_from_chat", "点击返回", log_label=f"点击返回（缓存）：{format_point(*cached)}")
        click_image_point(robot, *cached)
        return True
    pos = _template_center(current_image, config, BACK_BUTTON, group="buttons")
    if pos is None:
        _report(reporter, "return_from_chat", "返回按钮未找到")
        return False
    click_point = pos
    _cache_point_once(config, "template_points", "back", click_point)
    _report(reporter, "return_from_chat", "点击返回", log_label=f"点击返回：{format_point(*click_point)}")
    click_image_point(robot, *click_point)
    return True


def _click_message_button(
    robot: Any,
    config: dict[str, Any],
    image: Any | None = None,
    reporter: Reporter | None = None,
) -> bool:
    pos = (
        _cached_point(config, "screen_anchors", "bottom_message")
        if _optimization_enabled(config, "enable_bottom_nav_anchor_reuse")
        else None
    )
    if pos is not None:
        _report(reporter, "message_button_click", "点击消息按钮", log_label=f"点击消息按钮（缓存）：{format_point(*pos)}")
        click_image_point(robot, *pos)
        return True
    _report(reporter, "message_button_click", "消息按钮缓存不存在")
    return False


def ensure_list_screen(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
    image: Any | None = None,
) -> Any | None:
    cfg = _config(config)
    _report(reporter, "confirm_xinyu_list", "确认心遇聊天列表")
    initial_image = image if image is not None else _capture_image(robot, cfg, pre_move=True, label="startup_screen_detect")
    _raise_if_black_screen(initial_image, cfg)
    if _matches_screen(initial_image, cfg, CHAT_WINDOW_SCREEN):
        if not _click_back_once(robot, cfg, initial_image, reporter=reporter):
            raise StepUnavailable(f"当前在聊天界面，但未找到返回按钮模板：{_template_hint(BACK_BUTTON, group='buttons')}")
        time.sleep(0.4)
        image = capture_after_popups(robot, cfg, label="after_return_from_chat", reporter=reporter)
        _raise_if_black_screen(image, cfg)
        if _matches_chat_list_screen(image, cfg, reporter=reporter):
            return image
        raise StepUnavailable(f"已点击返回按钮，但未回到聊天列表，未匹配到模板：{_template_hint(CHAT_LIST_SCREEN, group='screens')}")
    elif _matches_screen(initial_image, cfg, MY_SCREEN):
        if not _click_message_button(robot, cfg, initial_image, reporter=reporter):
            raise StepUnavailable("当前在“我的”界面，但没有缓存的底部“消息”坐标")
        time.sleep(0.4)
        image = capture_after_popups(robot, cfg, label="after_message_button", reporter=reporter)
        _raise_if_black_screen(image, cfg)
        if _matches_chat_list_screen(image, cfg, reporter=reporter):
            return image
        raise StepUnavailable(f"已点击消息按钮，但未回到聊天列表，未匹配到模板：{_template_hint(CHAT_LIST_SCREEN, group='screens')}")
    elif _matches_chat_list_screen(initial_image, cfg, reporter=reporter):
        image = capture_after_popups(robot, cfg, label="confirm_list", reporter=reporter)
        _raise_if_black_screen(image, cfg)
        if _matches_chat_list_screen(image, cfg, reporter=reporter):
            return image
        raise StepUnavailable(f"当前界面疑似聊天列表，但复查后未匹配到模板：{_template_hint(CHAT_LIST_SCREEN, group='screens')}")
    else:
        if not _click_back_once(robot, cfg, initial_image, reporter=reporter):
            raise StepUnavailable(
                f"当前界面无法识别，且未找到返回按钮模板：{_template_hint(BACK_BUTTON, group='buttons')}"
            )
        time.sleep(0.4)
        image = capture_after_popups(robot, cfg, label="after_unknown_back", reporter=reporter)
        _raise_if_black_screen(image, cfg)
        if _matches_chat_list_screen(image, cfg, reporter=reporter):
            return image
        raise StepUnavailable(f"已尝试点击返回，但未回到聊天列表，未匹配到模板：{_template_hint(CHAT_LIST_SCREEN, group='screens')}")


def check_unread_button(
    robot: Any,
    config: dict[str, Any] | None = None,
    image: Any | None = None,
    reporter: Reporter | None = None,
    ocr_result: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    cfg = _config(config)
    current_image = image if image is not None else capture_after_popups(robot, cfg, label="unread_button", reporter=reporter)
    if ocr_result is None:
        ocr_result = _run_list_ocr(current_image, cfg, reporter=reporter)
    bottom_nav = ocr_result["result"].get("bottom_nav", [])
    unread_item = _bottom_nav_item(bottom_nav, ("未回复",))
    if unread_item is None:
        return current_image, ocr_result

    nav_item, unread_center = unread_item
    bounds = nav_item.get("bounds")
    if not isinstance(bounds, list) or len(bounds) != 4:
        return current_image, ocr_result
    _cache_unread_tab_band(cfg, bounds)
    red_stats = _red_pixel_stats(current_image, _match_result_from_bounds(bounds))
    has_red = bool(red_stats.get("has_red"))
    red_detail = (
        f"红色像素={red_stats.get('red_pixels', 0)}/{red_stats.get('total_pixels', 0)}"
        f"，阈值={red_stats.get('threshold', 0)}，区域={red_stats.get('region')}"
    )
    _report(
        reporter,
        "unread_button_check",
        "未回复入口已有红色" if has_red else "未回复入口无红色，点击切换",
        log_label=(
            f"未回复入口已有红色：{format_point(*unread_center)}，{red_detail}"
            if has_red
            else f"未回复入口无红色，点击切换：{format_point(*unread_center)}，{red_detail}"
        ),
    )
    if has_red:
        return current_image, ocr_result
    click_image_point(robot, *unread_center)
    time.sleep(0.4)
    current_image = capture_after_popups(robot, cfg, label="after_unread_button", reporter=reporter)
    ocr_result = _run_list_ocr(current_image, cfg, reporter=reporter, prefer_compact=True)
    return current_image, ocr_result


def find_unread_target(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
    image: Any | None = None,
) -> dict[str, Any] | None:
    cfg = _config(config)
    if image is None:
        image = ensure_list_screen(robot, cfg, reporter=reporter)
    else:
        image = ensure_list_screen(robot, cfg, reporter=reporter, image=image)
    if image is None:
        return None
    ocr_result = _run_list_ocr(image, cfg, reporter=reporter)
    image, ocr_result = check_unread_button(robot, cfg, image, reporter=reporter, ocr_result=ocr_result)
    rows = ocr_result["result"].get("message_list", [])
    unread_candidates: list[tuple[int, int, dict[str, Any]]] = []
    content_candidates: list[tuple[int, int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        if _is_empty_list_row(row):
            continue
        try:
            unread_count = int(row.get("unread_count") or 0)
        except (TypeError, ValueError):
            unread_count = 0
        candidate = _list_candidate_score(index, row)
        if unread_count >= 1:
            unread_candidates.append(candidate)
            continue
        if _clean_list_text(row.get("content")):
            content_candidates.append(candidate)
    candidates = unread_candidates or content_candidates
    if not candidates:
        return None
    _favorability, _position, selected_row = max(candidates, key=lambda item: (item[0], item[1]))
    left, top, right, bottom = selected_row["bounds"]
    center_x, center_y = _center_from_bounds([left, top, right, bottom])
    return {
        "name": selected_row.get("name", ""),
        "content": selected_row.get("content", ""),
        "favorability": selected_row.get("favorability"),
        "unread_count": int(selected_row.get("unread_count") or 0),
        "bounds": selected_row.get("bounds", [left, top, right, bottom]),
        "x": center_x,
        "y": center_y,
    }


def find_unread(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> tuple[float, float] | None:
    target = find_unread_target(robot, config, reporter=reporter)
    if target is None:
        return None
    return float(target["x"]), float(target["y"])


def keep_screen_awake(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> bool:
    cfg = _config(config)
    image = ensure_list_screen(robot, cfg, reporter=reporter)
    if image is None:
        return False

    cached_message = (
        _cached_point(cfg, "screen_anchors", "bottom_message")
        if _optimization_enabled(cfg, "enable_keep_awake_message_anchor")
        else None
    )
    if cached_message is not None and cached_message[1] >= 0.5:
        _report(
            reporter,
            "keep_awake_jump_message",
            "防熄屏点击消息",
            log_label=f"防熄屏点击消息（缓存）：{format_point(*cached_message)}",
        )
        click_image_point(robot, *cached_message)
        time.sleep(0.6)
        return True

    _report(reporter, "keep_awake_scan", "防熄屏扫描底部导航")
    ocr_result = _run_ocr_mode(image, cfg, "list", region=None)
    _cache_bottom_nav_from_ocr_result(cfg, ocr_result)
    bottom_nav = ocr_result["result"].get("bottom_nav", [])
    message = _bottom_nav_center(bottom_nav, ("消息",), min_center_y=0.5)
    if message is None:
        return False

    _report(
        reporter,
        "keep_awake_jump_message",
        "防熄屏点击消息",
        log_label=f"防熄屏点击消息：{format_point(*message)}",
    )
    click_image_point(robot, *message)
    time.sleep(0.6)
    return True


def read_my_points(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> str | None:
    cfg = _config(config)
    image = ensure_list_screen(robot, cfg, reporter=reporter)
    if image is None:
        return None

    cached_my = (
        _cached_point(cfg, "screen_anchors", "bottom_my")
        if _optimization_enabled(cfg, "enable_bottom_nav_anchor_reuse")
        else None
    )
    pos = cached_my
    if pos is None:
        _report(reporter, "points_scan_bottom_nav", "积分读取前扫描底部导航")
        ocr_result = _run_ocr_mode(image, cfg, "list", region=None)
        _cache_bottom_nav_from_ocr_result(cfg, ocr_result)
        bottom_nav = ocr_result.get("result", {}).get("bottom_nav", [])
        pos = _bottom_nav_center(bottom_nav if isinstance(bottom_nav, list) else [], ("我的",), min_center_y=0.5)
    if pos is None:
        _report(reporter, "points_skip_no_my", "底部导航未识别到我的，跳过积分读取")
        return None
    if _cached_point(cfg, "screen_anchors", "bottom_message") is None:
        _report(reporter, "points_skip_no_message", "底部导航未识别到消息，跳过积分读取")
        return None
    _cache_point_once(cfg, "screen_anchors", "bottom_my", pos)
    _report(reporter, "points_open_my", "打开我的界面", log_label=f"打开我的界面：{format_point(*pos)}")
    click_image_point(robot, *pos)
    time.sleep(0.6)

    image = capture_after_popups(robot, cfg, label="my_screen", reporter=reporter)
    _raise_if_black_screen(image, cfg)
    if not _matches_screen(image, cfg, MY_SCREEN):
        raise StepUnavailable(f"未匹配到“我的界面”模板：{_template_hint(MY_SCREEN, group='screens')}")

    _report(reporter, "points_ocr_read", "积分 OCR 读取")
    ocr_result = _run_ocr_mode(image, cfg, "points", region=_region(cfg, POINTS_REGION))
    result = ocr_result.get("result", {}) if isinstance(ocr_result.get("result", {}), dict) else {}
    points = str(result.get("points") or result.get("balance") or "").strip()
    _report(
        reporter,
        "points_ocr_read",
        "积分读取完成" if points else "积分未识别",
        log_label=f"积分读取完成：{points}" if points else "积分未识别",
    )

    if not _click_message_button(robot, cfg, image, reporter=reporter):
        raise StepUnavailable("读取积分后没有缓存的底部“消息”坐标")
    time.sleep(0.6)
    image = capture_after_popups(robot, cfg, label="after_points_message", reporter=reporter)
    _raise_if_black_screen(image, cfg)
    if not _matches_screen(image, cfg, CHAT_LIST_SCREEN):
        raise StepUnavailable(f"读取积分后未能返回聊天列表，未匹配到模板：{_template_hint(CHAT_LIST_SCREEN, group='screens')}")
    return points or None


def enter_chat(robot: Any, x: float, y: float) -> None:
    click_image_point(robot, x, y)


def confirm_chat_window(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> Any:
    cfg = _config(config)
    _report(reporter, "enter_chat", "确认聊天界面")
    image = capture_after_popups(robot, cfg, label="chat_window", reporter=reporter)
    _raise_if_black_screen(image, cfg)
    if not _matches_screen(image, cfg, CHAT_WINDOW_SCREEN):
        raise StepUnavailable(f"未匹配到聊天界面模板：{_template_hint(CHAT_WINDOW_SCREEN, group='screens')}")
    return image


def send_new_greeting_if_available(
    robot: Any,
    config: dict[str, Any] | None = None,
    image: Any | None = None,
    reporter: Reporter | None = None,
) -> bool:
    cfg = _config(config)
    current_image = image if image is not None else capture_after_popups(robot, cfg, label="new_greeting", reporter=reporter)
    pos = _template_center(current_image, cfg, NEW_GREETING_BUTTON, group="buttons")
    if pos is None:
        return False

    send_point = (min(0.98, pos[0] + 0.2), pos[1])
    _report(
        reporter,
        "new_greeting_send",
        "发送新招呼",
        log_label=f"发送新招呼：{format_point(*send_point)}",
    )
    click_image_point(robot, *send_point)
    time.sleep(0.4)

    image_after_send = capture_after_popups(robot, cfg, label="after_new_greeting", reporter=reporter)
    back_pos = _template_center(image_after_send, cfg, BACK_BUTTON_2, group="buttons")
    if back_pos is None:
        raise StepUnavailable(f"发送新招呼后未找到返回按钮模板：{_template_hint(BACK_BUTTON_2, group='buttons')}")
    _report(
        reporter,
        "new_greeting_return",
        "新招呼后返回列表",
        log_label=f"新招呼后返回列表：{format_point(*back_pos)}",
    )
    click_image_point(robot, *back_pos)
    time.sleep(0.4)
    return True


def read_message(
    robot: Any,
    config: dict[str, Any] | None = None,
    target: dict[str, Any] | None = None,
    image: Any | None = None,
    reporter: Reporter | None = None,
) -> dict[str, str]:
    cfg = _config(config)
    current_image = image if image is not None else confirm_chat_window(robot, cfg, reporter=reporter)
    message_region = _dialogue_message_region(current_image, cfg)
    if message_region is not None:
        _report(reporter, "dialogue_ocr_read", "聊天详情 OCR（消息小范围）")
        compact_result = _run_ocr_mode(current_image, cfg, "dialogue", region=message_region)
        message, has_content = _extract_message_from_dialogue_ocr(compact_result, cfg, target)
        if has_content:
            return message
        _increment_metric(cfg, "dialogue_compact_fallback_count")
        _report(reporter, "dialogue_ocr_read", "消息小范围无对方内容，回退完整 OCR")

    _report(reporter, "dialogue_ocr_read", "聊天详情 OCR")
    ocr_result = _run_ocr_mode(current_image, cfg, "dialogue", region=None)
    message, has_content = _extract_message_from_dialogue_ocr(ocr_result, cfg, target)
    if not ocr_result.get("ocr_items"):
        raise StepUnavailable("聊天详情 OCR 未识别到任何文字")
    if not has_content:
        target_sender = str(cfg["ocr_modes"]["dialogue"]["rules"].get("target_sender", "sender1"))
        raise StepUnavailable(f"聊天详情 OCR 未识别到对方消息：目标 sender={target_sender}")
    return message


def _dialogue_message_region(image: Any, config: dict[str, Any]) -> list[float] | None:
    if not _optimization_enabled(config, "enable_compact_dialogue_ocr_region"):
        return None
    chat_region = _valid_region(_region(config, CHAT_AREA_REGION)) or [0.0, 0.1, 1.0, 0.8]
    input_box = _cached_point(config, "template_points", "input_box")
    if input_box is None:
        try:
            input_box = _template_center(image, config, INPUT_BOX_BUTTON, group="buttons")
        except Exception:
            input_box = None
        if input_box is not None:
            _cache_point_once(config, "template_points", "input_box", input_box)
    if input_box is None:
        return None
    margin = float(config.get("dialogue_message_input_box_top_margin", 0.04) or 0.04)
    left = float(chat_region[0])
    top = float(chat_region[1])
    width = float(chat_region[2])
    configured_bottom = top + float(chat_region[3])
    bottom = min(configured_bottom, float(input_box[1]) - margin)
    if bottom - top < float(config.get("dialogue_compact_min_height", 0.2) or 0.2):
        return None
    region = [left, top, width, bottom - top]
    _cache_set(config, "ocr_regions", "dialogue_message", region)
    return region


def _extract_message_from_dialogue_ocr(
    ocr_result: dict[str, Any],
    config: dict[str, Any],
    target: dict[str, Any] | None,
) -> tuple[dict[str, str], bool]:
    items = ocr_result.get("ocr_items", [])
    dialogue_result = ocr_result.get("result", {})
    if not isinstance(dialogue_result, dict):
        dialogue_result = {}
    dialogues = dialogue_result.get("dialogues", [])
    dialogue_rules = config["ocr_modes"]["dialogue"]["rules"]
    target_sender = str(dialogue_rules.get("target_sender", "sender1"))
    trailing_target_messages: list[str] = []
    for item in reversed(dialogues if isinstance(dialogues, list) else []):
        if not isinstance(item, dict):
            continue
        sender = str(item.get("sender", ""))
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        if sender == target_sender:
            trailing_target_messages.append(content)
            continue
        if trailing_target_messages:
            break
    fallback_contact = str(target.get("name", "")) if isinstance(target, dict) else ""
    first_text = str(items[0].get("text", "")) if items and isinstance(items[0], dict) else ""
    contact = fallback_contact or str(dialogue_result.get("user_name") or first_text)
    content = "".join(reversed(trailing_target_messages))
    return {
        "user": contact,
        "contact": contact,
        "content": content,
        "sender": target_sender,
        "message_time": "",
        "unread_count": str(target.get("unread_count", "")) if isinstance(target, dict) else "",
        "favorability": str(target.get("favorability", "")) if isinstance(target, dict) else "",
    }, bool(content)


def build_list_message(target: dict[str, Any], content: str) -> dict[str, str]:
    contact = str(target.get("name", "") or "")
    return {
        "user": contact,
        "contact": contact,
        "content": content,
        "message_time": "",
        "unread_count": str(target.get("unread_count", "")),
        "favorability": str(target.get("favorability", "")),
    }


def focus_input_box(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> None:
    cfg = _config(config)
    image = capture_after_popups(robot, cfg, label="input_box", reporter=reporter)
    cached = _cached_point(cfg, "template_points", "input_box")
    if cached is not None:
        pos = cached
        _set_metric(cfg, "input_box_source", "cache")
        _set_metric(cfg, "input_box_template", "")
    else:
        match_result = _template_match(image, cfg, INPUT_BOX_BUTTON, group="buttons")
        if match_result is None:
            raise StepUnavailable(f"未找到输入框模板：{_template_hint(INPUT_BOX_BUTTON, group='buttons')}")
        pos = (
            match_result["x"] + match_result["w"] / 2,
            match_result["y"] + match_result["h"] / 2,
        )
        _cache_point_once(cfg, "template_points", "input_box", pos)
        _set_metric(cfg, "input_box_source", "template")
        _set_metric(cfg, "input_box_template", match_result.get("_template_name", ""))
    if pos is None:
        raise StepUnavailable(f"未找到输入框模板：{_template_hint(INPUT_BOX_BUTTON, group='buttons')}")
    raw_click_x = pos[0] - 0.5
    min_click_x = float(cfg.get("input_box_click_min_x", 0.12) or 0.12)
    max_click_x = float(cfg.get("input_box_click_max_x", 0.88) or 0.88)
    click_x = max(min_click_x, min(max_click_x, raw_click_x))
    click_point = (click_x, pos[1])
    if abs(click_x - raw_click_x) > 1e-6:
        _report(
            reporter,
            "input_box_click_adjusted",
            "输入框点击位置已修正",
            log_label=f"输入框点击位置已修正：原始X={raw_click_x:.3f}，修正后X={click_x:.3f}",
        )
    _report(reporter, "input_box_click", "点击输入框偏移位置", log_label=f"点击输入框偏移位置：{format_point(*click_point)}")
    click_image_point(robot, *click_point)


def play_voice_reply(
    robot: Any,
    text: str,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> None:
    cfg = _config(config)
    retry_count = max(0, int(cfg.get("tts_retry_attempts", 1) or 0))
    last_error: Exception | None = None
    _set_metric(cfg, "tts_success", False)
    _set_metric(cfg, "tts_retry_count", 0)
    for attempt in range(retry_count + 1):
        deadline = step_deadline(cfg, _step_timeout(cfg))
        while True:
            image = capture_after_popups(robot, cfg, label="microphone", reporter=reporter)
            microphone_match = _template_match(image, cfg, MICROPHONE_BUTTON, group="buttons")
            if microphone_match is not None:
                microphone = (
                    microphone_match["x"] + microphone_match["w"] / 2,
                    microphone_match["y"] + microphone_match["h"] / 2,
                )
                _set_metric(cfg, "microphone_template", microphone_match.get("_template_name", ""))
                label = "点击麦克风" if attempt == 0 else "重新点击麦克风"
                _report(reporter, "microphone_click", label, log_label=f"{label}：{format_point(*microphone)}")
                click_image_point(robot, *microphone)
                break
            if deadline.expired():
                raise StepUnavailable(f"未找到麦克风按钮模板：{_template_hint(MICROPHONE_BUTTON, group='buttons')}")
            time.sleep(0.4)

        try:
            _report(reporter, "tts_playback", "播放 TTS 回复" if attempt == 0 else "重新播放 TTS 回复")
            tts_input(robot, text)
            time.sleep(TTS_POST_DELAY_SECONDS)
            _set_metric(cfg, "tts_success", True)
            _set_metric(cfg, "tts_retry_count", attempt)
            return
        except OperationCancelled:
            raise
        except Exception as exc:
            last_error = exc
            if attempt >= retry_count:
                break
            _report(
                reporter,
                "tts_retry",
                "TTS 播放失败，准备重试",
                log_label=f"TTS 播放失败，准备重试：{exc}",
            )
            _set_metric(cfg, "tts_retry_count", attempt + 1)
            time.sleep(0.4)
    raise StepUnavailable(f"TTS 播放重试后仍失败：{last_error}") from last_error


def input_reply(
    robot: Any,
    text: str,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> None:
    focus_input_box(robot, config, reporter=reporter)
    play_voice_reply(robot, text, config, reporter=reporter)


def send_reply(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> int:
    cfg = _config(config)
    deadline = step_deadline(cfg, _step_timeout(cfg))
    clicked = 0
    _set_metric(cfg, "send_button_clicks", 0)
    while True:
        image = capture_after_popups(robot, cfg, label="send_button", reporter=reporter)
        pos = _template_center(image, cfg, SEND_BUTTON_KEY, group="buttons")
        if pos is None:
            if clicked > 0:
                return clicked
            if deadline.expired():
                raise StepUnavailable(f"未找到发送按钮模板：{_template_hint(SEND_BUTTON_KEY, group='buttons')}")
            time.sleep(0.4)
            continue
        _report(reporter, "send_button_click", "点击发送按钮", log_label=f"点击发送按钮：{format_point(*pos)}")
        click_image_point(robot, *pos)
        clicked += 1
        _set_metric(cfg, "send_button_clicks", clicked)
        if deadline.expired():
            raise StepUnavailable(f"点击发送后发送按钮仍未消失：{_template_hint(SEND_BUTTON_KEY, group='buttons')}")
        time.sleep(0.4)


def ensure_send_cleared(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> int:
    cfg = _config(config)
    image = capture_after_popups(robot, cfg, label="send_clear_check", reporter=reporter)
    pos = _template_center(image, cfg, SEND_BUTTON_KEY, group="buttons")
    if pos is None:
        _set_metric(cfg, "send_clear_extra_clicks", 0)
        _set_metric(cfg, "send_clear_reason", "send_button_absent")
        return 0
    _report(reporter, "send_button_click", "发送按钮仍存在，补点发送", log_label=f"发送按钮仍存在，补点发送：{format_point(*pos)}")
    click_image_point(robot, *pos)
    time.sleep(0.4)
    _set_metric(cfg, "send_clear_extra_clicks", 1)
    _set_metric(cfg, "send_clear_reason", "send_button_still_visible")
    return 1


def go_back(
    robot: Any,
    config: dict[str, Any] | None = None,
    *,
    double_click_on_first_attempt: bool = False,
    reporter: Reporter | None = None,
) -> int:
    cfg = _config(config)
    _report(reporter, "return_to_list", "返回聊天列表")
    ensure_send_cleared(robot, cfg, reporter=reporter)
    image = capture_after_popups(robot, cfg, label="go_back", reporter=reporter)
    if _matches_chat_list_screen(image, cfg, reporter=reporter):
        _report(reporter, "back_click", "已在聊天列表，无需返回")
        _set_metric(cfg, "return_clicks", 0)
        _set_metric(cfg, "return_reason", "already_in_list")
        return 0

    back_clicks = 0
    for attempt in range(2):
        image = capture_after_popups(robot, cfg, label="go_back_button", reporter=reporter)
        cached = _cached_point(cfg, "template_points", "back")
        pos = cached if cached is not None else _template_center(image, cfg, BACK_BUTTON, group="buttons")
        if pos is None:
            _report(reporter, "back_click", "返回按钮未找到")
            _set_metric(cfg, "return_reason", "back_template_missing")
            raise StepUnavailable(f"未找到返回按钮模板：{_template_hint(BACK_BUTTON, group='buttons')}")
        if cached is None:
            _cache_point_once(cfg, "template_points", "back", pos)
        click_point = pos
        click_times = 2 if attempt == 0 and double_click_on_first_attempt else 1
        if click_times == 2:
            _report(
                reporter,
                "back_click",
                "发送后连续点击返回两次",
                log_label=(
                    f"发送后连续点击返回两次（缓存）：{format_point(*click_point)}"
                    if cached is not None
                    else f"发送后连续点击返回两次：{format_point(*click_point)}"
                ),
            )
            click_image_point(robot, *click_point)
            click_image_point(robot, *click_point)
            back_clicks += 2
        else:
            _report(
                reporter,
                "back_click",
                "点击返回",
                log_label=(
                    f"点击返回（缓存）：{format_point(*click_point)}"
                    if cached is not None
                    else f"点击返回：{format_point(*click_point)}"
                ),
            )
            click_image_point(robot, *click_point)
            back_clicks += 1
        time.sleep(0.4)
        image = capture_after_popups(robot, cfg, label="after_go_back", reporter=reporter)
        if _matches_chat_list_screen(image, cfg, reporter=reporter):
            _report(
                reporter,
                "back_click",
                "返回聊天列表成功",
                log_label=f"返回聊天列表成功：返回点击 {back_clicks} 次",
            )
            _set_metric(cfg, "return_clicks", back_clicks)
            if back_clicks == 2 and double_click_on_first_attempt:
                _set_metric(cfg, "return_reason", "after_send_double_back")
            elif back_clicks == 1:
                _set_metric(cfg, "return_reason", "single_back_success")
            else:
                _set_metric(cfg, "return_reason", "retry_back_success")
            return back_clicks
        _set_metric(cfg, "return_reason", "cached_back_recheck_failed" if cached is not None else "template_back_recheck_failed")
        _report(
            reporter,
            "back_click",
            "点击返回后未到聊天列表，重新截图后再尝试",
            log_label=f"点击返回后未到聊天列表，重新截图后再尝试：{format_point(*click_point)}",
        )

    _set_metric(cfg, "return_clicks", back_clicks)
    _set_metric(cfg, "return_reason", "recheck_not_in_list")
    raise StepUnavailable(f"返回后仍未进入聊天列表，未匹配到模板：{_template_hint(CHAT_LIST_SCREEN, group='screens')}")

