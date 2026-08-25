from __future__ import annotations

import copy
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from apps.cancellation import OperationCancelled, cancellable_sleep, raise_if_cancelled
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
    format_template_match_diagnostic,
    match_template_candidates_with_region,
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
from vision.screen import ensure_portrait_image, load_image

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
CALL_SCREEN = "screen.call_window"
POINTS_REGION = "ocr.regions.points"
MESSAGE_LIST_REGION = "ocr.regions.message_list"
CHAT_AREA_REGION = "ocr.regions.chat_area"
INPUT_BOX_BUTTON = "button.input_box"
MICROPHONE_BUTTON = "button.microphone"
UNREAD_BUTTON = "button.unread"
UNREAD_BUTTON_NO_RED = "button.unread.no_red"
SEND_BUTTON_KEY = "button.send"
NEW_GREETING_BUTTON = "button.new_greeting"
PHOTO_BUTTON = "button.photo"
COMPLETE_BUTTON = "button.complete"
BACK_BUTTON_2 = "button.back.secondary"
UNREAD_TAB_KEYWORDS = ("\u672a\u56de\u590d",)
TTS_POST_DELAY_SECONDS = 3.0
POPUP_MAX_X = 0.9
BOTTOM_NAV_CACHE_Y_OFFSET = -0.02
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
    CALL_SCREEN: ("screen_通话界面.jpg",),
}

BUTTON_TEMPLATE_ALIASES: dict[str, tuple[str, ...]] = {
    BACK_BUTTON: ("btn_返回按钮.jpg", "btn_返回按钮_2.jpg", "btn_back.jpg"),
    UNREAD_BUTTON: ("btn_未读消息按钮.jpg", "btn_next_msg.jpg"),
    UNREAD_BUTTON_NO_RED: ("btn_未读消息-2.jpg",),
    INPUT_BOX_BUTTON: ("btn_输入框.jpg", "btn_input_box.jpg"),
    MICROPHONE_BUTTON: ("btn_麦克风按钮.jpg", "btn_voice_input.jpg"),
    SEND_BUTTON_KEY: ("btn_发送按钮.jpg", "btn_send.jpg"),
    NEW_GREETING_BUTTON: ("btn_新招呼按钮.jpg",),
    PHOTO_BUTTON: ("btn_照片按钮.jpg", "btn_相册按钮.jpg", "btn_photo.jpg"),
    COMPLETE_BUTTON: ("btn_完成按钮.jpg", "btn_complete.jpg"),
    BACK_BUTTON_2: ("btn_返回按钮_2.jpg",),
}


def _template_hint(key: str, *, group: str) -> str:
    aliases = SCREEN_TEMPLATE_ALIASES.get(key, ()) if group == "screens" else BUTTON_TEMPLATE_ALIASES.get(key, ())
    hint = " / ".join(aliases) if aliases else key
    return hint


def _path_signature(path: Path) -> tuple[str, int, int]:
    try:
        stat = path.stat()
    except OSError:
        return str(path), 0, 0
    return str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size)


def _configured_template_region(config: dict[str, Any], key: str) -> Any:
    regions = config.get("template_match_regions", {})
    if not isinstance(regions, dict):
        return None
    return regions.get(key)


def _template_failure_detail(config: dict[str, Any], key: str, *, group: str) -> str:
    aliases = SCREEN_TEMPLATE_ALIASES.get(key, ()) if group == "screens" else BUTTON_TEMPLATE_ALIASES.get(key, ())
    hint = " / ".join(aliases) if aliases else key
    diagnostic = format_template_match_diagnostic(config, _match_region_key(config, key))
    if diagnostic != "最高候选：无":
        return f"{hint}；{diagnostic}"
    return hint


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
    region_key = popup_template_region_key(popup_template)
    threshold = _popup_threshold(config, region_key)
    cache_key = "|".join(
        (
            "template_single",
            region_key,
            _json_cache_fragment(_path_signature(popup_template)),
            str(round(float(threshold), 4)),
            _json_cache_fragment(_configured_template_region(config, region_key)),
        )
    )
    signature = _ocr_reuse_signature(image)
    cached = _reused_detection_result(config, key=cache_key, signature=signature)
    if cached is not ...:
        result = cached
    else:
        result = match_template_with_region(
            image,
            popup_template,
            config,
            region_key,
            threshold=threshold,
            match_func=match,
        )
        _store_detection_result(config, key=cache_key, signature=signature, result=result)
    if not result:
        return None
    max_x = float(config.get("popup_max_x", POPUP_MAX_X) or POPUP_MAX_X)
    if float(result.get("x", 0.0)) > max_x:
        return None
    return result


def _step_timeout(config: dict[str, Any]) -> float:
    return float(config.get("step_timeout", 10))


def _json_cache_fragment(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return repr(value)


def _ocr_reuse_key(config: dict[str, Any], mode: str, region: Any) -> str:
    base_ocr = dict(config.get("ocr", {}) if isinstance(config.get("ocr"), dict) else {})
    base_ocr.pop("engine", None)
    mode_config = {}
    ocr_modes = config.get("ocr_modes", {})
    if isinstance(ocr_modes, dict) and isinstance(ocr_modes.get(mode), dict):
        mode_config = ocr_modes.get(mode, {})
    runtime = config.get("_runtime", {}) if isinstance(config.get("_runtime"), dict) else {}
    return "|".join(
        (
            str(runtime.get("device_ip", "")),
            str(runtime.get("app_name", "")),
            mode,
            _json_cache_fragment(region),
            _json_cache_fragment(base_ocr),
            _json_cache_fragment(mode_config),
        )
    )


def _ocr_reuse_signature(image: Any, region: Any = None) -> tuple[tuple[int, int], np.ndarray] | None:
    try:
        portrait = ensure_portrait_image(image).convert("L")
    except Exception:
        return None
    if isinstance(region, (list, tuple)) and len(region) == 4:
        x, y, w, h = (float(region[0]), float(region[1]), float(region[2]), float(region[3]))
        width, height = portrait.size
        left = int(max(0.0, min(1.0, x)) * width)
        top = int(max(0.0, min(1.0, y)) * height)
        right = int(max(0.0, min(1.0, x + w)) * width)
        bottom = int(max(0.0, min(1.0, y + h)) * height)
        if right > left and bottom > top:
            portrait = portrait.crop((left, top, right, bottom))
    thumb = portrait.resize((64, 64))
    return portrait.size, np.array(thumb, dtype=np.uint8)


def _same_ocr_signature(
    current: tuple[tuple[int, int], np.ndarray] | None,
    previous: tuple[tuple[int, int], np.ndarray] | None,
    config: dict[str, Any],
) -> bool:
    if current is None or previous is None:
        return False
    current_size, current_thumb = current
    previous_size, previous_thumb = previous
    if current_size != previous_size or current_thumb.shape != previous_thumb.shape:
        return False
    diff = np.abs(current_thumb.astype(np.int16) - previous_thumb.astype(np.int16))
    mean_threshold = float(config.get("ocr_reuse_image_mean_threshold", 0.8) or 0.8)
    if float(diff.mean()) > mean_threshold:
        return False
    pixel_threshold = int(config.get("ocr_reuse_image_pixel_threshold", 5) or 5)
    ratio_threshold = float(config.get("ocr_reuse_image_ratio_threshold", 0.003) or 0.003)
    changed_ratio = float((diff > pixel_threshold).mean())
    return changed_ratio <= ratio_threshold


def _reused_ocr_result(
    config: dict[str, Any],
    *,
    key: str,
    signature: tuple[tuple[int, int], np.ndarray] | None,
) -> dict[str, Any] | None:
    if not _optimization_enabled(config, "enable_screenshot_ocr_reuse"):
        return None
    reuse_cache = _runtime_cache(config).get("ocr_result_reuse", {})
    if not isinstance(reuse_cache, dict):
        return None
    cached = reuse_cache.get(key)
    if not isinstance(cached, dict):
        return None
    if not _same_ocr_signature(signature, cached.get("signature"), config):
        return None
    result = copy.deepcopy(cached.get("result"))
    if not isinstance(result, dict):
        return None
    result["_cache_hit"] = True
    result["_cache_source"] = "xinyu_screenshot_ocr_reuse"
    return result


def _reused_detection_result(
    config: dict[str, Any],
    *,
    key: str,
    signature: tuple[tuple[int, int], np.ndarray] | None,
) -> Any:
    if not _optimization_enabled(config, "enable_screenshot_detection_reuse"):
        return ...
    reuse_cache = _runtime_cache(config).get("detection_result_reuse", {})
    if not isinstance(reuse_cache, dict):
        return ...
    cached = reuse_cache.get(key)
    if not isinstance(cached, dict):
        return ...
    if not _same_ocr_signature(signature, cached.get("signature"), config):
        return ...
    metrics = _runtime_cache(config).get("metrics", {})
    if isinstance(metrics, dict):
        metrics["detection_cache_hit_count"] = int(metrics.get("detection_cache_hit_count", 0) or 0) + 1
    return copy.deepcopy(cached.get("result"))


def _store_detection_result(
    config: dict[str, Any],
    *,
    key: str,
    signature: tuple[tuple[int, int], np.ndarray] | None,
    result: Any,
) -> None:
    if not _optimization_enabled(config, "enable_screenshot_detection_reuse") or signature is None:
        return
    cache = _runtime_cache(config)
    reuse_cache = cache.setdefault("detection_result_reuse", {})
    if not isinstance(reuse_cache, dict):
        reuse_cache = {}
        cache["detection_result_reuse"] = reuse_cache
    if len(reuse_cache) >= 128 and key not in reuse_cache:
        reuse_cache.clear()
    reuse_cache[key] = {"signature": signature, "result": copy.deepcopy(result)}


def _store_ocr_reuse_result(
    config: dict[str, Any],
    *,
    key: str,
    signature: tuple[tuple[int, int], np.ndarray] | None,
    result: dict[str, Any],
) -> None:
    if not _optimization_enabled(config, "enable_screenshot_ocr_reuse") or signature is None:
        return
    cache = _runtime_cache(config)
    reuse_cache = cache.setdefault("ocr_result_reuse", {})
    if not isinstance(reuse_cache, dict):
        reuse_cache = {}
        cache["ocr_result_reuse"] = reuse_cache
    if len(reuse_cache) >= 16 and key not in reuse_cache:
        reuse_cache.clear()
    stored = copy.deepcopy(result)
    stored.pop("_cache_hit", None)
    stored.pop("_cache_source", None)
    reuse_cache[key] = {"signature": signature, "result": stored}


def _run_ocr_mode(image: Any, config: dict[str, Any], mode: str, region: Any = None) -> dict[str, Any]:
    started_at = time.monotonic()
    signature = _ocr_reuse_signature(image, region)
    reuse_key = _ocr_reuse_key(config, mode, region)
    reused = _reused_ocr_result(config, key=reuse_key, signature=signature)
    if reused is not None:
        metrics = _runtime_cache(config).get("metrics", {})
        if isinstance(metrics, dict):
            metrics["ocr_cache_hit_count"] = int(metrics.get("ocr_cache_hit_count", 0) or 0) + 1
            metrics["last_ocr"] = {
                "mode": mode,
                "region": region,
                "seconds": round(time.monotonic() - started_at, 4),
                "cache_hit": True,
                "cache_source": reused.get("_cache_source", ""),
            }
        return reused
    with exclude_ocr_wait_from_step_timeout(config):
        result = run_ocr_mode(image, config, mode, region=region)
    _store_ocr_reuse_result(config, key=reuse_key, signature=signature, result=result)
    metrics = _runtime_cache(config).get("metrics", {})
    if isinstance(metrics, dict):
        if result.get("_cache_hit"):
            metrics["ocr_cache_hit_count"] = int(metrics.get("ocr_cache_hit_count", 0) or 0) + 1
        else:
            metrics["ocr_count"] = int(metrics.get("ocr_count", 0) or 0) + 1
        metrics["last_ocr"] = {
            "mode": mode,
            "region": region,
            "seconds": round(time.monotonic() - started_at, 4),
            "cache_hit": bool(result.get("_cache_hit")),
            "cache_source": result.get("_cache_source", ""),
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
    threshold = _threshold(config, key)
    cache_key = "|".join(
        (
            "template_candidates",
            key,
            region_key,
            str(round(float(threshold), 4)),
            _json_cache_fragment([_path_signature(path) for path in paths]),
            _json_cache_fragment(_configured_template_region(config, region_key)),
            _json_cache_fragment(config.get("template_match_scales")),
            _json_cache_fragment(config.get("template_match_channels")),
        )
    )
    signature = _ocr_reuse_signature(image)
    cached = _reused_detection_result(config, key=cache_key, signature=signature)
    if cached is not ...:
        return cached
    result = match_template_candidates_with_region(
        image,
        paths,
        config,
        region_key,
        threshold=threshold,
        match_func=match,
        single_match_func=match_template_with_region,
    )
    _store_detection_result(config, key=cache_key, signature=signature, result=result)
    return result


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


def _select_list_target_from_rows(rows: Any, *, require_unread: bool = False) -> dict[str, Any] | None:
    if not isinstance(rows, list):
        return None
    unread_candidates: list[tuple[int, int, dict[str, Any]]] = []
    content_candidates: list[tuple[int, int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or _is_empty_list_row(row):
            continue
        bounds = row.get("bounds")
        if not isinstance(bounds, list) or len(bounds) != 4:
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
    candidates = unread_candidates if require_unread else (unread_candidates or content_candidates)
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


def _select_list_target_from_ocr_result(
    ocr_result: dict[str, Any],
    *,
    require_unread: bool = False,
) -> dict[str, Any] | None:
    result = ocr_result.get("result", {}) if isinstance(ocr_result, dict) else {}
    rows = result.get("message_list", []) if isinstance(result, dict) else []
    return _select_list_target_from_rows(rows, require_unread=require_unread)


def _photo_contact_key(value: Any) -> str:
    """Normalize a contact name for the per-run photo-send ledger."""
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _photo_sent_store_path(config: dict[str, Any]) -> Path | None:
    runtime = config.get("_runtime", {})
    device_ip = str(runtime.get("device_ip", "") or "").strip() if isinstance(runtime, dict) else ""
    if not device_ip:
        return None
    safe_device = re.sub(r"[^0-9A-Za-z_.-]+", "_", device_ip)
    return Path("runtime") / "xinyu" / "photo_sent_contacts" / f"{safe_device}.json"


def _photo_sent_cooldown_seconds(config: dict[str, Any]) -> float:
    try:
        return max(0.0, float(config.get("photo_sent_cooldown_seconds", 86400) or 0))
    except (TypeError, ValueError):
        return 86400.0


def _load_photo_sent_records(config: dict[str, Any]) -> dict[str, float]:
    path = _photo_sent_store_path(config)
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    records: dict[str, float] = {}
    now = time.time()
    cooldown = _photo_sent_cooldown_seconds(config)
    if cooldown <= 0:
        return {}
    for name, timestamp in payload.items():
        key = _photo_contact_key(name)
        try:
            sent_at = float(timestamp)
        except (TypeError, ValueError):
            continue
        if key and (cooldown <= 0 or now - sent_at < cooldown):
            records[key] = sent_at
    return records


def _photo_sent_contacts(config: dict[str, Any]) -> set[str]:
    cache = _runtime_cache(config)
    cooldown = _photo_sent_cooldown_seconds(config)
    if cooldown <= 0:
        cache["xinyu_photo_sent_contacts"] = []
        cache["xinyu_photo_sent_contact_times"] = {}
        return set()
    records = _load_photo_sent_records(config)
    now = time.time()
    runtime_times = cache.setdefault("xinyu_photo_sent_contact_times", {})
    if not isinstance(runtime_times, dict):
        runtime_times = {}
        cache["xinyu_photo_sent_contact_times"] = runtime_times
    runtime_times = {
        key: float(timestamp)
        for key, timestamp in runtime_times.items()
        if key and isinstance(timestamp, (int, float)) and now - float(timestamp) < cooldown
    }
    cache["xinyu_photo_sent_contact_times"] = runtime_times
    records.update(runtime_times)
    contacts = cache.setdefault("xinyu_photo_sent_contacts", [])
    if not isinstance(contacts, list):
        contacts = []
    for key in records:
        if not any(_photo_contact_key(value) == key for value in contacts):
            contacts.append(key)
    cache["xinyu_photo_sent_contacts"] = contacts
    return {
        key for key, timestamp in records.items()
        if key and now - timestamp < cooldown
    }


def _record_photo_sent_contact(config: dict[str, Any], name: Any) -> None:
    key = _photo_contact_key(name)
    if not key:
        return
    cache = _runtime_cache(config)
    cooldown = _photo_sent_cooldown_seconds(config)
    if cooldown <= 0:
        return
    contacts = cache.setdefault("xinyu_photo_sent_contacts", [])
    if not isinstance(contacts, list):
        contacts = []
        cache["xinyu_photo_sent_contacts"] = contacts
    if not any(_photo_contact_key(value) == key for value in contacts):
        contacts.append(str(name or "").strip())
    contact_times = cache.setdefault("xinyu_photo_sent_contact_times", {})
    if isinstance(contact_times, dict):
        contact_times[key] = time.time()
    path = _photo_sent_store_path(config)
    if path is not None:
        records = _load_photo_sent_records(config)
        records[key] = time.time()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass


def _select_online_target_from_ocr_result(
    ocr_result: dict[str, Any],
    *,
    excluded_contacts: set[str] | None = None,
) -> dict[str, Any] | None:
    result = ocr_result.get("result", {}) if isinstance(ocr_result, dict) else {}
    rows = result.get("message_list", []) if isinstance(result, dict) else []
    if not isinstance(rows, list):
        return None
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    excluded_contacts = excluded_contacts or set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or _is_empty_list_row(row):
            continue
        name = _clean_list_text(row.get("name"))
        bounds = row.get("bounds")
        if not name or not isinstance(bounds, list) or len(bounds) != 4:
            continue
        if _photo_contact_key(name) in excluded_contacts:
            continue
        try:
            favorability = int(row.get("favorability")) if row.get("favorability") is not None else -1
        except (TypeError, ValueError):
            favorability = -1
        candidates.append((favorability, -index, row))
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
        "bounds": selected_row["bounds"],
        "x": center_x,
        "y": center_y,
    }


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


def wait_for_call_screen_to_disappear(
    robot: Any,
    config: dict[str, Any],
    image: Any,
    *,
    label: str = "call_screen",
    reporter: Reporter | None = None,
) -> Any:
    current_image = image
    poll_seconds = float(config.get("call_screen_wait_poll_seconds", 2.0) or 2.0)
    timeout_seconds = float(config.get("call_screen_wait_timeout_seconds", 0) or 0)
    started_at = time.monotonic()
    reported = False
    while _matches_screen(current_image, config, CALL_SCREEN):
        if not reported:
            _report(reporter, "call_screen_wait", "检测到通话界面，等待通话结束")
            reported = True
        if timeout_seconds > 0 and time.monotonic() - started_at >= timeout_seconds:
            raise StepUnavailable("通话界面等待超时，仍未消失")
        cancellable_sleep(poll_seconds, robot, config)
        raise_if_cancelled(robot, config)
        current_image = _capture_image(robot, config, pre_move=True, label=f"{label}_wait")
    if reported:
        _report(reporter, "call_screen_wait", "通话界面已消失，继续流程")
    return current_image


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


def _unread_tab_item(ocr_result: dict[str, Any]) -> tuple[dict[str, Any], tuple[float, float]] | None:
    """Locate the top "未回复" filter when a banner pushes it below the normal OCR nav band."""
    result = ocr_result.get("result", {}) if isinstance(ocr_result, dict) else {}
    bottom_nav = result.get("bottom_nav", []) if isinstance(result, dict) else []
    if isinstance(bottom_nav, list):
        item = _bottom_nav_item(bottom_nav, UNREAD_TAB_KEYWORDS)
        if item is not None:
            return item

    raw_items = ocr_result.get("ocr_items", []) if isinstance(ocr_result, dict) else []
    for raw_item in raw_items if isinstance(raw_items, list) else []:
        if not isinstance(raw_item, dict):
            continue
        text = str(raw_item.get("text", "") or "")
        if not any(keyword in text for keyword in UNREAD_TAB_KEYWORDS):
            continue
        try:
            left = float(raw_item["x"])
            top = float(raw_item["y"])
            width = float(raw_item["w"])
            height = float(raw_item["h"])
        except (KeyError, TypeError, ValueError):
            continue
        bounds = [left, top, left + width, top + height]
        center = _center_from_bounds(bounds)
        if 0.08 <= center[1] <= 0.35:
            return {"name": text, "bounds": bounds}, center
    return None


def _cache_bottom_nav_from_ocr_result(config: dict[str, Any], ocr_result: dict[str, Any]) -> None:
    if not _optimization_enabled(config, "enable_bottom_nav_anchor_reuse"):
        return
    result = ocr_result.get("result", {})
    bottom_nav = result.get("bottom_nav", []) if isinstance(result, dict) else []
    if not isinstance(bottom_nav, list) or not bottom_nav:
        return
    bottom_nav_specs = (
        ("bottom_fate", ("缘分",)),
        ("bottom_feed", ("动态",)),
        ("bottom_entertainment", ("娱乐交友", "娱乐", "交友")),
        ("bottom_message", ("消息",)),
        ("bottom_my", ("我的",)),
    )
    detected: dict[str, tuple[float, float]] = {}
    for cache_key, keywords in bottom_nav_specs:
        item = _bottom_nav_item(bottom_nav, keywords, min_center_y=0.5)
        if item is None:
            continue
        center_x, center_y = item[1]
        corrected_center = (center_x, max(0.02, min(0.98, center_y + BOTTOM_NAV_CACHE_Y_OFFSET)))
        detected[cache_key] = corrected_center
        _cache_point_once(config, "screen_anchors", cache_key, corrected_center)

    message = detected.get("bottom_message")
    if message is not None:
        _cache_point_once(config, "screen_anchors", "bottom_message", message)
    my_pos = detected.get("bottom_my")
    if my_pos is not None:
        _cache_point_once(config, "screen_anchors", "bottom_my", my_pos)
    elif message is not None:
        message_x, message_y = message
        _cache_point_once(config, "screen_anchors", "bottom_my", (min(0.98, message_x + 0.2), message_y))
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


def _cache_message_from_cached_my(config: dict[str, Any], my_pos: tuple[float, float] | None) -> None:
    if _cached_point(config, "screen_anchors", "bottom_message") is not None:
        return
    if my_pos is None:
        return
    my_x, my_y = my_pos
    _cache_point_once(config, "screen_anchors", "bottom_message", (max(0.02, my_x - 0.2), my_y))


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
            compact_result = _run_ocr_mode(image, config, "list", region=compact_region)
            _report(
                reporter,
                "list_ocr_region",
                "复用列表主体小范围 OCR 结果" if compact_result.get("_cache_hit") else "使用列表主体小范围 OCR",
            )
            rows = compact_result.get("result", {}).get("message_list", [])
            if isinstance(rows, list) and rows:
                return compact_result
            _report(reporter, "list_ocr_region", "列表主体无有效行，回退完整 OCR")
    ocr_result = _run_ocr_mode(image, config, "list", region=None)
    _report(
        reporter,
        "list_ocr_scan",
        "复用上一轮 list OCR 结果" if ocr_result.get("_cache_hit") else "列表 OCR 扫描",
    )
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


def _mark_unread_tab_red_state(ocr_result: dict[str, Any] | None, has_red: bool) -> None:
    if isinstance(ocr_result, dict):
        ocr_result["_xinyu_unread_tab_has_red"] = bool(has_red)


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
    region = _chat_list_title_region(config)
    if region is not None:
        _report(reporter, "chat_list_title_ocr", "聊天列表标题 OCR")
        ocr_result = _run_ocr_mode(image, config, "list", region=region)
        if _chat_list_title_ocr_matches(ocr_result):
            _report(reporter, "chat_list_title_ocr", "识别到消息通话，判定聊天列表")
            return True
    return _matches_screen(image, config, CHAT_LIST_SCREEN)


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
    image = wait_for_call_screen_to_disappear(robot, config, image, label=label, reporter=reporter)
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
        image = wait_for_call_screen_to_disappear(robot, config, image, label=f"{label}_after_popup", reporter=reporter)
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
        _report(reporter, "return_from_chat", "使用缓存坐标", log_label=f"使用缓存坐标：返回按钮 {format_point(*cached)}")
        _report(reporter, "return_from_chat", "点击返回", log_label=f"点击返回：{format_point(*cached)}")
        click_image_point(robot, *cached)
        return True
    pos = _template_center(current_image, config, BACK_BUTTON, group="buttons")
    if pos is None:
        _report(reporter, "return_from_chat", "返回按钮未找到")
        return False
    click_point = pos
    _cache_point_once(config, "template_points", "back", click_point)
    _report(
        reporter,
        "return_from_chat",
        "刷新缓存坐标",
        log_label=f"刷新缓存坐标：返回按钮 {format_point(*click_point)}；{format_template_match_diagnostic(config, BACK_BUTTON)}",
    )
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
        _report(
            reporter,
            "message_button_click",
            "使用缓存坐标",
            log_label=f"使用缓存坐标：消息按钮 {format_point(*pos)}，来源=bottom_nav",
        )
        click_image_point(robot, *pos)
        return True
    if image is not None:
        _report(reporter, "message_button_click", "消息按钮缓存不存在，执行一次 list OCR 查找")
        ocr_result = _run_ocr_mode(image, config, "list", region=None)
        _cache_bottom_nav_from_ocr_result(config, ocr_result)
        pos = _cached_point(config, "screen_anchors", "bottom_message")
        if pos is not None and pos[1] >= 0.5:
            _report(
                reporter,
                "message_button_click",
                "OCR 找到消息位置",
                log_label=f"OCR 找到消息位置：{format_point(*pos)}",
            )
            click_image_point(robot, *pos)
            return True
    _report(reporter, "message_button_click", "未找到消息位置，本轮不复位")
    return False


def ensure_list_screen(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
    image: Any | None = None,
) -> Any | None:
    cfg = _config(config)
    _report(reporter, "confirm_xinyu_list", "确认心遇聊天列表")
    max_attempts = max(1, int(cfg.get("startup_recovery_max_attempts", 5) or 5))
    current_image = image if image is not None else _capture_image(
        robot, cfg, pre_move=True, label="startup_screen_detect"
    )
    last_state = "unknown"
    for attempt in range(1, max_attempts + 1):
        current_image = wait_for_call_screen_to_disappear(
            robot,
            cfg,
            current_image,
            label=f"startup_call_screen_{attempt}",
            reporter=reporter,
        )
        _raise_if_black_screen(current_image, cfg)

        # Chat pages must proceed normally; popup matching is skipped there.
        is_chat = _matches_screen(current_image, cfg, CHAT_WINDOW_SCREEN)
        if not is_chat:
            current_image = capture_after_popups(
                robot,
                cfg,
                label=f"startup_popup_check_{attempt}",
                reporter=reporter,
            )
            _raise_if_black_screen(current_image, cfg)
            is_chat = _matches_screen(current_image, cfg, CHAT_WINDOW_SCREEN)

        if is_chat:
            last_state = CHAT_WINDOW_SCREEN
            _report(reporter, "startup_recover", "启动时在聊天界面，匹配返回按钮回到列表")
            if not _click_back_once(robot, cfg, current_image, reporter=reporter):
                raise StepUnavailable(
                    f"当前在聊天界面，但未找到返回按钮模板：{_template_failure_detail(cfg, BACK_BUTTON, group='buttons')}"
                )
            time.sleep(0.4)
            current_image = capture_after_popups(
                robot, cfg, label="after_return_from_chat", reporter=reporter
            )
            continue

        if _matches_screen(current_image, cfg, MY_SCREEN):
            last_state = MY_SCREEN
            _report(reporter, "startup_recover", "启动时在我的界面，先 OCR 读取积分和底部导航")
            read_my_points(robot, cfg, reporter=reporter, image=current_image, assume_my=True)
            current_image = _capture_image(robot, cfg, pre_move=True, label="after_message_button")
            continue

        if _matches_chat_list_screen(current_image, cfg, reporter=reporter):
            _raise_if_black_screen(current_image, cfg)
            _report(reporter, "startup_recover", f"已确认聊天列表（第 {attempt} 轮）")
            return current_image

        last_state = "unknown"
        _report(
            reporter,
            "startup_recover",
            f"启动界面未识别，第 {attempt}/{max_attempts} 轮尝试匹配返回按钮",
        )
        if not _click_back_once(robot, cfg, current_image, reporter=reporter):
            _report(
                reporter,
                "startup_recover",
                "未知界面未找到返回按钮，等待下一轮截图，不执行设备复位",
                log_label=(
                    "未知界面未找到返回按钮，等待下一轮截图，不执行设备复位："
                    f"{_template_failure_detail(cfg, BACK_BUTTON, group='buttons')}"
                ),
            )
            return None
        time.sleep(0.4)
        current_image = capture_after_popups(
            robot, cfg, label="after_unknown_back", reporter=reporter
        )

    raise StepUnavailable(
        f"启动纠偏超过 {max_attempts} 轮，已点击返回按钮，但未回到聊天列表；当前界面={last_state}；"
        f"未匹配到聊天列表模板：{_template_failure_detail(cfg, CHAT_LIST_SCREEN, group='screens')}"
    )


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
    unread_item = _unread_tab_item(ocr_result)
    if unread_item is None:
        _mark_unread_tab_red_state(ocr_result, False)
        return current_image, ocr_result

    nav_item, unread_center = unread_item
    bounds = nav_item.get("bounds")
    if not isinstance(bounds, list) or len(bounds) != 4:
        _mark_unread_tab_red_state(ocr_result, False)
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
    _mark_unread_tab_red_state(ocr_result, has_red)
    if has_red:
        return current_image, ocr_result
    click_image_point(robot, *unread_center)
    time.sleep(0.4)
    current_image = capture_after_popups(robot, cfg, label="after_unread_button", reporter=reporter)
    ocr_result = _run_list_ocr(current_image, cfg, reporter=reporter, prefer_compact=True)
    unread_target = _select_list_target_from_ocr_result(ocr_result, require_unread=True)
    unread_item = _unread_tab_item(ocr_result)
    if unread_item is None:
        if unread_target is not None:
            _mark_unread_tab_red_state(ocr_result, True)
            _report(
                reporter,
                "unread_button_check",
                "切换后未识别到未回复入口文字，但发现未读消息，继续读取列表目标",
            )
            return current_image, ocr_result
        _mark_unread_tab_red_state(ocr_result, False)
        _report(reporter, "unread_button_check", "切换后未找到未回复入口，且无未读消息，不读取列表目标")
        return current_image, ocr_result
    nav_item, unread_center = unread_item
    bounds = nav_item.get("bounds")
    if not isinstance(bounds, list) or len(bounds) != 4:
        _mark_unread_tab_red_state(ocr_result, False)
        _report(reporter, "unread_button_check", "切换后未回复入口范围无效，不读取列表目标")
        return current_image, ocr_result
    _cache_unread_tab_band(cfg, bounds)
    red_stats = _red_pixel_stats(current_image, _match_result_from_bounds(bounds))
    has_red = bool(red_stats.get("has_red"))
    confirmed_unread_page = has_red or unread_target is not None
    _mark_unread_tab_red_state(ocr_result, confirmed_unread_page)
    red_detail = (
        f"红色像素={red_stats.get('red_pixels', 0)}/{red_stats.get('total_pixels', 0)}"
        f"，阈值={red_stats.get('threshold', 0)}，区域={red_stats.get('region')}"
    )
    _report(
        reporter,
        "unread_button_check",
        "切换后未回复入口已有红色" if has_red else (
            "切换后入口无红色，但发现未读消息，继续读取列表目标"
            if unread_target is not None
            else "切换后未回复入口仍无红色，且无未读消息，不读取列表目标"
        ),
        log_label=(
            f"切换后未回复入口已有红色：{format_point(*unread_center)}，{red_detail}"
            if has_red
            else (
                f"切换后入口无红色，但发现未读消息，继续读取列表目标：{format_point(*unread_center)}，{red_detail}"
                if unread_target is not None
                else f"切换后未回复入口仍无红色，且无未读消息，不读取列表目标：{format_point(*unread_center)}，{red_detail}"
            )
        ),
    )
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
    initial_ocr_result = _run_list_ocr(image, cfg, reporter=reporter)
    image, checked_ocr_result = check_unread_button(
        robot,
        cfg,
        image,
        reporter=reporter,
        ocr_result=initial_ocr_result,
    )
    if not isinstance(checked_ocr_result, dict) or checked_ocr_result.get("_xinyu_unread_tab_has_red") is not True:
        _report(reporter, "list_target_skipped", "未回复入口不是红色，不读取列表目标")
        return None
    target = _select_list_target_from_ocr_result(checked_ocr_result)
    if target is not None:
        return target
    return None


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
    image: Any | None = None,
    assume_my: bool = False,
) -> str | None:
    cfg = _config(config)
    already_my = assume_my or (image is not None and _matches_screen(image, cfg, MY_SCREEN))
    if image is None:
        image = ensure_list_screen(robot, cfg, reporter=reporter)
    if image is None:
        return None

    if already_my:
        _report(reporter, "points_open_my", "当前已在我的界面，直接读取积分")
        _report(reporter, "points_scan_bottom_nav", "积分读取前扫描底部导航")
        nav_ocr = _run_ocr_mode(image, cfg, "list", region=None)
        _cache_bottom_nav_from_ocr_result(cfg, nav_ocr)
        _cache_message_from_cached_my(cfg, _cached_point(cfg, "screen_anchors", "bottom_my"))
        _raise_if_black_screen(image, cfg)
        if not _cached_point(cfg, "screen_anchors", "bottom_message"):
            _report(reporter, "message_button_click", "OCR 未找到消息位置，暂不复位")
        points_image = image
        points = _run_ocr_mode(points_image, cfg, "points", region=_region(cfg, POINTS_REGION))
        result = points.get("result", {}) if isinstance(points.get("result", {}), dict) else {}
        point_text = str(result.get("points") or result.get("balance") or "").strip()
        _report(
            reporter,
            "points_ocr_read",
            "积分读取完成" if point_text else "积分未识别",
            log_label=f"积分读取完成：{point_text}" if point_text else "积分未识别",
        )
        if not _click_message_button(robot, cfg, points_image, reporter=reporter):
            return point_text or None
        time.sleep(0.6)
        return point_text or None

    cached_my = (
        _cached_point(cfg, "screen_anchors", "bottom_my")
        if _optimization_enabled(cfg, "enable_bottom_nav_anchor_reuse")
        else None
    )
    pos = cached_my
    if pos is not None:
        _cache_message_from_cached_my(cfg, pos)
        _report(
            reporter,
            "points_open_my",
            "使用缓存坐标",
            log_label=f"使用缓存坐标：我的 {format_point(*pos)}，来源=bottom_nav",
        )
    if pos is None:
        _report(reporter, "points_scan_bottom_nav", "积分读取前扫描底部导航")
        ocr_result = _run_ocr_mode(image, cfg, "list", region=None)
        _cache_bottom_nav_from_ocr_result(cfg, ocr_result)
        bottom_nav = ocr_result.get("result", {}).get("bottom_nav", [])
        pos = _cached_point(cfg, "screen_anchors", "bottom_my") or _bottom_nav_center(
            bottom_nav if isinstance(bottom_nav, list) else [],
            ("我的",),
            min_center_y=0.5,
        )
    if pos is None:
        _report(reporter, "points_skip_no_my", "底部导航未识别到我的，跳过积分读取")
        return None
    _cache_message_from_cached_my(cfg, pos)
    _cache_point_once(cfg, "screen_anchors", "bottom_my", pos)
    max_open_attempts = max(1, int(cfg.get("points_open_max_attempts", 3) or 3))
    my_image: Any | None = None
    for attempt in range(1, max_open_attempts + 1):
        _report(
            reporter,
            "points_open_my",
            "打开我的界面" if attempt == 1 else "我的界面未打开，重新点击",
            log_label=(
                f"打开我的界面：{format_point(*pos)}"
                if attempt == 1
                else f"我的界面未打开，第 {attempt}/{max_open_attempts} 次重新点击：{format_point(*pos)}"
            ),
        )
        click_image_point(robot, *pos)
        time.sleep(0.6)
        my_image = capture_after_popups(robot, cfg, label=f"my_screen_{attempt}", reporter=reporter)
        _raise_if_black_screen(my_image, cfg)
        if _matches_screen(my_image, cfg, MY_SCREEN):
            break
    if my_image is None or not _matches_screen(my_image, cfg, MY_SCREEN):
        raise StepUnavailable(
            f"点击我的 {max_open_attempts} 次后仍未匹配到“我的界面”模板："
            f"{_template_failure_detail(cfg, MY_SCREEN, group='screens')}"
        )
    image = my_image

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
        _report(reporter, "message_button_click", "读取积分后未找到消息位置，不触发复位")
        return points or None
    time.sleep(0.6)
    try:
        image = capture_after_popups(robot, cfg, label="after_points_message", reporter=reporter)
        _raise_if_black_screen(image, cfg)
        if not _matches_screen(image, cfg, CHAT_LIST_SCREEN):
            raise StepUnavailable(
                f"读取积分后未能返回聊天列表，未匹配到模板："
                f"{_template_failure_detail(cfg, CHAT_LIST_SCREEN, group='screens')}"
            )
    except Exception as exc:
        # 积分已经识别成功，回到列表失败不能吞掉积分结果，主 UI 仍需显示余额。
        _report(reporter, "points_return_list_failed", "积分已读取，返回聊天列表失败", log_label=str(exc))
        if not points:
            raise
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
        raise StepUnavailable(f"未匹配到聊天界面模板：{_template_failure_detail(cfg, CHAT_WINDOW_SCREEN, group='screens')}")
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
        raise StepUnavailable(f"发送新招呼后未找到返回按钮模板：{_template_failure_detail(cfg, BACK_BUTTON_2, group='buttons')}")
    _report(
        reporter,
        "new_greeting_return",
        "新招呼后返回列表",
        log_label=f"新招呼后返回列表：{format_point(*back_pos)}",
    )
    click_image_point(robot, *back_pos)
    time.sleep(0.4)
    return True


def send_photo_to_online_target(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> bool:
    """Open the online tab, enter one person, and send a randomly selected photo."""
    cfg = _config(config)
    image = ensure_list_screen(robot, cfg, reporter=reporter)
    if image is None:
        return False

    _report(reporter, "online_scan", "无消息，扫描在线入口")
    ocr_result = _run_list_ocr(image, cfg, reporter=reporter)
    result = ocr_result.get("result", {}) if isinstance(ocr_result, dict) else {}
    bottom_nav = result.get("bottom_nav", []) if isinstance(result, dict) else []
    online_item = _bottom_nav_item(bottom_nav, ("在线",))
    if online_item is None:
        _report(reporter, "online_skip", "未识别到在线入口，本轮不主动发图")
        return False

    nav_item, online_center = online_item
    bounds = nav_item.get("bounds")
    has_red = False
    red_detail = ""
    if isinstance(bounds, list) and len(bounds) == 4:
        red_stats = _red_pixel_stats(image, _match_result_from_bounds(bounds))
        has_red = bool(red_stats.get("has_red"))
        red_detail = (
            f"红色像素={red_stats.get('red_pixels', 0)}/{red_stats.get('total_pixels', 0)}，"
            f"阈值={red_stats.get('threshold', 0)}，区域={red_stats.get('region')}"
        )

    if has_red:
        _report(
            reporter,
            "online_open",
            "在线入口已有红色，直接读取在线列表",
            log_label=f"在线入口已有红色，直接读取在线列表：{format_point(*online_center)}，{red_detail}",
        )
        online_image = image
    else:
        _report(
            reporter,
            "online_open",
            "在线入口无红色，点击切换",
            log_label=f"在线入口无红色，点击切换：{format_point(*online_center)}，{red_detail}",
        )
        click_image_point(robot, *online_center)
        time.sleep(0.5)
        online_image = capture_after_popups(robot, cfg, label="online_list", reporter=reporter)

    online_ocr = _run_list_ocr(online_image, cfg, reporter=reporter)
    online_nav = online_ocr.get("result", {}).get("bottom_nav", [])
    online_state = _bottom_nav_item(online_nav, ("在线",)) if isinstance(online_nav, list) else None
    if online_state is not None:
        state_item, _ = online_state
        state_bounds = state_item.get("bounds")
        if isinstance(state_bounds, list) and len(state_bounds) == 4:
            after_red = _red_pixel_stats(online_image, _match_result_from_bounds(state_bounds))
            _report(
                reporter,
                "online_open",
                "已确认在线列表" if after_red.get("has_red") else "在线切换后未检测到红色，继续读取列表",
                log_label=(
                    "点击在线后已确认在线列表：" if after_red.get("has_red")
                    else "点击在线后未检测到在线红色，仍尝试读取列表："
                ) + f"{after_red.get('red_pixels', 0)}/{after_red.get('total_pixels', 0)}",
            )

    excluded_contacts = _photo_sent_contacts(cfg)
    if excluded_contacts:
        _report(reporter, "online_filter", "过滤冷却中的已发照片用户", log_label=f"过滤冷却中的已发照片用户：{len(excluded_contacts)} 位")

    max_swipes = max(0, int(cfg.get("online_list_max_swipes", 8) or 0))
    seen_pages: set[str] = set()
    target: dict[str, Any] | None = None
    for swipe_index in range(max_swipes + 1):
        rows = online_ocr.get("result", {}).get("message_list", []) if isinstance(online_ocr, dict) else []
        page_signature = _json_cache_fragment(rows)
        target = _select_online_target_from_ocr_result(online_ocr, excluded_contacts=excluded_contacts)
        if target is not None:
            break
        if swipe_index >= max_swipes:
            break
        if page_signature in seen_pages:
            _report(reporter, "online_skip", "在线列表上滑后页面未变化，停止查找")
            break
        seen_pages.add(page_signature)
        if not hasattr(robot, "swipe"):
            _report(reporter, "online_skip", "机械臂不支持上滑，无法继续查找未发照片用户")
            break
        start = cfg.get("online_list_swipe_start", [0.5, 0.72])
        end = cfg.get("online_list_swipe_end", [0.5, 0.32])
        if not isinstance(start, (list, tuple)) or not isinstance(end, (list, tuple)) or len(start) != 2 or len(end) != 2:
            raise StepUnavailable("在线列表上滑坐标配置无效，应为两个 [x, y] 坐标")
        start_x, start_y = float(start[0]), float(start[1])
        end_x, end_y = float(end[0]), float(end[1])
        _report(
            reporter,
            "online_scroll",
            "当前在线用户均已发过照片，上滑继续查找",
            log_label=f"当前在线用户均已发过照片，上滑继续查找 {swipe_index + 1}/{max_swipes}：{format_point(start_x, start_y)} -> {format_point(end_x, end_y)}",
        )
        robot.swipe(start_x, start_y, end_x, end_y)
        time.sleep(float(cfg.get("online_list_swipe_settle_seconds", 0.8) or 0.8))
        online_image = capture_after_popups(robot, cfg, label=f"online_list_scroll_{swipe_index + 1}", reporter=reporter)
        online_ocr = _run_list_ocr(online_image, cfg, reporter=reporter)
    if target is None:
        _report(
            reporter,
            "online_skip",
            "在线列表没有未发照片的用户，本轮不主动发图",
            log_label=f"在线列表没有未发照片的用户，已过滤 {len(excluded_contacts)} 个已发照片用户",
        )
        return False

    _report(
        reporter,
        "online_enter_chat",
        "进入在线用户聊天",
        log_label=f"进入在线用户聊天：{target.get('name', '')} {format_point(target['x'], target['y'])}",
    )
    click_image_point(robot, float(target["x"]), float(target["y"]))
    chat_image = confirm_chat_window(robot, cfg, reporter=reporter)

    photo_pos = _template_center(chat_image, cfg, PHOTO_BUTTON, group="buttons")
    if photo_pos is None:
        raise StepUnavailable(f"在线用户聊天界面未找到照片按钮模板：{_template_failure_detail(cfg, PHOTO_BUTTON, group='buttons')}")
    _report(reporter, "photo_open", "点击照片按钮", log_label=f"点击照片按钮：{format_point(*photo_pos)}")
    click_image_point(robot, *photo_pos)
    time.sleep(0.5)

    bounds = cfg.get("photo_select_random_bounds", [0.3, 0.2, 0.9, 0.5])
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        raise StepUnavailable("照片随机选择范围配置无效，应为 [0.3, 0.2, 0.9, 0.5]")
    left, top, right, bottom = (float(value) for value in bounds)
    if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
        raise StepUnavailable("照片随机选择范围配置无效，应满足 0<=左<右<=1 且 0<=上<下<=1")
    max_photo_attempts = max(1, int(cfg.get("photo_select_max_attempts", 3) or 3))
    complete_pos: tuple[float, float] | None = None
    for attempt in range(1, max_photo_attempts + 1):
        photo_point = (random.uniform(left, right), random.uniform(top, bottom))
        _report(
            reporter,
            "photo_select",
            "随机选择照片" if attempt == 1 else "未找到完成按钮，重新随机选择照片",
            log_label=(
                f"随机选择照片：{format_point(*photo_point)}"
                if attempt == 1
                else f"未找到完成按钮，第 {attempt}/{max_photo_attempts} 次重新随机选择照片：{format_point(*photo_point)}"
            ),
        )
        click_image_point(robot, *photo_point)
        time.sleep(0.5)
        complete_image = capture_after_popups(robot, cfg, label=f"photo_complete_{attempt}", reporter=reporter)
        complete_pos = _template_center(complete_image, cfg, COMPLETE_BUTTON, group="buttons")
        if complete_pos is not None:
            break
    if complete_pos is None:
        raise StepUnavailable(
            f"随机选择照片 {max_photo_attempts} 次后仍未找到完成按钮模板："
            f"{_template_failure_detail(cfg, COMPLETE_BUTTON, group='buttons')}"
        )
    _report(reporter, "photo_complete", "点击完成发送照片", log_label=f"点击完成发送照片：{format_point(*complete_pos)}")
    click_image_point(robot, *complete_pos)
    # “完成”即代表照片已提交；返回失败也必须保留冷却记录，避免重复发送。
    _record_photo_sent_contact(cfg, target.get("name"))
    _report(
        reporter,
        "photo_record",
        "记录已发照片用户",
        log_label=f"记录已发照片用户：{str(target.get('name', '') or '').strip()}",
    )
    time.sleep(0.6)
    # 主动发图后只需退出一次聊天；go_back 会截图确认是否已回到列表。
    go_back(robot, cfg, double_click_on_first_attempt=False, reporter=reporter)
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


def read_dialogue_continuity(image: Any, config: dict[str, Any] | None = None) -> dict[str, list[str]]:
    """Read visible dialogue bubbles solely for conversation-memory matching."""
    cfg = _config(config)
    ocr_result = _run_ocr_mode(image, cfg, "dialogue", region=None)
    dialogue_result = ocr_result.get("result", {})
    dialogues = dialogue_result.get("dialogues", []) if isinstance(dialogue_result, dict) else []
    result: dict[str, list[str]] = {"sender1": [], "sender2": []}
    for item in dialogues if isinstance(dialogues, list) else []:
        if not isinstance(item, dict):
            continue
        sender = str(item.get("sender", ""))
        content = str(item.get("content", "")).strip()
        if sender in result and content:
            result[sender].append(content)
    return {sender: contents[-3:] for sender, contents in result.items()}


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
    dialogue_mode = config.get("ocr_modes", {}).get("dialogue", {})
    dialogue_rules = dialogue_mode.get("rules", {}) if isinstance(dialogue_mode, dict) else {}
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
    continuity: dict[str, list[str]] = {"sender1": [], "sender2": []}
    for item in dialogues if isinstance(dialogues, list) else []:
        if not isinstance(item, dict):
            continue
        sender = str(item.get("sender", ""))
        dialogue_content = str(item.get("content", "")).strip()
        if sender in continuity and dialogue_content:
            continuity[sender].append(dialogue_content)
    return {
        "user": contact,
        "contact": contact,
        "content": content,
        "sender": target_sender,
        "message_time": "",
        "unread_count": str(target.get("unread_count", "")) if isinstance(target, dict) else "",
        "favorability": str(target.get("favorability", "")) if isinstance(target, dict) else "",
        "continuity": {sender: values[-3:] for sender, values in continuity.items()},
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
        _report(reporter, "input_box_locate", "使用缓存坐标", log_label=f"使用缓存坐标：输入框 {format_point(*pos)}")
    else:
        match_result = _template_match(image, cfg, INPUT_BOX_BUTTON, group="buttons")
        if match_result is None:
            raise StepUnavailable(f"未找到输入框模板：{_template_failure_detail(cfg, INPUT_BOX_BUTTON, group='buttons')}")
        pos = (
            match_result["x"] + match_result["w"] / 2,
            match_result["y"] + match_result["h"] / 2,
        )
        _cache_point_once(cfg, "template_points", "input_box", pos)
        _set_metric(cfg, "input_box_source", "template")
        _set_metric(cfg, "input_box_template", match_result.get("_template_name", ""))
        _report(
            reporter,
            "input_box_locate",
            "刷新缓存坐标",
            log_label=f"刷新缓存坐标：输入框 {format_point(*pos)}；{format_template_match_diagnostic(cfg, INPUT_BOX_BUTTON)}",
        )
    if pos is None:
        raise StepUnavailable(f"未找到输入框模板：{_template_failure_detail(cfg, INPUT_BOX_BUTTON, group='buttons')}")
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
                raise StepUnavailable(f"未找到麦克风按钮模板：{_template_failure_detail(cfg, MICROPHONE_BUTTON, group='buttons')}")
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
                raise StepUnavailable(f"未找到发送按钮模板：{_template_failure_detail(cfg, SEND_BUTTON_KEY, group='buttons')}")
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
            raise StepUnavailable(f"未找到返回按钮模板：{_template_failure_detail(cfg, BACK_BUTTON, group='buttons')}")
        if cached is None:
            _cache_point_once(cfg, "template_points", "back", pos)
            _report(
                reporter,
                "back_click",
                "刷新缓存坐标",
                log_label=f"刷新缓存坐标：返回按钮 {format_point(*pos)}；{format_template_match_diagnostic(cfg, BACK_BUTTON)}",
            )
        else:
            _report(reporter, "back_click", "使用缓存坐标", log_label=f"使用缓存坐标：返回按钮 {format_point(*pos)}")
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
    raise StepUnavailable(f"返回后仍未进入聊天列表，未匹配到模板：{_template_failure_detail(cfg, CHAT_LIST_SCREEN, group='screens')}")

