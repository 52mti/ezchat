from __future__ import annotations

from decimal import Decimal, InvalidOperation
import random
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
    CHAT_AREA_REGION,
    CHAT_LIST_SCREEN,
    CHAT_WINDOW_SCREEN,
    MESSAGE_LIST_REGION,
    NEXT_MESSAGE_BUTTON,
    SEND_BUTTON,
    all_screen_states,
    back_button_template_paths,
    popup_template_paths,
    region_rule,
    match_template_with_region,
    template_match_region,
    screen_state_path,
    template_variant_paths,
    template_path,
    template_root,
)
from vision import match, run_ocr_mode
from vision.screen import crop_region, load_image

from .config import DEFAULT_CONFIG


class StepUnavailable(RuntimeError):
    pass


class BlackScreenError(RuntimeError):
    pass


class NoPhoneScreenError(RuntimeError):
    pass


NO_PHONE_SCREEN = "screen.no_phone"
MY_SCREEN = "screen.my"
POINTS_REGION = "ocr.regions.points"
INPUT_BOX_BUTTON = "button.input_box"
MICROPHONE_BUTTON = "button.microphone"
NEW_GREETING_BUTTON = "button.new_greeting"
NEW_GREETING_SEARCH_REGION = (0.0, 0.0, 1.0, 0.85)
TTS_POST_DELAY_SECONDS = 3.0
POPUP_MAX_X = 0.9
NEXT_MESSAGE_SAME_NAME_RETRY_LIMIT = 3
CHAT_WINDOW_SCROLL_START = (0.5, 0.8)
CHAT_WINDOW_SCROLL_DISTANCE_Y = 0.6
CHAT_WINDOW_SCROLL_JITTER_MIN = 0.0
CHAT_WINDOW_SCROLL_JITTER_MAX = 0.0
CHAT_WINDOW_SCROLL_REPEAT = 1
INPUT_BOX_RETRACT_CENTER = (0.5, 0.3)
INPUT_BOX_RETRACT_JITTER = 0.02
INPUT_BOX_RETRACT_DELAY_SECONDS = 0.3
INPUT_BOX_FOCUS_SETTLE_SECONDS = 0.8

SCREEN_TEMPLATE_ALIASES: dict[str, tuple[str, ...]] = {
    CHAT_LIST_SCREEN: ("screen_聊天列表.jpg", "chat_list.jpg", "screen_message_list.jpg"),
    CHAT_WINDOW_SCREEN: ("screen_聊天界面.jpg", "chat_window.jpg"),
    NO_PHONE_SCREEN: ("screen_无手机界面.jpg",),
    MY_SCREEN: ("screen_我的界面.jpg", "my_screen.jpg"),
}

BUTTON_TEMPLATE_ALIASES: dict[str, tuple[str, ...]] = {
    BACK_BUTTON: ("btn_返回按钮.jpg", "btn_返回按钮_2.jpg", "btn_back.jpg"),
    NEXT_MESSAGE_BUTTON: ("btn_下一条按钮.jpg", "btn_下一条按钮_2.jpg", "btn_next_msg.jpg"),
    INPUT_BOX_BUTTON: ("btn_输入框.jpg", "btn_input_box.jpg"),
    MICROPHONE_BUTTON: ("btn_麦克风按钮.jpg", "btn_voice_input.jpg"),
    SEND_BUTTON: ("btn_发送按钮.jpg", "btn_send.jpg"),
    NEW_GREETING_BUTTON: ("btn_新招呼按钮.jpg",),
}

TEMPLATE_LABELS: dict[str, str] = {
    CHAT_LIST_SCREEN: "聊天列表界面",
    CHAT_WINDOW_SCREEN: "聊天界面",
    MY_SCREEN: "我的界面",
    NO_PHONE_SCREEN: "无手机界面",
    BACK_BUTTON: "返回按钮",
    NEXT_MESSAGE_BUTTON: "下一条按钮",
    INPUT_BOX_BUTTON: "输入框按钮",
    MICROPHONE_BUTTON: "麦克风按钮",
    SEND_BUTTON: "发送按钮",
    NEW_GREETING_BUTTON: "新招呼按钮",
}


def _report(reporter: Reporter | None, step: str, label: str, log_label: str | None = None) -> None:
    report_step(reporter, step, label, log_label)


def _balance_from_points(points: str) -> str:
    try:
        value = Decimal(str(points).strip()) / Decimal("10")
    except (InvalidOperation, ValueError):
        return ""
    return str(int(value))


def _config(config: dict[str, Any] | None) -> dict[str, Any]:
    return _deep_merge(DEFAULT_CONFIG, config or {})


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _threshold(config: dict[str, Any]) -> float:
    return float(config.get("match_threshold", 0.85))


def _step_timeout(config: dict[str, Any]) -> float:
    return float(config.get("step_timeout", 10))


def _run_ocr_mode(image: Any, config: dict[str, Any], mode: str, region: Any = None) -> dict[str, Any]:
    with exclude_ocr_wait_from_step_timeout(config):
        return run_ocr_mode(image, config, mode, region=region)


def _new_greeting_click_offset(config: dict[str, Any]) -> tuple[float, float]:
    x_offset = float(config.get("new_greeting_click_x_offset", 0.0) or 0.0)
    y_offset = float(config.get("new_greeting_click_y_offset", 0.0) or 0.0)
    return x_offset, y_offset


def _next_message_click_offset(config: dict[str, Any]) -> tuple[float, float]:
    x_offset = float(config.get("next_message_click_x_offset", 0.0) or 0.0)
    y_offset = float(config.get("next_message_click_y_offset", 0.0) or 0.0)
    return x_offset, y_offset


def _region(config: dict[str, Any], key: str) -> Any:
    current: Any = config
    for part in region_rule(key).config_path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


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
    if group == "buttons" and key == BACK_BUTTON:
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


def _template_names(config: dict[str, Any], key: str, *, group: str) -> str:
    candidates = _template_candidates(config, key, group=group)
    if candidates:
        return "、".join(path.name for path in candidates)
    aliases = SCREEN_TEMPLATE_ALIASES.get(key, ()) if group == "screens" else BUTTON_TEMPLATE_ALIASES.get(key, ())
    if aliases:
        return "、".join(aliases)
    return key


def _template_miss_message(config: dict[str, Any], key: str, *, group: str, action: str = "匹配") -> str:
    label = TEMPLATE_LABELS.get(key, key)
    return f"{action}失败：没有匹配到{label}，已尝试图片：{_template_names(config, key, group=group)}"


def _match_any(image: Any, paths: list[Path], config: dict[str, Any], key: str) -> dict[str, float] | None:
    for path in paths:
        result = match_template_with_region(
            image,
            path,
            config,
            key,
            threshold=_threshold(config),
            match_func=match,
        )
        if result:
            return result
    return None


def _match_any_with_path(
    image: Any,
    paths: list[Path],
    config: dict[str, Any],
    key: str,
) -> tuple[dict[str, float], Path] | None:
    for path in paths:
        result = match_template_with_region(
            image,
            path,
            config,
            key,
            threshold=_threshold(config),
            match_func=match,
        )
        if result:
            return result, path
    return None


def _template_match(image: Any, config: dict[str, Any], key: str, *, group: str) -> dict[str, float] | None:
    return _match_any(image, _template_candidates(config, key, group=group), config, key)


def _template_match_with_path(
    image: Any,
    config: dict[str, Any],
    key: str,
    *,
    group: str,
) -> tuple[dict[str, float], Path] | None:
    return _match_any_with_path(image, _template_candidates(config, key, group=group), config, key)


def _template_center(image: Any, config: dict[str, Any], key: str, *, group: str) -> tuple[float, float] | None:
    result = _template_match(image, config, key, group=group)
    if not result:
        return None
    return result["x"] + result["w"] / 2, result["y"] + result["h"] / 2


def _template_center_in_region(
    image: Any,
    config: dict[str, Any],
    key: str,
    *,
    group: str,
    region: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    configured_region = template_match_region(config, key)
    if configured_region is not None:
        region = configured_region
    local_config = dict(config)
    local_config.pop("template_match_regions", None)
    result = _match_any(
        crop_region(image, region),
        _template_candidates(config, key, group=group),
        local_config,
        key,
    )
    if not result:
        return None
    region_x, region_y, region_w, region_h = region
    center_x = region_x + (result["x"] + result["w"] / 2) * region_w
    center_y = region_y + (result["y"] + result["h"] / 2) * region_h
    return center_x, center_y


def _template_center_with_path(
    image: Any,
    config: dict[str, Any],
    key: str,
    *,
    group: str,
) -> tuple[float, float, Path] | None:
    matched = _template_match_with_path(image, config, key, group=group)
    if not matched:
        return None
    result, path = matched
    return result["x"] + result["w"] / 2, result["y"] + result["h"] / 2, path


def _matches_screen(image: Any, config: dict[str, Any], key: str) -> bool:
    return _template_match(image, config, key, group="screens") is not None


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


def _raise_if_abnormal_screen(image: Any, config: dict[str, Any]) -> None:
    if _matches_screen(image, config, NO_PHONE_SCREEN):
        raise NoPhoneScreenError(f"检测到无手机界面：{_template_names(config, NO_PHONE_SCREEN, group='screens')}")
    if _looks_like_black_screen(image, config):
        raise BlackScreenError("检测到黑屏：画面亮度过低，手机可能息屏或未亮屏")


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
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        if not name or not any(keyword in name for keyword in keywords):
            continue
        bounds = item.get("bounds")
        if not isinstance(bounds, list) or len(bounds) != 4:
            continue
        center = _center_from_bounds(bounds)
        if center[1] >= min_center_y:
            return item, center
    return None


def _click_bottom_nav_message(
    robot: Any,
    ocr_result: dict[str, Any],
    reporter: Reporter | None = None,
) -> bool:
    return _click_bottom_nav_item(
        robot,
        ocr_result,
        ("消息",),
        click_count=2,
        reporter=reporter,
        step="bottom_nav_click",
        label="点击底部消息",
    )


def _click_bottom_nav_item(
    robot: Any,
    ocr_result: dict[str, Any],
    keywords: tuple[str, ...],
    *,
    click_count: int = 1,
    reporter: Reporter | None = None,
    step: str = "bottom_nav_click",
    label: str = "点击底部导航",
    min_center_y: float = 0.5,
) -> bool:
    bottom_nav = ocr_result.get("result", {}).get("bottom_nav", [])
    if not isinstance(bottom_nav, list):
        return False
    match_item = _bottom_nav_item(bottom_nav, keywords, min_center_y=min_center_y)
    if match_item is None:
        return False
    nav_item, center = match_item
    bounds = nav_item.get("bounds")
    if not isinstance(bounds, list) or len(bounds) != 4:
        return False
    for index in range(max(1, click_count)):
        _report(
            reporter,
            step,
            label,
            log_label=f"{label}{index + 1}/{max(1, click_count)}：{format_point(*center)}",
        )
        click_image_point(robot, *center)
        time.sleep(0.15)
    return True


def _click_back_match(
    robot: Any,
    matched: tuple[float, float, Path],
    *,
    click_count: int = 1,
    reporter: Reporter | None = None,
) -> None:
    x, y, path = matched
    total_clicks = max(1, int(click_count or 1))
    for index in range(total_clicks):
        _report(
            reporter,
            "return_to_list",
            "点击返回",
            log_label=f"点击返回{index + 1}/{total_clicks}：{format_point(x, y)} [{path.name}]",
        )
        click_image_point(robot, x, y)
        time.sleep(0.4)


def _popup_match(image: Any, popup_template: Path, config: dict[str, Any]) -> dict[str, float] | None:
    result = match(image, popup_template, threshold=_threshold(config))
    if not result:
        return None
    max_x = float(config.get("popup_max_x", POPUP_MAX_X) or POPUP_MAX_X)
    if float(result.get("x", 0.0)) > max_x:
        return None
    return result


def current_screen(image: Any, config: dict[str, Any]) -> str:
    for state in all_screen_states(config):
        if match_template_with_region(
            image,
            state.path,
            config,
            state.name,
            threshold=_threshold(config),
            match_func=match,
        ):
            return state.name
    if _matches_screen(image, config, CHAT_LIST_SCREEN):
        return CHAT_LIST_SCREEN
    if _matches_screen(image, config, CHAT_WINDOW_SCREEN):
        return CHAT_WINDOW_SCREEN
    if _matches_screen(image, config, MY_SCREEN):
        return MY_SCREEN
    return "unknown"


def _popup_close_point(image: Any, config: dict[str, Any]) -> tuple[float, float] | None:
    for popup_template in popup_template_paths(config):
        result = _popup_match(image, popup_template, config)
        if not result:
            continue
        return min(0.98, result["x"] + result["w"] * 0.7), max(0.02, result["y"] + result["h"] * 0.1)
    return None


def is_popup_visible(image: Any, config: dict[str, Any]) -> bool:
    return _popup_close_point(image, config) is not None


def dismiss_popups(
    robot: Any,
    config: dict[str, Any],
    image: Any | None = None,
    reporter: Reporter | None = None,
) -> bool:
    current_image = image if image is not None else capture_transform_image(
        robot, config, pre_move=True, label="popup_check"
    )
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
    reporter: Reporter | None = None,
) -> Any:
    image = capture_transform_image(robot, config, pre_move=True, label=label)
    for _ in range(max_dismiss):
        close_point = _popup_close_point(image, config)
        if close_point is None:
            break
        _report(reporter, "popup_check", "关闭弹窗", log_label=f"关闭弹窗：{format_point(*close_point)}")
        try:
            click_image_point(robot, *close_point)
        except Exception:
            _report(reporter, "popup_check", "弹窗点击失败，忽略")
            break
        image = capture_transform_image(robot, config, pre_move=True, label=f"{label}_after_popup")
    return image


def _click_back_once(
    robot: Any,
    config: dict[str, Any],
    image: Any | None = None,
    reporter: Reporter | None = None,
) -> bool:
    current_image = image if image is not None else capture_after_popups(robot, config, label="back_check", reporter=reporter)
    matched = _template_center_with_path(current_image, config, BACK_BUTTON, group="buttons")
    if matched is None:
        _report(reporter, "return_to_list", "返回按钮未找到")
        return False
    _click_back_match(robot, matched, click_count=1, reporter=reporter)
    return True


def scroll_chat_window(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    cfg = _config(config)
    base_start_x, base_start_y = cfg.get("chat_window_scroll_start", CHAT_WINDOW_SCROLL_START)
    distance_y = float(cfg.get("chat_window_scroll_distance_y", CHAT_WINDOW_SCROLL_DISTANCE_Y) or CHAT_WINDOW_SCROLL_DISTANCE_Y)
    repeat = max(1, int(cfg.get("chat_window_scroll_repeat", CHAT_WINDOW_SCROLL_REPEAT) or CHAT_WINDOW_SCROLL_REPEAT))
    jitter_min = float(cfg.get("chat_window_scroll_jitter_min", CHAT_WINDOW_SCROLL_JITTER_MIN) or CHAT_WINDOW_SCROLL_JITTER_MIN)
    jitter_max = float(cfg.get("chat_window_scroll_jitter_max", CHAT_WINDOW_SCROLL_JITTER_MAX) or CHAT_WINDOW_SCROLL_JITTER_MAX)
    first_start: tuple[float, float] | None = None
    last_end: tuple[float, float] | None = None
    if not hasattr(robot, "swipe"):
        raise StepUnavailable("机械臂不支持滑动操作：缺少 swipe 方法")
    for index in range(repeat):
        jitter_x = random.uniform(jitter_min, jitter_max)
        jitter_y = random.uniform(jitter_min, jitter_max)
        start_x = min(0.98, max(0.02, float(base_start_x) + jitter_x))
        start_y = min(0.98, max(0.02, float(base_start_y) + jitter_y))
        end_x = start_x
        end_y = max(0.02, start_y - distance_y)
        if first_start is None:
            first_start = (start_x, start_y)
        last_end = (end_x, end_y)
        _report(
            reporter,
            "chat_window_scroll",
            "进入聊天后上滑",
            log_label=f"进入聊天后上滑{index + 1}/{repeat}：{format_point(start_x, start_y)} -> {format_point(end_x, end_y)}",
        )
        robot.swipe(start_x, start_y, end_x, end_y)
        time.sleep(float(cfg.get("chat_window_scroll_between_seconds", 0.25) or 0.25))
    time.sleep(float(cfg.get("chat_window_scroll_delay_seconds", 1.0) or 1.0))
    return (
        first_start if first_start is not None else CHAT_WINDOW_SCROLL_START,
        last_end if last_end is not None else CHAT_WINDOW_SCROLL_START,
    )


def ensure_list_screen(robot: Any, config: dict[str, Any] | None = None, reporter: Reporter | None = None) -> Any | None:
    cfg = _config(config)
    _report(reporter, "screen_detect", "确认欢友聊天列表")
    image = capture_after_popups(robot, cfg, label="confirm_list", reporter=reporter)
    _raise_if_abnormal_screen(image, cfg)
    if _matches_screen(image, cfg, CHAT_LIST_SCREEN):
        return image

    if _matches_screen(image, cfg, CHAT_WINDOW_SCREEN):
        if not _click_back_once(robot, cfg, image, reporter=reporter):
            return None
        time.sleep(0.4)
        image = capture_after_popups(robot, cfg, label="after_return_from_chat", reporter=reporter)
        _raise_if_abnormal_screen(image, cfg)
        if _matches_screen(image, cfg, CHAT_LIST_SCREEN):
            return image

    if _matches_screen(image, cfg, MY_SCREEN):
        if not _click_back_once(robot, cfg, image, reporter=reporter):
            return None
        time.sleep(0.4)
        image = capture_after_popups(robot, cfg, label="after_return_from_my", reporter=reporter)
        _raise_if_abnormal_screen(image, cfg)
        if _matches_screen(image, cfg, CHAT_LIST_SCREEN):
            return image

    _report(reporter, "unknown_screen_recover", "未知界面，点击返回恢复")
    if not _click_back_once(robot, cfg, image, reporter=reporter):
        return None
    time.sleep(0.4)
    image = capture_after_popups(robot, cfg, label="after_unknown_back", reporter=reporter)
    _raise_if_abnormal_screen(image, cfg)
    if _matches_screen(image, cfg, CHAT_LIST_SCREEN):
        return image
    return None


def _favorability_value(row: dict[str, Any]) -> int:
    favorability = row.get("favorability")
    try:
        return int(favorability) if favorability is not None else -1
    except (TypeError, ValueError):
        return -1


def _unread_rank(row: dict[str, Any]) -> int:
    try:
        return 1 if int(row.get("unread_count") or 0) > 0 else 0
    except (TypeError, ValueError):
        return 0


def find_unread_target(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> dict[str, Any] | None:
    cfg = _config(config)
    image = ensure_list_screen(robot, cfg, reporter=reporter)
    if image is None:
        return None

    _report(reporter, "chat_list_ocr", "列表 OCR 扫描")
    ocr_result = run_unread_list_ocr(image, cfg)
    clicked_bottom_nav = _click_bottom_nav_message(robot, ocr_result, reporter=reporter)
    if clicked_bottom_nav:
        time.sleep(0.3)
        image = capture_after_popups(robot, cfg, label="after_bottom_nav_message", reporter=reporter)
        _report(reporter, "chat_list_ocr", "列表 OCR 复扫")
        ocr_result = run_unread_list_ocr(image, cfg)

    return _select_unread_target_from_ocr_result(ocr_result, reporter=reporter)


def run_unread_list_ocr(image: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = _config(config)
    return _run_ocr_mode(image, cfg, "list", region=_region(cfg, MESSAGE_LIST_REGION))


def _select_unread_target_from_ocr_result(
    ocr_result: dict[str, Any],
    reporter: Reporter | None = None,
) -> dict[str, Any] | None:
    
    rows = ocr_result["result"].get("message_list", [])
    candidates: list[tuple[int, int, int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        bounds = row.get("bounds")
        if not isinstance(bounds, list) or len(bounds) != 4:
            continue
        if _unread_rank(row) <= 0:
            continue
        candidates.append((_favorability_value(row), _unread_rank(row), -index, row))
    if not candidates:
        return None

    _favorability, _unread, _position, selected_row = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    left, top, right, bottom = selected_row["bounds"]
    center_x, center_y = _center_from_bounds([left, top, right, bottom])
    target = {
        "name": selected_row.get("name", ""),
        "content": selected_row.get("content", ""),
        "favorability": selected_row.get("favorability"),
        "unread_count": selected_row.get("unread_count"),
        "bounds": selected_row.get("bounds", [left, top, right, bottom]),
        "x": center_x,
        "y": center_y,
    }
    _report(
        reporter,
        "target_selected",
        "选择好感度最高目标",
        log_label=f"选择好感度最高目标：{target.get('name', '')} {format_point(center_x, center_y)}",
    )
    return target


def rescan_unread_target(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> dict[str, Any] | None:
    cfg = _config(config)
    _report(reporter, "chat_list_ocr", "列表 OCR 复扫")
    image = ensure_list_screen(robot, cfg, reporter=reporter)
    if image is None:
        return None
    ocr_result = run_unread_list_ocr(image, cfg)
    return _select_unread_target_from_ocr_result(ocr_result, reporter=reporter)


def find_unread(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> tuple[float, float] | None:
    target = find_unread_target(robot, config, reporter=reporter)
    if target is None:
        return None
    return float(target["x"]), float(target["y"])


def read_my_points(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> str | None:
    cfg = _config(config)
    image = ensure_list_screen(robot, cfg, reporter=reporter)
    if image is None:
        return None

    _report(reporter, "points_open_my", "打开我的界面")
    list_ocr_result = _run_ocr_mode(image, cfg, "list", region=_region(cfg, MESSAGE_LIST_REGION))
    clicked_my = _click_bottom_nav_item(
        robot,
        list_ocr_result,
        ("我的",),
        reporter=reporter,
        step="points_open_my",
        label="打开我的界面",
    )
    if not clicked_my:
        raise StepUnavailable("打开我的界面失败：list OCR 没有识别到底部导航“我的”")
    time.sleep(0.6)

    image = capture_after_popups(robot, cfg, label="my_screen", reporter=reporter)
    _report(reporter, "points_ocr_read", "积分 OCR 读取")
    ocr_result = _run_ocr_mode(image, cfg, "points", region=_region(cfg, POINTS_REGION))
    result = ocr_result.get("result", {}) if isinstance(ocr_result.get("result", {}), dict) else {}
    points = str(result.get("points") or result.get("balance") or "").strip()
    _report(
        reporter,
        "points_ocr_read",
        "积分读取完成" if points else "积分未识别",
        log_label=(
            f"积分读取完成：{points}，主UI余额：{_balance_from_points(points)}"
            if points
            else "积分未识别"
        ),
    )

    my_ocr_result = _run_ocr_mode(image, cfg, "list", region=_region(cfg, MESSAGE_LIST_REGION))
    clicked_message = _click_bottom_nav_item(
        robot,
        my_ocr_result,
        ("消息",),
        reporter=reporter,
        step="points_return_message",
        label="积分读取后返回消息",
    )
    if not clicked_message:
        # If the selected "我的" page OCR misses the bottom tabs, reuse the last
        # successful list-mode nav coordinates rather than relying on templates.
        clicked_message = _click_bottom_nav_item(
            robot,
            list_ocr_result,
            ("消息",),
            reporter=reporter,
            step="points_return_message",
            label="积分读取后返回消息",
        )
    if not clicked_message:
        raise StepUnavailable("积分读取后返回消息失败：list OCR 没有识别到底部导航“消息”")
    time.sleep(0.6)
    capture_after_popups(robot, cfg, label="after_points_message", reporter=reporter)
    return points or None


def keep_screen_awake(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> bool:
    cfg = _config(config)
    image = ensure_list_screen(robot, cfg, reporter=reporter)
    if image is None:
        return False

    _report(reporter, "keep_awake_scan", "防熄屏扫描底部消息")
    ocr_result = _run_ocr_mode(image, cfg, "list", region=_region(cfg, MESSAGE_LIST_REGION))
    clicked = _click_bottom_nav_item(
        robot,
        ocr_result,
        ("消息",),
        reporter=reporter,
        step="bottom_nav_click",
        label="点击底部消息",
    )
    if clicked:
        _report(reporter, "keep_awake_wait", "防熄屏等待下一轮")
    return clicked


def enter_chat(robot: Any, x: float, y: float) -> None:
    click_image_point(robot, x, y)


def confirm_chat_window(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> Any:
    cfg = _config(config)
    _report(reporter, "chat_window_confirm", "确认聊天界面")
    image = capture_after_popups(robot, cfg, label="chat_window", reporter=reporter)
    if not _matches_screen(image, cfg, CHAT_WINDOW_SCREEN):
        raise StepUnavailable(_template_miss_message(cfg, CHAT_WINDOW_SCREEN, group="screens"))
    return image


def click_new_greeting(
    robot: Any,
    config: dict[str, Any] | None = None,
    image: Any | None = None,
    reporter: Reporter | None = None,
) -> bool:
    cfg = _config(config)
    current_image = image if image is not None else capture_after_popups(robot, cfg, label="new_greeting", reporter=reporter)
    pos = _template_center_in_region(
        current_image,
        cfg,
        NEW_GREETING_BUTTON,
        group="buttons",
        region=NEW_GREETING_SEARCH_REGION,
    )
    if pos is None:
        return False
    offset_x, offset_y = _new_greeting_click_offset(cfg)
    click_point = (min(0.98, max(0.02, pos[0] + offset_x)), min(0.98, max(0.02, pos[1] + offset_y)))
    _report(reporter, "new_greeting_click", "点击新招呼", log_label=f"点击新招呼：{format_point(*click_point)}")
    click_image_point(robot, *click_point)
    time.sleep(0.4)
    return True


def click_next_message(
    robot: Any,
    config: dict[str, Any] | None = None,
    image: Any | None = None,
    reporter: Reporter | None = None,
) -> bool:
    cfg = _config(config)
    current_image = image if image is not None else capture_after_popups(robot, cfg, label="next_chat", reporter=reporter)
    pos = _template_center(current_image, cfg, NEXT_MESSAGE_BUTTON, group="buttons")
    if pos is None:
        return False
    offset_x, offset_y = _next_message_click_offset(cfg)
    click_point = (min(0.98, pos[0] + offset_x), min(0.98, max(0.02, pos[1] + offset_y)))
    _report(reporter, "next_chat_click", "点击下一条", log_label=f"点击下一条：{format_point(*click_point)}")
    click_image_point(robot, *click_point)
    time.sleep(0.6)
    return True


def read_chat_contact_identity(
    robot: Any,
    config: dict[str, Any] | None = None,
    image: Any | None = None,
    reporter: Reporter | None = None,
) -> tuple[str, tuple[float, float] | None]:
    cfg = _config(config)
    current_image = image if image is not None else confirm_chat_window(robot, cfg, reporter=reporter)
    _report(reporter, "dialogue_ocr_chat_name", "识别当前聊天对象")
    ocr_result = _run_ocr_mode(current_image, cfg, "dialogue", region=_region(cfg, CHAT_AREA_REGION))
    dialogue_result = ocr_result["result"]
    current_name = str(dialogue_result.get("user_name") or "").strip()
    dialogue_rules = cfg["ocr_modes"]["dialogue"]["rules"]
    top_ratio = float(dialogue_rules.get("user_name_top_ratio", 0.12))
    center_x_min = float(dialogue_rules.get("user_name_center_x_min_ratio", 0.0))
    center_x_max = float(dialogue_rules.get("user_name_center_x_max_ratio", 1.0))
    normalized_current_name = re.sub(r"\s+", "", current_name)
    name_candidates: list[tuple[int, float, float, str, tuple[float, float]]] = []
    for item in ocr_result.get("ocr_items", []):
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        item_x = float(item.get("x", 0.0) or 0.0)
        item_y = float(item.get("y", 0.0) or 0.0)
        item_w = float(item.get("w", 0.0) or 0.0)
        item_h = float(item.get("h", 0.0) or 0.0)
        center_x = item_x + item_w / 2
        center_y = item_y + item_h / 2
        if center_y > top_ratio or center_x < center_x_min or center_x > center_x_max:
            continue
        normalized_text = re.sub(r"\s+", "", text)
        same_name = bool(normalized_current_name and normalized_text == normalized_current_name)
        if not same_name and current_name:
            continue
        name_candidates.append((0 if same_name else 1, center_y, abs(center_x - 0.5), text, (center_x, center_y)))
    if name_candidates:
        _score, _center_y, _center_gap, text, point = sorted(name_candidates)[0]
        return current_name or text, point
    if current_name:
        return current_name, None
    items = ocr_result.get("ocr_items", [])
    if items:
        return str(items[0].get("text", "")).strip(), None
    return "", None


def read_chat_contact_name(
    robot: Any,
    config: dict[str, Any] | None = None,
    image: Any | None = None,
    reporter: Reporter | None = None,
) -> str:
    name, _point = read_chat_contact_identity(robot, config, image=image, reporter=reporter)
    return name


def read_chat_contact_name_point(
    robot: Any,
    config: dict[str, Any] | None = None,
    image: Any | None = None,
    reporter: Reporter | None = None,
) -> tuple[float, float] | None:
    _name, point = read_chat_contact_identity(robot, config, image=image, reporter=reporter)
    return point


def click_next_message_until_changed(
    robot: Any,
    config: dict[str, Any] | None = None,
    expected_name: str | None = None,
    image: Any | None = None,
    reporter: Reporter | None = None,
    *,
    retry_limit: int | None = None,
) -> tuple[Any, str, int, tuple[float, float] | None]:
    cfg = _config(config)
    current_image = image if image is not None else capture_after_popups(robot, cfg, label="next_chat", reporter=reporter)
    expected = re.sub(r"\s+", "", str(expected_name or "")).strip()
    limit = max(1, int(retry_limit or cfg.get("next_message_same_name_retry_limit", NEXT_MESSAGE_SAME_NAME_RETRY_LIMIT) or NEXT_MESSAGE_SAME_NAME_RETRY_LIMIT))
    total_clicks = 0
    last_name = ""
    last_name_point: tuple[float, float] | None = None
    _report(
        reporter,
        "next_chat_verify",
        "校验下一条是否切人",
        log_label=f"校验下一条是否切人：{expected}" if expected else "校验下一条是否切人",
    )
    for attempt in range(limit):
        if not click_next_message(robot, cfg, image=current_image, reporter=reporter):
            raise StepUnavailable(_template_miss_message(cfg, NEXT_MESSAGE_BUTTON, group="buttons", action="查找"))
        total_clicks += 1
        current_image = confirm_chat_window(robot, cfg, reporter=reporter)
        last_name, last_name_point = read_chat_contact_identity(robot, cfg, image=current_image, reporter=reporter)
        normalized_name = re.sub(r"\s+", "", str(last_name or "")).strip()
        if not expected or not normalized_name or normalized_name != expected:
            _report(
                reporter,
                "next_chat_verified",
                "下一条已切到新联系人",
                log_label=f"下一条已切到新联系人：{last_name or normalized_name}",
            )
            return current_image, last_name, total_clicks, last_name_point
        if attempt < limit - 1:
            _report(
                reporter,
                "next_chat_retry",
                "下一条仍是同一个人，继续点击",
                log_label=f"下一条仍是同一个人：{last_name}",
            )
    raise StepUnavailable(f"连续点击下一条 {limit} 次后仍未切换联系人")


def consume_new_greetings(
    robot: Any,
    config: dict[str, Any] | None = None,
    image: Any | None = None,
    expected_name: str | None = None,
    reporter: Reporter | None = None,
) -> tuple[Any, int, bool]:
    cfg = _config(config)
    current_image = image if image is not None else confirm_chat_window(robot, cfg, reporter=reporter)
    max_rounds = max(1, int(cfg.get("new_greeting_max_rounds", 10) or 10))
    handled = 0
    stopped_without_next = False
    for _ in range(max_rounds):
        if not click_new_greeting(robot, cfg, image=current_image, reporter=reporter):
            return current_image, handled, stopped_without_next
        handled += 1
        current_image = capture_after_popups(robot, cfg, label="after_new_greeting", reporter=reporter)
        if expected_name:
            try:
                current_image, _current_name, _clicks, _name_point = click_next_message_until_changed(
                    robot,
                    cfg,
                    expected_name=expected_name,
                    image=current_image,
                    reporter=reporter,
                )
            except StepUnavailable as exc:
                if "下一条按钮" not in str(exc):
                    raise
                stopped_without_next = True
                _report(
                    reporter,
                    "new_greeting_no_next",
                    "新招呼后未找到下一条，返回聊天列表",
                )
                return current_image, handled, stopped_without_next
        else:
            if not click_next_message(robot, cfg, image=current_image, reporter=reporter):
                stopped_without_next = True
                return current_image, handled, stopped_without_next
            current_image = confirm_chat_window(robot, cfg, reporter=reporter)
    return current_image, handled, stopped_without_next


def read_message(
    robot: Any,
    config: dict[str, Any] | None = None,
    target: dict[str, Any] | None = None,
    image: Any | None = None,
    reporter: Reporter | None = None,
) -> dict[str, str]:
    cfg = _config(config)
    current_image = image if image is not None else confirm_chat_window(robot, cfg, reporter=reporter)
    _report(reporter, "dialogue_ocr_read", "聊天详情 OCR")
    ocr_result = _run_ocr_mode(current_image, cfg, "dialogue", region=_region(cfg, CHAT_AREA_REGION))
    items = ocr_result["ocr_items"]
    if not items:
        raise StepUnavailable("聊天详情 OCR 没有识别到任何文字")

    dialogue_result = ocr_result["result"]
    dialogues = dialogue_result.get("dialogues", [])
    dialogue_rules = cfg["ocr_modes"]["dialogue"]["rules"]
    target_sender = str(dialogue_rules["target_sender"])
    trailing_target_messages: list[str] = []
    for item in reversed(dialogues):
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
    contact = fallback_contact or str(dialogue_result.get("user_name") or items[0]["text"])
    content = "".join(reversed(trailing_target_messages))
    if not content:
        content = "你好"
        _report(
            reporter,
            "dialogue_ocr_fallback",
            "未识别到对方消息，按“你好”处理",
            log_label=f"未识别到对方消息，按“你好”处理：目标 sender={target_sender}",
        )
    return {
        "user": contact,
        "contact": contact,
        "content": content,
        "sender": target_sender,
        "message_time": "",
        "unread_count": str(target.get("unread_count", "")) if isinstance(target, dict) else "",
        "favorability": str(target.get("favorability", "")) if isinstance(target, dict) else "",
    }


def focus_input_box(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> None:
    cfg = _config(config)
    image = capture_after_popups(robot, cfg, label="input_box", reporter=reporter)
    pos = _template_center(image, cfg, INPUT_BOX_BUTTON, group="buttons")
    if pos is None:
        raise StepUnavailable(_template_miss_message(cfg, INPUT_BOX_BUTTON, group="buttons", action="查找"))
    click_point = (pos[0] - 0.5, pos[1])
    _report(reporter, "input_box_click", "点击输入框偏移位置", log_label=f"点击输入框偏移位置：{format_point(*click_point)}")
    click_image_point(robot, *click_point)
    time.sleep(float(cfg.get("input_box_focus_settle_seconds", INPUT_BOX_FOCUS_SETTLE_SECONDS) or INPUT_BOX_FOCUS_SETTLE_SECONDS))


def play_voice_reply(
    robot: Any,
    text: str,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> None:
    cfg = _config(config)
    retry_count = max(0, int(cfg.get("tts_retry_attempts", 1) or 0))
    last_error: Exception | None = None
    for attempt in range(retry_count + 1):
        deadline = step_deadline(cfg, _step_timeout(cfg))
        while True:
            image = capture_after_popups(robot, cfg, label="microphone", reporter=reporter)
            microphone = _template_center(image, cfg, MICROPHONE_BUTTON, group="buttons")
            if microphone is not None:
                label = "点击麦克风" if attempt == 0 else "重新点击麦克风"
                _report(reporter, "microphone_click", label, log_label=f"{label}：{format_point(*microphone)}")
                click_image_point(robot, *microphone)
                break
            if deadline.expired():
                _report(
                    reporter,
                    "microphone_not_found",
                    "麦克风按钮未找到",
                    log_label=f"麦克风按钮未找到：等待{deadline.elapsed_seconds():.1f}s，超时{_step_timeout(cfg):.1f}s",
                )
                raise StepUnavailable(_template_miss_message(cfg, MICROPHONE_BUTTON, group="buttons", action="查找"))
            time.sleep(0.4)

        try:
            _report(reporter, "tts_playback", "播放 TTS 回复" if attempt == 0 else "重新播放 TTS 回复")
            tts_input(robot, text)
            time.sleep(TTS_POST_DELAY_SECONDS)
            return
        except OperationCancelled:
            raise
        except Exception as exc:
            last_error = exc
            if attempt >= retry_count:
                break
            _report(reporter, "tts_retry", "TTS 播放失败，准备重试", log_label=f"TTS 播放失败，准备重试：{exc}")
            time.sleep(0.4)
    raise StepUnavailable(f"TTS 播放失败，重试后仍失败：{last_error}") from last_error


def input_reply(
    robot: Any,
    text: str,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> None:
    focus_input_box(robot, config, reporter=reporter)
    play_voice_reply(robot, text, config, reporter=reporter)


def retract_input_box(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
    click_point: tuple[float, float] | None = None,
) -> tuple[float, float]:
    cfg = _config(config)
    if click_point is not None:
        click_x = min(0.98, max(0.02, float(click_point[0])))
        click_y = min(0.98, max(0.02, float(click_point[1])))
        label = "收回输入框：点击名字位置"
    else:
        center_x, center_y = INPUT_BOX_RETRACT_CENTER
        jitter = float(cfg.get("input_box_retract_jitter", INPUT_BOX_RETRACT_JITTER) or INPUT_BOX_RETRACT_JITTER)
        offset_x = random.uniform(-jitter, jitter)
        offset_y = random.uniform(-jitter, jitter)
        click_x = min(0.98, max(0.02, center_x + offset_x))
        click_y = min(0.98, max(0.02, center_y + offset_y))
        label = "收回输入框"
    _report(
        reporter,
        "input_box_retract",
        "收回输入框",
        log_label=f"{label}：{format_point(click_x, click_y)}",
    )
    click_image_point(robot, click_x, click_y)
    time.sleep(float(cfg.get("input_box_retract_delay_seconds", INPUT_BOX_RETRACT_DELAY_SECONDS) or INPUT_BOX_RETRACT_DELAY_SECONDS))
    return click_x, click_y


def send_reply(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> int:
    cfg = _config(config)
    deadline = step_deadline(cfg, _step_timeout(cfg))
    clicked = 0
    while True:
        image = capture_after_popups(robot, cfg, label="send_button", reporter=reporter)
        pos = _template_center(image, cfg, SEND_BUTTON, group="buttons")
        if pos is None:
            if clicked > 0:
                return clicked
            if deadline.expired():
                raise StepUnavailable(_template_miss_message(cfg, SEND_BUTTON, group="buttons", action="查找"))
            time.sleep(0.4)
            continue
        _report(reporter, "send_button_click", "点击发送按钮", log_label=f"点击发送按钮：{format_point(*pos)}")
        click_image_point(robot, *pos)
        clicked += 1
        if deadline.expired():
            raise StepUnavailable("发送按钮点击后仍未消失：可能发送失败或按钮未生效")
        time.sleep(0.4)


def ensure_send_cleared(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> int:
    cfg = _config(config)
    image = capture_after_popups(robot, cfg, label="send_clear_check", reporter=reporter)
    pos = _template_center(image, cfg, SEND_BUTTON, group="buttons")
    if pos is None:
        return 0
    _report(reporter, "send_button_click", "发送按钮仍存在，补点发送", log_label=f"发送按钮仍存在，补点发送：{format_point(*pos)}")
    click_image_point(robot, *pos)
    time.sleep(0.4)
    return 1


def go_back(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
    *,
    click_count: int = 1,
) -> None:
    cfg = _config(config)
    _report(reporter, "return_to_list", "返回聊天列表")
    ensure_send_cleared(robot, cfg, reporter=reporter)
    image = capture_after_popups(robot, cfg, label="go_back", reporter=reporter)
    if _matches_screen(image, cfg, CHAT_LIST_SCREEN):
        return

    image = capture_after_popups(robot, cfg, label="go_back_button", reporter=reporter)
    matched = _template_center_with_path(image, cfg, BACK_BUTTON, group="buttons")
    if matched is None:
        _report(reporter, "return_to_list", "返回按钮未找到")
        raise StepUnavailable(_template_miss_message(cfg, BACK_BUTTON, group="buttons", action="查找"))
    _click_back_match(robot, matched, click_count=click_count, reporter=reporter)

    image = capture_after_popups(robot, cfg, label="after_go_back", reporter=reporter)
    if not _matches_screen(image, cfg, CHAT_LIST_SCREEN):
        raise StepUnavailable(_template_miss_message(cfg, CHAT_LIST_SCREEN, group="screens"))

