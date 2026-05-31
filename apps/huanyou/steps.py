from __future__ import annotations

from decimal import Decimal, InvalidOperation
import random
import re
import time
from pathlib import Path
from typing import Any

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
    popup_template_paths,
    region_rule,
    screen_state_path,
    template_path,
    template_root,
)
from vision import match, run_ocr_mode

from .config import DEFAULT_CONFIG


class StepUnavailable(RuntimeError):
    pass


MY_SCREEN = "screen.my"
INPUT_BOX_BUTTON = "button.input_box"
MICROPHONE_BUTTON = "button.microphone"
NEW_GREETING_BUTTON = "button.new_greeting"
TTS_POST_DELAY_SECONDS = 3.0
POPUP_MAX_X = 0.9
NEXT_MESSAGE_SAME_NAME_RETRY_LIMIT = 3
CHAT_WINDOW_SCROLL_START = (0.5, 0.8)
CHAT_WINDOW_SCROLL_DISTANCE_Y = 0.6
CHAT_WINDOW_SCROLL_JITTER_MIN = 0.0
CHAT_WINDOW_SCROLL_JITTER_MAX = 0.0
CHAT_WINDOW_SCROLL_REPEAT = 1
BACK_BUTTON_DOUBLE_CLICK_NAME = "btn_返回按钮.jpg"
BACK_BUTTON_SINGLE_CLICK_NAME = "btn_返回按钮_2.jpg"
INPUT_BOX_RETRACT_CENTER = (0.5, 0.3)
INPUT_BOX_RETRACT_JITTER = 0.02
INPUT_BOX_RETRACT_DELAY_SECONDS = 0.3

SCREEN_TEMPLATE_ALIASES: dict[str, tuple[str, ...]] = {
    CHAT_LIST_SCREEN: ("screen_聊天列表.jpg", "chat_list.jpg", "screen_message_list.jpg"),
    CHAT_WINDOW_SCREEN: ("screen_聊天界面.jpg", "chat_window.jpg"),
    MY_SCREEN: ("screen_我的界面.jpg",),
}

BUTTON_TEMPLATE_ALIASES: dict[str, tuple[str, ...]] = {
    BACK_BUTTON: ("btn_返回按钮.jpg", "btn_返回按钮_2.jpg", "btn_back.jpg"),
    NEXT_MESSAGE_BUTTON: ("btn_下一条按钮.jpg", "btn_下一条按钮_2.jpg", "btn_next_msg.jpg"),
    INPUT_BOX_BUTTON: ("btn_输入框.jpg", "btn_input_box.jpg"),
    MICROPHONE_BUTTON: ("btn_麦克风按钮.jpg", "btn_voice_input.jpg"),
    SEND_BUTTON: ("btn_发送按钮.jpg", "btn_send.jpg"),
    NEW_GREETING_BUTTON: ("btn_新招呼按钮.jpg",),
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
    try:
        existing = screen_state_path(config, key) if group == "screens" else template_path(config, key)
        if existing is not None:
            candidates.append(existing)
    except KeyError:
        pass

    aliases = SCREEN_TEMPLATE_ALIASES.get(key, ()) if group == "screens" else BUTTON_TEMPLATE_ALIASES.get(key, ())
    root = template_root(config, "screens" if group == "screens" else "buttons")
    for filename in aliases:
        path = root / filename
        if path.exists():
            candidates.append(path)
    return _unique_paths(candidates)


def _match_any(image: Any, paths: list[Path], config: dict[str, Any]) -> dict[str, float] | None:
    for path in paths:
        result = match(image, path, threshold=_threshold(config))
        if result:
            return result
    return None


def _match_any_with_path(image: Any, paths: list[Path], config: dict[str, Any]) -> tuple[dict[str, float], Path] | None:
    for path in paths:
        result = match(image, path, threshold=_threshold(config))
        if result:
            return result, path
    return None


def _template_match(image: Any, config: dict[str, Any], key: str, *, group: str) -> dict[str, float] | None:
    return _match_any(image, _template_candidates(config, key, group=group), config)


def _template_match_with_path(
    image: Any,
    config: dict[str, Any],
    key: str,
    *,
    group: str,
) -> tuple[dict[str, float], Path] | None:
    return _match_any_with_path(image, _template_candidates(config, key, group=group), config)


def _template_center(image: Any, config: dict[str, Any], key: str, *, group: str) -> tuple[float, float] | None:
    result = _template_match(image, config, key, group=group)
    if not result:
        return None
    return result["x"] + result["w"] / 2, result["y"] + result["h"] / 2


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


def _back_button_click_count(path: Path) -> int:
    if path.name == BACK_BUTTON_SINGLE_CLICK_NAME:
        return 1
    if path.name == BACK_BUTTON_DOUBLE_CLICK_NAME:
        return 2
    return 2


def _click_back_match(
    robot: Any,
    matched: tuple[float, float, Path],
    reporter: Reporter | None = None,
) -> None:
    x, y, path = matched
    click_count = _back_button_click_count(path)
    for index in range(click_count):
        _report(
            reporter,
            "return_to_list",
            "点击返回",
            log_label=f"点击返回{index + 1}/{click_count}：{format_point(x, y)} [{path.name}]",
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
        if match(image, state.path, threshold=_threshold(config)):
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
        return min(0.98, result["x"] + result["w"] * 0.9), max(0.02, result["y"] + result["h"] * 0.1)
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
    _click_back_match(robot, matched, reporter=reporter)
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
        raise StepUnavailable("robot does not support swipe")
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
    if _matches_screen(image, cfg, CHAT_LIST_SCREEN):
        return image

    if _matches_screen(image, cfg, CHAT_WINDOW_SCREEN):
        if not _click_back_once(robot, cfg, image, reporter=reporter):
            return None
        time.sleep(0.4)
        image = capture_after_popups(robot, cfg, label="after_return_from_chat", reporter=reporter)
        if _matches_screen(image, cfg, CHAT_LIST_SCREEN):
            return image

    if _matches_screen(image, cfg, MY_SCREEN):
        if not _click_back_once(robot, cfg, image, reporter=reporter):
            return None
        time.sleep(0.4)
        image = capture_after_popups(robot, cfg, label="after_return_from_my", reporter=reporter)
        if _matches_screen(image, cfg, CHAT_LIST_SCREEN):
            return image

    if not _click_back_once(robot, cfg, image, reporter=reporter):
        return None
    time.sleep(0.4)
    image = capture_after_popups(robot, cfg, label="after_unknown_back", reporter=reporter)
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
    image = capture_transform_image(robot, cfg, pre_move=True, label="idle_list_rescan")
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
        raise StepUnavailable("my bottom nav was not found")
    time.sleep(0.6)

    image = capture_after_popups(robot, cfg, label="my_screen", reporter=reporter)
    _report(reporter, "points_ocr_read", "积分 OCR 读取")
    ocr_result = _run_ocr_mode(image, cfg, "points", region=None)
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
        raise StepUnavailable("message bottom nav was not found after points read")
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
    clicked = _click_bottom_nav_message(robot, ocr_result, reporter=reporter)
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
        raise StepUnavailable("chat window screen template did not match")
    return image


def click_new_greeting(
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


def read_chat_contact_name(
    robot: Any,
    config: dict[str, Any] | None = None,
    image: Any | None = None,
    reporter: Reporter | None = None,
) -> str:
    cfg = _config(config)
    current_image = image if image is not None else confirm_chat_window(robot, cfg, reporter=reporter)
    _report(reporter, "dialogue_ocr_chat_name", "识别当前聊天对象")
    ocr_result = _run_ocr_mode(current_image, cfg, "dialogue", region=_region(cfg, CHAT_AREA_REGION))
    dialogue_result = ocr_result["result"]
    current_name = str(dialogue_result.get("user_name") or "").strip()
    if current_name:
        return current_name
    items = ocr_result.get("ocr_items", [])
    if items:
        return str(items[0].get("text", "")).strip()
    return ""


def click_next_message_until_changed(
    robot: Any,
    config: dict[str, Any] | None = None,
    expected_name: str | None = None,
    image: Any | None = None,
    reporter: Reporter | None = None,
    *,
    retry_limit: int | None = None,
) -> tuple[Any, str, int]:
    cfg = _config(config)
    current_image = image if image is not None else capture_after_popups(robot, cfg, label="next_chat", reporter=reporter)
    expected = re.sub(r"\s+", "", str(expected_name or "")).strip()
    limit = max(1, int(retry_limit or cfg.get("next_message_same_name_retry_limit", NEXT_MESSAGE_SAME_NAME_RETRY_LIMIT) or NEXT_MESSAGE_SAME_NAME_RETRY_LIMIT))
    total_clicks = 0
    last_name = ""
    _report(
        reporter,
        "next_chat_verify",
        "校验下一条是否切人",
        log_label=f"校验下一条是否切人：{expected}" if expected else "校验下一条是否切人",
    )
    for attempt in range(limit):
        if not click_next_message(robot, cfg, image=current_image, reporter=reporter):
            raise StepUnavailable("next message button was not found")
        total_clicks += 1
        current_image = confirm_chat_window(robot, cfg, reporter=reporter)
        last_name = read_chat_contact_name(robot, cfg, image=current_image, reporter=reporter)
        normalized_name = re.sub(r"\s+", "", str(last_name or "")).strip()
        if not expected or not normalized_name or normalized_name != expected:
            _report(
                reporter,
                "next_chat_verified",
                "下一条已切到新联系人",
                log_label=f"下一条已切到新联系人：{last_name or normalized_name}",
            )
            return current_image, last_name, total_clicks
        if attempt < limit - 1:
            _report(
                reporter,
                "next_chat_retry",
                "下一条仍是同一个人，继续点击",
                log_label=f"下一条仍是同一个人：{last_name}",
            )
    raise StepUnavailable(f"next message did not switch contact after {limit} attempts")


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
            current_image, _current_name, _clicks = click_next_message_until_changed(
                robot,
                cfg,
                expected_name=expected_name,
                image=current_image,
                reporter=reporter,
            )
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
        raise StepUnavailable("OCR did not return chat text")

    dialogue_result = ocr_result["result"]
    dialogues = dialogue_result.get("dialogues", [])
    dialogue_rules = cfg["ocr_modes"]["dialogue"]["rules"]
    target_sender = str(dialogue_rules["target_sender"])
    target_messages = [
        str(item.get("content", "")).strip()
        for item in dialogues
        if str(item.get("sender", "")) == target_sender and str(item.get("content", "")).strip()
    ]
    if not target_messages:
        raise StepUnavailable(f"dialogue OCR did not return a {target_sender} message")

    fallback_contact = str(target.get("name", "")) if isinstance(target, dict) else ""
    contact = fallback_contact or str(dialogue_result.get("user_name") or items[0]["text"])
    return {
        "user": contact,
        "contact": contact,
        "content": target_messages[-1],
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
        raise StepUnavailable("input box button was not found")
    click_point = (pos[0] - 0.5, pos[1])
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
                raise StepUnavailable("microphone button was not found")
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
    raise StepUnavailable(f"TTS playback failed after retry: {last_error}") from last_error


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
) -> tuple[float, float]:
    cfg = _config(config)
    center_x, center_y = INPUT_BOX_RETRACT_CENTER
    jitter = float(cfg.get("input_box_retract_jitter", INPUT_BOX_RETRACT_JITTER) or INPUT_BOX_RETRACT_JITTER)
    offset_x = random.uniform(-jitter, jitter)
    offset_y = random.uniform(-jitter, jitter)
    click_x = min(0.98, max(0.02, center_x + offset_x))
    click_y = min(0.98, max(0.02, center_y + offset_y))
    _report(
        reporter,
        "input_box_retract",
        "收回输入框",
        log_label=f"收回输入框：{format_point(click_x, click_y)}",
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
                raise StepUnavailable("send button was not found")
            time.sleep(0.4)
            continue
        _report(reporter, "send_button_click", "点击发送按钮", log_label=f"点击发送按钮：{format_point(*pos)}")
        click_image_point(robot, *pos)
        clicked += 1
        if deadline.expired():
            raise StepUnavailable("send button did not disappear after click")
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
        raise StepUnavailable("back button was not found")
    _click_back_match(robot, matched, reporter=reporter)

    image = capture_after_popups(robot, cfg, label="after_go_back", reporter=reporter)
    if not _matches_screen(image, cfg, CHAT_LIST_SCREEN):
        raise StepUnavailable("failed to return to chat list")

