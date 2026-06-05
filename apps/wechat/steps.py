from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from apps.coordinates import click_image_point
from apps.photo_capture import capture_transform_image
from apps.reporting import Reporter, format_point as _format_point, report_step
from apps.step_timing import exclude_ocr_wait_from_step_timeout, step_deadline
from apps.utils import tts_input
from apps.visual_assets import (
    BACK_BUTTON,
    CHAT_AREA_REGION,
    CHAT_LIST_SCREEN,
    CHAT_WINDOW_SCREEN,
    all_screen_states,
    back_button_template_paths,
    popup_template_paths,
    region_rule,
    screen_state_path,
    template_variant_paths,
    template_path,
    template_root,
)
from vision import match, run_ocr_mode

from .config import DEFAULT_CONFIG


class StepUnavailable(RuntimeError):
    pass


INPUT_BOX_BUTTON = "button.input_box"
MICROPHONE_BUTTON = "button.microphone"

SCREEN_TEMPLATE_ALIASES: dict[str, tuple[str, ...]] = {
    CHAT_LIST_SCREEN: ("screen_聊天列表.jpg", "chat_list.jpg", "screen_message_list.jpg"),
    CHAT_WINDOW_SCREEN: ("screen_聊天界面.jpg", "chat_window.jpg"),
}

BUTTON_TEMPLATE_ALIASES: dict[str, tuple[str, ...]] = {
    INPUT_BOX_BUTTON: ("btn_输入框.jpg", "btn_输入框按钮.jpg", "btn_input_box.jpg"),
    MICROPHONE_BUTTON: ("btn_麦克风按钮.jpg", "btn_voice_input.jpg"),
    BACK_BUTTON: ("btn_返回按钮_2.jpg", "btn_返回按钮.jpg", "btn_back.jpg"),
    "button.send": ("btn_发送按钮.jpg", "btn_send.jpg"),
}
POPUP_MAX_X = 0.9

SEND_BUTTON_KEY = "button.send"
SYSTEM_TARGET_KEYWORDS = ("用户通知", "系统通知", "微信团队", "服务通知")
SYSTEM_CONTENT_KEYWORDS = ("好友申请", "已通过", "系统", "通知")
TTS_POST_DELAY_SECONDS = 3.0


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


def _popup_match(image: Any, popup_template: Path, config: dict[str, Any]) -> dict[str, float] | None:
    result = match(image, popup_template, threshold=_threshold(config))
    if not result:
        return None
    max_x = float(config.get("popup_max_x", POPUP_MAX_X) or POPUP_MAX_X)
    if float(result.get("x", 0.0)) > max_x:
        return None
    return result


def _step_timeout(config: dict[str, Any]) -> float:
    return float(config.get("step_timeout", 10))


def _run_ocr_mode(image: Any, config: dict[str, Any], mode: str, region: Any = None) -> dict[str, Any]:
    with exclude_ocr_wait_from_step_timeout(config):
        return run_ocr_mode(image, config, mode, region=region)


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
            if key in {CHAT_LIST_SCREEN, CHAT_WINDOW_SCREEN, BACK_BUTTON}:
                existing = screen_state_path(config, key) if group == "screens" else template_path(config, key)
                if existing is not None:
                    candidates.append(existing)
            else:
                existing = template_path(config, key)
                if existing is not None:
                    candidates.append(existing)
        except KeyError:
            pass
        aliases = SCREEN_TEMPLATE_ALIASES.get(key, ()) if group == "screens" else BUTTON_TEMPLATE_ALIASES.get(key, ())
        root = template_root(config, "screens" if group == "screens" else "buttons")
        for filename in aliases:
            candidates.extend(template_variant_paths(root, filename))
    return _unique_paths(candidates)


def _match_any(image: Any, paths: list[Path], config: dict[str, Any]) -> dict[str, float] | None:
    for path in paths:
        result = match(image, path, threshold=_threshold(config))
        if result:
            return result
    return None


def _template_center(image: Any, config: dict[str, Any], key: str, *, group: str) -> tuple[float, float] | None:
    result = _match_any(image, _template_candidates(config, key, group=group), config)
    if not result:
        return None
    return result["x"] + result["w"] / 2, result["y"] + result["h"] / 2


def _matches_screen(image: Any, config: dict[str, Any], key: str) -> bool:
    return _match_any(image, _template_candidates(config, key, group="screens"), config) is not None


def _report(reporter: Reporter | None, step: str, label: str, log_label: str | None = None) -> None:
    report_step(reporter, step, label, log_label)


def _looks_like_system_target(row: dict[str, Any]) -> bool:
    name = str(row.get("name", "") or "")
    content = str(row.get("content", "") or "")
    if any(keyword in name for keyword in SYSTEM_TARGET_KEYWORDS):
        return True
    return any(keyword in content for keyword in SYSTEM_CONTENT_KEYWORDS)


def _center_from_bounds(bounds: list[float]) -> tuple[float, float]:
    left, top, right, bottom = bounds
    return (left + right) / 2, (top + bottom) / 2


def _bottom_nav_center(items: list[dict[str, Any]], keywords: tuple[str, ...]) -> tuple[float, float] | None:
    for item in items:
        name = str(item.get("name", "") or "")
        if any(keyword in name for keyword in keywords):
            bounds = item.get("bounds")
            if isinstance(bounds, list) and len(bounds) == 4:
                return _center_from_bounds(bounds)
    return None


def current_screen(image: Any, config: dict[str, Any]) -> str:
    for state in all_screen_states(config):
        if match(image, state.path, threshold=_threshold(config)):
            return state.name
    if _matches_screen(image, config, CHAT_LIST_SCREEN):
        return CHAT_LIST_SCREEN
    if _matches_screen(image, config, CHAT_WINDOW_SCREEN):
        return CHAT_WINDOW_SCREEN
    return "unknown"


def _popup_close_point(image: Any, config: dict[str, Any]) -> tuple[float, float] | None:
    for popup_template in popup_template_paths(config):
        result = _popup_match(image, popup_template, config)
        if not result:
            continue
        return min(0.98, result["x"] + result["w"] * 0.9), max(0.02, result["y"] + result["h"] * 0.1)
    return None


def dismiss_popups(robot: Any, config: dict[str, Any], image: Any | None = None, reporter: Reporter | None = None) -> bool:
    current_image = image if image is not None else capture_transform_image(
        robot, config, pre_move=True, label="popup_check"
    )
    close_point = _popup_close_point(current_image, config)
    if close_point is None:
        return False
    _report(reporter, "dismiss_popup", "关闭弹窗", log_label=f"关闭弹窗：{_format_point(*close_point)}")
    try:
        click_image_point(robot, *close_point)
    except Exception:
        _report(reporter, "dismiss_popup", "弹窗点击失败，忽略")
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
    _report(reporter, "capture_screen", "复位后截图")
    image = capture_transform_image(robot, config, pre_move=True, label=label)
    for _ in range(max_dismiss):
        close_point = _popup_close_point(image, config)
        if close_point is None:
            break
        _report(reporter, "dismiss_popup", "关闭弹窗", log_label=f"关闭弹窗：{_format_point(*close_point)}")
        try:
            click_image_point(robot, *close_point)
        except Exception:
            _report(reporter, "dismiss_popup", "弹窗点击失败，忽略")
            break
        _report(reporter, "capture_screen", "复位后截图")
        image = capture_transform_image(robot, config, pre_move=True, label=f"{label}_after_popup")
    return image


def _click_back_once(
    robot: Any,
    config: dict[str, Any],
    image: Any | None = None,
    reporter: Reporter | None = None,
) -> bool:
    current_image = image if image is not None else capture_after_popups(robot, config, label="back_check", reporter=reporter)
    pos = _template_center(current_image, config, BACK_BUTTON, group="buttons")
    if pos is not None:
        _report(reporter, "back_once", "点击返回", log_label=f"点击返回：{_format_point(*pos)}")
        click_image_point(robot, *pos)
        return True
    _report(reporter, "back_once", "点击返回", log_label=f"点击返回：{_format_point(0.08, 0.06)}")
    click_image_point(robot, 0.08, 0.06)
    return True


def _ensure_list_screen(robot: Any, config: dict[str, Any], reporter: Reporter | None = None) -> Any | None:
    _report(reporter, "confirm_chat_list", "确认聊天列表")
    image = capture_after_popups(robot, config, label="ensure_list", reporter=reporter)
    if _matches_screen(image, config, CHAT_LIST_SCREEN):
        return image
    if _matches_screen(image, config, CHAT_WINDOW_SCREEN):
        go_back(robot, config, reporter=reporter)
        _report(reporter, "confirm_chat_list", "确认聊天列表")
        image = capture_after_popups(robot, config, label="ensure_list_after_chat", reporter=reporter)
        if _matches_screen(image, config, CHAT_LIST_SCREEN):
            return image
    _click_back_once(robot, config, image, reporter=reporter)
    _report(reporter, "confirm_chat_list", "确认聊天列表")
    image = capture_after_popups(robot, config, label="ensure_list_after_unknown", reporter=reporter)
    if _matches_screen(image, config, CHAT_LIST_SCREEN):
        return image
    return None


def find_unread_target(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> dict[str, Any] | None:
    cfg = _config(config)
    image = _ensure_list_screen(robot, cfg, reporter=reporter)
    if image is None:
        return None

    _report(reporter, "list_ocr_scan", "列表OCR扫描")
    ocr_result = _run_ocr_mode(image, cfg, "list", region=None)
    primary_candidates: list[dict[str, Any]] = []
    fallback_candidates: list[dict[str, Any]] = []
    for row in ocr_result["result"].get("message_list", []):
        unread_count = row.get("unread_count")
        if unread_count is None or int(unread_count) < 1:
            continue
        fallback_candidates.append(row)
        if not _looks_like_system_target(row):
            primary_candidates.append(row)
    selected_row = primary_candidates[0] if primary_candidates else (fallback_candidates[0] if fallback_candidates else None)
    if selected_row is not None:
        left, top, right, bottom = selected_row["bounds"]
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        return {
            "name": selected_row.get("name", ""),
            "content": selected_row.get("content", ""),
            "unread_count": int(selected_row.get("unread_count") or 0),
            "bounds": selected_row.get("bounds", [left, top, right, bottom]),
            "x": center_x,
            "y": center_y,
        }
    return None


def keep_screen_awake(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> bool:
    cfg = _config(config)
    image = _ensure_list_screen(robot, cfg, reporter=reporter)
    if image is None:
        raise StepUnavailable("unable to return to chat list for keep-awake flow")

    _report(reporter, "keep_awake_scan", "防熄屏扫描底部导航")
    ocr_result = _run_ocr_mode(image, cfg, "list", region=None)
    bottom_nav = ocr_result["result"].get("bottom_nav", [])
    home = _bottom_nav_center(bottom_nav, ("微信", "寰俊", "消息", "娑堟伅"))
    if home is None:
        raise StepUnavailable("wechat bottom-nav item was not found")

    _report(reporter, "keep_awake_jump_home", "防熄屏点击微信", log_label=f"防熄屏点击微信：{_format_point(*home)}")
    click_image_point(robot, *home)
    time.sleep(0.6)
    return True


def find_unread(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> tuple[float, float] | None:
    target = find_unread_target(robot, config, reporter=reporter)
    if target is None:
        return None
    return float(target["x"]), float(target["y"])


def enter_chat(robot: Any, x: float, y: float) -> None:
    click_image_point(robot, x, y)


def read_message(
    robot: Any,
    config: dict[str, Any] | None = None,
    target: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> dict[str, str]:
    cfg = _config(config)
    _report(reporter, "confirm_chat_window", "确认聊天界面")
    image = capture_after_popups(robot, cfg, label="chat_window", reporter=reporter)
    if not _matches_screen(image, cfg, CHAT_WINDOW_SCREEN):
        if _matches_screen(image, cfg, CHAT_LIST_SCREEN):
            raise StepUnavailable("chat window screen template did not match; still on chat list")
        raise StepUnavailable("chat window screen template did not match")

    _report(reporter, "dialogue_ocr_read", "对话OCR读取")
    ocr_result = _run_ocr_mode(image, cfg, "dialogue", region=_region(cfg, CHAT_AREA_REGION))
    items = ocr_result["ocr_items"]
    if not items:
        raise StepUnavailable("OCR did not return chat text")

    dialogue_result = ocr_result["result"]
    dialogues = dialogue_result.get("dialogues", [])
    sender1_messages = [
        str(item.get("content", "")).strip()
        for item in dialogues
        if str(item.get("sender", "")) == "sender1" and str(item.get("content", "")).strip()
    ]
    if not sender1_messages:
        raise StepUnavailable("dialogue OCR did not return a sender1 message")

    fallback_contact = str(target.get("name", "")) if isinstance(target, dict) else ""
    contact = dialogue_result.get("user_name") or fallback_contact or items[0]["text"]
    unread_count = ""
    if isinstance(target, dict) and target.get("unread_count") is not None:
        unread_count = str(target.get("unread_count"))
    return {
        "user": str(contact),
        "contact": str(contact),
        "content": sender1_messages[-1],
        "message_time": "",
        "unread_count": unread_count,
    }


def focus_input_box(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> None:
    cfg = _config(config)
    _report(reporter, "input_box_locate", "定位输入框")
    image = capture_after_popups(robot, cfg, label="input_box", reporter=reporter)
    pos = _template_center(image, cfg, INPUT_BOX_BUTTON, group="buttons")
    if pos is None:
        raise StepUnavailable("input box button was not found")
    click_point = (pos[0] - 0.5, pos[1] - 0.02)
    _report(reporter, "input_box_click", "点击输入框", log_label=f"点击输入框：{_format_point(*click_point)}")
    click_image_point(robot, *click_point)


def play_voice_reply(
    robot: Any,
    text: str,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> None:
    cfg = _config(config)
    deadline = step_deadline(cfg, _step_timeout(cfg))
    while True:
        _report(reporter, "microphone_locate", "定位麦克风按钮")
        image = capture_after_popups(robot, cfg, label="microphone", reporter=reporter)
        microphone = _template_center(image, cfg, MICROPHONE_BUTTON, group="buttons")
        if microphone is not None:
            _report(reporter, "microphone_click", "点击麦克风按钮", log_label=f"点击麦克风按钮：{_format_point(*microphone)}")
            click_image_point(robot, *microphone)
            break
        if deadline.expired():
            raise StepUnavailable("microphone button was not found")
        time.sleep(0.4)

    _report(reporter, "tts_playback", "播放TTS回复")
    tts_input(robot, text)
    _report(reporter, "tts_wait", "等待3秒后检测下一张图")
    time.sleep(TTS_POST_DELAY_SECONDS)


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
) -> None:
    cfg = _config(config)
    deadline = step_deadline(cfg, _step_timeout(cfg))
    clicked = False
    while True:
        _report(reporter, "send_button_check", "检测发送按钮")
        image = capture_after_popups(robot, cfg, label="send_button", reporter=reporter)
        pos = _template_center(image, cfg, SEND_BUTTON_KEY, group="buttons")
        if pos is None:
            if clicked:
                return
            if deadline.expired():
                raise StepUnavailable("send button was not found")
            time.sleep(0.4)
            continue
        if not clicked:
            _report(reporter, "send_button_click", "点击发送按钮", log_label=f"点击发送按钮：{_format_point(*pos)}")
            click_image_point(robot, *pos)
            clicked = True
        if deadline.expired():
            raise StepUnavailable("send button did not disappear after click")
        time.sleep(0.4)


def go_back(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> None:
    cfg = _config(config)
    _report(reporter, "return_to_list", "返回聊天列表")
    image = capture_after_popups(robot, cfg, label="go_back", reporter=reporter)
    if _matches_screen(image, cfg, CHAT_LIST_SCREEN):
        return

    pos = _template_center(image, cfg, BACK_BUTTON, group="buttons")
    if pos is not None:
        _report(reporter, "back_double_click", "连续点击返回两次", log_label=f"连续点击返回两次：{_format_point(*pos)}")
        click_image_point(robot, *pos)
        click_image_point(robot, *pos)
        return
    _report(reporter, "back_double_click", "连续点击返回两次", log_label=f"连续点击返回两次：{_format_point(0.08, 0.06)}")
    click_image_point(robot, 0.08, 0.06)
    click_image_point(robot, 0.08, 0.06)

