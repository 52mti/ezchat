from __future__ import annotations

from pathlib import Path
from typing import Any

from apps.coordinates import click_image_point
from apps.photo_capture import capture_transform_image
from apps.reporting import Reporter, format_point, report_step
from apps.utils import tts_input
from apps.visual_assets import (
    BACK_BUTTON,
    CHAT_AREA_REGION,
    CHAT_LIST_SCREEN,
    CHAT_WINDOW_SCREEN,
    CHAT_WINDOW_SCREEN_ROLE,
    NEXT_MESSAGE_BUTTON,
    SEND_BUTTON,
    VOICE_INPUT_BUTTON,
    all_screen_states,
    popup_template_paths,
    region_rule,
    screen_matches_role,
    screen_role_name,
    screen_state_path,
    template_path,
)
from vision import crop_region, match, run_ocr_mode

from .config import DEFAULT_CONFIG


class StepUnavailable(RuntimeError):
    pass


POPUP_SEARCH_REGION = (0.0, 0.0, 0.9, 1.0)


def _report(reporter: Reporter | None, step: str, label: str, log_label: str | None = None) -> None:
    report_step(reporter, step, label, log_label)


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


def _popup_search_region(config: dict[str, Any]) -> tuple[float, float, float, float]:
    raw = config.get("popup_search_region", POPUP_SEARCH_REGION)
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return POPUP_SEARCH_REGION
    x, y, w, h = (float(value) for value in raw)
    left = max(0.0, min(1.0, x))
    top = max(0.0, min(1.0, y))
    right = max(left, min(1.0, x + w))
    bottom = max(top, min(1.0, y + h))
    return left, top, right - left, bottom - top


def _popup_match(image: Any, popup_template: Path, config: dict[str, Any]) -> dict[str, float] | None:
    x, y, w, h = _popup_search_region(config)
    if w <= 0 or h <= 0:
        return None
    try:
        target_image = crop_region(image, (x, y, w, h))
        cropped = True
    except Exception:
        target_image = image
        cropped = False
    result = match(target_image, popup_template, threshold=_threshold(config))
    if not result:
        return None
    if not cropped:
        return result
    return {
        "x": x + result["x"] * w,
        "y": y + result["y"] * h,
        "w": result["w"] * w,
        "h": result["h"] * h,
    }


def _region(config: dict[str, Any], key: str) -> Any:
    current: Any = config
    for part in region_rule(key).config_path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _template_center(image: Any, config: dict[str, Any], key: str) -> tuple[float, float] | None:
    template = template_path(config, key)
    if template is None:
        return None
    result = match(image, template, threshold=_threshold(config))
    if not result:
        return None
    return result["x"] + result["w"] / 2, result["y"] + result["h"] / 2


def is_screen(image: Any, config: dict[str, Any], key: str) -> bool:
    template = screen_state_path(config, key)
    if template is None:
        return False
    return match(image, template, threshold=_threshold(config)) is not None


def current_screen(image: Any, config: dict[str, Any]) -> str:
    for state in all_screen_states(config):
        if match(image, state.path, threshold=_threshold(config)):
            return state.name
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
    click_image_point(robot, *close_point)
    return True


def capture_after_popups(
    robot: Any,
    config: dict[str, Any],
    max_dismiss: int = 2,
    reporter: Reporter | None = None,
) -> bytes:
    image = capture_transform_image(robot, config, pre_move=True, label="screen")
    for _ in range(max_dismiss):
        close_point = _popup_close_point(image, config)
        if close_point is None:
            break
        _report(reporter, "popup_check", "关闭弹窗", log_label=f"关闭弹窗：{format_point(*close_point)}")
        click_image_point(robot, *close_point)
        image = capture_transform_image(robot, config, pre_move=True, label="screen_after_popup")
    return image


def ensure_list_context(robot: Any, config: dict[str, Any], image: Any, reporter: Reporter | None = None) -> Any:
    screen = current_screen(image, config)
    if screen_matches_role(screen, config, CHAT_WINDOW_SCREEN_ROLE, CHAT_WINDOW_SCREEN):
        go_back(robot, config, reporter=reporter)
        return capture_after_popups(robot, config, reporter=reporter)
    return image


def find_unread(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> tuple[float, float] | None:
    cfg = _config(config)
    _report(reporter, "list_ocr_scan", "查找未读消息")
    image = ensure_list_context(robot, cfg, capture_after_popups(robot, cfg, reporter=reporter), reporter=reporter)
    pos = _template_center(image, cfg, NEXT_MESSAGE_BUTTON)
    if pos is not None:
        return pos

    message_list_region = _region(cfg, "region.message_list")
    ocr_result = run_ocr_mode(image, cfg, "list", region=message_list_region)
    for row in ocr_result["result"].get("message_list", []):
        if row.get("unread_count"):
            left, top, right, bottom = row["bounds"]
            return (left + right) / 2, (top + bottom) / 2
    items = ocr_result["ocr_items"]
    for item in items:
        if "未读" in item["text"] or item["text"].startswith("["):
            return item["x"] + item["w"] / 2, item["y"] + item["h"] / 2
    return None


def enter_chat(robot: Any, x: float, y: float) -> None:
    click_image_point(robot, x, y)


def read_message(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> dict[str, str]:
    cfg = _config(config)
    _report(reporter, "dialogue_ocr_read", "对话 OCR 读取")
    image = capture_after_popups(robot, cfg, reporter=reporter)
    chat_window_name = screen_role_name(cfg, CHAT_WINDOW_SCREEN_ROLE, CHAT_WINDOW_SCREEN)
    chat_window = screen_matches_role(current_screen(image, cfg), cfg, CHAT_WINDOW_SCREEN_ROLE, CHAT_WINDOW_SCREEN)
    if screen_state_path(cfg, chat_window_name) is not None and not chat_window:
        raise StepUnavailable("chat window screen template did not match")
    ocr_result = run_ocr_mode(image, cfg, "dialogue", region=_region(cfg, CHAT_AREA_REGION))
    items = ocr_result["ocr_items"]
    if not items:
        raise StepUnavailable("OCR did not return chat text")
    dialogue_result = ocr_result["result"]
    dialogues = dialogue_result.get("dialogues", [])
    contact = dialogue_result.get("user_name") or items[0]["text"]
    content = dialogues[-1]["content"] if dialogues else items[-1]["text"]
    return {"user": contact, "contact": contact, "content": content, "message_time": ""}


def input_reply(
    robot: Any,
    text: str,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> None:
    cfg = _config(config)
    image = capture_after_popups(robot, cfg, reporter=reporter)
    pos = _template_center(image, cfg, VOICE_INPUT_BUTTON)
    if pos is None:
        if template_path(cfg, VOICE_INPUT_BUTTON) is None:
            raise StepUnavailable("voice input template is missing")
        raise StepUnavailable("voice input button was not found")
    _report(reporter, "voice_input_click", "点击语音输入", log_label=f"点击语音输入：{format_point(*pos)}")
    click_image_point(robot, *pos)
    _report(reporter, "tts_playback", "播放 TTS 回复")
    tts_input(robot, text)


def send_reply(robot: Any, config: dict[str, Any] | None = None, reporter: Reporter | None = None) -> None:
    cfg = _config(config)
    image = capture_after_popups(robot, cfg, reporter=reporter)
    pos = _template_center(image, cfg, SEND_BUTTON)
    if pos is None:
        if template_path(cfg, SEND_BUTTON) is None:
            raise StepUnavailable("send button template is missing")
        raise StepUnavailable("send button was not found")
    _report(reporter, "send_button_click", "点击发送按钮", log_label=f"点击发送按钮：{format_point(*pos)}")
    click_image_point(robot, *pos)


def go_back(robot: Any, config: dict[str, Any] | None = None, reporter: Reporter | None = None) -> None:
    cfg = _config(config)
    image = capture_after_popups(robot, cfg, reporter=reporter)
    pos = _template_center(image, cfg, BACK_BUTTON)
    if pos is not None:
        _report(reporter, "return_to_list", "点击返回", log_label=f"点击返回：{format_point(*pos)}")
        click_image_point(robot, *pos)
        return
    _report(reporter, "return_to_list", "点击返回", log_label=f"点击返回：{format_point(0.08, 0.06)}")
    click_image_point(robot, 0.08, 0.06)
