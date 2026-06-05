from __future__ import annotations

from pathlib import Path
from typing import Any

from apps.coordinates import click_image_point
from apps.photo_capture import capture_transform_image
from apps.reporting import Reporter, format_point, report_step
from apps.step_timing import exclude_ocr_wait_from_step_timeout
from apps.utils import tts_input
from vision import match, run_ocr_mode

from .visual_assets import (
    BACK_BUTTON,
    CHAT_AREA_REGION,
    CHAT_LIST_SCREEN,
    CHAT_WINDOW_SCREEN,
    NEXT_MESSAGE_BUTTON,
    SEND_BUTTON,
    VOICE_INPUT_BUTTON,
    back_button_template_paths,
    popup_template_paths,
    region_rule,
    match_template_with_region,
    template_path,
)


class StepUnavailable(RuntimeError):
    pass


POPUP_MAX_X = 0.9


def _report(reporter: Reporter | None, step: str, label: str, log_label: str | None = None) -> None:
    report_step(reporter, step, label, log_label)


def _threshold(config: dict[str, Any]) -> float:
    return float(config.get("match_threshold", 0.7))


def _run_ocr_mode(image: Any, config: dict[str, Any], mode: str, region: Any = None) -> dict[str, Any]:
    with exclude_ocr_wait_from_step_timeout(config):
        return run_ocr_mode(image, config, mode, region=region)


def _popup_match(image: Any, popup_template: Path, config: dict[str, Any]) -> dict[str, float] | None:
    result = match(image, popup_template, threshold=_threshold(config))
    if not result:
        return None
    max_x = float(config.get("popup_max_x", POPUP_MAX_X) or POPUP_MAX_X)
    if float(result.get("x", 0.0)) > max_x:
        return None
    return result


def _region(config: dict[str, Any], key: str) -> Any:
    current: Any = config
    for part in region_rule(key).config_path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _template_center(image: Any, config: dict[str, Any], key: str) -> tuple[float, float] | None:
    templates = back_button_template_paths(config) if key == BACK_BUTTON else []
    if not templates:
        template = template_path(config, key)
        if template is not None:
            templates.append(template)
    if not templates:
        return None
    for template in templates:
        result = match_template_with_region(
            image,
            template,
            config,
            key,
            threshold=_threshold(config),
            match_func=match,
        )
        if result:
            return result["x"] + result["w"] / 2, result["y"] + result["h"] / 2
    return None


def is_screen(image: Any, config: dict[str, Any], key: str) -> bool:
    template = template_path(config, key)
    if template is None:
        return False
    return match_template_with_region(
        image,
        template,
        config,
        key,
        threshold=_threshold(config),
        match_func=match,
    ) is not None


def current_screen(image: Any, config: dict[str, Any]) -> str:
    if is_screen(image, config, CHAT_WINDOW_SCREEN):
        return CHAT_WINDOW_SCREEN
    if is_screen(image, config, CHAT_LIST_SCREEN):
        return CHAT_LIST_SCREEN
    return "unknown"


def dismiss_popups(
    robot: Any,
    config: dict[str, Any],
    image: Any | None = None,
    reporter: Reporter | None = None,
) -> bool:
    current_image = image if image is not None else capture_transform_image(
        robot, config, pre_move=True, label="popup_check"
    )
    for popup_template in popup_template_paths(config):
        result = _popup_match(current_image, popup_template, config)
        if not result:
            continue
        close_x = min(0.98, result["x"] + result["w"] * 0.9)
        close_y = max(0.02, result["y"] + result["h"] * 0.1)
        _report(reporter, "popup_check", "关闭弹窗", log_label=f"关闭弹窗：{format_point(close_x, close_y)}")
        try:
            click_image_point(robot, close_x, close_y)
        except Exception:
            _report(reporter, "popup_check", "弹窗点击失败，忽略")
            return False
        return True
    return False


def capture_after_popups(
    robot: Any,
    config: dict[str, Any],
    max_dismiss: int = 2,
    reporter: Reporter | None = None,
) -> bytes:
    image = capture_transform_image(robot, config, pre_move=True, label="screen")
    for _ in range(max_dismiss):
        if not dismiss_popups(robot, config, image, reporter=reporter):
            break
        image = capture_transform_image(robot, config, pre_move=True, label="screen_after_popup")
    return image


def ensure_list_context(robot: Any, config: dict[str, Any], image: Any, reporter: Reporter | None = None) -> Any:
    screen = current_screen(image, config)
    if screen == CHAT_WINDOW_SCREEN:
        go_back(robot, config, reporter=reporter)
        return capture_after_popups(robot, config, reporter=reporter)
    return image


def find_unread(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> tuple[float, float] | None:
    cfg = config or {}
    _report(reporter, "list_ocr_scan", "查找未读消息")
    image = ensure_list_context(robot, cfg, capture_after_popups(robot, cfg, reporter=reporter), reporter=reporter)
    pos = _template_center(image, cfg, NEXT_MESSAGE_BUTTON)
    if pos is not None:
        return pos

    message_list_region = _region(cfg, "region.message_list")
    ocr_result = _run_ocr_mode(image, cfg, "list", region=message_list_region)
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
    cfg = config or {}
    _report(reporter, "dialogue_ocr_read", "对话 OCR 读取")
    image = capture_after_popups(robot, cfg, reporter=reporter)
    if template_path(cfg, CHAT_WINDOW_SCREEN) is not None and not is_screen(image, cfg, CHAT_WINDOW_SCREEN):
        raise StepUnavailable("chat window screen template did not match")
    ocr_result = _run_ocr_mode(image, cfg, "dialogue", region=_region(cfg, CHAT_AREA_REGION))
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
    cfg = config or {}
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
    cfg = config or {}
    image = capture_after_popups(robot, cfg, reporter=reporter)
    pos = _template_center(image, cfg, SEND_BUTTON)
    if pos is None:
        if template_path(cfg, SEND_BUTTON) is None:
            raise StepUnavailable("send button template is missing")
        raise StepUnavailable("send button was not found")
    _report(reporter, "send_button_click", "点击发送按钮", log_label=f"点击发送按钮：{format_point(*pos)}")
    click_image_point(robot, *pos)


def go_back(robot: Any, config: dict[str, Any] | None = None, reporter: Reporter | None = None) -> None:
    cfg = config or {}
    image = capture_after_popups(robot, cfg, reporter=reporter)
    pos = _template_center(image, cfg, BACK_BUTTON)
    if pos is not None:
        _report(reporter, "return_to_list", "点击返回", log_label=f"点击返回：{format_point(*pos)}")
        click_image_point(robot, *pos)
        return
    _report(reporter, "return_to_list", "点击返回", log_label=f"点击返回：{format_point(0.08, 0.06)}")
    click_image_point(robot, 0.08, 0.06)
