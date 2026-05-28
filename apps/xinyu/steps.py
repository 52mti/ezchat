from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from apps.cancellation import OperationCancelled
from apps.coordinates import click_image_point
from apps.photo_capture import capture_transform_image
from apps.reporting import Reporter, format_point, report_step
from apps.utils import tts_input
from apps.visual_assets import (
    BACK_BUTTON,
    CHAT_AREA_REGION,
    CHAT_LIST_SCREEN,
    CHAT_WINDOW_SCREEN,
    MESSAGE_LIST_REGION,
    all_screen_states,
    popup_template_paths,
    region_rule,
    screen_state_path,
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


BLACK_SCREEN = "screen.black_screen"
INPUT_BOX_BUTTON = "button.input_box"
MICROPHONE_BUTTON = "button.microphone"
UNREAD_BUTTON = "button.unread"
UNREAD_BUTTON_NO_RED = "button.unread.no_red"
SEND_BUTTON_KEY = "button.send"
TTS_POST_DELAY_SECONDS = 3.0

SCREEN_TEMPLATE_ALIASES: dict[str, tuple[str, ...]] = {
    CHAT_LIST_SCREEN: ("screen_聊天列表.jpg", "chat_list.jpg", "screen_message_list.jpg"),
    CHAT_WINDOW_SCREEN: ("screen_聊天界面.jpg", "chat_window.jpg"),
    BLACK_SCREEN: ("screen_黑屏异常.jpg",),
}

BUTTON_TEMPLATE_ALIASES: dict[str, tuple[str, ...]] = {
    BACK_BUTTON: ("btn_返回按钮.jpg", "btn_返回按钮_2.jpg", "btn_back.jpg"),
    UNREAD_BUTTON: ("btn_未读消息按钮.jpg", "btn_next_msg.jpg"),
    UNREAD_BUTTON_NO_RED: ("btn_未读消息-2.jpg",),
    INPUT_BOX_BUTTON: ("btn_输入框.jpg", "btn_input_box.jpg"),
    MICROPHONE_BUTTON: ("btn_麦克风按钮.jpg", "btn_voice_input.jpg"),
    SEND_BUTTON_KEY: ("btn_发送按钮.jpg", "btn_send.jpg"),
}


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


def _region(config: dict[str, Any], key: str) -> Any:
    current: Any = config
    for part in region_rule(key).config_path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


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


def _template_match(image: Any, config: dict[str, Any], key: str, *, group: str) -> dict[str, float] | None:
    return _match_any(image, _template_candidates(config, key, group=group), config)


def _template_center(image: Any, config: dict[str, Any], key: str, *, group: str) -> tuple[float, float] | None:
    result = _template_match(image, config, key, group=group)
    if not result:
        return None
    return result["x"] + result["w"] / 2, result["y"] + result["h"] / 2


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


def _raise_if_black_screen(image: Any, config: dict[str, Any]) -> None:
    if _matches_screen(image, config, BLACK_SCREEN) or _looks_like_black_screen(image, config):
        raise BlackScreenError("black screen detected; device may be powered off")


def _center_from_bounds(bounds: list[float]) -> tuple[float, float]:
    left, top, right, bottom = bounds
    return (left + right) / 2, (top + bottom) / 2


def _bottom_nav_center(
    items: list[dict[str, Any]],
    keywords: tuple[str, ...],
    *,
    min_center_y: float = 0.0,
) -> tuple[float, float] | None:
    for item in items:
        name = str(item.get("name", "") or "")
        if not any(keyword in name for keyword in keywords):
            continue
        bounds = item.get("bounds")
        if isinstance(bounds, list) and len(bounds) == 4:
            center = _center_from_bounds(bounds)
            if center[1] >= min_center_y:
                return center
    return None


def _has_red_pixels(image: Any, match_result: dict[str, float]) -> bool:
    try:
        pil_image = load_image(image)
    except Exception:
        return False
    width, height = pil_image.size
    pad_x = max(6, int(float(match_result["w"]) * width * 0.35))
    pad_y = max(6, int(float(match_result["h"]) * height * 0.35))
    left = max(0, int(float(match_result["x"]) * width) - pad_x)
    top = max(0, int(float(match_result["y"]) * height) - pad_y)
    right = min(width, int((float(match_result["x"]) + float(match_result["w"])) * width) + pad_x)
    bottom = min(height, int((float(match_result["y"]) + float(match_result["h"])) * height) + pad_y)
    if right <= left or bottom <= top:
        return False
    pixels = np.array(pil_image.crop((left, top, right, bottom)).convert("RGB"))
    red = pixels[:, :, 0].astype(np.int16)
    green = pixels[:, :, 1].astype(np.int16)
    blue = pixels[:, :, 2].astype(np.int16)
    strong_red = (red >= 150) & (red - green >= 55) & (red - blue >= 55)
    pink_red = (red >= 80) & (red - green >= 25) & (blue - green >= 10) & (red >= blue - 25)
    red_mask = strong_red | pink_red
    return int(red_mask.sum()) >= max(3, int(red_mask.size * 0.002))


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
        result = match(image, popup_template, threshold=_threshold(config))
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
    *,
    label: str = "screen",
    max_dismiss: int = 3,
    reporter: Reporter | None = None,
) -> Any:
    image = capture_transform_image(robot, config, pre_move=True, label=label)
    for _ in range(max_dismiss):
        close_point = _popup_close_point(image, config)
        if close_point is None:
            break
        _report(reporter, "popup_check", "关闭弹窗", log_label=f"关闭弹窗：{format_point(*close_point)}")
        click_image_point(robot, *close_point)
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
    click_point = pos if pos is not None else (0.08, 0.06)
    _report(reporter, "return_from_chat", "点击返回", log_label=f"点击返回：{format_point(*click_point)}")
    click_image_point(robot, *click_point)
    return True


def ensure_list_screen(robot: Any, config: dict[str, Any] | None = None, reporter: Reporter | None = None) -> Any | None:
    cfg = _config(config)
    _report(reporter, "confirm_xinyu_list", "确认心遇见聊天列表")
    image = capture_after_popups(robot, cfg, label="confirm_list", reporter=reporter)
    _raise_if_black_screen(image, cfg)
    if _matches_screen(image, cfg, CHAT_LIST_SCREEN):
        return image
    if _matches_screen(image, cfg, CHAT_WINDOW_SCREEN):
        _click_back_once(robot, cfg, image, reporter=reporter)
        time.sleep(0.4)
        image = capture_after_popups(robot, cfg, label="after_return_from_chat", reporter=reporter)
        _raise_if_black_screen(image, cfg)
        if _matches_screen(image, cfg, CHAT_LIST_SCREEN):
            return image
    _click_back_once(robot, cfg, image, reporter=reporter)
    time.sleep(0.4)
    image = capture_after_popups(robot, cfg, label="after_unknown_back", reporter=reporter)
    _raise_if_black_screen(image, cfg)
    if _matches_screen(image, cfg, CHAT_LIST_SCREEN):
        return image
    return None


def check_unread_button(
    robot: Any,
    config: dict[str, Any] | None = None,
    image: Any | None = None,
    reporter: Reporter | None = None,
) -> Any:
    cfg = _config(config)
    current_image = image if image is not None else capture_after_popups(robot, cfg, label="unread_button", reporter=reporter)
    result = _template_match(current_image, cfg, UNREAD_BUTTON, group="buttons")
    if result is None:
        return current_image
    has_red = _has_red_pixels(current_image, result)
    unread_center = (result["x"] + result["w"] / 2, result["y"] + result["h"] / 2)
    _report(
        reporter,
        "unread_button_check",
        "未读入口已有红色" if has_red else "未读入口无红色，点击切换",
        log_label=(
            f"未读入口已有红色：{format_point(*unread_center)}"
            if has_red
            else f"未读入口无红色，点击切换：{format_point(*unread_center)}"
        ),
    )
    if has_red:
        return current_image
    click_image_point(robot, *unread_center)
    time.sleep(0.4)
    return capture_after_popups(robot, cfg, label="after_unread_button", reporter=reporter)


def find_unread_target(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> dict[str, Any] | None:
    cfg = _config(config)
    image = ensure_list_screen(robot, cfg, reporter=reporter)
    if image is None:
        return None
    image = check_unread_button(robot, cfg, image, reporter=reporter)

    _report(reporter, "list_ocr_scan", "列表 OCR 扫描")
    ocr_result = run_ocr_mode(image, cfg, "list", region=None)
    rows = ocr_result["result"].get("message_list", [])
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        try:
            unread_count = int(row.get("unread_count") or 0)
        except (TypeError, ValueError):
            unread_count = 0
        if unread_count < 1:
            continue
        favorability = row.get("favorability")
        try:
            favorability_value = int(favorability) if favorability is not None else -1
        except (TypeError, ValueError):
            favorability_value = -1
        candidates.append((favorability_value, -index, row))
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

    _report(reporter, "keep_awake_scan", "防熄屏扫描底部导航")
    ocr_result = run_ocr_mode(image, cfg, "list", region=None)
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
        raise StepUnavailable("chat window screen template did not match")
    return image


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
    ocr_result = run_ocr_mode(current_image, cfg, "dialogue", region=_region(cfg, CHAT_AREA_REGION))
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
    contact = fallback_contact or str(dialogue_result.get("user_name") or items[0]["text"])
    return {
        "user": contact,
        "contact": contact,
        "content": sender1_messages[-1],
        "message_time": "",
        "unread_count": str(target.get("unread_count", "")) if isinstance(target, dict) else "",
        "favorability": str(target.get("favorability", "")) if isinstance(target, dict) else "",
    }


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
        deadline = time.monotonic() + _step_timeout(cfg)
        while True:
            image = capture_after_popups(robot, cfg, label="microphone", reporter=reporter)
            microphone = _template_center(image, cfg, MICROPHONE_BUTTON, group="buttons")
            if microphone is not None:
                label = "点击麦克风" if attempt == 0 else "重新点击麦克风"
                _report(reporter, "microphone_click", label, log_label=f"{label}：{format_point(*microphone)}")
                click_image_point(robot, *microphone)
                break
            if time.monotonic() >= deadline:
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
            _report(
                reporter,
                "tts_retry",
                "TTS 播放失败，准备重试",
                log_label=f"TTS 播放失败，准备重试：{exc}",
            )
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


def send_reply(
    robot: Any,
    config: dict[str, Any] | None = None,
    reporter: Reporter | None = None,
) -> int:
    cfg = _config(config)
    deadline = time.monotonic() + _step_timeout(cfg)
    clicked = 0
    while True:
        image = capture_after_popups(robot, cfg, label="send_button", reporter=reporter)
        pos = _template_center(image, cfg, SEND_BUTTON_KEY, group="buttons")
        if pos is None:
            if clicked > 0:
                return clicked
            if time.monotonic() >= deadline:
                raise StepUnavailable("send button was not found")
            time.sleep(0.4)
            continue
        _report(reporter, "send_button_click", "点击发送按钮", log_label=f"点击发送按钮：{format_point(*pos)}")
        click_image_point(robot, *pos)
        clicked += 1
        if time.monotonic() >= deadline:
            raise StepUnavailable("send button did not disappear after click")
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

    for _ in range(2):
        image = capture_after_popups(robot, cfg, label="go_back_button", reporter=reporter)
        pos = _template_center(image, cfg, BACK_BUTTON, group="buttons")
        click_point = pos if pos is not None else (0.08, 0.06)
        _report(reporter, "back_double_click", "点击返回", log_label=f"点击返回：{format_point(*click_point)}")
        click_image_point(robot, *click_point)
        time.sleep(0.4)

    image = capture_after_popups(robot, cfg, label="after_go_back", reporter=reporter)
    if not _matches_screen(image, cfg, CHAT_LIST_SCREEN):
        raise StepUnavailable("failed to return to chat list")


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

    pos = _template_center(image, cfg, BACK_BUTTON, group="buttons")
    click_point = pos if pos is not None else (0.08, 0.06)
    _report(reporter, "back_double_click", "连续点击返回两次", log_label=f"连续点击返回两次：{format_point(*click_point)}")
    click_image_point(robot, *click_point)
    click_image_point(robot, *click_point)
    time.sleep(0.4)

    image = capture_after_popups(robot, cfg, label="after_go_back", reporter=reporter)
    if not _matches_screen(image, cfg, CHAT_LIST_SCREEN):
        raise StepUnavailable("failed to return to chat list")
