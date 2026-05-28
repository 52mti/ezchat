from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple


TemplateGroup = Literal["buttons", "screens"]
AssetPhase = Literal["state", "action", "recovery", "region"]


@dataclass(frozen=True)
class TemplateRule:
    key: str
    label: str
    group: TemplateGroup
    filename: str
    phase: AssetPhase
    description: str = ""


@dataclass(frozen=True)
class PopupTemplateRule:
    key: str
    label: str
    group: TemplateGroup
    filename_prefix: str
    filename_pattern: str
    match_pattern: str
    phase: AssetPhase
    description: str = ""
    multiple: bool = True


@dataclass(frozen=True)
class RegionRule:
    key: str
    label: str
    config_path: tuple[str, ...]
    phase: AssetPhase
    description: str = ""


CHAT_LIST_SCREEN = "screen.chat_list"
CHAT_WINDOW_SCREEN = "screen.chat_window"
CUSTOM_SCREEN = "screen.custom"
NEXT_MESSAGE_BUTTON = "button.next_message"
VOICE_INPUT_BUTTON = "button.voice_input"
SEND_BUTTON = "button.send"
BACK_BUTTON = "button.back"
CUSTOM_POPUP = "popup.custom"
CHAT_AREA_REGION = "region.chat_area"
MESSAGE_LIST_REGION = "region.message_list"
MESSAGE_LIST_SCREEN_ROLE = "message_list"
CHAT_WINDOW_SCREEN_ROLE = "chat_window"
SCREEN_STATES_CONFIG_KEY = "screen_states"
SCREEN_ROLES_CONFIG_KEY = "screen_roles"


class ScreenState(NamedTuple):
    name: str
    label: str
    path: Path


FIXED_TEMPLATE_RULES: tuple[TemplateRule, ...] = (
    TemplateRule(
        key=CHAT_LIST_SCREEN,
        label="聊天列表界面",
        group="screens",
        filename="chat_list.jpg",
        phase="state",
        description="用于判断当前是否在消息列表界面。",
    ),
    TemplateRule(
        key=CHAT_WINDOW_SCREEN,
        label="聊天窗口界面",
        group="screens",
        filename="chat_window.jpg",
        phase="state",
        description="用于判断当前是否在聊天详情界面。",
    ),
    TemplateRule(
        key=NEXT_MESSAGE_BUTTON,
        label="未读消息入口",
        group="buttons",
        filename="btn_next_msg.jpg",
        phase="action",
        description="用于定位下一条未读消息入口。",
    ),
    TemplateRule(
        key=VOICE_INPUT_BUTTON,
        label="语音输入按钮",
        group="buttons",
        filename="btn_voice_input.jpg",
        phase="action",
        description="用于定位语音输入按钮。",
    ),
    TemplateRule(
        key=SEND_BUTTON,
        label="发送按钮",
        group="buttons",
        filename="btn_send.jpg",
        phase="action",
        description="用于定位发送按钮。",
    ),
    TemplateRule(
        key=BACK_BUTTON,
        label="返回按钮",
        group="buttons",
        filename="btn_back.jpg",
        phase="recovery",
        description="用于返回上一级界面或纠错回退。",
    ),
)


POPUP_TEMPLATE_RULE = PopupTemplateRule(
    key=CUSTOM_POPUP,
    label="弹窗模板",
    group="buttons",
    filename_prefix="popup_custom",
    filename_pattern="popup_custom_{index}.jpg",
    match_pattern="popup_*.jpg",
    phase="recovery",
    description="用于识别多种弹窗形态，匹配和点击坐标统一使用竖屏归一化坐标。",
)


REGION_RULES: tuple[RegionRule, ...] = (
    RegionRule(
        key=CHAT_AREA_REGION,
        label="聊天内容区域",
        config_path=("ocr", "regions", "chat_area"),
        phase="region",
        description="用于 OCR 读取聊天详情内容。",
    ),
    RegionRule(
        key=MESSAGE_LIST_REGION,
        label="消息列表区域",
        config_path=("ocr", "regions", "message_list"),
        phase="region",
        description="用于 OCR 扫描消息列表。",
    ),
)


def fixed_template_rules() -> tuple[TemplateRule, ...]:
    return FIXED_TEMPLATE_RULES


def region_rules() -> tuple[RegionRule, ...]:
    return REGION_RULES


def template_rule(key: str) -> TemplateRule:
    for rule in FIXED_TEMPLATE_RULES:
        if rule.key == key:
            return rule
    raise KeyError(f"unknown template rule: {key}")


def region_rule(key: str) -> RegionRule:
    for rule in REGION_RULES:
        if rule.key == key:
            return rule
    raise KeyError(f"unknown region rule: {key}")


def template_root(config: dict[str, Any], group: TemplateGroup) -> Path:
    return Path(config.get("templates", {}).get(group, ""))


def template_path(config: dict[str, Any], key: str) -> Path | None:
    rule = template_rule(key)
    path = template_root(config, rule.group) / rule.filename
    return path if path.exists() else None


def configured_screen_states(config: dict[str, Any]) -> list[ScreenState]:
    root = template_root(config, "screens")
    raw_states = config.get(SCREEN_STATES_CONFIG_KEY, {})
    states: list[ScreenState] = []
    if isinstance(raw_states, dict):
        for name, raw_value in raw_states.items():
            screen_name = str(name).strip()
            if not screen_name:
                continue
            label = screen_name
            filename = ""
            if isinstance(raw_value, dict):
                filename = str(raw_value.get("filename", "")).strip()
                label = str(raw_value.get("label", screen_name)).strip() or screen_name
            elif isinstance(raw_value, str):
                filename = raw_value.strip()
            if not filename:
                continue
            path = root / filename
            if path.exists():
                states.append(ScreenState(screen_name, label, path))
    elif isinstance(raw_states, list):
        for raw_value in raw_states:
            if not isinstance(raw_value, dict):
                continue
            screen_name = str(raw_value.get("name", "")).strip()
            filename = str(raw_value.get("filename", "")).strip()
            if not screen_name or not filename:
                continue
            label = str(raw_value.get("label", screen_name)).strip() or screen_name
            path = root / filename
            if path.exists():
                states.append(ScreenState(screen_name, label, path))
    return states


def screen_state_path(config: dict[str, Any], name: str) -> Path | None:
    for state in configured_screen_states(config):
        if state.name == name:
            return state.path
    fixed_path = template_path(config, name)
    return fixed_path if name in {CHAT_LIST_SCREEN, CHAT_WINDOW_SCREEN} else None


def all_screen_states(config: dict[str, Any]) -> list[ScreenState]:
    states = configured_screen_states(config)
    configured_names = {state.name for state in states}
    for key in (CHAT_WINDOW_SCREEN, CHAT_LIST_SCREEN):
        path = template_path(config, key)
        if path is not None and key not in configured_names:
            states.append(ScreenState(key, template_rule(key).label, path))
    return states


def screen_role_name(config: dict[str, Any], role: str, default: str) -> str:
    roles = config.get(SCREEN_ROLES_CONFIG_KEY, {})
    if not isinstance(roles, dict):
        return default
    value = str(roles.get(role, "")).strip()
    return value or default


def screen_matches_role(screen_name: str, config: dict[str, Any], role: str, default: str) -> bool:
    return screen_name == screen_role_name(config, role, default)


def popup_template_paths(config: dict[str, Any]) -> list[Path]:
    root = template_root(config, POPUP_TEMPLATE_RULE.group)
    if not root.exists():
        return []
    return sorted(root.glob(POPUP_TEMPLATE_RULE.match_pattern))


def next_popup_template_path(buttons_dir: Path) -> Path:
    index = 1
    while True:
        target = buttons_dir / f"popup_custom_{index}.jpg"
        if not target.exists():
            return target
        index += 1


def is_popup_template_name(filename: str) -> bool:
    return Path(filename).match(POPUP_TEMPLATE_RULE.match_pattern)


def import_template_rules() -> tuple[TemplateRule | PopupTemplateRule, ...]:
    return (*FIXED_TEMPLATE_RULES, POPUP_TEMPLATE_RULE)


def import_target_path(key: str, buttons_dir: Path, screens_dir: Path) -> Path:
    if key == POPUP_TEMPLATE_RULE.key:
        return next_popup_template_path(buttons_dir)
    rule = template_rule(key)
    root = buttons_dir if rule.group == "buttons" else screens_dir
    return root / rule.filename


def asset_rules_for_tool() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "save_types": [
            {
                "key": "screen",
                "label": "判定界面",
                "description": "用于判断当前在哪个界面，界面名称由用户自定义。",
            },
            {
                "key": "button",
                "label": "按钮",
                "description": "用于匹配按钮并返回竖屏归一化中心坐标，通常用于点击。",
            },
            {
                "key": "popup",
                "label": "弹窗",
                "description": "用于匹配多种弹窗并返回竖屏归一化点击坐标，通常用于关闭或确认。",
            },
        ],
        "templates": [
            {
                "key": rule.key,
                "label": rule.label,
                "group": rule.group,
                "filename": rule.filename,
                "phase": rule.phase,
                "description": rule.description,
            }
            for rule in FIXED_TEMPLATE_RULES
        ],
        "popups": [
            {
                "key": POPUP_TEMPLATE_RULE.key,
                "label": POPUP_TEMPLATE_RULE.label,
                "group": POPUP_TEMPLATE_RULE.group,
                "filename_prefix": POPUP_TEMPLATE_RULE.filename_prefix,
                "filename_pattern": POPUP_TEMPLATE_RULE.filename_pattern,
                "match_pattern": POPUP_TEMPLATE_RULE.match_pattern,
                "phase": POPUP_TEMPLATE_RULE.phase,
                "multiple": POPUP_TEMPLATE_RULE.multiple,
                "description": POPUP_TEMPLATE_RULE.description,
            }
        ],
        "custom_screens": [
            {
                "key": CUSTOM_SCREEN,
                "label": "自定义界面",
                "group": "screens",
                "filename_pattern": "screen_{name}.jpg",
                "config_key": SCREEN_STATES_CONFIG_KEY,
                "multiple": True,
                "phase": "state",
                "description": "用于保存用户自定义名称的界面判断图，可保存 3-5 个或更多界面。",
            }
        ],
        "custom_buttons": [
            {
                "key": "button.custom",
                "label": "自定义按钮",
                "group": "buttons",
                "filename_pattern": "btn_{name}.jpg",
                "multiple": True,
                "phase": "action",
                "description": "用于保存用户自定义名称的按钮模板，匹配后返回竖屏归一化中心坐标。",
            }
        ],
    }
