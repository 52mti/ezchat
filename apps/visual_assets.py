from __future__ import annotations

from hashlib import blake2b
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Literal, NamedTuple

from vision.screen import crop_region, load_image


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
MY_SCREEN = "screen.my"
CUSTOM_SCREEN = "screen.custom"
NEXT_MESSAGE_BUTTON = "button.next_message"
INPUT_BOX_BUTTON = "button.input_box"
MICROPHONE_BUTTON = "button.microphone"
VOICE_INPUT_BUTTON = MICROPHONE_BUTTON
SEND_BUTTON = "button.send"
NEW_GREETING_BUTTON = "button.new_greeting"
PHOTO_BUTTON = "button.photo"
COMPLETE_BUTTON = "button.complete"
BACK_BUTTON = "button.back"
CUSTOM_BACK_BUTTON = "button.back.custom"
CUSTOM_POPUP = "popup.custom"
CHAT_AREA_REGION = "region.chat_area"
MESSAGE_LIST_REGION = "region.message_list"
MESSAGE_LIST_SCREEN_ROLE = "message_list"
CHAT_WINDOW_SCREEN_ROLE = "chat_window"
SCREEN_STATES_CONFIG_KEY = "screen_states"
SCREEN_ROLES_CONFIG_KEY = "screen_roles"
TEMPLATE_MATCH_REGIONS_CONFIG_KEY = "template_match_regions"
GLOBAL_MATCH_TEMPLATE_KEYS = frozenset({"screen.no_phone", "screen.black_screen"})
TEMPLATE_MATCH_CACHE_MAX_SIZE = 512
_TEMPLATE_MATCH_CACHE: dict[tuple[Any, ...], dict[str, float] | None] = {}
_TEMPLATE_MATCH_CACHE_LOCK = Lock()


class ScreenState(NamedTuple):
    name: str
    label: str
    path: Path


class TemplateMatchRegionMissingError(RuntimeError):
    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"当前类别没有设置范围：{key}")


FIXED_TEMPLATE_RULES: tuple[TemplateRule, ...] = (
    TemplateRule(
        key=CHAT_LIST_SCREEN,
        label="聊天列表界面",
        group="screens",
        filename="screen_聊天列表.jpg",
        phase="state",
        description="用于判断当前是否在聊天列表界面。",
    ),
    TemplateRule(
        key=CHAT_WINDOW_SCREEN,
        label="聊天界面",
        group="screens",
        filename="screen_聊天界面.jpg",
        phase="state",
        description="用于判断当前是否在聊天详情界面。",
    ),
    TemplateRule(
        key=MY_SCREEN,
        label="我的界面",
        group="screens",
        filename="screen_我的界面.jpg",
        phase="state",
        description="用于判断当前是否在我的界面。",
    ),
    TemplateRule(
        key=NEXT_MESSAGE_BUTTON,
        label="下一条按钮",
        group="buttons",
        filename="btn_下一条按钮.jpg",
        phase="action",
        description="用于定位下一条按钮。",
    ),
    TemplateRule(
        key=INPUT_BOX_BUTTON,
        label="输入框",
        group="buttons",
        filename="btn_输入框.jpg",
        phase="action",
        description="用于定位输入框。",
    ),
    TemplateRule(
        key=MICROPHONE_BUTTON,
        label="麦克风",
        group="buttons",
        filename="btn_麦克风按钮.jpg",
        phase="action",
        description="用于定位麦克风按钮。",
    ),
    TemplateRule(
        key=SEND_BUTTON,
        label="发送按钮",
        group="buttons",
        filename="btn_发送按钮.jpg",
        phase="action",
        description="用于定位发送按钮。",
    ),
    TemplateRule(
        key=NEW_GREETING_BUTTON,
        label="新招呼",
        group="buttons",
        filename="btn_新招呼按钮.jpg",
        phase="action",
        description="用于定位新招呼按钮。",
    ),
    TemplateRule(
        key=PHOTO_BUTTON,
        label="照片按钮",
        group="buttons",
        filename="btn_照片按钮.jpg",
        phase="action",
        description="用于打开相册并选择照片。",
    ),
    TemplateRule(
        key=COMPLETE_BUTTON,
        label="完成按钮",
        group="buttons",
        filename="btn_完成按钮.jpg",
        phase="action",
        description="用于确认照片选择并返回发送流程。",
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


BACK_BUTTON_TEMPLATE_RULE = PopupTemplateRule(
    key=CUSTOM_BACK_BUTTON,
    label="返回按钮模板",
    group="buttons",
    filename_prefix="btn_返回按钮",
    filename_pattern="btn_返回按钮.jpg / btn_返回按钮_{index}.jpg",
    match_pattern="btn_返回按钮*.jpg",
    phase="recovery",
    description="用于识别当前 App 的返回按钮，文件名按 btn_返回按钮.jpg、btn_返回按钮_2.jpg 递增。",
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


def _coerce_region(raw_region: Any) -> tuple[float, float, float, float] | None:
    if isinstance(raw_region, dict):
        if {"x", "y", "w", "h"}.issubset(raw_region):
            raw_values = [raw_region.get("x"), raw_region.get("y"), raw_region.get("w"), raw_region.get("h")]
        elif {"left", "top", "right", "bottom"}.issubset(raw_region):
            left = float(raw_region.get("left", 0.0) or 0.0)
            top = float(raw_region.get("top", 0.0) or 0.0)
            right = float(raw_region.get("right", left) or left)
            bottom = float(raw_region.get("bottom", top) or top)
            raw_values = [left, top, right - left, bottom - top]
        else:
            return None
    elif isinstance(raw_region, (list, tuple)) and len(raw_region) == 4:
        raw_values = list(raw_region)
    else:
        return None
    try:
        x, y, w, h = (float(value) for value in raw_values)
    except (TypeError, ValueError):
        return None
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0 - x, w))
    h = max(0.0, min(1.0 - y, h))
    if w <= 0.0 or h <= 0.0:
        return None
    return x, y, w, h


def template_match_region(config: dict[str, Any], key: str) -> tuple[float, float, float, float] | None:
    regions = config.get(TEMPLATE_MATCH_REGIONS_CONFIG_KEY, {})
    if not isinstance(regions, dict):
        return None
    return _coerce_region(regions.get(key))


def map_match_result_from_region(
    result: dict[str, float],
    region: tuple[float, float, float, float],
) -> dict[str, float]:
    region_x, region_y, region_w, region_h = region
    mapped = dict(result)
    mapped["x"] = region_x + float(result["x"]) * region_w
    mapped["y"] = region_y + float(result["y"]) * region_h
    mapped["w"] = float(result["w"]) * region_w
    mapped["h"] = float(result["h"]) * region_h
    return mapped


def _template_file_signature(template: Path) -> tuple[str, int, int]:
    try:
        stat = template.stat()
    except OSError:
        return str(template), 0, 0
    return str(template.resolve()), int(stat.st_mtime_ns), int(stat.st_size)


def _image_fingerprint(image: Any) -> str:
    if isinstance(image, (bytes, bytearray)):
        return blake2b(bytes(image), digest_size=16).hexdigest()
    try:
        path = Path(image)
    except TypeError:
        path = None
    if path is not None and path.exists():
        try:
            return blake2b(path.read_bytes(), digest_size=16).hexdigest()
        except OSError:
            pass
    try:
        loaded = load_image(image)
        payload = (
            f"{loaded.mode}:{loaded.size[0]}x{loaded.size[1]}:".encode("utf-8")
            + loaded.tobytes()
        )
        return blake2b(payload, digest_size=16).hexdigest()
    except Exception:
        return blake2b(repr(image).encode("utf-8", errors="replace"), digest_size=16).hexdigest()


def _runtime_scope(config: dict[str, Any]) -> tuple[str, str]:
    runtime = config.get("_runtime", {})
    if not isinstance(runtime, dict):
        runtime = {}
    return str(runtime.get("device_ip") or ""), str(runtime.get("app_name") or "")


def _cached_template_match(cache_key: tuple[Any, ...]) -> dict[str, float] | None | object:
    with _TEMPLATE_MATCH_CACHE_LOCK:
        if cache_key not in _TEMPLATE_MATCH_CACHE:
            return ...
        cached = _TEMPLATE_MATCH_CACHE[cache_key]
    return dict(cached) if cached is not None else None


def _store_template_match(cache_key: tuple[Any, ...], result: dict[str, float] | None) -> None:
    with _TEMPLATE_MATCH_CACHE_LOCK:
        if len(_TEMPLATE_MATCH_CACHE) >= TEMPLATE_MATCH_CACHE_MAX_SIZE:
            _TEMPLATE_MATCH_CACHE.clear()
        _TEMPLATE_MATCH_CACHE[cache_key] = dict(result) if result is not None else None


def match_template_with_region(
    image: Any,
    template: Path,
    config: dict[str, Any],
    key: str,
    *,
    threshold: float,
    match_func: Any,
) -> dict[str, float] | None:
    if TEMPLATE_MATCH_REGIONS_CONFIG_KEY in config and isinstance(config.get(TEMPLATE_MATCH_REGIONS_CONFIG_KEY), dict):
        if key in GLOBAL_MATCH_TEMPLATE_KEYS:
            return match_func(image, template, threshold=threshold)
        regions = config[TEMPLATE_MATCH_REGIONS_CONFIG_KEY]
        region = _coerce_region(regions.get(key))
        if region is None:
            raise TemplateMatchRegionMissingError(key)
    else:
        region = None
    if config.get("enable_template_match_cache", True) is not False:
        cache_key = (
            _runtime_scope(config),
            _image_fingerprint(image),
            _template_file_signature(template),
            str(key),
            round(float(threshold), 4),
            tuple(round(float(value), 6) for value in region) if region is not None else None,
            id(match_func),
        )
        cached = _cached_template_match(cache_key)
        if cached is not ...:
            return cached
    else:
        cache_key = None
    if region is None:
        result = match_func(image, template, threshold=threshold)
        if cache_key is not None:
            _store_template_match(cache_key, result)
        return result
    result = match_func(crop_region(image, region), template, threshold=threshold)
    if not result:
        if cache_key is not None:
            _store_template_match(cache_key, None)
        return None
    mapped = map_match_result_from_region(result, region)
    if cache_key is not None:
        _store_template_match(cache_key, mapped)
    return mapped


def popup_template_paths(config: dict[str, Any]) -> list[Path]:
    root = template_root(config, POPUP_TEMPLATE_RULE.group)
    if not root.exists():
        return []
    return sorted(root.glob(POPUP_TEMPLATE_RULE.match_pattern))


def popup_template_region_key(path: Path | str) -> str:
    stem = Path(path).stem.strip()
    if not stem:
        return CUSTOM_POPUP
    return f"{CUSTOM_POPUP}.{stem}"


def _back_button_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    if name == "btn_返回按钮.jpg":
        return (1, name)
    match = re.fullmatch(r"btn_返回按钮_(\d+)\.jpg", name)
    if match:
        return (int(match.group(1)), name)
    return (9999, name)


def back_button_template_paths(config: dict[str, Any]) -> list[Path]:
    root = template_root(config, BACK_BUTTON_TEMPLATE_RULE.group)
    if not root.exists():
        return []
    paths = [path for path in root.glob(BACK_BUTTON_TEMPLATE_RULE.match_pattern) if is_back_button_template_name(path.name)]
    legacy_path = root / "btn_back.jpg"
    if legacy_path.exists():
        paths.append(legacy_path)
    return sorted(paths, key=_back_button_sort_key)


def _template_variant_pattern(filename: str) -> tuple[str, str]:
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".jpg"
    return stem, suffix


def _template_variant_sort_key(path: Path, base_filename: str) -> tuple[int, str]:
    base_stem, suffix = _template_variant_pattern(base_filename)
    if path.name == base_filename:
        return (1, path.name)
    match = re.fullmatch(rf"{re.escape(base_stem)}_(\d+){re.escape(suffix)}", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (9999, path.name)


def template_variant_paths(root: Path, filename: str) -> list[Path]:
    if not root.exists():
        return []
    base_stem, suffix = _template_variant_pattern(filename)
    paths = [
        path
        for path in root.glob(f"{base_stem}*{suffix}")
        if path.name == filename or re.fullmatch(rf"{re.escape(base_stem)}_\d+{re.escape(suffix)}", path.name)
    ]
    return sorted(paths, key=lambda path: _template_variant_sort_key(path, filename))


def next_template_variant_path(root: Path, filename: str) -> Path:
    first = root / filename
    if not first.exists():
        return first
    base_stem, suffix = _template_variant_pattern(filename)
    index = 2
    while True:
        target = root / f"{base_stem}_{index}{suffix}"
        if not target.exists():
            return target
        index += 1


def next_popup_template_path(buttons_dir: Path) -> Path:
    index = 1
    while True:
        target = buttons_dir / f"popup_custom_{index}.jpg"
        if not target.exists():
            return target
        index += 1


def next_back_button_template_path(buttons_dir: Path) -> Path:
    first = buttons_dir / "btn_返回按钮.jpg"
    if not first.exists():
        return first
    index = 2
    while True:
        target = buttons_dir / f"btn_返回按钮_{index}.jpg"
        if not target.exists():
            return target
        index += 1


def is_popup_template_name(filename: str) -> bool:
    return Path(filename).match(POPUP_TEMPLATE_RULE.match_pattern)


def is_back_button_template_name(filename: str) -> bool:
    return filename == "btn_返回按钮.jpg" or re.fullmatch(r"btn_返回按钮_\d+\.jpg", filename) is not None


def import_template_rules() -> tuple[TemplateRule | PopupTemplateRule, ...]:
    return (*FIXED_TEMPLATE_RULES, POPUP_TEMPLATE_RULE, BACK_BUTTON_TEMPLATE_RULE)


def import_target_path(key: str, buttons_dir: Path, screens_dir: Path) -> Path:
    if key == POPUP_TEMPLATE_RULE.key:
        return next_popup_template_path(buttons_dir)
    if key == BACK_BUTTON_TEMPLATE_RULE.key:
        return next_back_button_template_path(buttons_dir)
    rule = template_rule(key)
    root = buttons_dir if rule.group == "buttons" else screens_dir
    return next_template_variant_path(root, rule.filename)


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
                "match_region_supported": True,
                "description": rule.description,
            }
            for rule in FIXED_TEMPLATE_RULES
            if rule.key != BACK_BUTTON
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
                "match_region_supported": True,
                "description": POPUP_TEMPLATE_RULE.description,
            }
        ],
        "back_buttons": [
            {
                "key": BACK_BUTTON_TEMPLATE_RULE.key,
                "label": BACK_BUTTON_TEMPLATE_RULE.label,
                "group": BACK_BUTTON_TEMPLATE_RULE.group,
                "filename_prefix": BACK_BUTTON_TEMPLATE_RULE.filename_prefix,
                "filename_pattern": BACK_BUTTON_TEMPLATE_RULE.filename_pattern,
                "match_pattern": BACK_BUTTON_TEMPLATE_RULE.match_pattern,
                "phase": BACK_BUTTON_TEMPLATE_RULE.phase,
                "multiple": BACK_BUTTON_TEMPLATE_RULE.multiple,
                "match_region_supported": True,
                "description": BACK_BUTTON_TEMPLATE_RULE.description,
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
                "match_region_supported": True,
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
                "match_region_supported": True,
                "description": "用于保存用户自定义名称的按钮模板，匹配后返回竖屏归一化中心坐标。",
            }
        ],
    }
