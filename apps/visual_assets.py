from __future__ import annotations

from collections import OrderedDict
from hashlib import blake2b
import re
import uuid
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
_MACHINE_TEMPLATE_ID: str | None = None


def machine_template_id() -> str:
    """客户模板命名空间：本机 MAC 的 12 位小写十六进制。

    取不到真实 MAC 时退回固定标记 local（控制机械臂的机器都有网卡，
    实际不会走到）。
    """
    global _MACHINE_TEMPLATE_ID
    if _MACHINE_TEMPLATE_ID is None:
        node = uuid.getnode()
        _MACHINE_TEMPLATE_ID = "local" if (node >> 40) & 0x01 else f"{node:012x}"
    return _MACHINE_TEMPLATE_ID


_TEMPLATE_MATCH_CACHE: "OrderedDict[tuple[Any, ...], dict[str, float] | None]" = OrderedDict()
_TEMPLATE_MATCH_CACHE_LOCK = Lock()


@dataclass(frozen=True)
class TemplateMatchDiagnostic:
    key: str
    template_name: str
    score: float | None
    threshold: float
    region: tuple[float, float, float, float] | None
    channel: str
    matched: bool
    reason: str = ""


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


def expand_match_region(
    region: tuple[float, float, float, float],
    ratio: float,
) -> tuple[float, float, float, float]:
    if ratio <= 0.0:
        return region
    x, y, w, h = region
    dx = w * ratio
    dy = h * ratio
    left = max(0.0, x - dx)
    top = max(0.0, y - dy)
    right = min(1.0, x + w + dx)
    bottom = min(1.0, y + h + dy)
    return left, top, max(0.0, right - left), max(0.0, bottom - top)


def _match_region_expand_ratio(config: dict[str, Any], key: str) -> float:
    raw_by_key = config.get("template_match_region_expand_ratios")
    if isinstance(raw_by_key, dict) and key in raw_by_key:
        try:
            return max(0.0, min(0.2, float(raw_by_key[key])))
        except (TypeError, ValueError):
            return 0.0
    try:
        return max(0.0, min(0.2, float(config.get("template_match_region_expand_ratio", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def _match_region_for_key(config: dict[str, Any], key: str) -> tuple[float, float, float, float] | None:
    if TEMPLATE_MATCH_REGIONS_CONFIG_KEY in config and isinstance(config.get(TEMPLATE_MATCH_REGIONS_CONFIG_KEY), dict):
        if key in GLOBAL_MATCH_TEMPLATE_KEYS:
            return None
        region = _coerce_region(config[TEMPLATE_MATCH_REGIONS_CONFIG_KEY].get(key))
        if region is None:
            raise TemplateMatchRegionMissingError(key)
        return expand_match_region(region, _match_region_expand_ratio(config, key))
    return None


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


def _runtime_cache(config: dict[str, Any]) -> dict[str, Any]:
    cache = config.setdefault("_runtime_cache", {})
    if not isinstance(cache, dict):
        cache = {}
        config["_runtime_cache"] = cache
    diagnostics = cache.setdefault("template_match_diagnostics", {})
    if not isinstance(diagnostics, dict):
        cache["template_match_diagnostics"] = {}
    return cache


def _runtime_context(config: dict[str, Any]) -> dict[str, Any]:
    runtime = config.get("_runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        config["_runtime"] = runtime
    return runtime


def _append_runtime_path(runtime: dict[str, Any], key: str, path: str, *, limit: int = 10) -> None:
    values = runtime.get(key)
    if not isinstance(values, list):
        values = []
    path_text = str(path).strip()
    if not path_text:
        return
    values = [item for item in values if str(item) != path_text]
    values.append(path_text)
    runtime[key] = values[-limit:]


def _record_template_runtime_context(
    config: dict[str, Any],
    *,
    key: str,
    candidate_paths: list[Path] | None = None,
    matched_path: Path | None = None,
) -> None:
    runtime = _runtime_context(config)
    runtime["last_template_match_key"] = str(key)
    if candidate_paths is not None:
        runtime["last_template_paths"] = [str(path) for path in candidate_paths]
    if matched_path is not None:
        runtime["last_template_path"] = str(matched_path)
        _append_runtime_path(runtime, "template_paths", str(matched_path))


def _set_template_match_diagnostic(config: dict[str, Any], diagnostic: TemplateMatchDiagnostic) -> None:
    diagnostics = _runtime_cache(config).setdefault("template_match_diagnostics", {})
    if isinstance(diagnostics, dict):
        diagnostics[diagnostic.key] = diagnostic


def last_template_match_diagnostic(config: dict[str, Any], key: str) -> TemplateMatchDiagnostic | None:
    cache = config.get("_runtime_cache", {})
    if not isinstance(cache, dict):
        return None
    diagnostics = cache.get("template_match_diagnostics", {})
    if not isinstance(diagnostics, dict):
        return None
    value = diagnostics.get(key)
    return value if isinstance(value, TemplateMatchDiagnostic) else None


def format_template_match_diagnostic(config: dict[str, Any], key: str) -> str:
    diagnostic = last_template_match_diagnostic(config, key)
    if diagnostic is None:
        return "最高候选：无"
    region = (
        "[" + ", ".join(f"{value:.3f}" for value in diagnostic.region) + "]"
        if diagnostic.region is not None
        else "全图"
    )
    score = "无" if diagnostic.score is None else f"{diagnostic.score:.3f}"
    reason = f"，原因：{diagnostic.reason}" if diagnostic.reason else ""
    return (
        f"最高候选模板={diagnostic.template_name or '无'}，"
        f"最高相似度={score}，阈值={diagnostic.threshold:.3f}，"
        f"范围={region}，通道={diagnostic.channel or 'unknown'}{reason}"
    )


def _match_scales(config: dict[str, Any], key: str) -> tuple[float, ...]:
    raw_by_key = config.get("template_match_scales_by_key")
    raw: Any = None
    if isinstance(raw_by_key, dict):
        raw = raw_by_key.get(key)
    if raw is None:
        raw = config.get("template_match_scales", [0.92, 0.96, 1.0, 1.04, 1.08])
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    values: list[float] = []
    for value in raw:
        try:
            scale = float(value)
        except (TypeError, ValueError):
            continue
        if 0.2 <= scale <= 3.0 and scale not in values:
            values.append(scale)
    return tuple(values or [1.0])


def _match_channels(config: dict[str, Any], key: str) -> tuple[str, ...]:
    raw_by_key = config.get("template_match_channels_by_key")
    raw: Any = None
    if isinstance(raw_by_key, dict):
        raw = raw_by_key.get(key)
    if raw is None:
        raw = config.get("template_match_channels", ["gray_norm", "gray", "edge"])
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    channels = [str(value) for value in raw if str(value) in {"bgr", "gray", "gray_norm", "edge"}]
    return tuple(channels or ["gray"])


def _effective_match_func(config: dict[str, Any], key: str, match_func: Any) -> tuple[Any, tuple[Any, ...]]:
    if config.get("enable_enhanced_template_match", True) is False:
        return match_func, ("basic",)
    match_module = getattr(match_func, "__module__", "")
    match_name = getattr(match_func, "__name__", "")
    # Most app steps import the public lazy wrapper from ``vision``.  It is
    # still the built-in matcher and must receive the configured enhanced
    # channels instead of being mistaken for a caller-provided custom matcher.
    if match_name != "match" or match_module not in {"vision", "vision.matcher"}:
        return match_func, ("custom", id(match_func))
    try:
        from vision.matcher import match_enhanced
    except (ImportError, ModuleNotFoundError):
        # 更新覆盖过程中若仍加载到旧版视觉模块，模板测试至少可以使用基础匹配，
        # 不应因增强匹配模块暂不可用而直接失败。
        return match_func, ("basic-fallback",)

    scales = _match_scales(config, key)
    channels = _match_channels(config, key)

    def enhanced(image: Any, template: Any, threshold: float = 0.85) -> dict[str, float] | None:
        return match_enhanced(image, template, threshold=threshold, scales=scales, channels=channels)

    return enhanced, ("enhanced", scales, channels)


def _cached_template_match(cache_key: tuple[Any, ...]) -> dict[str, float] | None | object:
    with _TEMPLATE_MATCH_CACHE_LOCK:
        if cache_key not in _TEMPLATE_MATCH_CACHE:
            return ...
        cached = _TEMPLATE_MATCH_CACHE[cache_key]
        _TEMPLATE_MATCH_CACHE.move_to_end(cache_key)
    return dict(cached) if cached is not None else None


def _store_template_match(cache_key: tuple[Any, ...], result: dict[str, float] | None) -> None:
    with _TEMPLATE_MATCH_CACHE_LOCK:
        # LRU 淘汰最旧条目。原先满 512 条整体 clear()，会把所有其他设备的
        # 有效缓存一并抹掉，下一轮全部 miss 触发"全量重匹配风暴"（多设备
        # 周期性集体卡顿）；LRU 只逐出真正最久未用的条目。
        if len(_TEMPLATE_MATCH_CACHE) >= TEMPLATE_MATCH_CACHE_MAX_SIZE:
            _TEMPLATE_MATCH_CACHE.popitem(last=False)
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
    _record_template_runtime_context(config, key=key, candidate_paths=[template])
    region = _match_region_for_key(config, key)
    effective_match_func, strategy_key = _effective_match_func(config, key, match_func)
    if config.get("enable_template_match_cache", True) is not False:
        cache_key = (
            _runtime_scope(config),
            _image_fingerprint(image),
            _template_file_signature(template),
            str(key),
            round(float(threshold), 4),
            tuple(round(float(value), 6) for value in region) if region is not None else None,
            strategy_key,
        )
        cached = _cached_template_match(cache_key)
        if cached is not ...:
            return cached
    else:
        cache_key = None
    if region is None:
        result = effective_match_func(image, template, threshold=threshold)
        if cache_key is not None:
            _store_template_match(cache_key, result)
        if result:
            _record_template_runtime_context(config, key=key, matched_path=template)
        return result
    result = effective_match_func(crop_region(image, region), template, threshold=threshold)
    if not result:
        if cache_key is not None:
            _store_template_match(cache_key, None)
        return None
    mapped = map_match_result_from_region(result, region)
    if cache_key is not None:
        _store_template_match(cache_key, mapped)
    _record_template_runtime_context(config, key=key, matched_path=template)
    return mapped


def match_template_candidates_with_region(
    image: Any,
    templates: list[Path],
    config: dict[str, Any],
    key: str,
    *,
    threshold: float,
    match_func: Any,
    single_match_func: Any | None = None,
) -> dict[str, Any] | None:
    region: tuple[float, float, float, float] | None = None
    _record_template_runtime_context(config, key=key, candidate_paths=templates)
    if not templates:
        _set_template_match_diagnostic(
            config,
            TemplateMatchDiagnostic(
                key=key,
                template_name="",
                score=None,
                threshold=float(threshold),
                region=region,
                channel="",
                matched=False,
                reason="没有候选模板",
            ),
        )
        return None
    region = _match_region_for_key(config, key)
    best: dict[str, Any] | None = None
    best_path: Path | None = None
    best_error = ""
    run_single = single_match_func or match_template_with_region
    probe_threshold = -1.0 if run_single is match_template_with_region else float(threshold)
    for path in templates:
        try:
            result = run_single(
                image,
                path,
                config,
                key,
                threshold=probe_threshold,
                match_func=match_func,
            )
        except TemplateMatchRegionMissingError:
            raise
        except Exception as exc:
            best_error = str(exc)
            continue
        if not result:
            continue
        score = float(result.get("score", threshold) if "score" in result else threshold)
        if best is None or score > float(best.get("score", 0.0) or 0.0):
            best = dict(result)
            best["score"] = score
            best_path = path
    if best is None or best_path is None:
        _set_template_match_diagnostic(
            config,
            TemplateMatchDiagnostic(
                key=key,
                template_name="",
                score=None,
                threshold=float(threshold),
                region=region,
                channel="",
                matched=False,
                reason=best_error or "全部模板无有效结果",
            ),
        )
        return None
    channel = str(best.get("channel") or "gray")
    score = float(best.get("score", threshold) if "score" in best else threshold)
    matched = score >= float(threshold)
    best["_template_path"] = str(best_path)
    best["_template_name"] = best_path.name
    best["_template_threshold"] = float(threshold)
    best["_template_region"] = region
    best["_template_channel"] = channel
    diagnostic = TemplateMatchDiagnostic(
        key=key,
        template_name=best_path.name,
        score=score,
        threshold=float(threshold),
        region=region,
        channel=channel,
        matched=matched,
    )
    _set_template_match_diagnostic(config, diagnostic)
    if not matched:
        return None
    _record_template_runtime_context(config, key=key, matched_path=best_path)
    return best


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


CUSTOM_TEMPLATE_MARKER = "__customer"
DEVELOPER_TEMPLATE_MARKER = "__developer"

# 客户模板的 MAC 命名段：__customer_{12位小写hex}_{序号}。多台客户机的
# 模板汇总到开发侧不会重名，可直接复制进更新包向其他机器分发；旧格式
# __customer_{序号}（无 MAC 段）继续兼容识别。
_MACHINE_ID_PATTERN = r"[0-9a-f]{12}"
_VARIANT_TAIL_RE = rf"(?:_(?:{_MACHINE_ID_PATTERN}))?_\d+"


def _template_variant_sort_key(path: Path, base_filename: str) -> tuple[int, str]:
    base_stem, suffix = _template_variant_pattern(base_filename)
    if path.name == base_filename:
        return (1, path.name)
    match = re.fullmatch(rf"{re.escape(base_stem)}_(\d+){re.escape(suffix)}", path.name)
    if match:
        return (int(match.group(1)), path.name)
    match = re.fullmatch(
        rf"{re.escape(base_stem)}__(customer|developer){_VARIANT_TAIL_RE}{re.escape(suffix)}",
        path.name,
    )
    if match:
        # 旧的开发编号先保持原顺序，客户命名空间排在其后。
        return (1000 + int(path.stem.rsplit("_", 1)[-1]), path.name)
    return (9999, path.name)


def template_variant_paths(root: Path, filename: str) -> list[Path]:
    if not root.exists():
        return []
    base_stem, suffix = _template_variant_pattern(filename)
    paths = [
        path
        for path in root.glob(f"{base_stem}*{suffix}")
        if path.name == filename
        or re.fullmatch(rf"{re.escape(base_stem)}_\d+{re.escape(suffix)}", path.name)
        or re.fullmatch(
            rf"{re.escape(base_stem)}__(?:customer|developer){_VARIANT_TAIL_RE}{re.escape(suffix)}",
            path.name,
        )
    ]
    return sorted(paths, key=lambda path: _template_variant_sort_key(path, filename))


def next_template_variant_path(root: Path, filename: str, *, owner: str = "legacy") -> Path:
    base_stem, suffix = _template_variant_pattern(filename)
    if owner == "customer":
        # 客户模板一律带本机 MAC 命名空间，即使族里还没有基础文件也
        # 不占用基础名——更新包里的官方基础文件永远不会和客户文件互相
        # 覆盖；多台机器采集的文件汇总/分发也不重名。
        machine = machine_template_id()
        index = 1
        while True:
            target = root / f"{base_stem}{CUSTOM_TEMPLATE_MARKER}_{machine}_{index}{suffix}"
            if not target.exists():
                return target
            index += 1
    first = root / filename
    if not first.exists():
        return first
    if owner != "developer":
        index = 2
        while True:
            target = root / f"{base_stem}_{index}{suffix}"
            if not target.exists():
                return target
            index += 1
    index = 1
    while True:
        target = root / f"{base_stem}{DEVELOPER_TEMPLATE_MARKER}_{index}{suffix}"
        if not target.exists():
            return target
        index += 1


def next_popup_template_path(buttons_dir: Path, *, owner: str = "legacy") -> Path:
    if owner == "customer":
        machine = machine_template_id()
        index = 1
        while True:
            target = buttons_dir / f"popup_custom{CUSTOM_TEMPLATE_MARKER}_{machine}_{index}.jpg"
            if not target.exists():
                return target
            index += 1
    index = 1
    while True:
        target = buttons_dir / f"popup_custom_{index}.jpg"
        if not target.exists():
            return target
        index += 1


def next_back_button_template_path(buttons_dir: Path, *, owner: str = "legacy") -> Path:
    if owner == "customer":
        machine = machine_template_id()
        index = 1
        while True:
            target = buttons_dir / f"btn_返回按钮{CUSTOM_TEMPLATE_MARKER}_{machine}_{index}.jpg"
            if not target.exists():
                return target
            index += 1
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
    return (
        filename == "btn_返回按钮.jpg"
        or re.fullmatch(r"btn_返回按钮_\d+\.jpg", filename) is not None
        or re.fullmatch(rf"btn_返回按钮__(?:customer|developer){_VARIANT_TAIL_RE}\.jpg", filename) is not None
    )


def import_template_rules() -> tuple[TemplateRule | PopupTemplateRule, ...]:
    return (*FIXED_TEMPLATE_RULES, POPUP_TEMPLATE_RULE, BACK_BUTTON_TEMPLATE_RULE)


def import_target_path(
    key: str,
    buttons_dir: Path,
    screens_dir: Path,
    *,
    owner: str = "legacy",
) -> Path:
    if key == POPUP_TEMPLATE_RULE.key:
        return next_popup_template_path(buttons_dir, owner=owner)
    if key == BACK_BUTTON_TEMPLATE_RULE.key:
        return next_back_button_template_path(buttons_dir, owner=owner)
    rule = template_rule(key)
    root = buttons_dir if rule.group == "buttons" else screens_dir
    return next_template_variant_path(root, rule.filename, owner=owner)


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
