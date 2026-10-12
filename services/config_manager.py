from __future__ import annotations

import copy
import ipaddress
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "global": {
        "license_api_url": "",
        "project_secret": "",
        "greetings_dir": "runtime/greetings",
        "ocr_max_concurrency": 1,
        "rule_learning": {
            "upload_url": "",
            "rules_url": "https://jixiebi-1311918180.cos.ap-shanghai.myqcloud.com/rule-learning/{app}/rules.json",
            "bearer_token": "",
        },
        "persona": {
            "name": "",
            "age": "",
            "work": "",
            "location": "",
            "renshe": "",
        },
    },
    "devices": [],
    "log_retention_days": 7,
}

DEFAULT_OUTBOX: dict[str, Any] = {"schema_version": 1, "messages": []}
CURRENT_SCHEMA_VERSION = 3
DEFAULT_MATCH_THRESHOLD = 0.7

APP_LEVEL_KEYS = {
    "ocr",
    "ocr_modes",
    "dialogue_engine",
    "dialogue_llm",
    "templates",
    "match_threshold",
    "popup_match_threshold",
    "enable_enhanced_template_match",
    "template_match_region_expand_ratio",
    "template_match_region_expand_ratios",
    "template_match_scales",
    "template_match_scales_by_key",
    "template_match_channels",
    "template_match_channels_by_key",
    "template_match_regions",
    "template_match_thresholds",
    "screen_detection_rules",
}
DEVICE_LEVEL_KEYS = {
    "poll_interval",
    "step_timeout",
    "keep_awake_interval_seconds",
    "exclude_ocr_wait_from_step_timeout",
    "idle_poll_interval_seconds",
}
DEVICE_APP_LEVEL_KEYS = {
    "persona",
    "photo_sent_cooldown_seconds",
    "online_photo_send_count",
    # 人设 AI 新链路客户域键（docs/13 五/六节，2026-09-13）：设备 IP+App 粒度
    "persona_profile_json",
    "greetings_txt",
    "judge_enabled",
    "greeting_ai_enabled",
    "history_turns",
    "summary_interval",
}
APP_CONFIG_FILE_NAME = "config.json"
APP_DIALOGUE_OCR_CONFIG_FILE_NAME = "dialogue_ocr_config.json"
APP_LIST_OCR_CONFIG_FILE_NAME = "list_ocr_config.json"

# 开发域专属键（移植自 douyin）：视觉调参 / 模板路径 / OCR 引擎配置只随
# apps/<app>/ 随包默认值下发。客户层（runtime/configs/<app>/）在读取和
# 写盘两端剥离这些键，保证随包默认值的更新对客户机始终生效；客户层只
# 保留客户自己的设定（popup.* 模板规则、dialogue/list OCR 模式规则、
# 业务参数）。
DEVELOPER_SCOPED_APP_KEYS = frozenset({
    "screen_detection_rules",
    "match_threshold",
    "popup_match_threshold",
    "enable_enhanced_template_match",
    "template_match_region_expand_ratio",
    "template_match_scales",
    "template_match_scales_by_key",
    "template_match_channels",
    "template_match_channels_by_key",
    "templates",
    "ocr",
})


def _strip_customer_layer_template_scopes(payload: dict[str, Any]) -> dict[str, Any]:
    """客户层只保留客户自己的设定。

    - 开发域视觉键整体剔除；
    - template_match_regions / template_match_thresholds 只保留 popup.*；
    - ocr_modes 只保留 dialogue / list（points 等其余模式属开发域），且各模式
      剥离 preprocess——旧布局迁移曾把整份默认值（含 scale）冻进客户层，
      按键遮蔽随包默认值，开发端性能调优通道被冻结。
    """
    for config_key in DEVELOPER_SCOPED_APP_KEYS:
        payload.pop(config_key, None)
    for config_key in ("template_match_regions", "template_match_thresholds"):
        value = payload.get(config_key)
        if not isinstance(value, dict):
            continue
        popup_only = {k: v for k, v in value.items() if str(k).startswith("popup.")}
        if popup_only:
            payload[config_key] = popup_only
        else:
            payload.pop(config_key, None)
    ocr_modes = payload.get("ocr_modes")
    if isinstance(ocr_modes, dict):
        customer_modes = {}
        for mode_name, mode_payload in ocr_modes.items():
            if mode_name not in ("dialogue", "list") or not isinstance(mode_payload, dict):
                continue
            mode_payload.pop("preprocess", None)
            customer_modes[mode_name] = mode_payload
        if customer_modes:
            payload["ocr_modes"] = customer_modes
        else:
            payload.pop("ocr_modes", None)
    return payload


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _device_sort_key(device: dict[str, Any]) -> tuple[int, tuple[int, ...] | tuple[str]]:
    ip = str(device.get("ip", "")).strip()
    try:
        parsed = ipaddress.ip_address(ip)
    except ValueError:
        return (1, (ip,))
    return (0, tuple(int(part) for part in parsed.packed))


def _normalize_devices(devices: list[Any]) -> list[dict[str, Any]]:
    merged_by_ip: dict[str, dict[str, Any]] = {}
    ordered_ips: list[str] = []
    for raw_device in devices:
        if not isinstance(raw_device, dict):
            continue
        ip = str(raw_device.get("ip", "")).strip()
        if not ip:
            continue
        if ip not in merged_by_ip:
            merged_by_ip[ip] = {"ip": ip}
            ordered_ips.append(ip)
        merged_by_ip[ip].update(copy.deepcopy(raw_device))
        merged_by_ip[ip]["ip"] = ip
    normalized = [merged_by_ip[ip] for ip in ordered_ips]
    normalized.sort(key=_device_sort_key)
    return normalized


def _split_override_scopes(override: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    app_level: dict[str, Any] = {}
    device_level: dict[str, Any] = {}
    device_app_level: dict[str, Any] = {}
    for key, value in override.items():
        if key in APP_LEVEL_KEYS:
            app_level[key] = copy.deepcopy(value)
        elif key in DEVICE_LEVEL_KEYS:
            device_level[key] = copy.deepcopy(value)
        elif key in DEVICE_APP_LEVEL_KEYS:
            device_app_level[key] = copy.deepcopy(value)
    return app_level, device_level, device_app_level


def _normalize_app_templates(app_name: str, app_config: dict[str, Any]) -> None:
    templates = app_config.get("templates", {})
    if not isinstance(templates, dict):
        return
    for group in ("buttons", "screens"):
        value = templates.get(group)
        if not isinstance(value, str) or not value:
            continue
        normalized = value.replace("/", "\\")
        app_template_path = str(Path("apps") / str(app_name) / "templates" / group)
        if normalized.startswith("runtime\\templates\\") or "\\runtime\\templates\\" in normalized:
            templates[group] = app_template_path
            continue
        if normalized.startswith("apps\\") and f"\\{str(app_name)}\\templates\\{group}" in normalized:
            templates[group] = app_template_path


def _normalize_match_threshold(app_config: dict[str, Any]) -> None:
    if not isinstance(app_config, dict):
        return
    try:
        app_config["match_threshold"] = float(app_config.get("match_threshold", DEFAULT_MATCH_THRESHOLD))
    except (TypeError, ValueError):
        app_config["match_threshold"] = DEFAULT_MATCH_THRESHOLD


# App 层（apps/<app>/config_rules.py）在包导入时注册各自的配置规范化钩子。
# 配置服务只提供机制，不感知具体 App 的规则，保持 services 不反向依赖 apps。
_APP_CONFIG_NORMALIZERS: dict[str, Callable[[dict[str, Any]], None]] = {}


def register_app_config_normalizer(app_name: str, normalizer: Callable[[dict[str, Any]], None]) -> None:
    _APP_CONFIG_NORMALIZERS[str(app_name)] = normalizer


def _run_app_config_normalizer(app_name: str, app_config: dict[str, Any]) -> None:
    normalizer = _APP_CONFIG_NORMALIZERS.get(str(app_name))
    if normalizer is not None:
        normalizer(app_config)


def _normalize_app_configs(app_configs: dict[str, Any]) -> None:
    if not isinstance(app_configs, dict):
        return
    for app_name, app_config in app_configs.items():
        if isinstance(app_config, dict):
            _normalize_app_templates(str(app_name), app_config)
            _normalize_match_threshold(app_config)
            _run_app_config_normalizer(str(app_name), app_config)


def _load_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return copy.deepcopy(data) if isinstance(data, dict) else {}


def _normalize_secret_value(value: Any) -> str:
    text = str(value or "")
    for marker in ("\ufeff", "\u200b", "\u200c", "\u200d", "\u2060"):
        text = text.replace(marker, "")
    return text.strip()


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
        encoding="utf-8",
    ) as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        tmp_name = handle.name
    os.replace(tmp_name, path)


def _app_config_path(app_config_root: Path, app_name: str) -> Path:
    return app_config_root / str(app_name) / APP_CONFIG_FILE_NAME


def _app_dialogue_ocr_config_path(app_config_root: Path, app_name: str) -> Path:
    return app_config_root / str(app_name) / APP_DIALOGUE_OCR_CONFIG_FILE_NAME


def _app_list_ocr_config_path(app_config_root: Path, app_name: str) -> Path:
    return app_config_root / str(app_name) / APP_LIST_OCR_CONFIG_FILE_NAME


def _load_app_ocr_mode_override_files(app_config_root: Path, app_name: str) -> dict[str, Any]:
    ocr_modes: dict[str, Any] = {}
    for mode_name, path in (
        ("dialogue", _app_dialogue_ocr_config_path(app_config_root, app_name)),
        ("list", _app_list_ocr_config_path(app_config_root, app_name)),
    ):
        payload = _load_json_dict(path)
        if payload:
            ocr_modes[mode_name] = payload
    return {"ocr_modes": ocr_modes} if ocr_modes else {}


def _app_override_stamp(path: Path) -> tuple[int, int] | None:
    """文件存在性指纹：mtime_ns + size，用于跳过未变化文件的整读。

    多设备 worker 每个轮询周期都会读 App 级配置；stat（微秒级）代替
    6 次 open+read+parse（Windows 杀软实时扫描下单次数十毫秒），是
    ConfigManager._lock 持锁时长的数量级优化。
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _app_override_file_stamps(
    app_config_root: Path,
    app_name: str,
    defaults_root: Path | None = None,
) -> tuple[tuple[int, int] | None, ...]:
    roots = [app_config_root]
    if defaults_root is not None:
        roots.append(defaults_root)
    stamps: list[tuple[int, int] | None] = []
    for root in roots:
        stamps.append(_app_override_stamp(_app_config_path(root, app_name)))
        stamps.append(_app_override_stamp(_app_dialogue_ocr_config_path(root, app_name)))
        stamps.append(_app_override_stamp(_app_list_ocr_config_path(root, app_name)))
    return tuple(stamps)


def _load_app_override_file(
    app_config_root: Path,
    app_name: str,
    defaults_root: Path | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if defaults_root is not None:
        # 包内随包默认值（apps/<app>/，只读），客户层缺失的键由此兜底。
        payload = _load_json_dict(_app_config_path(defaults_root, app_name))
        payload = _deep_merge(payload, _load_app_ocr_mode_override_files(defaults_root, app_name))
    # 客户层先剥离开发域键再合并：随包默认值的视觉调参更新不被遮蔽。
    customer_payload = _load_json_dict(_app_config_path(app_config_root, app_name))
    customer_payload = _deep_merge(customer_payload, _load_app_ocr_mode_override_files(app_config_root, app_name))
    customer_payload = _strip_customer_layer_template_scopes(customer_payload)
    payload = _deep_merge(payload, customer_payload)
    app_level, _, legacy_device_app = _split_override_scopes(payload)
    # 旧布局把业务参数（时长/关键词等）写在 App 层文件里；读取时并回作为
    # 兜底值，设备层（devices[].app_overrides）合并时优先级更高。
    for key, value in legacy_device_app.items():
        app_level.setdefault(key, copy.deepcopy(value))
    _normalize_app_templates(app_name, app_level)
    _normalize_match_threshold(app_level)
    _run_app_config_normalizer(app_name, app_level)
    return app_level


def _default_app_name() -> str:
    # 延迟导入 App 注册表：仅旧版根配置迁移时需要，避免 services 在模块
    # 加载期反向依赖 apps 形成环。
    from apps import resolve_app_name

    return resolve_app_name(None)


def _normalize_device_scopes(device: dict[str, Any]) -> None:
    runtime_overrides = device.get("runtime_overrides", {})
    if not isinstance(runtime_overrides, dict):
        runtime_overrides = {}
    device["runtime_overrides"] = {
        key: copy.deepcopy(value)
        for key, value in runtime_overrides.items()
        if key in DEVICE_LEVEL_KEYS
    }

    app_overrides = device.get("app_overrides", {})
    if not isinstance(app_overrides, dict):
        device["app_overrides"] = {}
        return
    normalized: dict[str, dict[str, Any]] = {}
    for app_name, override in app_overrides.items():
        if not isinstance(override, dict):
            continue
        _, _, device_app_level = _split_override_scopes(override)
        if device_app_level:
            normalized[str(app_name)] = device_app_level
    device["app_overrides"] = normalized


def _merge_app_level_config(
    app_config_root: Path,
    app_name: str,
    updates: dict[str, Any],
    defaults_root: Path | None = None,
) -> None:
    if not updates:
        return
    # 保存基线只取客户层自身（不掺随包默认值），否则开发端默认会被复制
    # 进客户层文件，永久遮蔽后续默认值更新。
    existing = _load_json_dict(_app_config_path(app_config_root, app_name))
    existing = _deep_merge(existing, _load_app_ocr_mode_override_files(app_config_root, app_name))
    existing = _strip_customer_layer_template_scopes(existing)
    merged = _deep_merge(existing, updates)
    _normalize_app_templates(app_name, merged)
    _normalize_match_threshold(merged)
    # 客户层只保留客户自己的设定：dialogue/list 拆专属文件；其余开发端
    # 视觉键（含规范化器补回的默认值、points 等模式）写盘前统一剥离。
    ocr_modes = merged.get("ocr_modes", {})
    ocr_modes = copy.deepcopy(ocr_modes) if isinstance(ocr_modes, dict) else {}
    dialogue_mode = ocr_modes.pop("dialogue", None)
    list_mode = ocr_modes.pop("list", None)
    merged.pop("ocr_modes", None)
    merged = _strip_customer_layer_template_scopes(merged)
    _atomic_write_json(_app_config_path(app_config_root, app_name), merged)
    for mode_payload, path in (
        (dialogue_mode, _app_dialogue_ocr_config_path(app_config_root, app_name)),
        (list_mode, _app_list_ocr_config_path(app_config_root, app_name)),
    ):
        # preprocess（scale/rotate 等性能布局旋钮）属开发域，客户层 mode
        # 文件不落盘，与读端剥离互为镜像（读写两端剥离开发域键）。
        if isinstance(mode_payload, dict):
            mode_payload.pop("preprocess", None)
        if isinstance(mode_payload, dict) and mode_payload:
            _atomic_write_json(path, mode_payload)
        else:
            path.unlink(missing_ok=True)


def _merge_runtime_overrides(device: dict[str, Any], updates: dict[str, Any]) -> None:
    if not updates:
        return
    runtime_overrides = device.get("runtime_overrides", {})
    runtime_overrides = runtime_overrides if isinstance(runtime_overrides, dict) else {}
    runtime_overrides = _deep_merge(runtime_overrides, updates)
    device["runtime_overrides"] = runtime_overrides


def _merge_device_app_overrides(device: dict[str, Any], app_name: str, updates: dict[str, Any]) -> None:
    if not updates:
        return
    app_overrides = device.get("app_overrides", {})
    app_overrides = app_overrides if isinstance(app_overrides, dict) else {}
    existing = app_overrides.get(app_name, {})
    existing = existing if isinstance(existing, dict) else {}
    app_overrides[str(app_name)] = _deep_merge(existing, updates)
    device["app_overrides"] = app_overrides


def _apply_scoped_app_overrides(
    app_config_root: Path,
    device: dict[str, Any],
    app_overrides: dict[str, Any],
    defaults_root: Path | None = None,
) -> None:
    if not isinstance(app_overrides, dict):
        return
    for raw_app_name, raw_override in app_overrides.items():
        if not isinstance(raw_override, dict):
            continue
        app_name = str(raw_app_name)
        app_level, device_level, device_app_level = _split_override_scopes(raw_override)
        _merge_app_level_config(app_config_root, app_name, app_level, defaults_root=defaults_root)
        _merge_runtime_overrides(device, device_level)
        _merge_device_app_overrides(device, app_name, device_app_level)
    _normalize_device_scopes(device)


class ConfigManager:
    _lock = threading.Lock()

    def __init__(
        self,
        config_path: str | Path = "config.json",
        outbox_path: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        # 收件箱是运行数据（数据分层规范 L3），落 runtime/；显式传路径用于测试与隔离。
        self.outbox_path = (
            Path(outbox_path)
            if outbox_path is not None
            else self.config_path.parent / "runtime" / "outbox.json"
        )
        # 双文件分层：apps/<app>/ = 随包默认值（只读兜底，开发机上是源码）；
        # runtime/configs/<app>/ = 客户 App 级配置（UI 保存只写这里）。
        # 根 config.json 的 devices[].app_overrides = 客户对应设备的业务参数。
        fresh_install = not self.config_path.exists()
        self.app_config_root = self.config_path.parent / "runtime" / "configs"
        self.app_defaults_root = self.config_path.parent / "apps"
        self._migrate_legacy_app_configs(fresh_install=fresh_install)
        self._migrate_legacy_outbox()
        self._config = self._load_json(self.config_path, DEFAULT_CONFIG)
        self._outbox = self._load_json(self.outbox_path, DEFAULT_OUTBOX)
        # App 级配置按 (mtime_ns,size) 文件戳缓存：worker 每轮热刷新只需 stat，
        # 不再重复整读 6 个 JSON（见 _app_override_file_stams 注释）。
        self._app_overrides_cache: dict[str, tuple[tuple[tuple[int, int] | None, ...], dict[str, Any]]] = {}
        self._migrate_if_needed()
        self._normalize_loaded_device_scopes()
        self._apply_environment_overrides()

    def _migrate_legacy_outbox(self) -> None:
        """旧布局的 outbox.json 在根目录；一次性搬入 runtime/（数据分层规范）。

        冲突裁决：非空者优先于空文件（防止后写入的空壳盖掉真实账本），
        其次修改时间新者胜（旧版程序停写前的最后一份必须赢，否则已发送
        标记回退会导致重复回复）。
        """
        legacy = self.config_path.parent / "outbox.json"
        if self.outbox_path == legacy or not legacy.is_file():
            return

        def _message_count(path: Path) -> int:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return -1
            if not isinstance(payload, dict):
                return -1
            return len(payload.get("messages", []) or [])

        try:
            if not self.outbox_path.exists():
                self.outbox_path.parent.mkdir(parents=True, exist_ok=True)
                legacy.replace(self.outbox_path)
                return
            legacy_count = _message_count(legacy)
            target_count = _message_count(self.outbox_path)
            legacy_wins = (
                (legacy_count > 0 and target_count <= 0)
                or (
                    legacy_count > 0
                    and target_count > 0
                    and legacy.stat().st_mtime > self.outbox_path.stat().st_mtime
                )
            )
            if legacy_wins:
                legacy.replace(self.outbox_path)
        except OSError:
            pass

    def _migrate_legacy_app_configs(self, *, fresh_install: bool) -> None:
        """旧单层布局的一次性迁移（stamp: runtime/configs/.layout_v2）。

        - 旧布局把客户界面数据存放在 apps/<app>/ 三个 json；本迁移把它们
          复制到客户层 runtime/configs/<app>/，客户机上随后消费源文件。
        - runtime/configs/<app>/ 是客户层，只装客户自己的设定；全新安装
          从空开始——不把随包默认值整份复制进来，否则客户层按键遮蔽
          默认值，开发端后续更新默认值的通道会被冻结。
        - 新布局里 apps/<app>/ json 是开发端随包默认（defaults 层）：
          客户层按键遮蔽它，它遮蔽代码层默认；stamp 之后随更新包落地的
          json 不再被本迁移触碰。
        """
        stamp = self.app_config_root / ".layout_v2"
        if stamp.is_file():
            return
        packaged_launch = bool(str(os.environ.get("JIXIEBI_PACKAGE_ROOT", "") or "").strip())
        legacy_root = self.config_path.parent / "apps"
        stamp.parent.mkdir(parents=True, exist_ok=True)
        if fresh_install:
            stamp.write_text("fresh install: customer layers start empty\n", encoding="utf-8")
            return
        if not legacy_root.is_dir():
            stamp.write_text("no legacy apps configs\n", encoding="utf-8")
            return
        try:
            migrated = False
            for app_dir in sorted(legacy_root.iterdir()):
                if not app_dir.is_dir():
                    continue
                for file_name in (
                    APP_CONFIG_FILE_NAME,
                    APP_DIALOGUE_OCR_CONFIG_FILE_NAME,
                    APP_LIST_OCR_CONFIG_FILE_NAME,
                ):
                    source = app_dir / file_name
                    if not source.is_file():
                        continue
                    target = self.app_config_root / app_dir.name / file_name
                    if not target.is_file():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
                    if target.is_file() and target.read_bytes() == source.read_bytes():
                        if packaged_launch:
                            source.unlink()
                        migrated = True
            if migrated:
                marker = self.app_config_root / "migrated_from_apps.txt"
                marker.write_text(
                    f"legacy app configs migrated from {legacy_root}\n",
                    encoding="utf-8",
                )
        except OSError:
            return
        try:
            stamp.write_text("legacy app configs migrated\n", encoding="utf-8")
        except OSError:
            pass

    def _load_json(self, path: Path, default: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return copy.deepcopy(default)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return copy.deepcopy(default)
        if not isinstance(data, dict):
            return copy.deepcopy(default)
        return _deep_merge(default, data)

    def _atomic_write(self, path: Path, data: dict[str, Any]) -> None:
        _atomic_write_json(path, data)

    def _config_for_disk(self) -> dict[str, Any]:
        data = copy.deepcopy(self._config)
        global_config = data.get("global", {})
        if isinstance(global_config, dict):
            global_config.pop("apikey", None)
        return data

    def _write_config(self) -> None:
        self._atomic_write(self.config_path, self._config_for_disk())

    def _reload_config_for_write(self) -> None:
        self._config = self._load_json(self.config_path, DEFAULT_CONFIG)
        self._normalize_loaded_device_scopes()
        self._apply_environment_overrides()

    def _reload_outbox_for_write(self) -> None:
        self._outbox = self._load_json(self.outbox_path, DEFAULT_OUTBOX)

    def _backup_file(self, path: Path) -> Path | None:
        if not path.exists():
            return None
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        return backup

    def _migrate_if_needed(self) -> None:
        config_version = int(self._config.get("schema_version", 0) or 0)
        outbox_version = int(self._outbox.get("schema_version", 0) or 0)
        legacy_apps_present = "apps" in self._config
        if config_version < CURRENT_SCHEMA_VERSION or legacy_apps_present:
            self._backup_file(self.config_path)
            self._config = self._migrate_config(self._config, config_version, CURRENT_SCHEMA_VERSION)
            self._write_config()
        if outbox_version < CURRENT_SCHEMA_VERSION:
            self._backup_file(self.outbox_path)
            self._outbox = self._migrate_outbox(self._outbox, outbox_version, CURRENT_SCHEMA_VERSION)
            self._atomic_write(self.outbox_path, self._outbox)

    def _normalize_loaded_device_scopes(self) -> None:
        devices = self._config.get("devices", [])
        if not isinstance(devices, list):
            return
        changed = False
        for device in devices:
            if not isinstance(device, dict):
                continue
            before = copy.deepcopy(
                {
                    "runtime_overrides": device.get("runtime_overrides", {}),
                    "app_overrides": device.get("app_overrides", {}),
                }
            )
            _normalize_device_scopes(device)
            if before != {
                "runtime_overrides": device.get("runtime_overrides", {}),
                "app_overrides": device.get("app_overrides", {}),
            }:
                changed = True
        if changed:
            self._write_config()

    def _migrate_config(
        self,
        data: dict[str, Any],
        from_version: int,
        to_version: int,
    ) -> dict[str, Any]:
        migrated = _deep_merge(DEFAULT_CONFIG, data)
        migrated["schema_version"] = to_version
        legacy_app_configs = migrated.pop("apps", {})
        if not isinstance(legacy_app_configs, dict):
            legacy_app_configs = {}
        devices = migrated.get("devices", [])
        for app_name, app_config in list(legacy_app_configs.items()):
            if not isinstance(app_config, dict):
                continue
            normalized_app_config = copy.deepcopy(app_config)
            app_level, _, _ = _split_override_scopes(normalized_app_config)
            _merge_app_level_config(
                self.app_config_root, str(app_name), app_level, defaults_root=self.app_defaults_root
            )
        if isinstance(devices, list):
            migrated["devices"] = _normalize_devices(devices)
            for device in migrated["devices"]:
                if not isinstance(device, dict):
                    continue
                device.setdefault("selected_app", _default_app_name())
                device.setdefault("persona", {})
                runtime_overrides = device.get("runtime_overrides", {})
                runtime_overrides = runtime_overrides if isinstance(runtime_overrides, dict) else {}
                old_app_overrides = device.get("app_overrides", {})
                old_app_overrides = old_app_overrides if isinstance(old_app_overrides, dict) else {}
                device["runtime_overrides"] = runtime_overrides
                device["app_overrides"] = {}
                _apply_scoped_app_overrides(
                    self.app_config_root, device, old_app_overrides, defaults_root=self.app_defaults_root
                )
                _normalize_device_scopes(device)
        return migrated

    def _migrate_outbox(
        self,
        data: dict[str, Any],
        from_version: int,
        to_version: int,
    ) -> dict[str, Any]:
        migrated = _deep_merge(DEFAULT_OUTBOX, data)
        migrated["schema_version"] = to_version
        messages = migrated.get("messages", [])
        if not isinstance(messages, list):
            migrated["messages"] = []
        return migrated

    def _apply_environment_overrides(self) -> None:
        devices = self._config.get("devices", [])
        if isinstance(devices, list):
            for device in devices:
                if not isinstance(device, dict):
                    continue
                ip = str(device.get("ip", ""))
                env_key = "CHATBOT_LICENSE_" + ip.replace(".", "_")
                license_key = os.getenv(env_key)
                if license_key:
                    device["license_key"] = license_key

    def data(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._config)

    def devices(self) -> list[dict[str, Any]]:
        with self._lock:
            devices = self._config.get("devices", [])
            normalized = _normalize_devices(devices if isinstance(devices, list) else [])
            return copy.deepcopy(normalized)

    def find_device(self, device_ip: str) -> dict[str, Any] | None:
        with self._lock:
            devices = self._config.get("devices", [])
            if not isinstance(devices, list):
                return None
            for device in devices:
                if isinstance(device, dict) and str(device.get("ip", "")) == device_ip:
                    return copy.deepcopy(device)
            return None

    def update_device(self, device_ip: str, updates: dict[str, Any]) -> None:
        with self._lock:
            self._reload_config_for_write()
            devices = self._config.setdefault("devices", [])
            if not isinstance(devices, list):
                raise ValueError("devices must be a list")
            raw_updates = copy.deepcopy(updates)
            scoped_app_overrides = raw_updates.pop("app_overrides", None)
            runtime_updates = raw_updates.pop("runtime_overrides", None)
            for device in devices:
                if isinstance(device, dict) and str(device.get("ip", "")) == device_ip:
                    device.update(raw_updates)
                    if isinstance(runtime_updates, dict):
                        _merge_runtime_overrides(device, runtime_updates)
                    if isinstance(scoped_app_overrides, dict):
                        _apply_scoped_app_overrides(
                            self.app_config_root,
                            device,
                            scoped_app_overrides,
                            defaults_root=self.app_defaults_root,
                        )
                    _normalize_device_scopes(device)
                    self._config["devices"] = _normalize_devices(devices)
                    self._write_config()
                    return
            new_device = {"ip": device_ip, **raw_updates}
            if isinstance(runtime_updates, dict):
                _merge_runtime_overrides(new_device, runtime_updates)
            if isinstance(scoped_app_overrides, dict):
                _apply_scoped_app_overrides(
                    self.app_config_root, new_device, scoped_app_overrides, defaults_root=self.app_defaults_root
                )
            _normalize_device_scopes(new_device)
            devices.append(new_device)
            self._config["devices"] = _normalize_devices(devices)
            self._write_config()

    def add_device(self, device: dict[str, Any]) -> bool:
        device_ip = str(device.get("ip", "")).strip()
        if not device_ip:
            raise ValueError("device ip is required")
        with self._lock:
            self._reload_config_for_write()
            devices = self._config.setdefault("devices", [])
            if not isinstance(devices, list):
                raise ValueError("devices must be a list")
            if any(isinstance(existing, dict) and str(existing.get("ip", "")).strip() == device_ip for existing in devices):
                return False
            new_device = copy.deepcopy(device)
            _apply_scoped_app_overrides(
                self.app_config_root, new_device, new_device.get("app_overrides", {}), defaults_root=self.app_defaults_root
            )
            _normalize_device_scopes(new_device)
            devices.append(new_device)
            self._config["devices"] = _normalize_devices(devices)
            self._write_config()
            return True

    def remove_device(self, device_ip: str) -> bool:
        device_ip = str(device_ip).strip()
        if not device_ip:
            return False
        with self._lock:
            self._reload_config_for_write()
            devices = self._config.get("devices", [])
            if not isinstance(devices, list):
                return False
            remaining = [
                copy.deepcopy(device)
                for device in devices
                if not (isinstance(device, dict) and str(device.get("ip", "")).strip() == device_ip)
            ]
            if len(remaining) == len(devices):
                return False
            self._config["devices"] = _normalize_devices(remaining)
            self._write_config()
            return True

    def get_app_config(
        self,
        device_ip: str,
        app_name: str,
        default_config: dict[str, Any],
    ) -> dict[str, Any]:
        device = self.find_device(device_ip) or {}
        app_level = self.get_app_overrides(app_name)
        runtime_overrides = (
            copy.deepcopy(device.get("runtime_overrides", {}))
            if isinstance(device.get("runtime_overrides", {}), dict)
            else {}
        )
        device_app_overrides = (
            copy.deepcopy(device.get("app_overrides", {}).get(app_name, {}))
            if isinstance(device.get("app_overrides", {}), dict)
            and isinstance(device.get("app_overrides", {}).get(app_name, {}), dict)
            else {}
        )
        merged = _deep_merge(default_config, app_level)
        merged = _deep_merge(merged, runtime_overrides)
        merged = _deep_merge(merged, device_app_overrides)
        _normalize_match_threshold(merged)
        return merged

    def get_app_overrides(self, app_name: str) -> dict[str, Any]:
        with self._lock:
            cache_key = str(app_name)
            stamps = _app_override_file_stamps(self.app_config_root, cache_key, defaults_root=self.app_defaults_root)
            cached = self._app_overrides_cache.get(cache_key)
            if cached is not None and cached[0] == stamps:
                return copy.deepcopy(cached[1])
            payload = _load_app_override_file(
                self.app_config_root, str(app_name), defaults_root=self.app_defaults_root
            )
            self._app_overrides_cache[cache_key] = (stamps, copy.deepcopy(payload))
            return payload

    def update_app_config(self, app_name: str, updates: dict[str, Any]) -> None:
        scoped_updates, _, _ = _split_override_scopes(copy.deepcopy(updates))
        if not scoped_updates:
            return
        with self._lock:
            _merge_app_level_config(
                self.app_config_root, str(app_name), scoped_updates, defaults_root=self.app_defaults_root
            )
            # 写后主动失效：防极端情况下新写入与旧文件 mtime_ns+size 完全相同。
            self._app_overrides_cache.pop(str(app_name), None)

    def get_device_runtime_overrides(self, device_ip: str) -> dict[str, Any]:
        device = self.find_device(device_ip) or {}
        runtime_overrides = device.get("runtime_overrides", {})
        return copy.deepcopy(runtime_overrides if isinstance(runtime_overrides, dict) else {})

    def update_device_runtime(self, device_ip: str, updates: dict[str, Any]) -> None:
        runtime_updates = {
            key: copy.deepcopy(value)
            for key, value in updates.items()
            if key in DEVICE_LEVEL_KEYS
        }
        if runtime_updates:
            self.update_device(device_ip, {"runtime_overrides": runtime_updates})

    def get_device_app_overrides(self, device_ip: str, app_name: str) -> dict[str, Any]:
        device = self.find_device(device_ip) or {}
        app_overrides = device.get("app_overrides", {})
        if not isinstance(app_overrides, dict):
            return {}
        override = app_overrides.get(app_name, {})
        return copy.deepcopy(override if isinstance(override, dict) else {})

    def update_device_app_overrides(self, device_ip: str, app_name: str, updates: dict[str, Any]) -> None:
        _, _, device_app_updates = _split_override_scopes(copy.deepcopy(updates))
        if device_app_updates:
            self.update_device(device_ip, {"app_overrides": {app_name: device_app_updates}})

    def get(self, *keys: str, default: Any = None) -> Any:
        with self._lock:
            current: Any = self._config
            for key in keys:
                if not isinstance(current, dict) or key not in current:
                    return default
                current = current[key]
            return copy.deepcopy(current)

    def set(self, *keys: str, value: Any) -> None:
        if not keys:
            raise ValueError("at least one key is required")
        with self._lock:
            self._reload_config_for_write()
            current = self._config
            for key in keys[:-1]:
                current = current.setdefault(key, {})
                if not isinstance(current, dict):
                    raise ValueError(f"cannot set nested key under non-dict: {key}")
            current[keys[-1]] = value
            self._write_config()

    def update_global(self, key: str, value: Any) -> None:
        """写 global 段单键（如 persona_chat_model，docs/13 五节）。"""
        self.set("global", key, value=value)

    def save_outbox_entry(self, entry: dict[str, Any]) -> None:
        fingerprint = entry.get("message_fingerprint")
        if not fingerprint:
            raise ValueError("outbox entry requires message_fingerprint")
        with self._lock:
            self._reload_outbox_for_write()
            messages = self._outbox.setdefault("messages", [])
            for index, existing in enumerate(messages):
                if existing.get("message_fingerprint") == fingerprint:
                    merged = copy.deepcopy(existing)
                    merged.update(copy.deepcopy(entry))
                    if merged == existing:
                        # 重复标记同一状态（如重试路径的 mark_sent）：
                        # 内容未变化不重写盘，省一次全文件读改写。
                        return
                    messages[index] = merged
                    break
            else:
                messages.append(copy.deepcopy(entry))
            self._atomic_write(self.outbox_path, self._outbox)

    def load_outbox(self) -> list[dict[str, Any]]:
        with self._lock:
            messages = self._outbox.get("messages", [])
            return copy.deepcopy(messages if isinstance(messages, list) else [])

    def pending_outbox(self) -> list[dict[str, Any]]:
        return [entry for entry in self.load_outbox() if entry.get("pending_send")]
