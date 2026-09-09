"""按 App 共享系统消息规则的本地队列与同步客户端。

客户端发现新规则后先写本地待上传队列，再调用配置的上传入口；上传成功后
拉取同一 App 的共享规则。上传入口必须是 OSS 预签名/API 网关，客户端不
保存或上传 COS 密钥；公开 COS 地址只用于拉取规则。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

DEFAULT_ROOT = Path("runtime") / "rule-learning"
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def anonymous_id(value: str, *, salt: str = "") -> str:
    """生成不可逆标识，避免上传客户标识或设备 IP 明文。"""
    value = str(value or "").strip()
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()[:16] if value else ""


def _lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve()).lower()
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


def _safe_app_name(app_name: str) -> str:
    value = "".join(char for char in str(app_name).strip().lower() if char.isalnum() or char in "_-." )
    if not value or value in {".", ".."}:
        raise ValueError("invalid app name")
    return value


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return copy.deepcopy(default)
    return payload if isinstance(payload, dict) else copy.deepcopy(default)


class RuleLearningStore:
    """保存 App 级规则，并在上传成功后同步该 App 的共享规则。"""

    def __init__(self, root: str | Path = DEFAULT_ROOT, *, timeout: float = 10.0) -> None:
        self.root = Path(root)
        self.timeout = max(1.0, float(timeout))

    def cache_path(self, app_name: str) -> Path:
        return self.root / _safe_app_name(app_name) / "rules.json"

    def pending_dir(self, app_name: str) -> Path:
        return self.root / _safe_app_name(app_name) / "pending"

    def _cache_keywords(self, app_name: str) -> list[str]:
        payload = _load(self.cache_path(app_name), {"schema_version": 1, "app": app_name, "rules": []})
        return list(dict.fromkeys(
            " ".join(str(item.get("keyword", "")).split()).strip()
            for item in payload.get("rules", [])
            if isinstance(item, dict) and str(item.get("keyword", "")).strip()
        ))

    def append_keywords(
        self, app_name: str, keywords: Iterable[str], *, client_id: str = "", device_id: str = "", source: str = "vision_model"
    ) -> int:
        app = _safe_app_name(app_name)
        values = list(dict.fromkeys(" ".join(str(value or "").split()).strip() for value in keywords if str(value or "").strip()))
        if not values:
            return 0
        cache_path = self.cache_path(app)
        with _lock_for(cache_path):
            existing_values = self._cache_keywords(app)
            existing = set(existing_values)
            added = [value for value in values if value not in existing]
            if not added:
                return 0
            all_keywords = existing_values + added
            _atomic_write(cache_path, {"schema_version": 1, "app": app, "rules": [{"keyword": value} for value in all_keywords]})
            event = {
                "schema_version": 1, "event_id": uuid.uuid4().hex, "app": app, "keywords": added,
                "client_id": anonymous_id(client_id, salt=app), "device_id": anonymous_id(device_id, salt=app),
                "source": source, "created_at": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_write(self.pending_dir(app) / f"{event['event_id']}.json", event)
            return len(added)

    def load_keywords(self, app_name: str) -> list[str]:
        return self._cache_keywords(_safe_app_name(app_name))

    def sync(self, app_name: str, *, upload_url: str = "", rules_url: str = "", bearer_token: str = "") -> list[str]:
        """上传当前 App 的待处理事件，并拉取该 App 的最新共享规则。"""
        app = _safe_app_name(app_name)
        headers = {"Authorization": f"Bearer {bearer_token}"} if bearer_token.strip() else {}
        for path in sorted(self.pending_dir(app).glob("*.json")):
            if not upload_url.strip():
                break
            try:
                response = requests.post(upload_url.strip().format(app=app), json=_load(path, {}), headers=headers, timeout=self.timeout)
                response.raise_for_status()
            except (OSError, requests.RequestException, ValueError):
                break
            else:
                path.unlink(missing_ok=True)
        if not rules_url.strip():
            return self.load_keywords(app)
        try:
            response = requests.get(rules_url.strip().format(app=app), headers=headers, timeout=self.timeout)
            response.raise_for_status()
            remote = response.json()
        except (OSError, requests.RequestException, ValueError, json.JSONDecodeError):
            return self.load_keywords(app)
        remote_rules = remote.get("rules", []) if isinstance(remote, dict) else remote
        remote_keywords = []
        if isinstance(remote_rules, list):
            for item in remote_rules:
                value = item.get("keyword", "") if isinstance(item, dict) else item
                value = " ".join(str(value or "").split()).strip()
                if value and value not in remote_keywords:
                    remote_keywords.append(value)
        all_keywords = list(dict.fromkeys(self.load_keywords(app) + remote_keywords))
        cache_path = self.cache_path(app)
        with _lock_for(cache_path):
            _atomic_write(cache_path, {"schema_version": 1, "app": app, "rules": [{"keyword": value} for value in all_keywords]})
        return all_keywords
