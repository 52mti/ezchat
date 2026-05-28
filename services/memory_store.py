from __future__ import annotations

import hashlib
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PERSONA_FIELDS = ("requestPersona", "name", "age", "work", "location", "renshe")


def normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def build_persona_signature(persona: dict[str, Any]) -> str:
    raw = "|".join(normalize_text(str(persona.get(field, "") or "")) for field in PERSONA_FIELDS)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class MemoryStore:
    def __init__(self, root: str | Path = "runtime/memory") -> None:
        self.root = Path(root)
        self._lock = threading.Lock()

    def _path(self, device_ip: str, app_name: str) -> Path:
        safe_ip = normalize_text(device_ip).replace(":", "_")
        safe_app = normalize_text(app_name) or "unknown"
        return self.root / safe_ip / safe_app / "conversations.yaml"

    @staticmethod
    def _empty_payload() -> dict[str, Any]:
        return {"schema_version": 1, "records": {}}

    @staticmethod
    def build_record_key(
        device_ip: str,
        app_name: str,
        identity_mode: str,
        identity_key: str,
        persona_signature: str,
    ) -> str:
        raw = "|".join(
            [
                normalize_text(device_ip),
                normalize_text(app_name),
                normalize_text(identity_mode),
                normalize_text(identity_key),
                normalize_text(persona_signature),
            ]
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def key(self, contact_name: str, persona: dict[str, Any]) -> str:
        return f"{normalize_text(contact_name)}|{build_persona_signature(persona)}"

    def _backup_bad_file(self, path: Path) -> None:
        if not path.exists():
            return
        backup = path.with_name(f"{path.name}.bad.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        try:
            path.replace(backup)
        except OSError:
            return

    def _load(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return self._empty_payload()
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            self._backup_bad_file(path)
            return self._empty_payload()
        if not isinstance(data, dict):
            self._backup_bad_file(path)
            return self._empty_payload()

        records = data.get("records", {})
        if isinstance(records, list):
            migrated: dict[str, Any] = {}
            for item in records:
                if not isinstance(item, dict):
                    continue
                record_key = str(item.get("recordKey") or item.get("record_key") or "")
                if record_key:
                    migrated[record_key] = item
            records = migrated
        if not isinstance(records, dict):
            records = {}

        return {"schema_version": int(data.get("schema_version", 1) or 1), "records": records}

    def _save(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        tmp_path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)

    def _record_keys(
        self,
        device_ip: str,
        app_name: str,
        contact_name: str,
        persona: dict[str, Any],
    ) -> tuple[str, str]:
        identity_key = normalize_text(contact_name)
        persona_signature = build_persona_signature(persona)
        return (
            self.build_record_key(device_ip, app_name, "contact_name", identity_key, persona_signature),
            self.key(contact_name, persona),
        )

    def get_conversation_id(
        self,
        device_ip: str,
        app_name: str,
        contact_name: str,
        persona: dict[str, Any],
    ) -> str:
        path = self._path(device_ip, app_name)
        record_key, legacy_key = self._record_keys(device_ip, app_name, contact_name, persona)
        with self._lock:
            records = self._load(path).get("records", {})
            record = records.get(record_key) or records.get(legacy_key) or {}
            return str(record.get("conversation_id") or record.get("conversationId") or "")

    def upsert_record(
        self,
        device_ip: str,
        app_name: str,
        contact_name: str,
        persona: dict[str, Any],
        conversation_id: str,
        *,
        last_question_text: str = "",
        last_reply_text: str = "",
        last_error: str = "",
    ) -> None:
        path = self._path(device_ip, app_name)
        identity_key = normalize_text(contact_name)
        persona_signature = build_persona_signature(persona)
        record_key = self.build_record_key(device_ip, app_name, "contact_name", identity_key, persona_signature)
        with self._lock:
            data = self._load(path)
            records = data.setdefault("records", {})
            existing = records.get(record_key, {}) if isinstance(records.get(record_key), dict) else {}
            records[record_key] = {
                **existing,
                "record_key": record_key,
                "device_ip": normalize_text(device_ip),
                "app_name": normalize_text(app_name),
                "identity_mode": "contact_name",
                "identity_key": identity_key,
                "contact_name": contact_name,
                "persona_signature": persona_signature,
                "conversation_id": conversation_id,
                "last_question_text": last_question_text,
                "last_reply_text": last_reply_text,
                "last_error": last_error,
                "updated_at": _now_iso(),
            }
            self._save(path, data)
