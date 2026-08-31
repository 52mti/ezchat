from __future__ import annotations

import hashlib
import os
import re
import threading
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml


PERSONA_FIELDS = ("requestPersona", "name", "age", "work", "location", "renshe")


def normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def _normalize_message_text(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", str(value or "")).lower()


def _message_match_score(current: Any, historical: Any) -> int:
    current_text = _normalize_message_text(current)
    historical_text = _normalize_message_text(historical)
    if not current_text or not historical_text:
        return 0
    if current_text == historical_text:
        return 100
    shorter = min(len(current_text), len(historical_text))
    if shorter >= 4 and (current_text in historical_text or historical_text in current_text):
        return 92
    similarity = SequenceMatcher(None, current_text, historical_text).ratio()
    return int(round(similarity * 100)) if similarity >= 0.88 else 0


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

    def find_continuity_match(
        self,
        device_ip: str,
        app_name: str,
        persona: dict[str, Any],
        *,
        sender1_messages: list[str] | None = None,
        sender2_messages: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Find an unambiguous previous conversation from visible chat bubbles."""
        path = self._path(device_ip, app_name)
        persona_signature = build_persona_signature(persona)
        incoming = [str(item) for item in (sender1_messages or []) if str(item).strip()][-3:]
        outgoing = [str(item) for item in (sender2_messages or []) if str(item).strip()][-3:]
        if not outgoing:
            return None

        with self._lock:
            records = self._load(path).get("records", {})

        candidates: dict[str, dict[str, Any]] = {}
        for record in records.values() if isinstance(records, dict) else []:
            if not isinstance(record, dict) or str(record.get("persona_signature") or "") != persona_signature:
                continue
            conversation_id = str(record.get("conversation_id") or record.get("conversationId") or "")
            if not conversation_id:
                continue
            reply_score = max((_message_match_score(item, record.get("last_reply_text", "")) for item in outgoing), default=0)
            if reply_score < 92:
                continue
            question_score = max((_message_match_score(item, record.get("last_question_text", "")) for item in incoming), default=0)
            candidate = {
                "conversation_id": conversation_id,
                "contact_name": str(record.get("contact_name") or record.get("identity_key") or ""),
                "score": reply_score + (40 if question_score >= 88 else 0),
                "reply_score": reply_score,
                "question_score": question_score,
            }
            existing = candidates.get(conversation_id)
            if existing is None or candidate["score"] > existing["score"]:
                candidates[conversation_id] = candidate

        ranked = sorted(candidates.values(), key=lambda item: int(item["score"]), reverse=True)
        if not ranked:
            return None
        best = ranked[0]
        if len(ranked) > 1 and int(best["score"]) - int(ranked[1]["score"]) < 20:
            return None
        return best

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
            # OCR 昵称修正后仍应绑定到同一个 AI 会话，而不是拆出新会话。
            if conversation_id:
                for candidate_key, candidate in records.items():
                    if not isinstance(candidate, dict) or candidate_key == record_key:
                        continue
                    if (
                        str(candidate.get("conversation_id") or candidate.get("conversationId") or "")
                        == conversation_id
                        and str(candidate.get("persona_signature") or "") == persona_signature
                    ):
                        existing = candidate
                        break
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
