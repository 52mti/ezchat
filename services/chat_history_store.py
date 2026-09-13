from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .memory_store import build_persona_signature, normalize_text


GREETING_TEXTS = {"你好"}
GREETING_REASONS = {"greeting", "greeting_reply", "new_greeting"}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_name(value: str) -> str:
    normalized = normalize_text(value).replace(":", "_")
    normalized = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]+", "_", normalized)
    return normalized.strip("._") or "unknown"


def _clean_message_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", "", text)
    return text.strip("，。！？,.!?;；:：\"'“”‘’（）()[]【】")


def is_greeting_message(text: Any, reason: str = "") -> bool:
    if str(reason or "").strip() in GREETING_REASONS:
        return True
    return _clean_message_text(text) in GREETING_TEXTS


class ChatHistoryStore:
    def __init__(self, root: str | Path = "runtime/chat_history") -> None:
        self.root = Path(root)
        self._lock = threading.Lock()

    def _path(self, device_ip: str, app_name: str) -> Path:
        return self.root / _safe_name(device_ip) / _safe_name(app_name) / "history.sqlite"

    def _connect(self, device_ip: str, app_name: str) -> sqlite3.Connection:
        path = self._path(device_ip, app_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema(conn)
        return conn

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_ip TEXT NOT NULL,
                app_name TEXT NOT NULL,
                persona_signature TEXT NOT NULL,
                contact_name TEXT NOT NULL,
                direction TEXT NOT NULL,
                content TEXT NOT NULL,
                message_fingerprint TEXT NOT NULL DEFAULT '',
                conversation_id TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                round_key TEXT NOT NULL,
                created_at TEXT NOT NULL,
                meta_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_chat_history_message
            ON messages (
                device_ip,
                app_name,
                persona_signature,
                contact_name,
                round_key,
                direction
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS ix_chat_history_contacts
            ON messages (device_ip, app_name, persona_signature, contact_name, id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS ix_chat_history_conversation
            ON messages (device_ip, app_name, persona_signature, conversation_id, id)
            """
        )

    @staticmethod
    def _persona_signature(persona: dict[str, Any] | None) -> str:
        return build_persona_signature(persona if isinstance(persona, dict) else {})

    @staticmethod
    def _round_key(
        *,
        device_ip: str,
        app_name: str,
        persona_signature: str,
        contact_name: str,
        incoming_text: str,
        outgoing_text: str,
        message_fingerprint: str,
    ) -> str:
        if message_fingerprint:
            return message_fingerprint
        raw = "|".join([device_ip, app_name, persona_signature, contact_name, incoming_text, outgoing_text])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def append_exchange(
        self,
        *,
        device_ip: str,
        app_name: str,
        contact_name: str,
        persona: dict[str, Any] | None,
        incoming_text: str,
        outgoing_text: str,
        message_fingerprint: str = "",
        conversation_id: str = "",
        source: str = "",
        reason: str = "",
        created_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        contact = normalize_text(contact_name)
        incoming = str(incoming_text or "").strip()
        outgoing = str(outgoing_text or "").strip()
        if not contact or not incoming or not outgoing:
            return False
        if is_greeting_message(incoming, reason):
            return False

        persona_signature = self._persona_signature(persona)
        round_key = self._round_key(
            device_ip=normalize_text(device_ip),
            app_name=normalize_text(app_name),
            persona_signature=persona_signature,
            contact_name=contact,
            incoming_text=incoming,
            outgoing_text=outgoing,
            message_fingerprint=str(message_fingerprint or ""),
        )
        timestamp = created_at or _now_iso()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
        rows = [
            (
                normalize_text(device_ip),
                normalize_text(app_name),
                persona_signature,
                contact,
                "incoming",
                incoming,
                str(message_fingerprint or ""),
                str(conversation_id or ""),
                str(source or ""),
                round_key,
                timestamp,
                meta_json,
            ),
            (
                normalize_text(device_ip),
                normalize_text(app_name),
                persona_signature,
                contact,
                "outgoing",
                outgoing,
                str(message_fingerprint or ""),
                str(conversation_id or ""),
                str(source or ""),
                round_key,
                timestamp,
                meta_json,
            ),
        ]
        with self._lock:
            conn = self._connect(device_ip, app_name)
            try:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO messages (
                        device_ip,
                        app_name,
                        persona_signature,
                        contact_name,
                        direction,
                        content,
                        message_fingerprint,
                        conversation_id,
                        source,
                        round_key,
                        created_at,
                        meta_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                conn.commit()
                return conn.total_changes > 0
            except sqlite3.DatabaseError:
                self._backup_bad_database(self._path(device_ip, app_name))
                raise
            finally:
                conn.close()

    def list_contacts(
        self,
        *,
        device_ip: str,
        app_name: str,
        persona: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        persona_signature = self._persona_signature(persona)
        path = self._path(device_ip, app_name)
        if not path.exists():
            return []
        with self._lock:
            conn = self._connect(device_ip, app_name)
            try:
                rows = conn.execute(
                    """
                    WITH grouped AS (
                        SELECT
                            CASE WHEN conversation_id <> '' THEN 'conversation:' || conversation_id
                                 ELSE 'contact:' || contact_name END AS group_key,
                            MAX(id) AS max_id,
                            COUNT(*) AS message_count
                        FROM messages
                        WHERE device_ip = ?
                          AND app_name = ?
                          AND persona_signature = ?
                        GROUP BY group_key
                    )
                    SELECT
                        m.contact_name,
                        m.conversation_id,
                        m.content AS last_message,
                        m.direction AS last_direction,
                        m.created_at AS last_message_at,
                        grouped.message_count
                    FROM grouped
                    JOIN messages m ON m.id = grouped.max_id
                    ORDER BY m.created_at DESC, m.id DESC
                    """,
                    (normalize_text(device_ip), normalize_text(app_name), persona_signature),
                ).fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()

    def get_messages(
        self,
        *,
        device_ip: str,
        app_name: str,
        contact_name: str = "",
        persona: dict[str, Any] | None = None,
        conversation_id: str = "",
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        persona_signature = self._persona_signature(persona)
        path = self._path(device_ip, app_name)
        if not path.exists():
            return []
        with self._lock:
            conn = self._connect(device_ip, app_name)
            try:
                if conversation_id:
                    query = """
                    SELECT
                        id,
                        contact_name,
                        direction,
                        content,
                        created_at,
                        message_fingerprint,
                        conversation_id,
                        source
                    FROM messages
                    WHERE device_ip = ?
                      AND app_name = ?
                      AND persona_signature = ?
                      AND conversation_id = ?
                    ORDER BY id ASC
                    LIMIT ?
                    """
                    params = (
                        normalize_text(device_ip),
                        normalize_text(app_name),
                        persona_signature,
                        str(conversation_id).strip(),
                        max(1, int(limit)),
                    )
                else:
                    query = """
                    SELECT
                        id, contact_name, direction, content, created_at,
                        message_fingerprint, conversation_id, source
                    FROM messages
                    WHERE device_ip = ? AND app_name = ?
                      AND persona_signature = ? AND contact_name = ?
                    ORDER BY id ASC LIMIT ?
                    """
                    params = (
                        normalize_text(device_ip), normalize_text(app_name),
                        persona_signature, normalize_text(contact_name),
                        max(1, int(limit)),
                    )
                rows = conn.execute(query, params).fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()

    @staticmethod
    def _backup_bad_database(path: Path) -> None:
        if not path.exists():
            return
        backup = path.with_name(f"{path.name}.bad.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        try:
            os.replace(path, backup)
        except OSError:
            return
