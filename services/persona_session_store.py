"""人设对话会话存储（docs/13 第四节）。

runtime/persona_sessions/<设备IP>/<App>/sessions.sqlite，三表：
- sessions：组合键 (device_ip, app_name, contact_name) → 本地 session_id（uuid，不上送）
- turns：会话内单调 seq 的消息轮次（role=user 对方 / assistant 我方；kind=chat/greeting/synthetic_greeting）
- summaries：滚动"前情摘要"，summarized_up_to_seq 为已摘要水位

人设与会话身份解耦：人设文本变化不换会话（4.2 定稿）。
30 天未活跃会话由 cleanup_inactive_sessions 清理（4.1 保留策略）。
"""
from __future__ import annotations

import re
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

KIND_CHAT = "chat"
KIND_GREETING = "greeting"
KIND_SYNTHETIC_GREETING = "synthetic_greeting"

DEFAULT_INACTIVE_DAYS = 30


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_name(value: str) -> str:
    normalized = " ".join(str(value or "").strip().split())
    normalized = normalized.replace(":", "_")
    normalized = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]+", "_", normalized)
    return normalized.strip("._") or "unknown"


def _normalize_match_text(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", str(value or "")).lower()


def _message_match_score(current: Any, historical: Any) -> int:
    """气泡文本 vs 历史末轮相似度：一致 100、包含 92、比值 ≥0.88 取整，否则 0（沿用旧 find_continuity_match 口径）。"""
    current_text = _normalize_match_text(current)
    historical_text = _normalize_match_text(historical)
    if not current_text or not historical_text:
        return 0
    if current_text == historical_text:
        return 100
    shorter = min(len(current_text), len(historical_text))
    if shorter >= 4 and (current_text in historical_text or historical_text in current_text):
        return 92
    similarity = SequenceMatcher(None, current_text, historical_text).ratio()
    return int(round(similarity * 100)) if similarity >= 0.88 else 0


class PersonaSessionStore:
    def __init__(self, root: str | Path = "runtime/persona_sessions") -> None:
        self.root = Path(root)
        self._lock = threading.Lock()

    # ---------- 底层 ----------
    def _path(self, device_ip: str, app_name: str) -> Path:
        return self.root / _safe_name(device_ip) / _safe_name(app_name) / "sessions.sqlite"

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
            CREATE TABLE IF NOT EXISTS sessions (
              session_id TEXT PRIMARY KEY,
              device_ip TEXT NOT NULL,
              app_name TEXT NOT NULL,
              contact_name TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_persona_active_session
            ON sessions (device_ip, app_name, contact_name)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT NOT NULL,
              seq INTEGER NOT NULL,
              role TEXT NOT NULL,
              content TEXT NOT NULL,
              kind TEXT NOT NULL DEFAULT 'chat',
              created_at TEXT NOT NULL,
              UNIQUE (session_id, seq)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS ix_persona_turns_session
            ON turns (session_id, seq)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
              session_id TEXT PRIMARY KEY,
              summary TEXT NOT NULL,
              summarized_up_to_seq INTEGER NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )

    # ---------- 会话 ----------
    def find_or_create_session(self, device_ip: str, app_name: str, contact_name: str) -> dict[str, Any]:
        contact = _safe_name(contact_name)
        with self._lock:
            conn = self._connect(device_ip, app_name)
            try:
                row = conn.execute(
                    "SELECT * FROM sessions WHERE device_ip=? AND app_name=? AND contact_name=?",
                    (_safe_name(device_ip), _safe_name(app_name), contact),
                ).fetchone()
                if row is None:
                    session_id = uuid.uuid4().hex
                    now = _now_iso()
                    conn.execute(
                        "INSERT INTO sessions (session_id, device_ip, app_name, contact_name, created_at, updated_at)"
                        " VALUES (?,?,?,?,?,?)",
                        (session_id, _safe_name(device_ip), _safe_name(app_name), contact, now, now),
                    )
                    conn.commit()
                    row = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
                else:
                    # 活跃度触达：供 30 天清理判断（docs/13 4.1）
                    conn.execute(
                        "UPDATE sessions SET updated_at=? WHERE session_id=?",
                        (_now_iso(), row["session_id"]),
                    )
                    conn.commit()
                return dict(row)
            finally:
                conn.close()

    def rebind_contact(self, device_ip: str, app_name: str, session_id: str, new_contact_name: str) -> dict[str, Any] | None:
        """OCR 昵称漂移恢复：把识别到的新昵称绑回旧会话（docs/13 4.2）。

        若新昵称已占位（_resolve_session 先 find_or_create 出的空会话），
        且该占位会话无任何轮次，则删除占位后换绑；否则返回已占位会话。
        """
        new_contact = _safe_name(new_contact_name)
        with self._lock:
            conn = self._connect(device_ip, app_name)
            try:
                existing = conn.execute(
                    "SELECT * FROM sessions WHERE device_ip=? AND app_name=? AND contact_name=?",
                    (_safe_name(device_ip), _safe_name(app_name), new_contact),
                ).fetchone()
                if existing is not None:
                    has_turns = conn.execute(
                        "SELECT 1 FROM turns WHERE session_id=? LIMIT 1", (existing["session_id"],)
                    ).fetchone()
                    if has_turns is not None or str(existing["session_id"]) == str(session_id):
                        return dict(existing)
                    conn.execute("DELETE FROM sessions WHERE session_id=?", (existing["session_id"],))
                conn.execute(
                    "UPDATE sessions SET contact_name=?, updated_at=? WHERE session_id=?",
                    (new_contact, _now_iso(), session_id),
                )
                conn.commit()
                row = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
                return dict(row) if row is not None else None
            finally:
                conn.close()

    def list_sessions(self, device_ip: str, app_name: str) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect(device_ip, app_name)
            try:
                rows = conn.execute(
                    "SELECT * FROM sessions WHERE device_ip=? AND app_name=? ORDER BY updated_at DESC",
                    (_safe_name(device_ip), _safe_name(app_name)),
                ).fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()

    def last_exchange(self, session_id: str) -> dict[str, str]:
        """会话最后一组 user/assistant 文本（连续性匹配用）。"""
        with self._lock:
            conn = self._connect_by_session(session_id)
            if conn is None:
                return {"user": "", "assistant": ""}
            try:
                return self._last_exchange(conn, session_id)
            finally:
                conn.close()

    @staticmethod
    def _last_exchange(conn: sqlite3.Connection, session_id: str) -> dict[str, str]:
        result = {"user": "", "assistant": ""}
        for role in ("assistant", "user"):
            row = conn.execute(
                "SELECT content FROM turns WHERE session_id=? AND role=? ORDER BY seq DESC LIMIT 1",
                (session_id, role),
            ).fetchone()
            if row is not None:
                result[role] = str(row["content"] or "")
        return result

    def _connect_by_session(self, session_id: str) -> sqlite3.Connection | None:
        """session 分散在 <ip>/<app> 各库中；按需扫描定位（低频路径：连续性恢复）。"""
        if not self.root.exists():
            return None
        for path in sorted(self.root.glob("*/*/sessions.sqlite")):
            try:
                conn = sqlite3.connect(path)
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT 1 FROM sessions WHERE session_id=?", (session_id,)).fetchone()
                if row is not None:
                    return conn
                conn.close()
            except sqlite3.Error:
                continue
        return None

    def find_continuity_session(
        self,
        device_ip: str,
        app_name: str,
        *,
        outgoing_texts: list[str] | None = None,
        incoming_texts: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """OCR 昵称漂移：用屏幕可见气泡匹配旧会话（口径同旧 find_continuity_match）。"""
        outgoing = [str(item) for item in (outgoing_texts or []) if str(item).strip()][-3:]
        incoming = [str(item) for item in (incoming_texts or []) if str(item).strip()][-3:]
        if not outgoing:
            return None
        candidates: dict[str, dict[str, Any]] = {}
        for session in self.list_sessions(device_ip, app_name):
            session_id = str(session.get("session_id") or "")
            if not session_id:
                continue
            exchange = self.last_exchange(session_id)
            if not exchange["assistant"]:
                continue
            reply_score = max((_message_match_score(item, exchange["assistant"]) for item in outgoing), default=0)
            if reply_score < 92:
                continue
            question_score = max((_message_match_score(item, exchange["user"]) for item in incoming), default=0)
            candidate = {
                "session": session,
                "score": reply_score + (40 if question_score >= 88 else 0),
                "reply_score": reply_score,
                "question_score": question_score,
            }
            existing = candidates.get(session_id)
            if existing is None or candidate["score"] > existing["score"]:
                candidates[session_id] = candidate
        ranked = sorted(candidates.values(), key=lambda item: int(item["score"]), reverse=True)
        if not ranked:
            return None
        best = ranked[0]
        if len(ranked) > 1 and int(best["score"]) - int(ranked[1]["score"]) < 20:
            return None
        return {"session": best["session"], "score": best["score"], "reply_score": best["reply_score"], "question_score": best["question_score"]}

    # ---------- 轮次 ----------
    def append_turn(self, device_ip: str, app_name: str, session_id: str, role: str, content: str, kind: str = KIND_CHAT) -> int:
        if role not in ("user", "assistant"):
            raise ValueError(f"invalid role: {role}")
        text = str(content or "").strip()
        if not text:
            raise ValueError("empty turn content")
        with self._lock:
            conn = self._connect(device_ip, app_name)
            try:
                row = conn.execute("SELECT COALESCE(MAX(seq), 0) AS max_seq FROM turns WHERE session_id=?", (session_id,)).fetchone()
                seq = int(row["max_seq"]) + 1
                conn.execute(
                    "INSERT INTO turns (session_id, seq, role, content, kind, created_at) VALUES (?,?,?,?,?,?)",
                    (session_id, seq, role, text, kind or KIND_CHAT, _now_iso()),
                )
                conn.execute("UPDATE sessions SET updated_at=? WHERE session_id=?", (_now_iso(), session_id))
                conn.commit()
                return seq
            finally:
                conn.close()

    def record_exchange(
        self,
        device_ip: str,
        app_name: str,
        session_id: str,
        *,
        user_text: str = "",
        assistant_text: str = "",
        kind: str = KIND_CHAT,
    ) -> None:
        """record_sent 统一入口（docs/13 4.3）：只记实际发送内容；普通回复与招呼共用。"""
        if user_text:
            self.append_turn(device_ip, app_name, session_id, "user", user_text, kind)
        if assistant_text:
            self.append_turn(device_ip, app_name, session_id, "assistant", assistant_text, kind)

    def recent_turns(self, device_ip: str, app_name: str, session_id: str, n: int = 6) -> list[dict[str, Any]]:
        if n <= 0:
            return []
        with self._lock:
            conn = self._connect(device_ip, app_name)
            try:
                rows = conn.execute(
                    "SELECT seq, role, content, kind, created_at FROM turns"
                    " WHERE session_id=? ORDER BY seq DESC LIMIT ?",
                    (session_id, int(n)),
                ).fetchall()
                return [dict(row) for row in reversed(rows)]
            finally:
                conn.close()

    def max_seq(self, device_ip: str, app_name: str, session_id: str) -> int:
        with self._lock:
            conn = self._connect(device_ip, app_name)
            try:
                row = conn.execute("SELECT COALESCE(MAX(seq), 0) AS max_seq FROM turns WHERE session_id=?", (session_id,)).fetchone()
                return int(row["max_seq"])
            finally:
                conn.close()

    def recent_greetings(self, device_ip: str, app_name: str, n: int = 5) -> list[str]:
        """跨会话最近招呼语（AI 改写防重复负例，docs/13 4.4）。"""
        with self._lock:
            conn = self._connect(device_ip, app_name)
            try:
                rows = conn.execute(
                    "SELECT content FROM turns"
                    " WHERE role='assistant' AND kind IN (?,?)"
                    " ORDER BY id DESC LIMIT ?",
                    (KIND_GREETING, KIND_SYNTHETIC_GREETING, int(n)),
                ).fetchall()
                return [str(row["content"]) for row in rows]
            finally:
                conn.close()

    # ---------- 摘要 ----------
    def get_summary(self, device_ip: str, app_name: str, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect(device_ip, app_name)
            try:
                row = conn.execute("SELECT * FROM summaries WHERE session_id=?", (session_id,)).fetchone()
                return dict(row) if row is not None else None
            finally:
                conn.close()

    def save_summary(self, device_ip: str, app_name: str, session_id: str, summary: str, up_to_seq: int) -> None:
        with self._lock:
            conn = self._connect(device_ip, app_name)
            try:
                conn.execute(
                    "INSERT INTO summaries (session_id, summary, summarized_up_to_seq, updated_at)"
                    " VALUES (?,?,?,?)"
                    " ON CONFLICT(session_id) DO UPDATE SET"
                    " summary=excluded.summary, summarized_up_to_seq=excluded.summarized_up_to_seq,"
                    " updated_at=excluded.updated_at",
                    (session_id, str(summary or ""), int(up_to_seq), _now_iso()),
                )
                conn.commit()
            finally:
                conn.close()

    def aged_turns_for_summary(
        self,
        device_ip: str,
        app_name: str,
        session_id: str,
        *,
        window_n: int,
        min_aged: int,
    ) -> tuple[list[dict[str, Any]], int, int]:
        """返回 (已老化轮次, 老化最大 seq, 当前水位 W)。

        window_n 为问答轮数（一轮 = user+assistant 两条消息）；
        仅摘要 W < seq <= max_seq - 2*window_n（docs/13 4.5，摘要与窗口不重叠）。
        """
        with self._lock:
            conn = self._connect(device_ip, app_name)
            try:
                max_seq = int(
                    conn.execute(
                        "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM turns WHERE session_id=?", (session_id,)
                    ).fetchone()["max_seq"]
                )
                summary_row = conn.execute(
                    "SELECT summarized_up_to_seq FROM summaries WHERE session_id=?", (session_id,)
                ).fetchone()
                watermark = int(summary_row["summarized_up_to_seq"]) if summary_row is not None else 0
                upper = max_seq - 2 * max(0, int(window_n))
                if upper <= watermark:
                    return [], watermark, watermark
                rows = conn.execute(
                    "SELECT seq, role, content, kind FROM turns"
                    " WHERE session_id=? AND seq>? AND seq<=? ORDER BY seq ASC",
                    (session_id, watermark, upper),
                ).fetchall()
                aged = [dict(row) for row in rows]
                if len(aged) < max(1, int(min_aged)):
                    return [], watermark, watermark
                return aged, int(aged[-1]["seq"]), watermark
            finally:
                conn.close()

    # ---------- 保留策略 ----------
    def cleanup_inactive_sessions(self, device_ip: str, app_name: str, *, inactive_days: int = DEFAULT_INACTIVE_DAYS) -> int:
        """删除 updated_at 早于 inactive_days 的会话及其轮次与摘要（docs/13 4.1）。"""
        cutoff = (datetime.now() - timedelta(days=max(1, int(inactive_days)))).isoformat(timespec="seconds")
        removed = 0
        with self._lock:
            conn = self._connect(device_ip, app_name)
            try:
                stale = conn.execute(
                    "SELECT session_id FROM sessions WHERE device_ip=? AND app_name=? AND updated_at < ?",
                    (_safe_name(device_ip), _safe_name(app_name), cutoff),
                ).fetchall()
                for row in stale:
                    session_id = str(row["session_id"])
                    conn.execute("DELETE FROM turns WHERE session_id=?", (session_id,))
                    conn.execute("DELETE FROM summaries WHERE session_id=?", (session_id,))
                    conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
                    removed += 1
                if stale:
                    conn.commit()
                return removed
            finally:
                conn.close()
