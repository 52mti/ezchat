from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from apps.base import ReplyAction
from .chat_history_store import ChatHistoryStore
from .config_manager import ConfigManager
from .message_state import build_message_fingerprint, contact_key, find_outbox_entry
from .persona import resolve_persona
from .persona_chat import KIND_CHAT, KIND_GREETING, PersonaChat

GREETING_REASONS = {"greeting", "greeting_reply", "new_greeting"}


class MessageProcessor:
    def __init__(
        self,
        *,
        device: dict[str, Any],
        app_name: str,
        config_manager: ConfigManager,
        persona_chat: PersonaChat,
        chat_history_store: ChatHistoryStore | None = None,
        logger: Callable[[str, str, dict[str, Any] | None], None] | None = None,
    ) -> None:
        self.device = device
        self.app_name = app_name
        self.config_manager = config_manager
        self.persona_chat = persona_chat
        self.chat_history_store = chat_history_store
        self.logger = logger

    @property
    def device_ip(self) -> str:
        return str(self.device.get("ip", ""))

    def _log(self, level: str, message: str, payload: dict[str, Any] | None = None) -> None:
        if self.logger:
            self.logger(level, message, payload)

    def _persona_for(self, contact_name: str) -> dict[str, Any]:
        return resolve_persona(
            self.config_manager,
            self.device_ip,
            self.app_name,
            contact_name,
            device=self.device,
        )

    def needs_continuity_context(self, message: dict[str, Any]) -> bool:
        """新会话（无任何轮次）时才需要读屏幕气泡做连续性匹配（docs/13 4.2）。"""
        contact = str(message.get("contact") or message.get("user") or "")
        if not contact:
            return False
        try:
            session = self.persona_chat.store.find_or_create_session(self.device_ip, self.app_name, contact)
            turns = self.persona_chat.store.recent_turns(self.device_ip, self.app_name, str(session["session_id"]), n=2)
        except Exception as exc:
            self._log("warning", "检查人设会话失败，跳过连续性匹配", {"contact": contact, "error": str(exc)})
            return False
        return not turns

    def prepare_reply(self, message: dict[str, Any]) -> ReplyAction:
        contact = message.get("contact") or message.get("user") or ""
        content = message.get("content", "")
        message_time = message.get("message_time", "")
        unread_count = message.get("unread_count", "")
        fingerprint = build_message_fingerprint(
            self.device_ip,
            self.app_name,
            contact,
            content,
            message_time,
            unread_count,
        )

        existing = find_outbox_entry(self.config_manager.load_outbox(), fingerprint)
        if existing and existing.get("pending_send") and existing.get("pending_reply_text"):
            self._log("info", "using pending reply", {"fingerprint": fingerprint, "contact": contact, "content": content})
            return ReplyAction(
                text=str(existing["pending_reply_text"]),
                message_fingerprint=fingerprint,
                reason="pending_send",
            )

        reply = self.persona_chat.generate_reply(
            device_ip=self.device_ip,
            app_name=self.app_name,
            contact_name=contact,
            message=content,
            continuity=message.get("continuity"),
        )
        self._log(
            "info",
            "persona reply generated",
            {"contact": contact, "stats": reply.stats},
        )
        self.config_manager.save_outbox_entry(
            {
                "message_fingerprint": fingerprint,
                "contact_key": contact_key(self.app_name, self.device_ip, contact),
                "last_processed_at": "",
                "pending_reply_text": reply.text,
                "pending_send": True,
                "conversation_id": reply.session_id,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        return ReplyAction(text=reply.text, message_fingerprint=fingerprint, reason="ai_reply")

    def mark_sent(self, action: ReplyAction, message: dict[str, str]) -> None:
        if not action.message_fingerprint or action.skip:
            # 招呼动作无指纹：仍需落人设记忆（docs/13 4.4 record_sent 统一入口）
            self._record_persona_turn(action, message)
            return
        contact = message.get("contact") or message.get("user") or ""
        outbox_entry = find_outbox_entry(self.config_manager.load_outbox(), action.message_fingerprint) or {}
        self.config_manager.save_outbox_entry(
            {
                "message_fingerprint": action.message_fingerprint,
                "contact_key": contact_key(self.app_name, self.device_ip, contact),
                "last_processed_at": datetime.now().isoformat(timespec="seconds"),
                "pending_reply_text": "",
                "pending_send": False,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self._record_persona_turn(action, message, conversation_id=str(outbox_entry.get("conversation_id", "") or ""))
        if self.chat_history_store is None:
            return
        try:
            persona = self._persona_for(contact)
            self.chat_history_store.append_exchange(
                device_ip=self.device_ip,
                app_name=self.app_name,
                contact_name=contact,
                persona=persona,
                incoming_text=message.get("content", ""),
                outgoing_text=action.text,
                message_fingerprint=action.message_fingerprint,
                conversation_id=str(outbox_entry.get("conversation_id", "") or ""),
                source=str(message.get("source", "") or ""),
                reason=action.reason,
                metadata={
                    "message_time": message.get("message_time", ""),
                    "unread_count": message.get("unread_count", ""),
                },
            )
        except Exception as exc:
            self._log("warning", "chat history update failed", {"contact": contact, "error": str(exc)})

    def _record_persona_turn(self, action: ReplyAction, message: dict[str, str], conversation_id: str = "") -> None:
        contact = message.get("contact") or message.get("user") or ""
        kind = KIND_GREETING if str(action.reason or "") in GREETING_REASONS else KIND_CHAT
        try:
            self.persona_chat.record_sent(
                device_ip=self.device_ip,
                app_name=self.app_name,
                contact_name=contact,
                user_text=str(message.get("content", "") or ""),
                assistant_text=str(action.text or ""),
                kind=kind,
            )
        except Exception as exc:
            self._log("warning", "人设记忆落库失败", {"contact": contact, "error": str(exc)})
