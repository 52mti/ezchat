from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from apps.base import ReplyAction
from .ai_client import AIClient
from .chat_history_store import ChatHistoryStore
from .config_manager import ConfigManager
from .memory_store import MemoryStore
from .message_state import build_message_fingerprint, contact_key, find_outbox_entry
from .persona import resolve_persona


class MessageProcessor:
    def __init__(
        self,
        *,
        device: dict[str, Any],
        app_name: str,
        config_manager: ConfigManager,
        ai_client: AIClient,
        memory_store: MemoryStore,
        chat_history_store: ChatHistoryStore | None = None,
        logger: Callable[[str, str, dict[str, Any] | None], None] | None = None,
    ) -> None:
        self.device = device
        self.app_name = app_name
        self.config_manager = config_manager
        self.ai_client = ai_client
        self.memory_store = memory_store
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
        contact = str(message.get("contact") or message.get("user") or "")
        if not contact:
            return False
        persona = self._persona_for(contact)
        return not bool(self.memory_store.get_conversation_id(self.device_ip, self.app_name, contact, persona))

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

        apikey = self.config_manager.get("global", "apikey", default="")
        persona = self._persona_for(contact)
        try:
            conversation_id = self.memory_store.get_conversation_id(
                self.device_ip,
                self.app_name,
                contact,
                persona,
            )
        except Exception as exc:
            conversation_id = ""
            self._log("warning", "memory lookup failed", {"contact": contact, "error": str(exc)})
        else:
            self._log(
                "info",
                "memory hit" if conversation_id else "memory miss",
                {"contact": contact, "conversation_id": conversation_id},
            )
        if not conversation_id:
            continuity = message.get("continuity", {})
            if isinstance(continuity, dict):
                try:
                    matched = self.memory_store.find_continuity_match(
                        self.device_ip,
                        self.app_name,
                        persona,
                        sender1_messages=continuity.get("sender1"),
                        sender2_messages=continuity.get("sender2"),
                    )
                except Exception as exc:
                    self._log("warning", "近期消息连续性匹配失败", {"contact": contact, "error": str(exc)})
                else:
                    if matched:
                        conversation_id = str(matched["conversation_id"])
                        self._log("info", "近期消息连续性命中，复用会话记忆", {"contact": contact, **matched})
        reply = self.ai_client.chat(content, persona, apikey, conversation_id=conversation_id)
        final_conversation_id = reply.conversation_id or conversation_id
        if final_conversation_id:
            try:
                self.memory_store.upsert_record(
                    self.device_ip,
                    self.app_name,
                    contact,
                    persona,
                    final_conversation_id,
                    last_question_text=content,
                    last_reply_text=reply.text,
                )
            except Exception as exc:
                self._log("warning", "memory update failed", {"contact": contact, "error": str(exc)})
            else:
                self._log(
                    "info",
                    "memory updated",
                    {"contact": contact, "conversation_id": final_conversation_id},
                )
        self.config_manager.save_outbox_entry(
            {
                "message_fingerprint": fingerprint,
                "contact_key": contact_key(self.app_name, self.device_ip, contact),
                "last_processed_at": "",
                "pending_reply_text": reply.text,
                "pending_send": True,
                "conversation_id": final_conversation_id,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        return ReplyAction(text=reply.text, message_fingerprint=fingerprint, reason="ai_reply")

    def mark_sent(self, action: ReplyAction, message: dict[str, str]) -> None:
        if not action.message_fingerprint or action.skip:
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
