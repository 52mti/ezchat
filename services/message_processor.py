from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from apps.base import ReplyAction
from apps.defaults import with_app_defaults

from .ai_client import AIClient
from .config_manager import ConfigManager, _deep_merge
from .memory_store import MemoryStore
from .message_state import build_message_fingerprint, contact_key, find_outbox_entry


DUPLICATE_PROTECTION_THRESHOLD = 5


class MessageProcessor:
    def __init__(
        self,
        *,
        device: dict[str, Any],
        app_name: str,
        config_manager: ConfigManager,
        ai_client: AIClient,
        memory_store: MemoryStore,
        logger: Callable[[str, str, dict[str, Any] | None], None] | None = None,
    ) -> None:
        self.device = device
        self.app_name = app_name
        self.config_manager = config_manager
        self.ai_client = ai_client
        self.memory_store = memory_store
        self.logger = logger

    @property
    def device_ip(self) -> str:
        return str(self.device.get("ip", ""))

    def _log(self, level: str, message: str, payload: dict[str, Any] | None = None) -> None:
        if self.logger:
            self.logger(level, message, payload)

    def _persona_for(self, contact_name: str) -> dict[str, Any]:
        global_persona = self.config_manager.get("global", "persona", default={})
        app_overrides = self.device.get("app_overrides", {})
        app_override = app_overrides.get(self.app_name, {}) if isinstance(app_overrides, dict) else {}
        app_persona = app_override.get("persona", {}) if isinstance(app_override, dict) else {}
        app_persona = with_app_defaults({"persona": app_persona if isinstance(app_persona, dict) else {}})["persona"]
        device_persona = self.device.get("persona", {})
        device_persona = device_persona if isinstance(device_persona, dict) else {}
        contact_personas = self.device.get("contact_personas", {})
        contact_persona = {}
        if isinstance(contact_personas, dict):
            contact_persona = contact_personas.get(contact_name, {}) or {}
        persona = _deep_merge(global_persona if isinstance(global_persona, dict) else {}, app_persona)
        persona = _deep_merge(persona, device_persona)
        return _deep_merge(persona, contact_persona if isinstance(contact_persona, dict) else {})

    def prepare_reply(self, message: dict[str, str]) -> ReplyAction:
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
        if existing and not existing.get("pending_send"):
            duplicate_hits = int(existing.get("duplicate_hit_count") or 0) + 1
            self.config_manager.save_outbox_entry(
                {
                    "message_fingerprint": fingerprint,
                    "duplicate_hit_count": duplicate_hits,
                    "duplicate_last_hit_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            payload = {
                "fingerprint": fingerprint,
                "duplicate_hit_count": duplicate_hits,
                "duplicate_threshold": DUPLICATE_PROTECTION_THRESHOLD,
            }
            if duplicate_hits >= DUPLICATE_PROTECTION_THRESHOLD:
                self._log("warning", "duplicate message protection triggered", payload)
                return ReplyAction(
                    message_fingerprint=fingerprint,
                    skip=True,
                    reason="duplicate_message_protection",
                )
            self._log("info", "duplicate message hit", payload)
        if existing and existing.get("pending_send") and existing.get("pending_reply_text"):
            self._log("info", "using pending reply", {"fingerprint": fingerprint})
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
