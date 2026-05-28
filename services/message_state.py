from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any


def normalize_message_part(value: str | None) -> str:
    return " ".join(str(value or "").strip().split())


def contact_key(app_name: str, device_ip: str, contact_name: str) -> str:
    return "|".join(
        [
            normalize_message_part(app_name),
            normalize_message_part(device_ip),
            normalize_message_part(contact_name),
        ]
    )


def build_message_fingerprint(
    device_ip: str,
    app_name: str,
    contact_name: str,
    message_text: str,
    message_time: str = "",
    unread_count: str = "",
) -> str:
    aux_key = normalize_message_part(message_time) or normalize_message_part(unread_count)
    raw = "|".join(
        [
            normalize_message_part(device_ip),
            normalize_message_part(app_name),
            normalize_message_part(contact_name),
            normalize_message_part(message_text),
            aux_key,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def make_outbox_entry(
    *,
    device_ip: str,
    app_name: str,
    contact_name: str,
    message_text: str,
    message_time: str = "",
    unread_count: str = "",
    pending_reply_text: str = "",
    pending_send: bool = False,
    conversation_id: str = "",
) -> dict[str, Any]:
    now = datetime.now().isoformat(timespec="seconds")
    fingerprint = build_message_fingerprint(
        device_ip,
        app_name,
        contact_name,
        message_text,
        message_time,
        unread_count,
    )
    return {
        "message_fingerprint": fingerprint,
        "contact_key": contact_key(app_name, device_ip, contact_name),
        "last_processed_at": now if not pending_send else "",
        "pending_reply_text": pending_reply_text,
        "pending_send": pending_send,
        "conversation_id": conversation_id,
        "updated_at": now,
    }


def find_outbox_entry(
    entries: list[dict[str, Any]],
    fingerprint: str,
) -> dict[str, Any] | None:
    for entry in entries:
        if entry.get("message_fingerprint") == fingerprint:
            return entry
    return None
