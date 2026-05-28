from __future__ import annotations

from services.message_state import build_message_fingerprint, contact_key, make_outbox_entry


def test_fingerprint_uses_time_or_unread_fallback():
    fp1 = build_message_fingerprint("1.1.1.1", "wechat", "张三", "在吗", message_time="12:00")
    fp2 = build_message_fingerprint("1.1.1.1", "wechat", "张三", "在吗", message_time="12:01")
    fp3 = build_message_fingerprint("1.1.1.1", "wechat", "张三", "在吗", unread_count="[2]")

    assert fp1 != fp2
    assert fp1 != fp3
    assert contact_key("wechat", "1.1.1.1", "张三") == "wechat|1.1.1.1|张三"


def test_make_outbox_entry_shape():
    entry = make_outbox_entry(
        device_ip="1.1.1.1",
        app_name="wechat",
        contact_name="张三",
        message_text="你好",
        pending_reply_text="你好呀",
        pending_send=True,
    )

    assert entry["pending_send"] is True
    assert entry["pending_reply_text"] == "你好呀"
    assert entry["contact_key"] == "wechat|1.1.1.1|张三"
