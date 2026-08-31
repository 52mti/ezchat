from __future__ import annotations

from services.chat_history_store import ChatHistoryStore


PERSONA = {"requestPersona": "app-test", "name": "bot"}


def test_chat_history_contacts_are_sorted_by_last_message_time(tmp_path):
    store = ChatHistoryStore(tmp_path / "history")
    store.append_exchange(
        device_ip="1.1.1.1",
        app_name="xinyu",
        contact_name="张三",
        persona=PERSONA,
        incoming_text="最近忙吗",
        outgoing_text="还好，你呢",
        message_fingerprint="fp1",
        created_at="2026-06-09T10:00:00",
    )
    store.append_exchange(
        device_ip="1.1.1.1",
        app_name="xinyu",
        contact_name="李四",
        persona=PERSONA,
        incoming_text="吃饭了吗",
        outgoing_text="刚吃完",
        message_fingerprint="fp2",
        created_at="2026-06-09T11:00:00",
    )
    store.append_exchange(
        device_ip="1.1.1.1",
        app_name="xinyu",
        contact_name="张三",
        persona=PERSONA,
        incoming_text="那就好",
        outgoing_text="嗯嗯",
        message_fingerprint="fp3",
        created_at="2026-06-09T12:00:00",
    )

    contacts = store.list_contacts(device_ip="1.1.1.1", app_name="xinyu", persona=PERSONA)

    assert [contact["contact_name"] for contact in contacts] == ["张三", "李四"]
    assert contacts[0]["last_message"] == "嗯嗯"
    assert contacts[0]["last_direction"] == "outgoing"


def test_chat_history_get_messages_keeps_incoming_left_and_outgoing_right_order(tmp_path):
    store = ChatHistoryStore(tmp_path / "history")
    store.append_exchange(
        device_ip="1.1.1.1",
        app_name="huanyou",
        contact_name="小小茹",
        persona=PERSONA,
        incoming_text="今天忙不忙",
        outgoing_text="还行，刚忙完",
        message_fingerprint="fp1",
        created_at="2026-06-09T10:00:00",
    )

    messages = store.get_messages(
        device_ip="1.1.1.1",
        app_name="huanyou",
        contact_name="小小茹",
        persona=PERSONA,
    )

    assert [message["direction"] for message in messages] == ["incoming", "outgoing"]
    assert messages[0]["content"] == "今天忙不忙"
    assert messages[1]["content"] == "还行，刚忙完"


def test_chat_history_skips_greeting_records(tmp_path):
    store = ChatHistoryStore(tmp_path / "history")

    stored = store.append_exchange(
        device_ip="1.1.1.1",
        app_name="xinyu",
        contact_name="张三",
        persona=PERSONA,
        incoming_text="你好",
        outgoing_text="你好呀",
        message_fingerprint="fp1",
    )

    assert stored is False
    assert store.list_contacts(device_ip="1.1.1.1", app_name="xinyu", persona=PERSONA) == []


def test_chat_history_is_scoped_by_persona(tmp_path):
    store = ChatHistoryStore(tmp_path / "history")
    other_persona = {"requestPersona": "app-other", "name": "bot"}
    store.append_exchange(
        device_ip="1.1.1.1",
        app_name="xinyu",
        contact_name="张三",
        persona=PERSONA,
        incoming_text="最近忙吗",
        outgoing_text="还好",
        message_fingerprint="fp1",
    )

    assert store.list_contacts(device_ip="1.1.1.1", app_name="xinyu", persona=other_persona) == []


def test_chat_history_groups_name_drift_by_conversation_id(tmp_path):
    store = ChatHistoryStore(tmp_path / "history")
    for contact_name, incoming, outgoing, fingerprint in [
        ("清雅如茶", "你好呀", "最近怎么样", "fp1"),
        ("清雅如荼", "还不错", "那就好", "fp2"),
    ]:
        store.append_exchange(
            device_ip="1.1.1.1",
            app_name="xinyu",
            contact_name=contact_name,
            persona=PERSONA,
            incoming_text=incoming,
            outgoing_text=outgoing,
            conversation_id="conversation-1",
            message_fingerprint=fingerprint,
        )

    contacts = store.list_contacts(device_ip="1.1.1.1", app_name="xinyu", persona=PERSONA)
    messages = store.get_messages(
        device_ip="1.1.1.1",
        app_name="xinyu",
        persona=PERSONA,
        conversation_id="conversation-1",
    )

    assert len(contacts) == 1
    assert contacts[0]["conversation_id"] == "conversation-1"
    assert [item["content"] for item in messages] == ["你好呀", "最近怎么样", "还不错", "那就好"]
