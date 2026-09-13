from __future__ import annotations

import json

from services.chat_history_store import ChatHistoryStore
from services.config_manager import ConfigManager
from services.message_processor import MessageProcessor
from services.persona_chat import PersonaReply
from services.persona_session_store import KIND_GREETING, PersonaSessionStore


class FakePersonaChat:
    """替代旧 FakeAI：与 PersonaChat 同签名，会话用真实 store。"""

    def __init__(self, tmp_path):
        self.store = PersonaSessionStore(root=tmp_path / "persona_sessions")
        self.calls = 0
        self.last_message = ""
        self.last_continuity = None
        self.record_calls: list[dict] = []

    def generate_reply(self, *, device_ip, app_name, contact_name, message, continuity=None):
        self.calls += 1
        self.last_message = message
        self.last_continuity = continuity
        session = self.store.find_or_create_session(device_ip, app_name, contact_name)
        return PersonaReply(text="回复", session_id=str(session["session_id"]), contact_name=contact_name)

    def record_sent(self, *, device_ip, app_name, contact_name, user_text="", assistant_text="", kind="chat"):
        session = self.store.find_or_create_session(device_ip, app_name, contact_name)
        self.store.record_exchange(
            device_ip, app_name, str(session["session_id"]), user_text=user_text, assistant_text=assistant_text, kind=kind
        )
        self.record_calls.append({"contact": contact_name, "user": user_text, "assistant": assistant_text, "kind": kind})


def _processor(tmp_path, persona_chat, history=None, logger=None, config_seed=None, app_name="wechat"):
    config_path = tmp_path / "config.json"
    if not config_path.exists():
        config_path.write_text(
            json.dumps(config_seed or {"global": {}, "devices": [{"ip": "1.1.1.1", "selected_app": app_name}]}, ensure_ascii=False),
            encoding="utf-8",
        )
    return MessageProcessor(
        device={"ip": "1.1.1.1"},
        app_name=app_name,
        config_manager=ConfigManager(config_path, tmp_path / "outbox.json"),
        persona_chat=persona_chat,
        chat_history_store=history,
        logger=logger,
    )


def test_message_processor_writes_pending_then_marks_sent(tmp_path):
    persona = FakePersonaChat(tmp_path)
    processor = _processor(tmp_path, persona)
    message = {"contact": "张三", "content": "你好", "message_time": "12:00"}
    action = processor.prepare_reply(message)

    assert action.text == "回复"
    outbox = processor.config_manager.load_outbox()
    assert outbox[0]["pending_send"] is True
    assert outbox[0]["conversation_id"]  # 人设会话 id（本地 uuid）
    session_id = outbox[0]["conversation_id"]

    processor.mark_sent(action, message)

    outbox = processor.config_manager.load_outbox()
    assert outbox[0]["pending_send"] is False
    assert outbox[0]["pending_reply_text"] == ""
    assert outbox[0]["conversation_id"] == session_id


def test_message_processor_replies_for_repeated_message(tmp_path):
    persona = FakePersonaChat(tmp_path)
    processor = _processor(tmp_path, persona)
    message = {"contact": "张三", "content": "你好", "message_time": "12:00"}
    action = processor.prepare_reply(message)
    processor.mark_sent(action, message)

    second = processor.prepare_reply(message)

    assert second.skip is False
    assert second.text == "回复"
    assert persona.calls == 2


def test_message_processor_logs_pending_reply_with_contact_and_content(tmp_path):
    persona = FakePersonaChat(tmp_path)
    logs = []
    processor = _processor(tmp_path, persona, logger=lambda level, message, payload=None: logs.append((level, message, payload)))
    message = {"contact": "张三", "content": "你好", "message_time": "12:00"}
    processor.prepare_reply(message)

    second = processor.prepare_reply(message)

    assert second.reason == "pending_send"
    assert any(
        msg == "using pending reply"
        and payload is not None
        and payload.get("contact") == "张三"
        and payload.get("content") == "你好"
        for _, msg, payload in logs
    )


def test_message_processor_passes_continuity_to_persona_chat(tmp_path):
    persona = FakePersonaChat(tmp_path)
    processor = _processor(tmp_path, persona)
    continuity = {"sender1": ["在吗"], "sender2": ["上一句我方回复"]}
    processor.prepare_reply({"contact": "张三", "content": "你好", "message_time": "12:00", "continuity": continuity})
    assert persona.last_continuity == continuity


def test_message_processor_degrades_when_persona_record_fails(tmp_path):
    persona = FakePersonaChat(tmp_path)

    def failing_record(**kwargs):
        raise RuntimeError("record failed")

    persona.record_sent = failing_record
    logs = []
    processor = _processor(tmp_path, persona, logger=lambda level, message, payload=None: logs.append((level, message, payload)))
    message = {"contact": "张三", "content": "你好", "message_time": "12:00"}

    action = processor.prepare_reply(message)
    processor.mark_sent(action, message)

    assert action.text == "回复"
    assert processor.config_manager.load_outbox()[0]["pending_send"] is False
    assert any(message == "人设记忆落库失败" for _, message, _ in logs)


def test_message_processor_writes_chat_history_after_send(tmp_path):
    persona = FakePersonaChat(tmp_path)
    history = ChatHistoryStore(tmp_path / "history")
    processor = _processor(
        tmp_path,
        persona,
        history=history,
        app_name="xinyu",
        config_seed={"global": {}, "devices": [{"ip": "1.1.1.1", "selected_app": "xinyu"}]},
    )
    message = {"contact": "张三", "content": "最近忙吗", "message_time": "12:00", "source": "dialogue"}

    action = processor.prepare_reply(message)
    processor.mark_sent(action, message)

    audit_persona = processor._persona_for("张三")
    contacts = history.list_contacts(device_ip="1.1.1.1", app_name="xinyu", persona=audit_persona)
    messages = history.get_messages(device_ip="1.1.1.1", app_name="xinyu", contact_name="张三", persona=audit_persona)
    assert [contact["contact_name"] for contact in contacts] == ["张三"]
    assert [entry["direction"] for entry in messages] == ["incoming", "outgoing"]
    assert messages[0]["content"] == "最近忙吗"
    assert messages[1]["content"] == action.text


def test_message_processor_records_greeting_into_persona_memory_not_audit(tmp_path):
    """招呼语：审计库（ChatHistoryStore 旧过滤保留）不记录，人设记忆必须记录（docs/13 4.4）。"""
    persona = FakePersonaChat(tmp_path)
    history = ChatHistoryStore(tmp_path / "history")
    processor = _processor(
        tmp_path,
        persona,
        history=history,
        app_name="xinyu",
        config_seed={"global": {}, "devices": [{"ip": "1.1.1.1", "selected_app": "xinyu"}]},
    )
    from apps.base import ReplyAction

    message = {"contact": "张三", "content": "你好", "message_time": "12:00"}
    action = ReplyAction(text="哈喽呀，冒个泡", reason="greeting_reply")  # 招呼动作无指纹
    processor.mark_sent(action, message)

    assert history.list_contacts(device_ip="1.1.1.1", app_name="xinyu", persona=processor._persona_for("张三")) == []
    assert persona.record_calls and persona.record_calls[0]["kind"] == KIND_GREETING
    session = persona.store.find_or_create_session("1.1.1.1", "xinyu", "张三")
    turns = persona.store.recent_turns("1.1.1.1", "xinyu", str(session["session_id"]), n=2)
    assert [t["content"] for t in turns] == ["你好", "哈喽呀，冒个泡"]
    assert turns[-1]["kind"] == KIND_GREETING
