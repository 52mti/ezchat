from __future__ import annotations

import json

from services.ai_client import AIReply
from services.config_manager import ConfigManager
from services.memory_store import MemoryStore
from services.message_processor import MessageProcessor


DEFAULT_PERSONA_ID = "REMOVED-BEARER-TOKEN"


class FakeAI:
    def __init__(self):
        self.calls = 0
        self.last_persona = None
        self.last_conversation_id = None

    def chat(self, query, persona, apikey, conversation_id="", retries=2, retry_delay=3.0):
        self.calls += 1
        self.last_persona = persona
        self.last_conversation_id = conversation_id
        return AIReply("回复", "cid", {"answer": json.dumps({"huifu": "回复"})})


class FailingMemoryStore:
    def get_conversation_id(self, *args, **kwargs):
        raise RuntimeError("memory read failed")

    def upsert_record(self, *args, **kwargs):
        raise RuntimeError("memory write failed")


def test_message_processor_writes_pending_then_marks_sent(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "global": {"apikey": "key", "persona": {"name": "bot"}},
                "devices": [{"ip": "1.1.1.1", "selected_app": "wechat"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = ConfigManager(config_path, tmp_path / "outbox.json")
    ai = FakeAI()
    processor = MessageProcessor(
        device={"ip": "1.1.1.1"},
        app_name="wechat",
        config_manager=config,
        ai_client=ai,  # type: ignore[arg-type]
        memory_store=MemoryStore(tmp_path / "memory"),
    )

    message = {"contact": "张三", "content": "你好", "message_time": "12:00"}
    action = processor.prepare_reply(message)

    assert action.text == "回复"
    assert config.load_outbox()[0]["pending_send"] is True
    assert config.load_outbox()[0]["conversation_id"] == "cid"

    processor.mark_sent(action, message)

    assert config.load_outbox()[0]["pending_send"] is False
    assert config.load_outbox()[0]["pending_reply_text"] == ""
    assert config.load_outbox()[0]["conversation_id"] == "cid"


def test_message_processor_replies_for_repeated_message(tmp_path):
    config = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")
    ai = FakeAI()
    processor = MessageProcessor(
        device={"ip": "1.1.1.1"},
        app_name="wechat",
        config_manager=config,
        ai_client=ai,  # type: ignore[arg-type]
        memory_store=MemoryStore(tmp_path / "memory"),
    )
    message = {"contact": "张三", "content": "你好", "message_time": "12:00"}
    action = processor.prepare_reply(message)
    processor.mark_sent(action, message)

    second = processor.prepare_reply(message)

    assert second.skip is False
    assert second.text == "回复"
    assert ai.calls == 2


def test_message_processor_uses_app_scoped_persona(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "global": {"apikey": "key", "persona": {"name": "global-name"}},
                "devices": [
                    {
                        "ip": "1.1.1.1",
                        "selected_app": "wechat",
                        "app_overrides": {
                            "wechat": {
                                "persona": {
                                    "name": "app-name",
                                    "renshe": "app-renshe",
                                }
                            }
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = ConfigManager(config_path, tmp_path / "outbox.json")
    ai = FakeAI()
    device = config.find_device("1.1.1.1") or {"ip": "1.1.1.1"}
    processor = MessageProcessor(
        device=device,
        app_name="wechat",
        config_manager=config,
        ai_client=ai,  # type: ignore[arg-type]
        memory_store=MemoryStore(tmp_path / "memory"),
    )

    processor.prepare_reply({"contact": "张三", "content": "你好", "message_time": "12:00"})

    assert ai.last_persona["name"] == "app-name"
    assert ai.last_persona["renshe"] == "app-renshe"
    assert ai.last_persona["requestPersona"] == DEFAULT_PERSONA_ID


def test_message_processor_prefers_app_persona_over_device_persona(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "global": {"apikey": "key", "persona": {"name": "global-name"}},
                "devices": [
                    {
                        "ip": "1.1.1.1",
                        "selected_app": "wechat",
                        "persona": {"name": "device-name"},
                        "app_overrides": {
                            "wechat": {
                                "persona": {
                                    "name": "app-name",
                                    "renshe": "app-renshe",
                                }
                            }
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = ConfigManager(config_path, tmp_path / "outbox.json")
    ai = FakeAI()
    device = config.find_device("1.1.1.1") or {"ip": "1.1.1.1"}
    processor = MessageProcessor(
        device=device,
        app_name="wechat",
        config_manager=config,
        ai_client=ai,  # type: ignore[arg-type]
        memory_store=MemoryStore(tmp_path / "memory"),
    )

    processor.prepare_reply({"contact": "寮犱笁", "content": "浣犲ソ", "message_time": "12:00"})

    assert ai.last_persona["name"] == "app-name"
    assert ai.last_persona["renshe"] == "app-renshe"


def test_message_processor_degrades_when_memory_store_fails(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "global": {"apikey": "key", "persona": {"name": "bot"}},
                "devices": [{"ip": "1.1.1.1", "selected_app": "wechat"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = ConfigManager(config_path, tmp_path / "outbox.json")
    ai = FakeAI()
    logs = []
    processor = MessageProcessor(
        device={"ip": "1.1.1.1"},
        app_name="wechat",
        config_manager=config,
        ai_client=ai,  # type: ignore[arg-type]
        memory_store=FailingMemoryStore(),  # type: ignore[arg-type]
        logger=lambda level, message, payload=None: logs.append((level, message, payload)),
    )

    action = processor.prepare_reply({"contact": "张三", "content": "你好", "message_time": "12:00"})

    assert action.text == "回复"
    assert ai.last_conversation_id == ""
    assert config.load_outbox()[0]["conversation_id"] == "cid"
    assert any(message == "memory lookup failed" for _, message, _ in logs)
    assert any(message == "memory update failed" for _, message, _ in logs)
