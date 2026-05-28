from __future__ import annotations

import yaml

from services.memory_store import MemoryStore, build_persona_signature


def test_memory_store_scopes_conversation_by_device_app_contact_and_persona(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    persona = {"name": "bot", "renshe": "friendly"}

    store.upsert_record(
        "192.168.1.108",
        "wechat",
        "Alice",
        persona,
        "cid-1",
        last_question_text="hello",
        last_reply_text="hi",
    )

    assert store.get_conversation_id("192.168.1.108", "wechat", "Alice", persona) == "cid-1"
    assert store.get_conversation_id("192.168.1.109", "wechat", "Alice", persona) == ""
    assert store.get_conversation_id("192.168.1.108", "xinyu", "Alice", persona) == ""
    assert store.get_conversation_id("192.168.1.108", "wechat", "Alice", {"name": "other"}) == ""

    data = yaml.safe_load((tmp_path / "memory" / "192.168.1.108" / "wechat" / "conversations.yaml").read_text(encoding="utf-8"))
    record = next(iter(data["records"].values()))
    assert record["device_ip"] == "192.168.1.108"
    assert record["app_name"] == "wechat"
    assert record["identity_mode"] == "contact_name"
    assert record["identity_key"] == "Alice"
    assert record["persona_signature"] == build_persona_signature(persona)
    assert record["last_question_text"] == "hello"
    assert record["last_reply_text"] == "hi"


def test_memory_store_backs_up_bad_yaml_and_falls_back_empty(tmp_path):
    path = tmp_path / "memory" / "192.168.1.108" / "wechat" / "conversations.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("records: [unterminated", encoding="utf-8")

    store = MemoryStore(tmp_path / "memory")

    assert store.get_conversation_id("192.168.1.108", "wechat", "Alice", {"name": "bot"}) == ""
    assert not path.exists()
    assert list(path.parent.glob("conversations.yaml.bad.*"))


def test_memory_store_reads_legacy_contact_persona_key(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    persona = {"name": "bot"}
    legacy_key = store.key("Alice", persona)
    path = tmp_path / "memory" / "192.168.1.108" / "wechat" / "conversations.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "records": {legacy_key: {"conversation_id": "legacy-cid"}}},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    assert store.get_conversation_id("192.168.1.108", "wechat", "Alice", persona) == "legacy-cid"
