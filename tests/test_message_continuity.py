from __future__ import annotations

from services.ai_client import AIReply
from services.config_manager import ConfigManager
from services.memory_store import MemoryStore
from services.message_processor import MessageProcessor


class FakeAI:
    def __init__(self) -> None:
        self.last_conversation_id = ""

    def chat(self, query, persona, apikey, conversation_id="", retries=2, retry_delay=3.0):
        self.last_conversation_id = conversation_id
        return AIReply("回复", "new-conversation", {})


def test_xinyu_reuses_memory_when_contact_name_ocr_drifts(tmp_path):
    config = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")
    ai = FakeAI()
    memory = MemoryStore(tmp_path / "memory")
    processor = MessageProcessor(
        device={"ip": "1.1.1.1"},
        app_name="xinyu",
        config_manager=config,
        ai_client=ai,  # type: ignore[arg-type]
        memory_store=memory,
    )
    persona = processor._persona_for("清雅如茶")
    memory.upsert_record(
        "1.1.1.1", "xinyu", "清雅如茶", persona, "old-conversation",
        last_question_text="你在忙什么", last_reply_text="我刚忙完，正想找你聊天",
    )

    processor.prepare_reply({
        "contact": "清雅如荼",
        "content": "哦哦哦",
        "continuity": {
            "sender1": ["你在忙什么", "哦哦哦"],
            "sender2": ["我刚忙完，正想找你聊天"],
        },
    })

    assert ai.last_conversation_id == "old-conversation"


def test_xinyu_does_not_merge_ambiguous_continuity_matches(tmp_path):
    config = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")
    ai = FakeAI()
    memory = MemoryStore(tmp_path / "memory")
    processor = MessageProcessor(
        device={"ip": "1.1.1.1"}, app_name="xinyu", config_manager=config,
        ai_client=ai, memory_store=memory,  # type: ignore[arg-type]
    )
    persona = processor._persona_for("甲")
    for contact, conversation_id in (("甲", "conversation-a"), ("乙", "conversation-b")):
        memory.upsert_record(
            "1.1.1.1", "xinyu", contact, persona, conversation_id,
            last_question_text="在吗", last_reply_text="好的",
        )

    processor.prepare_reply({
        "contact": "丙", "content": "继续聊", "continuity": {"sender1": [], "sender2": ["好的"]},
    })

    assert ai.last_conversation_id == ""
