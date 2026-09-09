from __future__ import annotations

from apps.huanyou import steps


class _FakeReply:
    def __init__(self, payload, provider="token_plan", model="MiniMax-M3") -> None:
        self.payload = payload
        self.provider = provider
        self.model = model


def test_huanyou_read_message_uses_llm_dialogue_result(monkeypatch):
    reports: list[tuple[str, str, str | None]] = []
    monkeypatch.setattr(steps, "confirm_chat_window", lambda robot, config, reporter=None: b"image")
    monkeypatch.setattr(
        steps._DIALOGUE_LLM_CLIENT,
        "parse_dialogue",
        lambda image, config: _FakeReply(
            {
                "chat_title": "风中行者",
                "senders": {
                    "sender1": {"alias": "风中行者", "role": "对方"},
                    "sender2": {"alias": "我", "role": "我方"},
                },
                "messages": [
                    {"id": 1, "sender": "sender2", "type": "text", "content": "在吗"},
                    {"id": 2, "sender": "sender1", "type": "text", "content": "刚忙完"},
                    {"id": 3, "sender": "sender1", "type": "text", "content": "现在有空"},
                ],
            }
        ),
    )

    message = steps.read_message(
        robot=object(),
        config={"dialogue_engine": "llm_vision"},
        target={"name": "风中行者", "unread_count": 2, "favorability": 9},
        reporter=lambda step, label, log_label=None: reports.append((step, label, log_label)),
    )

    assert message["contact"] == "风中行者"
    assert message["content"] == "刚忙完现在有空"
    assert any(step == "dialogue_llm_read" for step, _label, _log_label in reports)
    assert any(step == "dialogue_llm_provider" for step, _label, _log_label in reports)
