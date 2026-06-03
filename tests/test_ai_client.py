from __future__ import annotations

import json

import pytest

from services.ai_client import AIClient, AIClientError, DEFAULT_CHAT_BEARER, DEFAULT_CREDITS_BEARER


def test_parse_reply_payload_extracts_huifu():
    payload = {
        "conversation_id": "cid",
        "answer": json.dumps({"huifu": "你好"}),
    }

    reply = AIClient.parse_reply_payload(payload)

    assert reply.text == "你好"
    assert reply.conversation_id == "cid"


def test_parse_reply_payload_extracts_huifu_from_json_code_block():
    payload = {
        "conversation_id": "cid",
        "answer": """```json
{
  "name": "",
  "xinli": "对方回复平淡，需引导话题",
  "zhuangtai": "线上(打字聊天)",
  "huifu": "怎么啦，感觉你有点兴致缺缺"
}
```""",
    }

    reply = AIClient.parse_reply_payload(payload)

    assert reply.text == "怎么啦，感觉你有点兴致缺缺"
    assert reply.conversation_id == "cid"


def test_parse_reply_payload_rejects_missing_huifu():
    with pytest.raises(AIClientError):
        AIClient.parse_reply_payload({"answer": json.dumps({"other": "value"})})


def test_parse_reply_payload_rejects_insufficient_credit_text():
    with pytest.raises(AIClientError, match="insufficient credits"):
        AIClient.parse_reply_payload({"answer": "次数不足，请充值"})


def test_ai_client_uses_default_bearers_when_config_is_empty():
    client = AIClient(chat_bearer="", credits_bearer="")

    assert client.chat_bearer == DEFAULT_CHAT_BEARER
    assert client.credits_bearer == DEFAULT_CREDITS_BEARER


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"conversation_id": "cid", "answer": json.dumps({"huifu": "回复"})}


class FakeSession:
    def __init__(self):
        self.requests = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.requests.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse()


def test_ai_client_uses_persona_request_persona_as_chat_bearer():
    session = FakeSession()
    client = AIClient(chat_bearer="global-bearer", session=session)

    client.chat("你好", {"requestPersona": "app-persona-bearer", "name": "bot"}, "apikey", retries=0)

    assert session.requests[0]["headers"]["Authorization"] == "Bearer app-persona-bearer"


def test_ai_client_falls_back_to_global_chat_bearer_when_persona_is_empty():
    session = FakeSession()
    client = AIClient(chat_bearer="global-bearer", session=session)

    client.chat("你好", {"requestPersona": ""}, "apikey", retries=0)

    assert session.requests[0]["headers"]["Authorization"] == "Bearer global-bearer"
