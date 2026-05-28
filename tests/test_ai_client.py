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
