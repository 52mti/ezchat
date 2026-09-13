from __future__ import annotations

import pytest

from services.deepseek_client import (
    FALLBACK_MODELS,
    DeepSeekClient,
    DeepSeekError,
    derive_user_id,
    load_models_with_fallback,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.post_calls = 0
        self.get_calls = 0
        self.last_payload = None

    def post(self, url, headers=None, json=None, timeout=None):
        self.post_calls += 1
        self.last_payload = json
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url, headers=None, timeout=None):
        self.get_calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _ok_reply():
    return FakeResponse(200, {"choices": [{"message": {"content": "好的"}, "finish_reason": "stop"}], "usage": {"prompt_cache_hit_tokens": 10}})


def _client(responses, **kwargs):
    session = FakeSession(responses)
    client = DeepSeekClient(api_key="sk-test", session=session, **kwargs)
    return client, session


def test_user_id_derived_from_key_and_stable():
    assert derive_user_id("sk-test") == derive_user_id("sk-test")
    assert derive_user_id("sk-test").startswith("jixie-")
    assert derive_user_id("sk-test") != derive_user_id("sk-other")
    assert len(derive_user_id("sk-test")) == len("jixie-") + 16


def test_chat_success_shape_and_payload():
    client, session = _client([_ok_reply()])
    reply = client.chat([{"role": "user", "content": "hi"}], user_id="u1")
    assert reply.text == "好的" and reply.finish_reason == "stop"
    assert reply.usage["prompt_cache_hit_tokens"] == 10
    payload = session.last_payload
    assert payload["model"] == "deepseek-flash"
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["user_id"] == "u1"
    assert payload["stream"] is False
    assert "frequency_penalty" not in payload and "top_p" not in payload


def test_auth_and_payment_fail_fast_without_retry():
    for status, kind in ((401, "auth"), (402, "payment"), (400, "bad_request")):
        client, session = _client([FakeResponse(status, {"error": {"message": "x"}})])
        with pytest.raises(DeepSeekError) as exc_info:
            client.chat([{"role": "user", "content": "hi"}])
        assert exc_info.value.kind == kind
        assert session.post_calls == 1, f"{status} 不应重试"


def test_rate_limit_retries_then_raises():
    client, session = _client([FakeResponse(429, {}), FakeResponse(429, {}), FakeResponse(429, {})])
    with pytest.raises(DeepSeekError) as exc_info:
        client.chat([{"role": "user", "content": "hi"}], retry_delay=0)
    assert exc_info.value.kind == "rate_limit"
    assert session.post_calls == 3  # 1 + 2 次重试


def test_server_error_retries_then_succeeds():
    client, session = _client([FakeResponse(503, {}), _ok_reply()])
    reply = client.chat([{"role": "user", "content": "hi"}], retry_delay=0)
    assert reply.text == "好的" and session.post_calls == 2


def test_missing_key_raises_auth(tmp_path):
    client = DeepSeekClient(api_key="", workspace_root=tmp_path)  # 隔离 .env，避免读到本机真实 key
    with pytest.raises(DeepSeekError) as exc_info:
        client.chat([{"role": "user", "content": "hi"}])
    assert exc_info.value.kind == "auth"


def test_list_models_and_fallback():
    client, session = _client([FakeResponse(200, {"object": "list", "data": [{"id": "deepseek-flash"}, {"id": "deepseek-v4-pro"}]})])
    assert client.list_models() == ["deepseek-flash", "deepseek-v4-pro"]

    failing, _ = _client([FakeResponse(500, {})])
    assert load_models_with_fallback(failing) == list(FALLBACK_MODELS)
