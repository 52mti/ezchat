from __future__ import annotations

import json

from PIL import Image

from services.vision_dialogue_client import VisionDialogueClient, VisionDialogueError


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.requests.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return self.responses.pop(0)


def ok_payload(content: str = "你好"):
    return {"choices": [{"message": {"content": json.dumps({
        "chat_title": "用户A",
        "senders": {"sender1": {"alias": "用户A", "role": "对方"}, "sender2": {"alias": "我", "role": "我方"}},
        "messages": [{"sender": "sender1", "type": "text", "content": content}],
    }, ensure_ascii=False)}}]}


def config(prompt_path):
    return {"prompt_path": str(prompt_path),
            "openrouter_ensemble": {"analysis_models": ["z-ai/glm-5.3-flash", "minimax/minimax-m3", "x-ai/grok-4.6"], "judge_model": "z-ai/glm-5.3-flash"}}


def setup(tmp_path):
    env_dir = tmp_path / ".env"
    env_dir.mkdir()
    (env_dir / "openrouter_key.txt").write_text("openrouter-key\n", encoding="utf-8")
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("只输出 JSON", encoding="utf-8")
    return prompt_path


def test_openrouter_ensemble_direct_requests_and_judge(tmp_path):
    prompt_path = setup(tmp_path)
    session = FakeSession([FakeResponse(422, {"error": {"message": "sensitive content risk detected"}})] + [FakeResponse(200, ok_payload()) for _ in range(4)])
    client = VisionDialogueClient(workspace_root=tmp_path, session=session)
    reply = client.parse_dialogue(Image.new("RGB", (200, 400), "white"), {"dialogue_llm": config(prompt_path)})
    assert reply.provider == "openrouter_ensemble"
    assert reply.payload["messages"][0]["content"] == "你好"
    assert session.requests[0]["headers"]["Authorization"] == "Bearer openrouter-key"
    assert session.requests[0]["json"]["model"] == "z-ai/glm-5.3-flash"
    assert {item["json"]["model"] for item in session.requests[1:]} == {"z-ai/glm-5.3-flash", "minimax/minimax-m3", "x-ai/grok-4.6"}
    assert all("plugins" not in item["json"] for item in session.requests[1:])
    assert any("独立分析结果" in item["json"]["messages"][0]["content"] for item in session.requests[1:])


def test_one_analysis_failure_does_not_block_judge(tmp_path):
    prompt_path = setup(tmp_path)
    session = FakeSession([FakeResponse(422, {"error": {"message": "sensitive risk"}}), FakeResponse(200, ok_payload()), FakeResponse(500, text="offline"), FakeResponse(200, ok_payload()), FakeResponse(200, ok_payload())])
    client = VisionDialogueClient(workspace_root=tmp_path, session=session)
    reply = client.parse_dialogue(Image.new("RGB", (20, 20), "white"), {"dialogue_llm": config(prompt_path)})
    assert reply.provider == "openrouter_ensemble"


def test_all_analysis_failures_raise(tmp_path):
    prompt_path = setup(tmp_path)
    session = FakeSession([FakeResponse(422, {"error": {"message": "sensitive risk"}})] + [FakeResponse(500, text="offline") for _ in range(3)])
    client = VisionDialogueClient(workspace_root=tmp_path, session=session)
    try:
        client.parse_dialogue(Image.new("RGB", (20, 20), "white"), {"dialogue_llm": config(prompt_path)})
    except VisionDialogueError as exc:
        assert "all OpenRouter analysis models failed" in str(exc)
    else:
        raise AssertionError("expected VisionDialogueError")
