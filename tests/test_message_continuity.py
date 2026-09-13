from __future__ import annotations

"""OCR 昵称漂移恢复（docs/13 4.2）：逻辑已从旧 MemoryStore 迁至 PersonaSessionStore +
PersonaChat._resolve_session，用真实 PersonaChat + 脚本化客户端在会话层验证。"""

import json

from services.deepseek_client import DeepSeekReply
from services.persona_chat import PersonaChat
from services.persona_session_store import PersonaSessionStore


class ScriptedClient:
    user_id = "jixie-test"
    usage: dict = {}

    def __init__(self):
        self.script: list[str] = []

    def chat(self, messages, **kwargs):
        return DeepSeekReply(text=self.script.pop(0), finish_reason="stop", usage={}, raw={})


def _make_chat(tmp_path, client):
    profile = {"基础设定": "测试人设：简短口语化回复。", "人物信息": "", "边界条件": "", "审核标准": ""}
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")
    store = PersonaSessionStore(root=tmp_path / "sessions")
    chat = PersonaChat(
        client=client,
        store=store,
        workspace_root=tmp_path,
        read_settings=lambda: {"persona_profile_json": str(profile_path), "judge_enabled": False},
    )
    return chat, store


def test_reuses_session_when_contact_name_ocr_drifts(tmp_path):
    client = ScriptedClient()
    chat, store = _make_chat(tmp_path, client)

    # 旧会话：联系人"清雅如茶"，末轮问答与屏幕气泡一致
    old_session = store.find_or_create_session("1.1.1.1", "xinyu", "清雅如茶")
    store.record_exchange(
        "1.1.1.1", "xinyu", str(old_session["session_id"]),
        user_text="你在忙什么", assistant_text="我刚忙完，正想找你聊天",
    )

    client.script = ["好呀那你继续说"]
    reply = chat.generate_reply(
        device_ip="1.1.1.1", app_name="xinyu", contact_name="清雅如荼",  # OCR 漂移
        message="哦哦哦",
        continuity={"sender1": ["你在忙什么", "哦哦哦"], "sender2": ["我刚忙完，正想找你聊天"]},
    )

    assert reply.session_id == str(old_session["session_id"])
    # 换绑后组合键指向同一会话
    assert store.find_or_create_session("1.1.1.1", "xinyu", "清雅如荼")["session_id"] == old_session["session_id"]


def test_does_not_merge_ambiguous_continuity_matches(tmp_path):
    client = ScriptedClient()
    chat, store = _make_chat(tmp_path, client)

    for contact in ("甲", "乙"):
        session = store.find_or_create_session("1.1.1.1", "xinyu", contact)
        store.record_exchange(
            "1.1.1.1", "xinyu", str(session["session_id"]),
            user_text="在吗", assistant_text="好的",
        )

    client.script = ["嗯嗯你说"]
    reply = chat.generate_reply(
        device_ip="1.1.1.1", app_name="xinyu", contact_name="丙",
        message="继续聊",
        continuity={"sender1": [], "sender2": ["好的"]},
    )

    # 两个候选同分 → 不合并，丙拿到全新会话
    existing_ids = {
        str(store.find_or_create_session("1.1.1.1", "xinyu", name)["session_id"]) for name in ("甲", "乙")
    }
    assert reply.session_id not in existing_ids


def test_continuity_ignored_when_session_already_has_turns(tmp_path):
    client = ScriptedClient()
    chat, store = _make_chat(tmp_path, client)

    session = store.find_or_create_session("1.1.1.1", "xinyu", "丁")
    store.record_exchange(
        "1.1.1.1", "xinyu", str(session["session_id"]),
        user_text="在吗", assistant_text="好的",
    )

    client.script = ["收到收到"]
    reply = chat.generate_reply(
        device_ip="1.1.1.1", app_name="xinyu", contact_name="丁",
        message="继续",
        continuity={"sender2": ["好的"]},
    )
    assert reply.session_id == str(session["session_id"])
