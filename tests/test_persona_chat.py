from __future__ import annotations

"""A3-A8 验收：persona_chat 编排层（打回环/招呼/审核 fail-open/人设解耦，docs/13 十一）。"""

import json
import time

from services.deepseek_client import DeepSeekReply
from services.persona_chat import PersonaChat, PersonaReply
from services.persona_reply_check import check_greeting, check_reply
from services.persona_session_store import PersonaSessionStore


class ScriptedClient:
    user_id = "jixie-test"
    usage = {"prompt_cache_hit_tokens": 100, "prompt_tokens": 200}

    def __init__(self):
        self.script: list[tuple[str, str]] = []
        self.calls: list[dict] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": [dict(m) for m in messages], "kwargs": dict(kwargs)})
        text, finish = self.script.pop(0)
        return DeepSeekReply(text=text, finish_reason=finish, usage=dict(self.usage), raw={})


def _setup(tmp_path, settings=None):
    profile = {
        "基础设定": "测试人设：温柔女生，说话简短口语化。",
        "人物信息": "小测，22岁。",
        "边界条件": "不见面、不给联系方式。",
        "审核标准": "",
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")
    greetings = tmp_path / "pingtai.txt"
    greetings.write_text("你好呀\nhi~在吗\n", encoding="utf-8")
    base_settings = {
        "persona_profile_json": str(profile_path),
        "greetings_txt": str(greetings),
        "history_turns": 2,
        "summary_interval": 1,
        "judge_enabled": True,
    }
    base_settings.update(settings or {})
    client = ScriptedClient()
    store = PersonaSessionStore(root=tmp_path / "sessions")
    chat = PersonaChat(client=client, store=store, workspace_root=tmp_path, read_settings=lambda: dict(base_settings))
    return chat, client, store, base_settings


def _ok_judge():
    return ('{"ok": true, "reason": ""}', "stop")


def test_a4_prog_rejection_loop_and_clean_final(tmp_path):
    chat, client, store, _ = _setup(tmp_path)
    client.script = [
        ("加我微信聊聊吧", "stop"),                      # 平台名 → 系统打回
        ("今天上班好累呀，你们那边下雨了吗", "stop"),    # 干净回复
        _ok_judge(),
    ]
    reply = chat.generate_reply(device_ip="1.1.1.1", app_name="xinyu", contact_name="tk", message="在忙吗")
    assert reply.text == "今天上班好累呀，你们那边下雨了吗"
    assert reply.stats["prog_retries"] == 1 and reply.stats["attempts"] == 2
    rejected_call = client.calls[1]["messages"]
    assert rejected_call[-2]["content"] == "加我微信聊聊吧"
    assert "【系统打回】" in rejected_call[-1]["content"] and "平台名" in rejected_call[-1]["content"]


def test_a4_judge_rejection_loop(tmp_path):
    chat, client, store, _ = _setup(tmp_path)
    client.script = [
        ("好呀好呀你继续说", "stop"),
        ('{"ok": false, "reason": "答非所问"}', "stop"),
        ("嗯我在听你说呢", "stop"),
        _ok_judge(),
    ]
    reply = chat.generate_reply(device_ip="1.1.1.1", app_name="xinyu", contact_name="tk", message="讲讲你的工作")
    assert reply.text == "嗯我在听你说呢"
    assert reply.stats["judge_retries"] == 1
    judge_call = client.calls[1]
    assert judge_call["kwargs"].get("temperature") == 0.3
    regen_call = client.calls[2]["messages"]
    assert "【审核打回】" in regen_call[-1]["content"] and "答非所问" in regen_call[-1]["content"]


def test_a6_judge_invalid_json_fails_open(tmp_path):
    chat, client, store, _ = _setup(tmp_path)
    client.script = [("嗯嗯我知道了", "stop"), ("完全不是JSON", "stop")]
    reply = chat.generate_reply(device_ip="1.1.1.1", app_name="xinyu", contact_name="tk", message="在吗")
    assert reply.text == "嗯嗯我知道了" and reply.stats["judge_retries"] == 0


def test_a6_judge_disabled_skips_call(tmp_path):
    chat, client, store, _ = _setup(tmp_path, {"judge_enabled": False})
    client.script = [("嗯嗯我知道了", "stop")]
    reply = chat.generate_reply(device_ip="1.1.1.1", app_name="xinyu", contact_name="tk", message="在吗")
    assert len(client.calls) == 1  # 只有生成，无审核调用


def test_a4_budget_exhausted_outputs_last_text(tmp_path):
    chat, client, store, _ = _setup(tmp_path, {"judge_enabled": False})
    client.script = [
        ("加我微信聊聊吧", "stop"),
        ("加我微信聊聊吧", "stop"),  # 与上次完全相同 → 近似卡死退出，不再消耗预算
    ]
    reply = chat.generate_reply(device_ip="1.1.1.1", app_name="xinyu", contact_name="tk", message="在忙吗")
    assert reply.text  # 卡死退出仍输出非空（归一化后平台名已替换）
    assert reply.stats["attempts"] == 2 and reply.stats["prog_retries"] == 1


def test_a3_persona_change_keeps_session(tmp_path):
    chat, client, store, settings = _setup(tmp_path)
    sid = str(store.find_or_create_session("1.1.1.1", "xinyu", "tk")["session_id"])

    client.script = [("今天上班好累呀", "stop"), _ok_judge()]
    reply1 = chat.generate_reply(device_ip="1.1.1.1", app_name="xinyu", contact_name="tk", message="在忙吗")
    chat.record_sent(device_ip="1.1.1.1", app_name="xinyu", contact_name="tk", user_text="在忙吗", assistant_text=reply1.text)

    # 改人设（换 profile 内容），会话必须延续
    import threading
    profile_path = tmp_path / "profile.json"
    new_profile = {
        "基础设定": "另一个人设：酷酷的滑板少女。",
        "人物信息": "", "边界条件": "", "审核标准": "",
    }
    # mtime 粒度：稍等确保 mtime 变化，或直接清缓存
    chat._profile_cache.clear()
    profile_path.write_text(json.dumps(new_profile, ensure_ascii=False), encoding="utf-8")

    client.script = [("嗯我知道了", "stop"), _ok_judge()]
    reply2 = chat.generate_reply(device_ip="1.1.1.1", app_name="xinyu", contact_name="tk", message="在吗")
    assert reply2.session_id == sid == reply1.session_id


def test_history_and_summary_injection(tmp_path):
    chat, client, store, _ = _setup(tmp_path)
    sid = str(store.find_or_create_session("1.1.1.1", "xinyu", "tk")["session_id"])
    for i in range(3):
        client.script = [(f"这是第{i}轮的回复呀", "stop"), _ok_judge()]
        reply = chat.generate_reply(device_ip="1.1.1.1", app_name="xinyu", contact_name="tk", message=f"消息{i}")
        chat.record_sent(device_ip="1.1.1.1", app_name="xinyu", contact_name="tk", user_text=f"消息{i}", assistant_text=reply.text)

    # 3 次交换=6 条消息，窗口 2 轮=4 条 → 老化 2 条 ≥ min_aged(1) → 后台摘要触发
    client.script = [("【关系进展】刚认识。\n【关于对方】无。\n【关于我自己】无。\n【情绪与事件】无。\n【未完结话题】无", "stop")]
    deadline = time.time() + 5
    summary = None
    while time.time() < deadline:
        summary = store.get_summary("1.1.1.1", "xinyu", sid)
        if summary:
            break
        time.sleep(0.1)
    assert summary is not None and summary["summarized_up_to_seq"] == 2

    client.script = [("嗯嗯我在听呢", "stop"), _ok_judge()]
    chat.generate_reply(device_ip="1.1.1.1", app_name="xinyu", contact_name="tk", message="然后呢")
    last_gen_messages = client.calls[-2]["messages"]
    assert any("前情摘要" in m["content"] for m in last_gen_messages)
    assert "消息0" not in json.dumps(last_gen_messages, ensure_ascii=False)  # 已摘要的原文不再入窗


def test_a5_greeting_switch_off_no_ai_calls(tmp_path):
    chat, client, store, settings = _setup(tmp_path, {"greeting_ai_enabled": False})
    greeting = chat.generate_greeting(device_ip="1.1.1.1", app_name="xinyu", contact_name="new1")
    assert greeting in ("你好呀", "hi~在吗") and len(client.calls) == 0


def test_a5_greeting_rewrite_with_negatives_and_fallback(tmp_path):
    chat, client, store, _ = _setup(tmp_path)
    sid = str(store.find_or_create_session("1.1.1.1", "xinyu", "old")["session_id"])
    store.record_exchange("1.1.1.1", "xinyu", sid, user_text="你好", assistant_text="你好呀", kind="greeting")

    # 改写产物含英文 → 检查不过 → 重试一次 → 通过
    client.script = [("this is english!!", "stop"), ("哈喽呀，冒个泡", "stop")]
    greeting = chat.generate_greeting(device_ip="1.1.1.1", app_name="xinyu", contact_name="new1")
    assert greeting == "哈喽呀，冒个泡"
    greet_call = client.calls[0]["messages"]
    negatives = [m["content"] for m in greet_call if m["role"] == "system" and m["content"].startswith("最近已用过的开场白（禁止相似）")]
    assert negatives and "你好呀" in negatives[0]

    # 两次都不过 → 回退底稿
    client.script = [("bad 微信", "stop"), ("still bad 微信", "stop")]
    fallback = chat.generate_greeting(device_ip="1.1.1.1", app_name="xinyu", contact_name="new2")
    assert fallback in ("你好呀", "hi~在吗")


def test_record_sent_only_records_actual_exchange(tmp_path):
    chat, client, store, _ = _setup(tmp_path)
    chat.record_sent(device_ip="1.1.1.1", app_name="xinyu", contact_name="tk", user_text="u1", assistant_text="a1")
    sid = str(store.find_or_create_session("1.1.1.1", "xinyu", "tk")["session_id"])
    turns = store.recent_turns("1.1.1.1", "xinyu", sid, n=2)
    assert [(t["role"], t["content"]) for t in turns] == [("user", "u1"), ("assistant", "a1")]


def test_check_reply_rule_families():
    assert check_reply("今天上班好累呀，你们那边下雨了吗") == []
    assert any("平台名" in r for r in check_reply("加我微信聊聊吧"))
    assert any("换行" in r for r in check_reply("第一行\n第二行"))
    assert any("英文字母" in r for r in check_reply("ok 好的"))
    assert any("离场" in r for r in check_reply("我先回了啊"))
    assert check_reply("我先回了啊", user_messages=["我困了先睡了"]) == []  # 用户主动收尾放宽
    assert any("重复" in r for r in check_reply("今天上班好累呀", dedup_pool=["今天上班好累呀"]))


def test_check_greeting_rules():
    assert check_greeting("哈喽呀，冒个泡") == []
    assert check_greeting("hello!!") != []
    assert check_greeting("加我微信") != []
    assert check_greeting("") != []
