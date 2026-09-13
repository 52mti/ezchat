from __future__ import annotations

"""A1/A2 验收：messages 结构不变式 + 会话存储（docs/13 十一）。"""

import json
import sqlite3
import time

from services.persona_prompt import (
    append_rejection,
    build_chat_messages,
    build_greeting_messages,
    build_judge_messages,
    build_summary_messages,
)
from services.persona_session_store import PersonaSessionStore


def _history(n=3):
    turns = []
    for i in range(n):
        turns.append({"role": "user", "content": f"u{i}", "seq": i * 2 + 1, "kind": "chat"})
        turns.append({"role": "assistant", "content": f"a{i}", "seq": i * 2 + 2, "kind": "chat"})
    return turns


def test_a1_prefix_stable_across_rounds():
    """同会话两轮：第 1 条 message 逐字节相同（缓存前缀不变式）。"""
    kwargs = dict(base_prompt="基础设定文本", profile_text="人物信息文本", boundary="边界文本")
    m1 = build_chat_messages(query="第一轮", history_turns=[], now_text="T1", **kwargs)
    m2 = build_chat_messages(query="第二轮", history_turns=_history(2), now_text="T2", summary="摘要", **kwargs)
    assert m1[0] == m2[0] and m1[0]["role"] == "system"


def test_a1_segment_order_time_and_boundary_before_query():
    messages = build_chat_messages(
        base_prompt="B", profile_text="P", summary="S", history_turns=_history(2),
        query="Q", boundary="BOUNDARY", now_text="2026-09-13 21:40----星期日",
    )
    assert messages[-1] == {"role": "user", "content": "Q"}
    assert "BOUNDARY" in messages[-2]["content"] and messages[-2]["role"] == "system"
    assert "当前真实时间" in messages[-3]["content"]
    assert messages[1]["content"].startswith("前情摘要")
    assert [m["role"] for m in messages[2:-3]] == ["user", "assistant"] * 2


def test_a1_empty_persona_segment_raises():
    import pytest

    with pytest.raises(ValueError):
        build_chat_messages(base_prompt="", profile_text="", query="Q")


def test_a1_rejection_appends_assistant_then_instruction():
    messages = [{"role": "user", "content": "Q"}]
    append_rejection(messages, draft_text="草稿", reason="违规", label="系统打回")
    assert messages[-2] == {"role": "assistant", "content": "草稿"}
    assert messages[-1]["role"] == "user" and "【系统打回】" in messages[-1]["content"] and "只输出新的角色台词" in messages[-1]["content"]


def test_greeting_messages_include_negatives():
    messages = build_greeting_messages(
        base_prompt="B", profile_text="P", draft="你好呀",
        negative_examples=["哈喽呀", ""],
    )
    sys_texts = [m["content"] for m in messages if m["role"] == "system"]
    assert any("最近已用过的开场白" in t and "哈喽呀" in t for t in sys_texts)
    assert messages[-1]["content"] == "底稿：你好呀"


def test_judge_messages_transcript_and_time():
    messages = build_judge_messages(
        judge_prompt="", history_turns=_history(2), query="在吗",
        reply_text="在的呀", now_text="T", judge_turns=1,
    )
    assert messages[0]["role"] == "system" and "JSON" in messages[0]["content"]  # 内置默认
    assert "当前真实时间" in messages[1]["content"]
    user_text = messages[-1]["content"]
    assert "对方：在吗" in user_text and "我（本轮新回复）：在的呀" in user_text and "合理吗" in user_text


def test_summary_messages_previous_and_aged():
    aged = [
        {"role": "user", "content": "u1", "seq": 1},
        {"role": "assistant", "content": "a1", "seq": 2},
    ]
    messages = build_summary_messages(previous_summary="", aged_turns=aged)
    assert "无（本次为首轮提炼）" in messages[-1]["content"]
    assert "对方：u1" in messages[-1]["content"] and "我：a1" in messages[-1]["content"]


def test_a2_session_identity_and_turn_window(tmp_path):
    store = PersonaSessionStore(root=tmp_path)
    s1 = store.find_or_create_session("1.1.1.1", "xinyu", "tk")
    assert store.find_or_create_session("1.1.1.1", "xinyu", "tk")["session_id"] == s1["session_id"]
    other = store.find_or_create_session("1.1.1.1", "xinyu", "other")
    assert other["session_id"] != s1["session_id"]

    sid = str(s1["session_id"])
    for i in range(6):
        store.record_exchange("1.1.1.1", "xinyu", sid, user_text=f"u{i}", assistant_text=f"a{i}")
    turns = store.recent_turns("1.1.1.1", "xinyu", sid, n=4)
    assert [t["content"] for t in turns] == ["u4", "a4", "u5", "a5"]


def test_a2_summary_watermark_never_overlaps_window(tmp_path):
    store = PersonaSessionStore(root=tmp_path)
    sid = str(store.find_or_create_session("1.1.1.1", "xinyu", "tk")["session_id"])
    for i in range(8):  # 16 条消息
        store.record_exchange("1.1.1.1", "xinyu", sid, user_text=f"u{i}", assistant_text=f"a{i}")

    aged, aged_max, watermark = store.aged_turns_for_summary("1.1.1.1", "xinyu", sid, window_n=6, min_aged=2)
    max_seq = store.max_seq("1.1.1.1", "xinyu", sid)
    assert max_seq == 16
    assert [t["seq"] for t in aged] == [1, 2, 3, 4] and aged_max == 4 and watermark == 0
    store.save_summary("1.1.1.1", "xinyu", sid, "v1", aged_max)

    aged2, aged_max2, watermark2 = store.aged_turns_for_summary("1.1.1.1", "xinyu", sid, window_n=6, min_aged=2)
    assert aged2 == [] and watermark2 == 4

    # 窗口内的轮次永远不会进入老化区间
    window_seqs = {t["seq"] for t in store.recent_turns("1.1.1.1", "xinyu", sid, n=12)}
    aged_seqs = {t["seq"] for t in aged} | {t["seq"] for t in aged2}
    assert not (window_seqs & aged_seqs)


def test_a2_min_aged_threshold_respected(tmp_path):
    store = PersonaSessionStore(root=tmp_path)
    sid = str(store.find_or_create_session("1.1.1.1", "xinyu", "tk")["session_id"])
    for i in range(4):  # 8 条消息；window 6 → 老化 1..? upper=8-12<0 → 无
        store.record_exchange("1.1.1.1", "xinyu", sid, user_text=f"u{i}", assistant_text=f"a{i}")
    aged, _, _ = store.aged_turns_for_summary("1.1.1.1", "xinyu", sid, window_n=2, min_aged=10)
    assert aged == []


def test_a2_cleanup_removes_only_inactive_sessions(tmp_path):
    store = PersonaSessionStore(root=tmp_path)
    active = str(store.find_or_create_session("1.1.1.1", "xinyu", "active")["session_id"])
    stale = str(store.find_or_create_session("1.1.1.1", "xinyu", "stale")["session_id"])
    store.record_exchange("1.1.1.1", "xinyu", stale, user_text="u", assistant_text="a")
    store.save_summary("1.1.1.1", "xinyu", stale, "s", 2)

    conn = sqlite3.connect(store._path("1.1.1.1", "xinyu"))
    conn.execute("UPDATE sessions SET updated_at='2020-01-01T00:00:00' WHERE session_id=?", (stale,))
    conn.commit()
    conn.close()

    assert store.cleanup_inactive_sessions("1.1.1.1", "xinyu", inactive_days=30) == 1
    assert store.get_summary("1.1.1.1", "xinyu", stale) is None
    assert store.recent_turns("1.1.1.1", "xinyu", active, n=2) == []
    assert store.find_or_create_session("1.1.1.1", "xinyu", "active")["session_id"] == active


def test_a2_recent_greetings_across_sessions(tmp_path):
    store = PersonaSessionStore(root=tmp_path)
    for contact in ("a", "b"):
        sid = str(store.find_or_create_session("1.1.1.1", "xinyu", contact)["session_id"])
        store.record_exchange("1.1.1.1", "xinyu", sid, user_text="你好", assistant_text=f"招呼-{contact}", kind="greeting")
    assert sorted(store.recent_greetings("1.1.1.1", "xinyu", n=5)) == ["招呼-a", "招呼-b"]
