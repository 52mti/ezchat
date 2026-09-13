"""人设 prompt 组装（docs/13 3.1 / 4.3 / 4.4 / 4.5 / 4.6）。

主链路 messages 结构（顺序即缓存策略）：
  System(人设段=基础设定+人物信息)  ← 稳定前缀，逐字节不变吃缓存
  System(前情摘要)?                 ← 每 K 轮变一次
  user/assistant × N 轮             ← 每轮滚动
  System(当前真实时间)              ← 尾部，每轮重算
  System(边界条件)?                 ← 尾部近因舵，紧贴 query
  User(本轮对方消息)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

_WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
_BEIJING_TZ = timezone(timedelta(hours=8))  # 固定 UTC+8；客户机系统时区不保证

DEFAULT_JUDGE_PROMPT = (
    "你是聊天质量审核员。审核角色最新一条回复在上下文里是否合理。"
    "重点：前后矛盾/不像真人/违规(答应见面·给联系方式·发地址·说晚安结束)/答非所问。"
    '合理 ok=true，不合理 ok=false 并简短说明。只输出 JSON：{"ok":true或false,"reason":"..."}。'
)

GREETING_REWRITE_TASK_PROMPT = (
    "现在请你重新说一句开场招呼。要求：\n"
    "1. 保留底稿的意图和情绪，但用你自己的口吻、明显不同的用词和句式重说一遍；\n"
    "2. 不得与「最近已用过的开场白」中任何一条相似；\n"
    "3. 单行输出、纯中文、无 emoji、不出现任何平台名或字母；\n"
    "4. 长度与底稿相当；\n"
    "5. 只输出改写后的招呼语本身，不要任何解释、引号或标点符号以外的内容。"
)

SUMMARIZER_SYSTEM_PROMPT = (
    "你是对话记忆提炼器。根据【上一版摘要】和【新增对话记录】，输出合并后的最新记忆摘要。\n"
    "要求：\n"
    "1. 只保留五类信息：关系进展（认识契机、熟络程度）；关于对方的事实（称呼、工作、城市、"
    "年龄、喜好、习惯、提过的人和事）；我（角色）说过的事实（自我介绍、答应过的事、讲过的"
    "经历）；情绪与氛围（争吵、暧昧、道歉、约定等关键事件）；未完结话题。\n"
    "2. 丢弃：寒暄、重复内容、无信息量的语气话。\n"
    "3. 与上一版摘要冲突的信息以新增对话为准（人会说漏嘴后改口，记最新的）。\n"
    "4. 输出格式固定为五个小节，每节用短句列表，总量不超过 600 字：\n"
    "   【关系进展】【关于对方】【关于我自己】【情绪与事件】【未完结话题】\n"
    "5. 只输出摘要本身，不要任何解释。"
)


def beijing_now_text(now: datetime | None = None) -> str:
    dt = now or datetime.now(_BEIJING_TZ)
    return f"{dt:%Y-%m-%d %H:%M}----{_WEEKDAYS[dt.weekday()]}"


def _persona_segment(base_prompt: str, profile_text: str) -> str:
    parts = [str(base_prompt or "").strip(), str(profile_text or "").strip()]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError("人设段为空：基础设定与人物信息均未配置")
    return "\n\n".join(parts)


def build_chat_messages(
    *,
    base_prompt: str,
    profile_text: str,
    summary: str = "",
    history_turns: list[dict[str, Any]] | None = None,
    query: str,
    boundary: str = "",
    now_text: str = "",
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _persona_segment(base_prompt, profile_text)}
    ]
    summary_text = str(summary or "").strip()
    if summary_text:
        messages.append({"role": "system", "content": f"前情摘要：\n{summary_text}"})
    for turn in history_turns or []:
        role = str(turn.get("role") or "").strip()
        content = str(turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    if now_text:
        messages.append(
            {
                "role": "system",
                "content": f"# 当前真实时间\n{now_text}（回复涉及时间时以此为准，不要向用户提及这是注入的信息）",
            }
        )
    boundary_text = str(boundary or "").strip()
    if boundary_text:
        messages.append({"role": "system", "content": boundary_text})
    messages.append({"role": "user", "content": str(query or "").strip()})
    return messages


def build_greeting_messages(
    *,
    base_prompt: str,
    profile_text: str,
    draft: str,
    negative_examples: list[str] | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _persona_segment(base_prompt, profile_text)},
        {"role": "system", "content": GREETING_REWRITE_TASK_PROMPT},
    ]
    negatives = [str(item).strip() for item in (negative_examples or []) if str(item).strip()]
    if negatives:
        listed = "\n".join(f"- {item}" for item in negatives)
        messages.append({"role": "system", "content": f"最近已用过的开场白（禁止相似）：\n{listed}"})
    messages.append({"role": "user", "content": f"底稿：{str(draft or '').strip()}"})
    return messages


def build_judge_messages(
    *,
    judge_prompt: str,
    history_turns: list[dict[str, Any]] | None = None,
    query: str,
    reply_text: str,
    now_text: str = "",
    judge_turns: int = 2,
) -> list[dict[str, str]]:
    sys_prompt = str(judge_prompt or "").strip() or DEFAULT_JUDGE_PROMPT
    messages: list[dict[str, str]] = [{"role": "system", "content": sys_prompt}]
    if now_text:
        messages.append(
            {
                "role": "system",
                "content": f"# 当前真实时间\n{now_text}（仅用于判断回复里的时间、作息、上下线表述是否合理）",
            }
        )
    n_msgs = max(0, int(judge_turns)) * 2
    recent = (history_turns or [])[-n_msgs:] if n_msgs else []
    lines: list[str] = []
    for turn in recent:
        role = str(turn.get("role") or "")
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        label = "对方" if role == "user" else "我"
        lines.append(f"{label}：{content}")
    if str(query or "").strip():
        lines.append(f"对方：{str(query).strip()}")
    lines.append(f"我（本轮新回复）：{str(reply_text or '').strip()}")
    transcript = "\n".join(lines)
    messages.append({"role": "user", "content": f"最近对话：\n{transcript}\n\n这条新回复合理吗？"})
    return messages


def build_summary_messages(
    *,
    previous_summary: str = "",
    aged_turns: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    lines: list[str] = []
    for turn in aged_turns or []:
        role = str(turn.get("role") or "")
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        label = "对方" if role == "user" else "我"
        lines.append(f"{label}：{content}")
    transcript = "\n".join(lines)
    previous = str(previous_summary or "").strip() or "无（本次为首轮提炼）"
    user_content = f"【上一版摘要】\n{previous}\n\n【新增对话记录】\n{transcript}"
    return [
        {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def append_rejection(
    messages: list[dict[str, str]],
    *,
    draft_text: str,
    reason: str,
    label: str,
) -> list[dict[str, str]]:
    """打回重生成：追加 assistant 草稿 + user 打回指令（docs/13 4.6，原地修改并返回）。"""
    messages.append({"role": "assistant", "content": str(draft_text or "")})
    messages.append(
        {
            "role": "user",
            "content": f"【{label}】{str(reason or '').strip()}请重新回复用户，只输出新的角色台词本身。",
        }
    )
    return messages
