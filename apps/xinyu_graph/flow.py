from __future__ import annotations

from typing import Any

from apps.xinyu_graph.classify import (
    DEFAULT_SELECTED_TAB_REGION,
    append_lines,
    bubble_kind,
    classify_screen,
    first_unread_row,
    merge_new_messages,
    split_bubbles,
)

MATCH_LIMIT = 3
KEYWORD_FALLBACK = {
    "unread_tab": ("未回复",),
    "next_unread": ("下一个未读", "下一条"),
}


def _region(state: dict[str, Any]) -> tuple[float, float, float, float]:
    value = state.get("selected_tab_region")
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(float(part) for part in value)  # type: ignore[return-value]
    return DEFAULT_SELECTED_TAB_REGION


def _capture_items(ports: Any) -> list[dict[str, Any]]:
    image = ports.capture()
    items = ports.ocr(image)
    return [item for item in items if isinstance(item, dict)]


def _keyword_hit(items: list[dict[str, Any]], words: tuple[str, ...]) -> tuple[float, float] | None:
    hits = []
    for item in items:
        if not isinstance(item, dict) or "x" not in item:
            continue
        text = str(item.get("text") or "").replace(" ", "")
        if any(word in text for word in words):
            hits.append(item)
    if not hits:
        return None
    item = min(hits, key=lambda row: float(row["y"]))
    return float(item["x"]) + float(item["w"]) / 2, float(item["y"]) + float(item["h"]) / 2


def match_retries(ports: Any, key: str, times: int = MATCH_LIMIT) -> tuple[float, float] | None:
    """模板匹配最多三次。模板图不在时，用同一次拍照的 OCR 文字框兜底。失败后先复位，再拍下一张。"""
    words = KEYWORD_FALLBACK.get(key, ())
    for attempt in range(times):
        image = ports.capture()
        hit = ports.match(image, key)
        if not hit and words:
            hit = _keyword_hit(ports.ocr(image), words)
        if hit:
            return float(hit[0]), float(hit[1])
        if attempt < times - 1:
            ports.reset()
    return None


def click_and_assert(ports: Any, x: float, y: float) -> list[dict[str, Any]]:
    """等点击完成并复位后，才拍照做断言。"""
    ports.click(x, y)
    ports.reset()
    return _capture_items(ports)


def enter(state: dict[str, Any], ports: Any) -> dict[str, Any]:
    items = _capture_items(ports)
    screen = classify_screen(items, _region(state))
    clicks = 0 if screen != "list_all" else int(state.get("list_all_clicks") or 0)
    if screen == "list_all" and clicks >= MATCH_LIMIT:
        return {"screen": screen, "phase": "done", "list_all_clicks": clicks, "last_error": "未回复已点击三次，仍停在全部"}
    if screen == "list_unread" and state.get("row_attempted"):
        return {"screen": screen, "phase": "done", "list_all_clicks": clicks}
    phase = {"list_all": "click_unread", "list_unread": "open_row", "chat": "bubbles"}.get(screen, "done")
    update: dict[str, Any] = {"screen": screen, "phase": phase, "list_all_clicks": clicks}
    if screen == "unknown":
        update["last_error"] = "无法判断当前页面"
    return update


def click_unread(state: dict[str, Any], ports: Any) -> dict[str, Any]:
    hit = match_retries(ports, "unread_tab")
    if hit is None:
        return {"phase": "done", "last_error": "未回复按钮三次未匹配"}
    click_and_assert(ports, hit[0], hit[1])
    return {"phase": "entry", "list_all_clicks": int(state.get("list_all_clicks") or 0) + 1}


def open_row(state: dict[str, Any], ports: Any) -> dict[str, Any]:
    items = _capture_items(ports)
    row = first_unread_row(items)
    if row is None:
        return {"phase": "done", "row_attempted": True, "last_error": "未回复列表没有第一条网名"}
    contact = str(row["name"])
    history = append_lines(state.get("history"), contact, [], "in")
    click_and_assert(ports, float(row["x"]), float(row["y"]))
    return {
        "phase": "entry",
        "row_attempted": True,
        "contact_name": contact,
        "history": history,
    }


def read_bubbles(state: dict[str, Any], ports: Any) -> dict[str, Any]:
    items = _capture_items(ports)
    kind = bubble_kind(items)
    opponent, _own = split_bubbles(items)
    contact = str(state.get("contact_name") or "未知")
    if kind == "opponent":
        history = append_lines(state.get("history"), contact, opponent, "in")
        pending = list(state.get("pending_opponent") or [])
        for line in opponent:
            if line not in pending:
                pending.append(line)
        signature = list(opponent)
        unchanged = int(state.get("scroll_unchanged") or 0)
        if state.get("just_scrolled"):
            previous = list(state.get("last_opponent_sig") or [])
            unchanged = unchanged + 1 if signature == previous else 0
        else:
            unchanged = 0
        shared = {
            "history": history,
            "pending_opponent": pending,
            "last_opponent_sig": signature,
            "scroll_unchanged": unchanged,
            "just_scrolled": False,
            "contact_name": contact,
        }
        if unchanged >= 2:
            return {**shared, "phase": "next_or_back"}
        return {**shared, "phase": "scroll"}
    if kind == "self":
        return {"phase": "next_or_back", "just_scrolled": False, "contact_name": contact}
    if kind == "both":
        merged = merge_new_messages(state.get("history"), contact, state.get("pending_opponent"), opponent)
        if not merged:
            return {"phase": "next_or_back", "just_scrolled": False, "pending_opponent": [], "contact_name": contact}
        history = append_lines(state.get("history"), contact, merged, "in")
        return {
            "phase": "reply",
            "history": history,
            "pending_opponent": [],
            "reply_input": "\n".join(merged),
            "just_scrolled": False,
            "scroll_unchanged": 0,
            "contact_name": contact,
        }
    return {"phase": "done", "last_error": "聊天页没有气泡"}


def scroll_down(state: dict[str, Any], ports: Any) -> dict[str, Any]:
    del state
    ports.swipe_down()
    ports.reset()
    return {"phase": "bubbles", "just_scrolled": True}


def next_or_back(state: dict[str, Any], ports: Any) -> dict[str, Any]:
    del state
    hit = match_retries(ports, "next_unread")
    if hit is not None:
        click_and_assert(ports, hit[0], hit[1])
        return {"phase": "bubbles", "just_scrolled": False, "scroll_unchanged": 0}
    hit = match_retries(ports, "back")
    if hit is not None:
        click_and_assert(ports, hit[0], hit[1])
        return {"phase": "entry", "just_scrolled": False, "row_attempted": False}
    return {"phase": "done", "last_error": "下一个未读和返回按钮都未匹配"}


def make_reply(state: dict[str, Any], ports: Any) -> dict[str, Any]:
    contact = str(state.get("contact_name") or "未知")
    text = str(state.get("reply_input") or "")
    try:
        reply_text, fingerprint = ports.reply(contact, text)
    except Exception as exc:
        return {"phase": "done", "last_error": str(exc)}
    return {"phase": "mic", "reply_text": reply_text, "reply_fingerprint": fingerprint}


def play_reply(state: dict[str, Any], ports: Any) -> dict[str, Any]:
    contact = str(state.get("contact_name") or "未知")
    reply_text = str(state.get("reply_text") or "")
    incoming = str(state.get("reply_input") or "")
    chunks = ports.prepare_tts(reply_text)
    hit = match_retries(ports, "microphone")
    if hit is None:
        return {"phase": "done", "last_error": "麦克风三次未匹配"}
    ports.click(hit[0], hit[1])
    ports.reset()
    ports.capture()
    ports.play(chunks)
    ports.mark_sent(contact, incoming, reply_text, str(state.get("reply_fingerprint") or ""))
    history = append_lines(state.get("history"), contact, [reply_text], "out")
    return {"phase": "bubbles", "history": history, "just_scrolled": False}
