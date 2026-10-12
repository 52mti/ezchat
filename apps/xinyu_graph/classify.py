from __future__ import annotations

from typing import Any

# 顶栏 tab 的默认选中槽。全部通常在左侧，未回复在它右边。
# 哪个关键词的中心落进这块区域，就认为那个 tab 被选中。
DEFAULT_SELECTED_TAB_REGION = (0.02, 0.06, 0.20, 0.10)
TOP_BAND_Y = 0.22
SELF_X = 0.55
OPPONENT_X = 0.45

TAB_SKIP = ("全部", "未回复", "消息", "通话", "在线", "距离", "亲密度", "下一个未读")


def _text(item: dict[str, Any]) -> str:
    return str(item.get("text") or "").replace(" ", "").strip()


def _center(item: dict[str, Any]) -> tuple[float, float]:
    return float(item["x"]) + float(item["w"]) / 2, float(item["y"]) + float(item["h"]) / 2


def _region(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(float(part) for part in value)  # type: ignore[return-value]
    return DEFAULT_SELECTED_TAB_REGION


def _inside(item: dict[str, Any], region: tuple[float, float, float, float]) -> bool:
    cx, cy = _center(item)
    x, y, w, h = region
    return x <= cx <= x + w and y <= cy <= y + h


def _tab_kind(text: str) -> str | None:
    if "未回复" in text:
        return "unread"
    if "全部" in text:
        return "all"
    return None


def classify_screen(items: list[dict[str, Any]], selected_region: Any = None) -> str:
    """用关键词和文字框位置区分全部列表、未回复列表和聊天页。"""
    region = _region(selected_region)
    tabs = []
    for item in items:
        if not isinstance(item, dict) or "x" not in item:
            continue
        kind = _tab_kind(_text(item))
        if kind is None:
            continue
        _cx, cy = _center(item)
        if cy <= TOP_BAND_Y:
            tabs.append((kind, item))
    if tabs:
        unread_hit = [item for kind, item in tabs if kind == "unread" and _inside(item, region)]
        all_hit = [item for kind, item in tabs if kind == "all" and _inside(item, region)]
        if unread_hit:
            return "list_unread"
        if all_hit:
            return "list_all"
        slot_x = region[0] + region[2] / 2
        slot_y = region[1] + region[3] / 2

        def distance(item: dict[str, Any]) -> float:
            cx, cy = _center(item)
            return (cx - slot_x) ** 2 + (cy - slot_y) ** 2

        kind, _item = min(tabs, key=lambda pair: distance(pair[1]))
        return "list_unread" if kind == "unread" else "list_all"
    if any(_is_bubble(item) for item in items if isinstance(item, dict)):
        return "chat"
    return "unknown"


def _is_bubble(item: dict[str, Any]) -> bool:
    if "x" not in item:
        return False
    text = _text(item)
    if not text or text.isdigit() or any(word in text for word in TAB_SKIP):
        return False
    _cx, cy = _center(item)
    return cy > TOP_BAND_Y


def split_bubbles(items: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """左气泡是对方，右气泡是自己。返回从上到下的文字。"""
    opponent: list[tuple[float, str]] = []
    own: list[tuple[float, str]] = []
    for item in items:
        if not isinstance(item, dict) or not _is_bubble(item):
            continue
        cx, cy = _center(item)
        text = _text(item)
        if cx <= OPPONENT_X:
            opponent.append((cy, text))
        elif cx >= SELF_X:
            own.append((cy, text))
    opponent.sort()
    own.sort()
    return [text for _cy, text in opponent], [text for _cy, text in own]


def bubble_kind(items: list[dict[str, Any]]) -> str:
    opponent, own = split_bubbles(items)
    if opponent and own:
        return "both"
    if opponent:
        return "opponent"
    if own:
        return "self"
    return "empty"


def first_unread_row(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """未回复列表里第一条网名和它的中心点。"""
    rows: list[tuple[float, dict[str, Any]]] = []
    for item in items:
        if not isinstance(item, dict) or "x" not in item:
            continue
        text = _text(item)
        if not text or text.isdigit() or _tab_kind(text) or any(word in text for word in TAB_SKIP):
            continue
        cx, cy = _center(item)
        if cy <= TOP_BAND_Y:
            continue
        rows.append((cy, {"name": text, "x": cx, "y": cy}))
    if not rows:
        return None
    rows.sort(key=lambda pair: pair[0])
    return rows[0][1]


def append_lines(history: dict[str, Any] | None, contact: str, lines: list[str], direction: str) -> dict[str, list[dict[str, str]]]:
    copied: dict[str, list[dict[str, str]]] = {}
    for key, value in (history or {}).items():
        copied[str(key)] = [dict(item) for item in value if isinstance(item, dict)]
    bucket = copied.setdefault(contact, [])
    existing = {item.get("content") for item in bucket if item.get("direction") == direction}
    for line in lines:
        text = str(line).strip()
        if not text or text in existing:
            continue
        bucket.append({"direction": direction, "content": text})
        existing.add(text)
    return copied


def merge_new_messages(
    history: dict[str, Any] | None,
    contact: str,
    pending: list[str] | None,
    current: list[str] | None,
) -> list[str]:
    """去掉已经回复过的前缀，留下尚未回复的对方消息。"""
    bucket = [item for item in (history or {}).get(contact, []) if isinstance(item, dict)]
    last_out = -1
    for index, item in enumerate(bucket):
        if item.get("direction") == "out":
            last_out = index
    replied = {
        str(item.get("content") or "")
        for item in bucket[: last_out + 1]
        if item.get("direction") == "in"
    }
    tail = [
        str(item.get("content") or "")
        for item in bucket[last_out + 1 :]
        if item.get("direction") == "in" and str(item.get("content") or "").strip()
    ]
    merged: list[str] = []
    for line in tail + list(pending or []) + list(current or []):
        text = str(line).strip()
        if not text or text in replied or text in merged:
            continue
        merged.append(text)
    return merged
