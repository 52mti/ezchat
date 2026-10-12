from __future__ import annotations

from apps.xinyu_graph.classify import classify_screen, merge_new_messages
from apps.xinyu_graph.graph import PORTS, build_graph

REGION = [0.02, 0.06, 0.20, 0.10]
ALL_ITEMS = [
    {"text": "全部", "x": 0.05, "y": 0.09, "w": 0.12, "h": 0.04},
    {"text": "未回复", "x": 0.30, "y": 0.09, "w": 0.14, "h": 0.04},
]
UNREAD_ITEMS = [
    {"text": "全部", "x": 0.30, "y": 0.09, "w": 0.12, "h": 0.04},
    {"text": "未回复", "x": 0.05, "y": 0.09, "w": 0.12, "h": 0.04},
    {"text": "小明", "x": 0.16, "y": 0.32, "w": 0.20, "h": 0.05},
]
OPPONENT_ITEMS = [{"text": "在吗", "x": 0.08, "y": 0.40, "w": 0.22, "h": 0.05}]
SELF_ITEMS = [{"text": "我在", "x": 0.62, "y": 0.48, "w": 0.20, "h": 0.05}]
BOTH_ITEMS = [
    {"text": "在吗", "x": 0.08, "y": 0.40, "w": 0.22, "h": 0.05},
    {"text": "我在", "x": 0.62, "y": 0.55, "w": 0.20, "h": 0.05},
]


class FakePorts:
    def __init__(self, items, matches=None):
        self.items = items
        self.matches = dict(matches or {})
        self.clicks = []
        self.resets = 0
        self.swipes = 0
        self.replies = []
        self.played = []
        self.sent = []
        self.match_calls = []

    def capture(self):
        return object()

    def ocr(self, _image):
        return self.items

    def match(self, _image, key):
        self.match_calls.append(key)
        spec = self.matches.get(key)
        if isinstance(spec, list):
            if not spec:
                return None
            return spec.pop(0)
        return spec

    def click(self, x, y):
        self.clicks.append((round(float(x), 3), round(float(y), 3)))

    def reset(self):
        self.resets += 1

    def swipe_down(self):
        self.swipes += 1

    def reply(self, contact, text):
        self.replies.append((contact, text))
        return "回复:" + text, "fp"

    def prepare_tts(self, text):
        return [text] if text else []

    def play(self, chunks):
        self.played.extend(chunks)

    def mark_sent(self, contact, incoming, outgoing, fingerprint):
        self.sent.append((contact, incoming, outgoing, fingerprint))


def _run(ports, *, max_hops=12, contact="", thread="t"):
    from langgraph.checkpoint.memory import MemorySaver

    graph = build_graph(MemorySaver())
    token = PORTS.set(ports)
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 80}
    try:
        graph.invoke(
            {
                "hops": 0,
                "max_hops": max_hops,
                "row_attempted": False,
                "list_all_clicks": 0,
                "history": {},
                "pending_opponent": [],
                "contact_name": contact,
                "just_scrolled": False,
                "scroll_unchanged": 0,
                "selected_tab_region": REGION,
                "phase": "entry",
            },
            config,
        )
    finally:
        PORTS.reset(token)
    return graph.get_state(config).values


def test_all_tab_clicks_unread_three_times_then_stops():
    ports = FakePorts(ALL_ITEMS, {"unread_tab": (0.4, 0.12)})
    state = _run(ports, thread="all")
    assert classify_screen(ALL_ITEMS, REGION) == "list_all"
    assert len(ports.clicks) == 3
    assert state["list_all_clicks"] == 3
    assert ports.resets >= 3


def test_unread_tab_records_first_name_and_clicks_its_center():
    ports = FakePorts(UNREAD_ITEMS, {"unread_tab": (0.2, 0.12)})
    state = _run(ports, thread="unread")
    assert classify_screen(UNREAD_ITEMS, REGION) == "list_unread"
    assert state["contact_name"] == "小明"
    assert "小明" in state["history"]
    assert ports.clicks[0] == (0.26, 0.345)


def test_opponent_only_caches_message_and_scrolls_back_to_bubbles():
    ports = FakePorts(OPPONENT_ITEMS)
    state = _run(ports, max_hops=3, contact="小明", thread="opponent")
    assert ports.swipes == 1
    assert state["history"]["小明"][0]["content"] == "在吗"
    assert "在吗" in state["pending_opponent"]


def test_self_only_falls_back_to_back_button_after_three_misses():
    ports = FakePorts(SELF_ITEMS, {"next_unread": [None, None, None], "back": (0.08, 0.05)})
    _run(ports, max_hops=4, contact="小明", thread="self")
    assert ports.match_calls.count("next_unread") == 3
    assert "back" in ports.match_calls
    assert (0.08, 0.05) in ports.clicks


def test_both_sides_merge_then_reply_and_play():
    ports = FakePorts(BOTH_ITEMS, {"microphone": (0.2, 0.9)})
    state = _run(ports, max_hops=4, contact="小明", thread="both")
    assert ports.replies == [("小明", "在吗")]
    assert ports.played == ["回复:在吗"]
    assert ports.sent[0][0] == "小明"
    assert state["history"]["小明"][-1]["direction"] == "out"


def test_merge_drops_prefix_already_replied():
    history = {"小明": [{"direction": "in", "content": "在吗"}, {"direction": "out", "content": "在的"}]}
    assert merge_new_messages(history, "小明", ["在吗"], ["在吗", "新消息"]) == ["新消息"]
