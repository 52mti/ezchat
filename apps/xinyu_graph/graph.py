from __future__ import annotations

import contextvars
from typing import Any, TypedDict

from apps.xinyu_graph.flow import (
    click_unread,
    enter,
    make_reply,
    next_or_back,
    open_row,
    play_reply,
    read_bubbles,
    scroll_down,
)

PORTS: contextvars.ContextVar[Any] = contextvars.ContextVar("xinyu_graph_ports")


class GraphState(TypedDict, total=False):
    hops: int
    max_hops: int
    phase: str
    screen: str
    list_all_clicks: int
    row_attempted: bool
    contact_name: str
    history: dict[str, list[dict[str, str]]]
    pending_opponent: list[str]
    just_scrolled: bool
    scroll_unchanged: int
    last_opponent_sig: list[str]
    selected_tab_region: list[float]
    reply_text: str
    reply_input: str
    reply_fingerprint: str
    last_error: str


def _guard(state: GraphState) -> dict[str, Any] | None:
    if int(state.get("hops") or 0) >= int(state.get("max_hops") or 12):
        return {"phase": "done"}
    return None


def _finish(state: GraphState, update: dict[str, Any]) -> dict[str, Any]:
    if update.get("phase") == "done" and "hops" not in update:
        update = {**update, "hops": int(state.get("hops") or 0) + 1}
        return update
    return {**update, "hops": int(state.get("hops") or 0) + 1}


def _step(state: GraphState, action: Any) -> dict[str, Any]:
    stopped = _guard(state)
    if stopped is not None:
        return stopped
    return _finish(state, action(state, PORTS.get()))


def entry_node(state: GraphState) -> dict[str, Any]:
    return _step(state, enter)


def click_unread_node(state: GraphState) -> dict[str, Any]:
    return _step(state, click_unread)


def open_row_node(state: GraphState) -> dict[str, Any]:
    return _step(state, open_row)


def bubbles_node(state: GraphState) -> dict[str, Any]:
    return _step(state, read_bubbles)


def scroll_node(state: GraphState) -> dict[str, Any]:
    return _step(state, scroll_down)


def next_or_back_node(state: GraphState) -> dict[str, Any]:
    return _step(state, next_or_back)


def reply_node(state: GraphState) -> dict[str, Any]:
    return _step(state, make_reply)


def mic_node(state: GraphState) -> dict[str, Any]:
    return _step(state, play_reply)


def route_phase(state: GraphState) -> str:
    phase = str(state.get("phase") or "done")
    if phase in {"entry", "click_unread", "open_row", "bubbles", "scroll", "next_or_back", "reply", "mic"}:
        return phase
    return "done"


def build_graph(checkpointer: Any) -> Any:
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(GraphState)
    graph.add_node("entry", entry_node)
    graph.add_node("click_unread", click_unread_node)
    graph.add_node("open_row", open_row_node)
    graph.add_node("bubbles", bubbles_node)
    graph.add_node("scroll", scroll_node)
    graph.add_node("next_or_back", next_or_back_node)
    graph.add_node("reply", reply_node)
    graph.add_node("mic", mic_node)
    graph.add_edge(START, "entry")
    destinations = {
        "entry": "entry",
        "click_unread": "click_unread",
        "open_row": "open_row",
        "bubbles": "bubbles",
        "scroll": "scroll",
        "next_or_back": "next_or_back",
        "reply": "reply",
        "mic": "mic",
        "done": END,
    }
    for name in ("entry", "click_unread", "open_row", "bubbles", "scroll", "next_or_back", "reply", "mic"):
        graph.add_conditional_edges(name, route_phase, destinations)
    return graph.compile(checkpointer=checkpointer)
