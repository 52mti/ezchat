from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Callable

from apps.base import BaseApp, ReplyAction
from apps.xinyu_graph.classify import DEFAULT_SELECTED_TAB_REGION
from apps.xinyu_graph.graph import PORTS, build_graph

CHECKPOINT_DIR = Path("runtime") / "graph_checkpoints"
MAX_HOPS = 12


class XinyuGraphApp(BaseApp):
    name = "心遇图"
    description = "LangGraph 值守：全部、未回复、聊天页三条分支"

    def __init__(
        self,
        robot: Any,
        config: dict[str, Any] | None = None,
        reply_provider: Callable[[dict[str, Any]], str | ReplyAction] | None = None,
        sent_callback: Callable[[ReplyAction, dict[str, str]], None] | None = None,
        **_kwargs: Any,
    ) -> None:
        active = dict(config or {})
        active.setdefault("selected_tab_region", list(DEFAULT_SELECTED_TAB_REGION))
        ocr_config = active.get("ocr") if isinstance(active.get("ocr"), dict) else {}
        ocr_config = dict(ocr_config)
        ocr_config.setdefault("runtime_enabled", True)
        ocr_config.setdefault("backend", "auto")
        active["ocr"] = ocr_config
        if isinstance(config, dict):
            config.clear()
            config.update(active)
            super().__init__(robot, config)
        else:
            super().__init__(robot, active)
        self.reply_provider = reply_provider
        self.sent_callback = sent_callback
        self._graph = None
        self._checkpoint = None
        self._checkpoint_conn: sqlite3.Connection | None = None

    def _device_ip(self) -> str:
        runtime = self._runtime_context()
        return str(runtime.get("device_ip") or "unknown")

    def _ensure_graph(self) -> Any:
        if self._graph is not None:
            return self._graph
        from langgraph.checkpoint.sqlite import SqliteSaver

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self._checkpoint_conn = sqlite3.connect(str(CHECKPOINT_DIR / "xinyu_graph.sqlite"), check_same_thread=False)
        self._checkpoint = SqliteSaver(self._checkpoint_conn)
        self._graph = build_graph(self._checkpoint)
        return self._graph

    def run_once(self) -> None:
        from apps.xinyu_graph.ports import ArmPorts

        graph = self._ensure_graph()
        ports = ArmPorts(self.robot, self.config, self.reply_provider, self.sent_callback)
        token = PORTS.set(ports)
        config = {"configurable": {"thread_id": self._device_ip()}, "recursion_limit": 80}
        region = self.config.get("selected_tab_region") or list(DEFAULT_SELECTED_TAB_REGION)
        try:
            self.set_runtime_step("entry", "透视拍照并粗分类")
            graph.invoke(
                {
                    "hops": 0,
                    "max_hops": MAX_HOPS,
                    "row_attempted": False,
                    "selected_tab_region": [float(value) for value in region],
                },
                config,
            )
        finally:
            PORTS.reset(token)
        snapshot = graph.get_state(config)
        values = snapshot.values if snapshot is not None else {}
        phase = str(values.get("phase") or "")
        error = str(values.get("last_error") or "")
        contact = str(values.get("contact_name") or "")
        label = error or f"{phase} {contact}".strip()
        self.set_runtime_step(phase or "done", label)
        if error:
            self.status.last_error = error
            self.emit_runtime_log("warning", error, {"contact": contact, "screen": values.get("screen", "")})
