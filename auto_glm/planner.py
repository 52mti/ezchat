"""Strict plan schema shared by a GPT planner and AutoGLM executor."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlanNode:
    id: str
    goal: str
    executor_task: str
    verification: str
    on_failure: str
    executor_type: str = "autoglm"
    retry: int = 1
    timeout_seconds: int = 30


@dataclass(frozen=True)
class ExecutionPlan:
    summary: str
    nodes: tuple[PlanNode, ...] = field(default_factory=tuple)

    @classmethod
    def from_json(cls, content: str | dict[str, Any]) -> "ExecutionPlan":
        payload = json.loads(content) if isinstance(content, str) else content
        if not isinstance(payload, dict):
            raise ValueError("GPT 规划结果必须是 JSON 对象")
        raw_nodes = payload.get("nodes")
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise ValueError("GPT 规划结果必须包含至少一个 nodes 节点")
        nodes: list[PlanNode] = []
        identifiers: set[str] = set()
        for index, raw in enumerate(raw_nodes, start=1):
            if not isinstance(raw, dict):
                raise ValueError(f"规划节点 {index} 必须是对象")
            required = ("id", "goal", "executor_task", "verification", "on_failure")
            missing = [key for key in required if not str(raw.get(key, "")).strip()]
            if missing:
                raise ValueError(f"规划节点 {index} 缺少字段：{', '.join(missing)}")
            identifier = str(raw["id"]).strip()
            if identifier in identifiers:
                raise ValueError(f"规划节点 id 重复：{identifier}")
            identifiers.add(identifier)
            nodes.append(
                PlanNode(
                    id=identifier,
                    goal=str(raw["goal"]).strip(),
                    executor_task=str(raw["executor_task"]).strip(),
                    verification=str(raw["verification"]).strip(),
                    on_failure=str(raw["on_failure"]).strip(),
                    executor_type=str((raw.get("executor") or {}).get("type", raw.get("executor_type", "autoglm"))).strip(),
                    retry=max(0, int(raw.get("retry", 1))),
                    timeout_seconds=max(1, int(raw.get("timeout_seconds", 30))),
                )
            )
        return cls(summary=str(payload.get("summary", "")).strip(), nodes=tuple(nodes))
