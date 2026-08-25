"""GPT planner that produces verifiable atomic tasks for AutoGLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from auto_glm.planner import ExecutionPlan


_PLANNER_PROMPT = """你是自动化流程规划器。将用户目标拆成小而可验证的原子节点。
每个节点必须包含 id、goal、executor_task、verification、on_failure、executor、retry、timeout_seconds。
executor.type 只能是 template、ocr、color、program、autoglm、hybrid。
本地模板/OCR/颜色规则优先；只有本地规则不能可靠定位时才使用 autoglm。
AutoGLM 每次只能执行一个短动作，动作后必须重新截图验证。所有文本必须是简体中文。
只输出 JSON：{"summary":"...","nodes":[...]}。"""


@dataclass(frozen=True)
class GPTPlannerConfig:
    api_key: str
    model: str
    endpoint: str
    timeout_seconds: float = 75.0
    max_tokens: int = 3000


class GPTPlannerClient:
    """OpenAI-compatible chat-completions planner.

    The caller owns provider selection and credentials, so this module remains
    usable with the project's existing GPT provider configuration.
    """

    def __init__(self, config: GPTPlannerConfig, *, session: requests.Session | None = None) -> None:
        if not config.api_key.strip() or not config.model.strip() or not config.endpoint.strip():
            raise ValueError("GPT 规划器缺少 API Key、模型名或接口地址")
        self.config = config
        self.session = session or requests.Session()

    def plan(
        self,
        goal: str,
        *,
        current_state: str,
        capabilities: dict[str, Any],
        project_skills: str = "",
    ) -> ExecutionPlan:
        user_prompt = (
            f"用户目标：{goal.strip()}\n"
            f"当前界面状态：{current_state.strip() or '未知'}\n"
            f"可用能力：{capabilities}\n"
            f"项目技能：\n{project_skills.strip() or '无'}"
        )
        response = self.session.post(
            self.config.endpoint,
            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.config.model,
                "temperature": 0,
                "max_tokens": self.config.max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": _PLANNER_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("GPT 规划器未返回 JSON 规划结果") from error
        return ExecutionPlan.from_json(str(content))
