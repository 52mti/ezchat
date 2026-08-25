"""GPT planner that produces verifiable atomic tasks for AutoGLM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from auto_glm.planner import ExecutionPlan
from auto_glm.secrets import read_local_secret


_PLANNER_PROMPT = """你是自动化流程规划器。将用户目标拆成小而可验证的原子节点。
每个节点必须包含 id、goal、executor_task、verification、on_failure、executor、retry、timeout_seconds。
executor.type 只能是 template、ocr、color、program、autoglm、hybrid。
必须严格遵守调用方声明的可用能力，不能凭空添加模板、OCR、颜色检测或 App 专项知识。
AutoGLM 每次只能执行一个短动作，动作后必须重新截图验证。所有文本必须是简体中文。
节点描述只表达可观察的目标状态，不能臆测界面一定存在某个中间面板、按钮或入口。
例如目标是进入相册时，节点应为“进入照片选择界面”，不能擅自拆成“打开附件面板再进入相册”；具体从哪个可见入口进入由视觉执行器根据截图决定。
每个节点只能完成一个状态迁移，不得把后续节点的操作合并到当前节点。
只输出 JSON：{"summary":"...","nodes":[...]}。"""


@dataclass(frozen=True)
class GPTPlannerConfig:
    api_key: str
    model: str
    endpoint: str
    timeout_seconds: float = 75.0
    max_tokens: int = 3000

    @classmethod
    def from_workspace(cls, workspace_root: Path, *, model: str, endpoint: str, **overrides: Any) -> "GPTPlannerConfig":
        """Load GPT planner credentials from ``.env/gpt_planner.txt``."""
        api_key = read_local_secret(workspace_root, "gpt_planner.txt")
        return cls(api_key=api_key, model=model, endpoint=endpoint, **overrides)


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
