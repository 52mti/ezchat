"""OpenAI-compatible AutoGLM client for one screenshot-led action at a time."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from auto_glm.protocol import AutoGLMAction, parse_action
from auto_glm.secrets import read_local_secret


SYSTEM_PROMPT = """你是一个手机界面操作智能体。根据当前截图和任务，每次只执行一个动作。
必须严格输出以下格式，标签外不得输出任何文字：
<think>简短理由</think>
<answer>动作</answer>

动作只能是以下 Python 风格调用之一：
do(action=\"Launch\", app=\"xxx\")
do(action=\"Tap\", element=[x,y])
do(action=\"Type\", text=\"xxx\")
do(action=\"Type_Name\", text=\"xxx\")
do(action=\"Swipe\", start=[x1,y1], end=[x2,y2])
do(action=\"Long Press\", element=[x,y])
do(action=\"Double Tap\", element=[x,y])
do(action=\"Back\")
do(action=\"Home\")
do(action=\"Wait\", duration=\"x seconds\")
do(action=\"Take_over\", message=\"xxx\")
do(action=\"Interact\")
do(action=\"Note\", message=\"True\")
do(action=\"Call_API\", instruction=\"xxx\")
finish(message=\"任务完成\")

坐标范围为 0 到 999，左上角是 (0,0)，右下角是 (999,999)。
当前截图已经满足当前任务时，必须输出 finish，不得继续点击。
finish 的 message 必须简短，不要包含未转义的引号、换行或 Markdown。
不确定时使用 Take_over，禁止猜测坐标或输出多个动作。"""


@dataclass(frozen=True)
class AutoGLMConfig:
    api_key: str
    model: str = "autoglm-phone"
    endpoint: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    timeout_seconds: float = 60.0
    max_tokens: int = 512

    @classmethod
    def from_workspace(cls, workspace_root: Path, **overrides: Any) -> "AutoGLMConfig":
        """Load the local AutoGLM key without exposing it in app configuration."""
        api_key = read_local_secret(workspace_root, "autoglm.txt", "zhipu.txt")
        return cls(api_key=api_key, **overrides)


@dataclass(frozen=True)
class ModelDecision:
    action: AutoGLMAction
    thinking: str
    raw_content: str


class AutoGLMProtocolError(ValueError):
    """A model reply was received but did not contain an executable action."""

    def __init__(self, message: str, raw_content: str) -> None:
        super().__init__(message)
        self.raw_content = raw_content


class AutoGLMClient:
    def __init__(self, config: AutoGLMConfig, *, session: requests.Session | None = None) -> None:
        if not config.api_key.strip():
            raise ValueError("AutoGLM API Key 不能为空")
        self.config = config
        self.session = session or requests.Session()

    def decide(
        self,
        task: str,
        screenshot: Any,
        *,
        history: list[str] | None = None,
        system_prompt: str | None = None,
        reference_images: list[tuple[str, Any]] | None = None,
        conversation: list[dict[str, Any]] | None = None,
    ) -> ModelDecision:
        instruction = f"当前原子任务：{task.strip()}"
        if history:
            instruction += "\n最近动作结果：\n" + "\n".join(history[-5:])
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "当前完整截图（动作坐标必须基于此图输出）："},
            {"type": "image_url", "image_url": {"url": _as_data_url(screenshot)}},
        ]
        for label, reference_image in reference_images or []:
            content.extend(
                [
                    {"type": "text", "text": f"视觉参考图：{label}。只用于识别控件外观，不得以此图坐标输出动作。"},
                    {"type": "image_url", "image_url": {"url": _as_data_url(reference_image)}},
                ]
            )
        content.append({"type": "text", "text": instruction})
        user_message = {"role": "user", "content": content}
        messages = conversation if conversation is not None else [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT}
        ]
        messages.append(user_message)
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": 0,
            "messages": messages,
        }
        response = self.session.post(
            self.config.endpoint,
            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        try:
            content = _content_to_text(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("AutoGLM 返回中没有可解析的动作内容") from error
        thinking, action_text = _split_thinking(content)
        try:
            action = parse_action(action_text)
        except ValueError as error:
            if conversation is not None:
                messages.pop()
            raise AutoGLMProtocolError(str(error), content) from error
        decision = ModelDecision(action=action, thinking=thinking, raw_content=content)
        if conversation is not None:
            # Mirror Open-AutoGLM: keep the action trace as text, but discard
            # old screenshot bytes after each turn to control context size.
            messages[-1] = {
                "role": "user",
                "content": [item for item in user_message["content"] if item.get("type") == "text"],
            }
            messages.append(
                {"role": "assistant", "content": f"<think>{thinking}</think><answer>{_action_text(content)}</answer>"}
            )
        return decision


def _as_data_url(image: Any) -> str:
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        pil_image = Image.fromarray(np.asarray(image)).convert("RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _split_thinking(content: str) -> tuple[str, str]:
    text = str(content).strip()
    thinking = ""
    if "<think>" in text:
        thinking = text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    else:
        # Some AutoGLM replies omit the requested XML tags but still explain the
        # choice before the executable action. Keep that explanation visible.
        starts = [index for index in (text.find("do("), text.find("finish(")) if index >= 0]
        if starts:
            thinking = text[: min(starts)].strip()
    return thinking, text


def _action_text(content: str) -> str:
    text = str(content).strip()
    if "<answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    starts = [index for index in (text.find("do("), text.find("finish(")) if index >= 0]
    return text[min(starts) :].strip() if starts else text


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "".join(parts).strip()
    return str(content)
