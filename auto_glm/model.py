"""OpenAI-compatible AutoGLM client for one screenshot-led action at a time."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests
from PIL import Image

from auto_glm.protocol import AutoGLMAction, parse_action


SYSTEM_PROMPT = """你是手机界面操作执行器。每次只输出一个动作，不要输出多个步骤。
坐标原点为左上角，右下角为 (999,999)。只能输出：
<think>简短理由</think><answer>do(action=\"Tap\", element=[x,y])</answer>
或 Swipe、Type、Back、Wait、Take_over、finish(message=\"...\")。
必须根据当前截图执行当前原子任务；不确定时使用 Take_over，不能猜测点击位置。"""


@dataclass(frozen=True)
class AutoGLMConfig:
    api_key: str
    model: str = "autoglm-phone"
    endpoint: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    timeout_seconds: float = 60.0
    max_tokens: int = 512


@dataclass(frozen=True)
class ModelDecision:
    action: AutoGLMAction
    thinking: str
    raw_content: str


class AutoGLMClient:
    def __init__(self, config: AutoGLMConfig, *, session: requests.Session | None = None) -> None:
        if not config.api_key.strip():
            raise ValueError("AutoGLM API Key 不能为空")
        self.config = config
        self.session = session or requests.Session()

    def decide(self, task: str, screenshot: Any, *, history: list[str] | None = None) -> ModelDecision:
        instruction = f"当前原子任务：{task.strip()}"
        if history:
            instruction += "\n最近动作结果：\n" + "\n".join(history[-5:])
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _as_data_url(screenshot)}},
                        {"type": "text", "text": instruction},
                    ],
                },
            ],
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
            content = str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("AutoGLM 返回中没有可解析的动作内容") from error
        thinking, action_text = _split_thinking(content)
        return ModelDecision(action=parse_action(action_text), thinking=thinking, raw_content=content)


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
    return thinking, text
