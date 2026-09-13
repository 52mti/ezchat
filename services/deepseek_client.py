"""DeepSeek 官方直连客户端（人设 AI 新链路，docs/13 第三节）。

OpenAI 格式：POST /chat/completions、GET /models；Bearer 认证。
错误分类：401 auth / 402 payment / 400 bad_request 立即失败不重试；
429 rate_limit、5xx server、网络错误退避重试。
user_id 按客户 apikey 派生（jixie-+sha1 前 16 位），同一 key 恒定，
使该客户名下所有请求共享 KVCache 前缀缓存池。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

DEFAULT_API_BASE = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
# GET /models 失败时 UI 下拉回退此清单（docs/13 3.1）
FALLBACK_MODELS = ["deepseek-flash", "deepseek-v4-pro"]
USER_ID_PREFIX = "jixie-"
ENV_KEY_NAME = "DEEPSEEK_API_KEY"
LOCAL_KEY_FILE = "deepseek_api_key.txt"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 512
# 聊天场景默认关 thinking：快速出文本，否则 temperature 无效且首字慢（docs/13 第二节）
THINKING_DISABLED = {"type": "disabled"}

RETRYABLE_KINDS = {"rate_limit", "server", "network"}


class DeepSeekError(RuntimeError):
    """kind ∈ auth(401) / payment(402) / bad_request(400) / rate_limit(429) / server(5xx) / network"""

    def __init__(self, message: str, *, kind: str = "network", status: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status


@dataclass(frozen=True)
class DeepSeekReply:
    text: str
    finish_reason: str
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def _normalize_secret_value(value: Any) -> str:
    text = str(value or "")
    for marker in ("\ufeff", "\u200b", "\u200c", "\u200d", "\u2060"):
        text = text.replace(marker, "")
    return text.strip()


def _default_workspace_root() -> Path:
    configured = str(os.environ.get("JIXIEBI_PACKAGE_ROOT", "") or "").strip()
    return Path(configured).resolve() if configured else Path.cwd().resolve()


def read_local_secret(workspace_root: Path, *names: str) -> str:
    env_dir = workspace_root / ".env"
    for name in names:
        path = env_dir / name
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = _normalize_secret_value(line)
            if not stripped or stripped.startswith("#"):
                continue
            if "=" in stripped:
                _key, value = stripped.split("=", 1)
                stripped = _normalize_secret_value(value).strip('"').strip("'")
            return stripped
    return ""


def resolve_api_key(workspace_root: str | Path | None = None) -> str:
    env_value = _normalize_secret_value(os.environ.get(ENV_KEY_NAME))
    if env_value:
        return env_value
    root = Path(workspace_root).resolve() if workspace_root else _default_workspace_root()
    return read_local_secret(root, LOCAL_KEY_FILE)


def write_local_api_key(workspace_root: str | Path | None, api_key: str) -> None:
    """UI 保存入口：key 只写 .env/deepseek_api_key.txt（原子写，绝不进 config.json）。"""
    root = Path(workspace_root).resolve() if workspace_root else _default_workspace_root()
    env_dir = root / ".env"
    env_dir.mkdir(parents=True, exist_ok=True)
    value = _normalize_secret_value(api_key)
    path = env_dir / LOCAL_KEY_FILE
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(value, encoding="utf-8")
    os.replace(tmp_path, path)


def derive_user_id(api_key: str) -> str:
    digest = hashlib.sha1((api_key or "").encode("utf-8")).hexdigest()[:16]
    return f"{USER_ID_PREFIX}{digest}"


def _error_kind(status: int) -> str:
    if status == 401:
        return "auth"
    if status == 402:
        return "payment"
    if status == 400:
        return "bad_request"
    if status == 429:
        return "rate_limit"
    return "server"


def _error_body_snippet(response: requests.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return (response.text or "")[:200]
    err = data.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or err)[:200]
    return str(data)[:200]


class DeepSeekClient:
    def __init__(
        self,
        *,
        api_key: str = "",
        api_base: str = DEFAULT_API_BASE,
        timeout: float = 60.0,
        workspace_root: str | Path | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else _default_workspace_root()
        self.api_key = api_key or resolve_api_key(self.workspace_root)
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.user_id = derive_user_id(self.api_key) if self.api_key else ""

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise DeepSeekError(
                "DeepSeek API key 未配置（.env/deepseek_api_key.txt 或环境变量 DEEPSEEK_API_KEY）",
                kind="auth",
            )
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        disable_thinking: bool = True,
        user_id: str = "",
        retries: int = 2,
        retry_delay: float = 3.0,
        timeout: float | None = None,
    ) -> DeepSeekReply:
        payload: dict[str, Any] = {
            "model": model or DEFAULT_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if disable_thinking:
            payload["thinking"] = dict(THINKING_DISABLED)
        payload["user_id"] = (user_id or self.user_id)[:512]
        last_error: DeepSeekError | None = None
        for attempt in range(retries + 1):
            try:
                response = self.session.post(
                    f"{self.api_base}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=timeout or self.timeout,
                )
            except requests.RequestException as exc:
                last_error = DeepSeekError(f"DeepSeek 网络请求失败：{exc}", kind="network")
            else:
                if response.status_code >= 400:
                    last_error = DeepSeekError(
                        f"DeepSeek HTTP {response.status_code}：{_error_body_snippet(response)}",
                        kind=_error_kind(response.status_code),
                        status=response.status_code,
                    )
                else:
                    try:
                        data = response.json()
                    except ValueError as exc:
                        last_error = DeepSeekError(
                            f"DeepSeek 响应不是 JSON：{exc}",
                            kind="server",
                        )
                    else:
                        choices = data.get("choices") or []
                        if not choices:
                            last_error = DeepSeekError("DeepSeek 响应缺少 choices", kind="server")
                        else:
                            choice = choices[0]
                            message = choice.get("message") or {}
                            content = message.get("content")
                            text = content if isinstance(content, str) else ""
                            return DeepSeekReply(
                                text=text,
                                finish_reason=str(choice.get("finish_reason") or ""),
                                usage=data.get("usage") or {},
                                raw=data,
                            )
            if last_error is not None:
                if last_error.kind not in RETRYABLE_KINDS:
                    raise last_error
                if attempt < retries:
                    time.sleep(retry_delay)
                    continue
        raise last_error or DeepSeekError("DeepSeek 调用失败", kind="network")

    def list_models(self, *, timeout: float = 10.0) -> list[str]:
        response = self.session.get(
            f"{self.api_base}/models",
            headers=self._headers(),
            timeout=timeout,
        )
        if response.status_code >= 400:
            raise DeepSeekError(
                f"DeepSeek HTTP {response.status_code}：{_error_body_snippet(response)}",
                kind=_error_kind(response.status_code),
                status=response.status_code,
            )
        data = response.json()
        models = [
            str(item.get("id"))
            for item in (data.get("data") or [])
            if isinstance(item, dict) and item.get("id")
        ]
        return models


def load_models_with_fallback(client: DeepSeekClient, *, timeout: float = 10.0) -> list[str]:
    """UI 下拉数据源：获取失败回退内置清单，绝不抛异常。"""
    try:
        models = client.list_models(timeout=timeout)
    except DeepSeekError:
        return list(FALLBACK_MODELS)
    return models or list(FALLBACK_MODELS)
