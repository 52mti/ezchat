from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from vision.screen import ensure_portrait_image


class VisionDialogueError(RuntimeError):
    pass


JSON_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)

_PROMPT_CACHE_LOCK = threading.Lock()
_PROMPT_CACHE: dict[str, tuple[int, str]] = {}

DEFAULT_OPENROUTER_ENSEMBLE = {
    "analysis_models": [
        "z-ai/glm-5.3-flash",
        "minimax/minimax-m3",
        "x-ai/grok-4.6",
    ],
    "judge_model": "z-ai/glm-5.3-flash",
}


@dataclass(frozen=True)
class VisionDialogueReply:
    payload: dict[str, Any]
    provider: str
    model: str
    raw_text: str


def _normalize_secret_value(value: Any) -> str:
    text = str(value or "")
    for marker in ("\ufeff", "\u200b", "\u200c", "\u200d", "\u2060"):
        text = text.replace(marker, "")
    return text.strip()


def _read_local_secret(workspace_root: Path, *names: str) -> str:
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


def _read_local_json_secret(workspace_root: Path, key: str) -> str:
    env_dir = workspace_root / ".env"
    for name in ("vision_dialogue_keys.json", "minimax_keys.json"):
        path = env_dir / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        value = _normalize_secret_value(payload.get(key, ""))
        if value:
            return value
    return ""


def _extract_json_text(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    match = JSON_CODE_BLOCK_RE.search(raw)
    if match:
        return match.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start : end + 1].strip()
    return raw


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content or "")


def _normalize_messages(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload if isinstance(payload, dict) else {})
    senders = result.get("senders")
    if not isinstance(senders, dict):
        senders = {}
    result["senders"] = {
        "sender1": senders.get("sender1") if isinstance(senders.get("sender1"), dict) else {"alias": "", "role": "对方"},
        "sender2": senders.get("sender2") if isinstance(senders.get("sender2"), dict) else {"alias": "我", "role": "我方"},
    }
    normalized_messages: list[dict[str, Any]] = []
    for item in result.get("messages", []):
        if not isinstance(item, dict):
            continue
        sender = str(item.get("sender", "")).strip()
        if sender not in {"sender1", "sender2"}:
            continue
        message_type = str(item.get("type", "text") or "text").strip() or "text"
        content = item.get("content")
        if content is None:
            normalized_content = ""
        else:
            normalized_content = str(content).strip()
        normalized_messages.append(
            {
                "id": len(normalized_messages) + 1,
                "sender": sender,
                "type": message_type,
                "content": normalized_content,
            }
        )
    result["messages"] = normalized_messages
    stats = result.get("stats")
    if not isinstance(stats, dict):
        stats = {}
    stats.setdefault("total_messages", len(normalized_messages))
    stats.setdefault("sender1_count", sum(1 for item in normalized_messages if item["sender"] == "sender1"))
    stats.setdefault("sender2_count", sum(1 for item in normalized_messages if item["sender"] == "sender2"))
    result["stats"] = stats
    result["chat_title"] = str(result.get("chat_title", "") or "").strip()
    return result


class VisionDialogueClient:
    def __init__(
        self,
        *,
        workspace_root: str | Path | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self._session_is_custom = session is not None
        self.session = session or requests.Session()

    def _prompt_text(self, prompt_path: str | Path) -> str:
        path = Path(prompt_path)
        if not path.is_absolute():
            configured = str(os.environ.get("JIXIEBI_PACKAGE_ROOT", "") or "").strip()
            base = Path(configured).resolve() if configured else self.workspace_root
            path = (base / path).resolve()
        try:
            stat = path.stat()
        except OSError as exc:
            raise VisionDialogueError(f"dialogue prompt not found: {path}") from exc
        cache_key = str(path)
        with _PROMPT_CACHE_LOCK:
            cached = _PROMPT_CACHE.get(cache_key)
            if cached and cached[0] == int(stat.st_mtime_ns):
                return cached[1]
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise VisionDialogueError(f"failed to read dialogue prompt: {path}") from exc
        if not text:
            raise VisionDialogueError(f"dialogue prompt is empty: {path}")
        with _PROMPT_CACHE_LOCK:
            _PROMPT_CACHE[cache_key] = (int(stat.st_mtime_ns), text)
        return text

    def _image_to_data_uri(self, image: Any) -> str:
        portrait = ensure_portrait_image(image)
        from io import BytesIO

        buffer = BytesIO()
        portrait.save(buffer, format="JPEG", quality=95)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        mime = mimetypes.guess_type("image.jpg")[0] or "image/jpeg"
        return f"data:{mime};base64,{encoded}"

    def _secret(self, env_name: str, *local_names: str) -> str:
        return _normalize_secret_value(os.getenv(env_name)) or _read_local_secret(self.workspace_root, *local_names)

    def _openrouter_key(self) -> str:
        return (
            "REMOVED-OPENROUTER-KEY"
            or _read_local_json_secret(self.workspace_root, "openrouter_key")
            or self._secret(
            "OPENROUTER_API_KEY",
            "openrouter_key.txt",
            "openrouter_api_key.txt",
            "openrouter_key",
            "openrouter_api_key",
        )
        )

    @staticmethod
    def _provider_settings(config: dict[str, Any], provider: str) -> tuple[str, str, str]:
        model = str(config.get("model", "minimax/minimax-m3") or "minimax/minimax-m3")
        model_map = config.get("openrouter_models", {})
        if isinstance(model_map, dict):
            model = str(model_map.get(model, model) or model)
        return provider, str(config.get("openrouter_url", "https://openrouter.ai/api/v1/chat/completions")), model

    @staticmethod
    def _user_message(config: dict[str, Any], data_uri: str) -> dict[str, Any]:
        user_text = str(config.get("user_text", "请识别这张欢游聊天截图，只输出 JSON。") or "").strip()
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }

    def _request_payload(
        self, config: dict[str, Any], prompt: str, data_uri: str, provider: str,
        *, model_override: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        _provider, _base_url, model = self._provider_settings(config, provider)
        if model_override:
            model = str(model_override).strip()
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                self._user_message(config, data_uri),
            ],
            "stream": False,
            "temperature": float(config.get("temperature", 0.1) or 0.1),
        }
        if provider == "openrouter":
            payload["max_tokens"] = int(config.get("max_tokens", 4096) or 4096)
            payload["reasoning"] = {"enabled": False}
            # 自定义编排直接请求模型，不附加 openrouter/fusion 插件。
        else:
            payload["max_completion_tokens"] = int(config.get("max_completion_tokens", 4096) or 4096)
            payload["thinking"] = {"type": "disabled"}
        return payload, model

    @staticmethod
    def _error_message(response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text[:1000]
        if isinstance(payload, dict):
            base_resp = payload.get("base_resp")
            if isinstance(base_resp, dict) and base_resp.get("status_msg"):
                return str(base_resp.get("status_msg"))
            error = payload.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error.get("message"))
            if payload.get("message"):
                return str(payload.get("message"))
        return json.dumps(payload, ensure_ascii=False, default=str)

    @staticmethod
    def _extract_response_text(payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    return _content_to_text(message.get("content"))
        return ""

    def _request_once(
        self, config: dict[str, Any], prompt: str, data_uri: str, provider: str,
        *, model_override: str | None = None, session: Any | None = None,
    ) -> VisionDialogueReply:
        if provider != "openrouter":
            raise VisionDialogueError("仅支持 OpenRouter 视觉识别通道")
        provider_name, base_url, _model = self._provider_settings(config, provider)
        payload, model = self._request_payload(
            config, prompt, data_uri, provider_name, model_override=model_override
        )
        api_key = self._openrouter_key()
        if not api_key:
            raise VisionDialogueError(f"{provider_name} key is not configured")
        request_session = session or self.session
        response = request_session.post(
            base_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=float(config.get("timeout", 90.0) or 90.0),
        )
        # 部分 OpenRouter 端点强制开启 reasoning；仅针对该明确错误去掉参数重试一次。
        if provider_name == "openrouter" and response.status_code == 400:
            message = self._error_message(response)
            if "reasoning" in message.lower() and "mandatory" in message.lower():
                retry_payload = dict(payload)
                retry_payload.pop("reasoning", None)
                response = request_session.post(
                    base_url,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=retry_payload,
                    timeout=float(config.get("timeout", 90.0) or 90.0),
                )
        if response.status_code != 200:
            message = self._error_message(response)
            raise VisionDialogueError(f"{provider_name} returned {response.status_code}: {message}")
        try:
            response_payload = response.json()
        except ValueError as exc:
            raise VisionDialogueError(f"{provider_name} returned invalid JSON") from exc
        raw_text = self._extract_response_text(response_payload)
        if not raw_text:
            raise VisionDialogueError(f"{provider_name} response did not contain choices[0].message.content")
        normalized_text = _extract_json_text(raw_text)
        try:
            parsed = json.loads(normalized_text)
        except json.JSONDecodeError as exc:
            raise VisionDialogueError(f"{provider_name} response was not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise VisionDialogueError(f"{provider_name} response JSON root must be an object")
        return VisionDialogueReply(
            payload=_normalize_messages(parsed),
            provider=provider_name,
            model=model,
            raw_text=raw_text,
        )

    def _request_openrouter_ensemble(
        self, config: dict[str, Any], prompt: str, data_uri: str
    ) -> VisionDialogueReply:
        """并行请求分析模型，再单独请求汇总模型，绕开 Fusion 路由。"""
        ensemble = config.get("openrouter_ensemble", DEFAULT_OPENROUTER_ENSEMBLE)
        if not isinstance(ensemble, dict):
            ensemble = DEFAULT_OPENROUTER_ENSEMBLE
        models = [str(item).strip() for item in ensemble.get("analysis_models", []) if str(item).strip()]
        if not models:
            raise VisionDialogueError("openrouter ensemble analysis_models is empty")

        def run(model: str) -> tuple[str, VisionDialogueReply]:
            request_session = self.session if self._session_is_custom else requests.Session()
            return model, self._request_once(
                config, prompt, data_uri, "openrouter", model_override=model, session=request_session
            )

        results: list[tuple[str, VisionDialogueReply]] = []
        with ThreadPoolExecutor(max_workers=len(models), thread_name_prefix="ocr-llm") as pool:
            futures = {pool.submit(run, model): model for model in models}
            for future in as_completed(futures):
                model = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    # 单模型失败只影响该模型，剩余结果继续参与汇总。
                    print(f"[vision-dialogue] OpenRouter analysis failed ({model}): {exc}")
        if not results:
            raise VisionDialogueError("all OpenRouter analysis models failed")

        judge_model = str(ensemble.get("judge_model", "z-ai/glm-5.3-flash") or "z-ai/glm-5.3-flash").strip()
        order = {model: index for index, model in enumerate(models)}
        analyses = [
            {"model": model, "result": reply.payload}
            for model, reply in sorted(results, key=lambda item: order[item[0]])
        ]
        judge_prompt = (
            f"{prompt}\n\n"
            "你是最终汇总模型。以下是多个视觉模型对同一张截图的独立分析结果。"
            "请以原图为最终依据，比较共识并纠正冲突，只输出上述格式的最终 JSON。"
            "真实聊天放入 messages；明确的系统提示、广告等放入 excluded_messages；"
            "底部系统快捷语、推荐开场白和快捷回复放入 quick_replies，绝不能放入 excluded_messages，"
            "不要输出分析过程。\n独立分析结果：\n"
            f"{json.dumps(analyses, ensure_ascii=False, default=str)}"
        )
        judge = self._request_once(
            config, judge_prompt, data_uri, "openrouter", model_override=judge_model,
            session=self.session if self._session_is_custom else requests.Session(),
        )
        return VisionDialogueReply(
            payload=judge.payload, provider="openrouter_ensemble", model=judge_model, raw_text=judge.raw_text
        )

    def parse_dialogue(self, image: Any, config: dict[str, Any]) -> VisionDialogueReply:
        dialogue_llm = config.get("dialogue_llm", {})
        if not isinstance(dialogue_llm, dict):
            raise VisionDialogueError("dialogue_llm config must be a dict")
        prompt_path = dialogue_llm.get("prompt_path") or "apps/huanyou/dialogue_llm_prompt.txt"
        prompt = self._prompt_text(prompt_path)
        data_uri = self._image_to_data_uri(image)
        return self._request_openrouter_ensemble(dialogue_llm, prompt, data_uri)
