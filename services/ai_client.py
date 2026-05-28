from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests


class AIClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class AIReply:
    text: str
    conversation_id: str
    raw: dict[str, Any]


DEFAULT_CHAT_BEARER = "REMOVED-BEARER-TOKEN"
DEFAULT_CREDITS_BEARER = "REMOVED-BEARER-TOKEN-12.13"
INSUFFICIENT_CREDIT_HINTS = ("次数不足", "请充值", "余额不足", "剩余积分不足")


class AIClient:
    def __init__(
        self,
        api_base: str = "http://150.158.173.140",
        chat_bearer: str = "",
        credits_bearer: str = "",
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.chat_bearer = chat_bearer or DEFAULT_CHAT_BEARER
        self.credits_bearer = credits_bearer or DEFAULT_CREDITS_BEARER
        self.timeout = timeout
        self.session = session or requests.Session()

    def _headers(self, bearer: str) -> dict[str, str]:
        if not bearer:
            raise AIClientError("AI bearer token is not configured")
        return {"Authorization": f"Bearer {bearer}", "Content-Type": "application/json"}

    @staticmethod
    def parse_reply_payload(payload: dict[str, Any]) -> AIReply:
        answer = payload.get("answer", "")
        text = ""
        if isinstance(answer, str):
            try:
                parsed = json.loads(answer)
            except json.JSONDecodeError:
                text = answer
            else:
                text = str(parsed.get("huifu", ""))
        elif isinstance(answer, dict):
            text = str(answer.get("huifu", ""))

        normalized = str(text or "").strip()
        if normalized and any(hint in normalized for hint in INSUFFICIENT_CREDIT_HINTS):
            raise AIClientError(f"AI service reported insufficient credits: {normalized}")
        if not text:
            raise AIClientError("AI response did not contain answer.huifu")
        return AIReply(
            text=text,
            conversation_id=str(payload.get("conversation_id", "")),
            raw=payload,
        )

    def chat(
        self,
        query: str,
        persona: dict[str, Any],
        apikey: str,
        conversation_id: str = "",
        retries: int = 2,
        retry_delay: float = 3.0,
    ) -> AIReply:
        if not apikey:
            raise AIClientError("apikey is not configured")
        payload = {
            "inputs": {
                "name": persona.get("name", ""),
                "age": persona.get("age", ""),
                "work": persona.get("work", ""),
                "location": persona.get("location", ""),
                "renshe": persona.get("renshe", ""),
                "apikey": apikey,
            },
            "query": query,
            "response_mode": "blocking",
            "conversation_id": conversation_id,
            "user": apikey,
        }
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = self.session.post(
                    f"{self.api_base}/v1/chat-messages",
                    headers=self._headers(self.chat_bearer),
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return self.parse_reply_payload(response.json())
            except (requests.RequestException, ValueError, AIClientError) as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(retry_delay)
        raise AIClientError(f"AI chat request failed: {last_error}") from last_error

    def remaining_credits(self, apikey: str, timeout: float = 10.0) -> int | None:
        if not apikey:
            raise AIClientError("apikey is not configured")
        payload = {
            "inputs": {"apikey": apikey},
            "response_mode": "blocking",
            "user": apikey,
        }
        response = self.session.post(
            f"{self.api_base}/v1/workflows/run",
            headers=self._headers(self.credits_bearer),
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        value = data.get("data", {}).get("outputs", {}).get("remaining_count")
        return int(value) if value is not None and str(value).isdigit() else None
