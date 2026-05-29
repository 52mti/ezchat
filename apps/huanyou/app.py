from __future__ import annotations

import time
from typing import Any, Callable

from apps.base import BaseApp, ReplyAction

from . import steps
from .config import DEFAULT_CONFIG


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class HuanyouApp(BaseApp):
    name = "欢友"
    description = "Huanyou unread message scan and TTS reply skeleton"

    def __init__(
        self,
        robot: Any,
        config: dict[str, Any] | None = None,
        reply_provider: Callable[[dict[str, str]], str | ReplyAction] | None = None,
        sent_callback: Callable[[ReplyAction, dict[str, str]], None] | None = None,
    ) -> None:
        merged_config = _deep_merge(DEFAULT_CONFIG, config or {})
        super().__init__(robot, merged_config)
        self.reply_provider = reply_provider or (lambda message: "")
        self.sent_callback = sent_callback

    def run_once(self) -> None:
        if self.status.is_circuit_open:
            return
        try:
            reporter = self.set_runtime_step
            pos = steps.find_unread(self.robot, self.config, reporter=reporter)
            if not pos:
                return
            round_started_at = time.monotonic()
            reporter("enter_chat", "进入聊天界面", log_label=f"进入聊天界面：{steps.format_point(*pos)}")
            steps.enter_chat(self.robot, *pos)
            message = steps.read_message(self.robot, self.config, reporter=reporter)
            raw_action = self.reply_provider(message)
            action = raw_action if isinstance(raw_action, ReplyAction) else ReplyAction(text=str(raw_action))
            if action.skip:
                steps.go_back(self.robot, self.config, reporter=reporter)
                if action.reason == "duplicate_message_protection":
                    self.request_pause("重复消息保护", "duplicate_message_protection")
                    return
                self.mark_success()
                return
            if not action.text:
                raise RuntimeError("reply provider returned empty text")
            steps.input_reply(self.robot, action.text, self.config, reporter=reporter)
            steps.send_reply(self.robot, self.config, reporter=reporter)
            if self.sent_callback:
                self.sent_callback(action, message)
            steps.go_back(self.robot, self.config, reporter=reporter)
            self.record_reply_round(time.monotonic() - round_started_at)
            self.mark_success()
        except Exception as exc:
            self.on_error(exc)
