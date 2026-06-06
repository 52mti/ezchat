from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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


class WechatApp(BaseApp):
    name = "微信"
    description = "WeChat-like unread message scan and TTS reply skeleton"

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
        self.last_activity_at = time.monotonic()

    def _mark_activity(self) -> None:
        self.last_activity_at = time.monotonic()

    def run_once(self) -> None:
        if self.status.is_circuit_open:
            return
        try:
            reporter = self.set_runtime_step
            keep_awake_interval = float(self.config.get("keep_awake_interval_seconds", 0) or 0)
            if keep_awake_interval > 0 and time.monotonic() - self.last_activity_at >= keep_awake_interval:
                reporter("keep_awake_due", "执行防熄屏跳转")
                try:
                    steps.keep_screen_awake(self.robot, self.config, reporter=reporter)
                except Exception as keep_awake_exc:
                    self.emit_runtime_log("warning", "keep_awake_failed", {"error": str(keep_awake_exc)})
                else:
                    self._mark_activity()

            entry_image = self.capture_entry_image()
            reporter("popup_check", "检查弹窗")
            target = self.call_step_with_optional_image(
                steps.find_unread_target,
                self.robot,
                self.config,
                reporter=reporter,
                image=entry_image,
            )
            if not target:
                return

            round_started_at = time.monotonic()
            target_name = str(target.get("name", "") or "").strip()
            point_text = steps._format_point(float(target["x"]), float(target["y"]))
            reporter(
                "enter_chat",
                f"进入聊天：{target_name}" if target_name else "进入聊天",
                log_label=f"进入聊天：{target_name} {point_text}" if target_name else f"进入聊天：{point_text}",
            )
            steps.enter_chat(self.robot, float(target["x"]), float(target["y"]))
            self._mark_activity()
            message = steps.read_message(self.robot, self.config, target, reporter=reporter)
            self.emit_runtime_log(
                "info",
                "dialogue_ocr_last_message",
                {
                    "contact": message.get("contact", ""),
                    "content": message.get("content", ""),
                },
            )

            reporter("request_ai", "请求AI回复")
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(self.reply_provider, message)
            try:
                steps.focus_input_box(self.robot, self.config, reporter=reporter)
                self._mark_activity()
                raw_action = future.result()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            action = raw_action if isinstance(raw_action, ReplyAction) else ReplyAction(text=str(raw_action))
            self.emit_runtime_log(
                "info",
                "ai_reply_text",
                {
                    "contact": message.get("contact", ""),
                    "reply_text": action.text,
                    "skip": action.skip,
                    "reason": action.reason,
                },
            )
            if action.skip:
                reporter("skip_reply", "跳过本次回复")
                steps.go_back(self.robot, self.config, reporter=reporter)
                self._mark_activity()
                self.mark_success()
                return
            if not action.text:
                raise RuntimeError("reply provider returned empty text")

            steps.play_voice_reply(self.robot, action.text, self.config, reporter=reporter)
            self._mark_activity()
            steps.send_reply(self.robot, self.config, reporter=reporter)
            self._mark_activity()
            if self.sent_callback:
                reporter("mark_sent", "记录发送结果")
                self.sent_callback(action, message)
            steps.go_back(self.robot, self.config, reporter=reporter)
            self._mark_activity()
            reporter("wait_next_round", "等待下一轮扫描")
            self.record_reply_round(time.monotonic() - round_started_at)
            self.mark_success()
        except Exception as exc:
            self.on_error(exc)
