from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
import random
import re
import time
from pathlib import Path
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


class XinyuApp(BaseApp):
    name = "心遇"
    description = "Xinyu unread message scan and TTS reply flow"

    def __init__(
        self,
        robot: Any,
        config: dict[str, Any] | None = None,
        reply_provider: Callable[[dict[str, str]], str | ReplyAction] | None = None,
        sent_callback: Callable[[ReplyAction, dict[str, str]], None] | None = None,
    ) -> None:
        merged_config = _deep_merge(DEFAULT_CONFIG, config or {})
        steps._disable_dialogue_region_filter(merged_config)
        super().__init__(robot, merged_config)
        self.reply_provider = reply_provider or (lambda message: "")
        self.sent_callback = sent_callback
        self.last_activity_at = time.monotonic()
        self.idle_points_checked = False

    def _mark_activity(self) -> None:
        self.last_activity_at = time.monotonic()

    @staticmethod
    def _balance_from_points(points: Any) -> str:
        try:
            value = Decimal(str(points).strip()) / Decimal("10")
        except (InvalidOperation, ValueError):
            return ""
        return str(int(value))

    def _run_idle_maintenance(self, reporter: Callable[..., None]) -> None:
        now = time.monotonic()
        keep_awake_interval = float(self.config.get("keep_awake_interval_seconds", 0) or 0)
        if keep_awake_interval > 0 and now - self.last_activity_at >= keep_awake_interval:
            reporter("keep_awake_due", "执行防熄屏跳转")
            try:
                did_keep_awake = steps.keep_screen_awake(self.robot, self.config, reporter=reporter)
            except Exception as keep_awake_exc:
                self.emit_runtime_log("warning", "keep_awake_failed", {"error": str(keep_awake_exc)})
            else:
                if did_keep_awake:
                    self._mark_activity()

        if self.idle_points_checked:
            return
        self.idle_points_checked = True
        reporter("points_check_due", "首次防熄屏读取余额")
        try:
            points = steps.read_my_points(self.robot, self.config, reporter=reporter)
        except Exception as points_exc:
            self.emit_runtime_log("warning", "xinyu_balance_failed", {"error": str(points_exc)})
            return
        if points:
            balance = self._balance_from_points(points)
            self.emit_runtime_log("info", "xinyu_balance", {"balance": balance, "points": points})
            self._mark_activity()

    def _greeting_path(self) -> Path:
        runtime = self.config.get("runtime", {})
        runtime_dir = runtime.get("greetings_dir") if isinstance(runtime, dict) else None
        configured_dir = self.config.get("greetings_dir") or runtime_dir
        if not configured_dir:
            configured_dir = "runtime/greetings"
        return Path(str(configured_dir)) / "pingtai.txt"

    def _pick_greeting_reply(self) -> str:
        path = self._greeting_path()
        try:
            lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError:
            return ""
        return random.choice(lines) if lines else ""

    @staticmethod
    def _clean_content(content: Any) -> str:
        text = str(content or "").strip()
        text = re.sub(r"\s+", "", text)
        return text.strip("，。！？、,.!?;；:：\"'“”‘’（）()[]【】")

    @classmethod
    def _can_reply_from_list(cls, content: Any) -> bool:
        cleaned = cls._clean_content(content)
        if not cleaned:
            return False
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", cleaned))
        comparable_length = cjk_count if cjk_count > 0 else len(cleaned)
        return comparable_length < 10

    @classmethod
    def _is_empty_list_target(cls, target: dict[str, Any]) -> bool:
        name = cls._clean_content(target.get("name", ""))
        content = cls._clean_content(target.get("content", ""))
        return not content and name in {"暂无消息", "暂无信息", "没有消息", "无消息"}

    def run_once(self) -> None:
        if self.status.is_circuit_open:
            return
        try:
            reporter = self.set_runtime_step
            reporter("popup_check", "检查并关闭弹窗")
            target = steps.find_unread_target(self.robot, self.config, reporter=reporter)
            if not target:
                reporter("empty_list_target", "暂无消息，进入防熄屏等待")
                self._run_idle_maintenance(reporter)
                return

            round_started_at = time.monotonic()
            self.emit_runtime_log(
                "info",
                "target_selected",
                {
                    "name": target.get("name", ""),
                    "favorability": target.get("favorability"),
                    "unread_count": target.get("unread_count"),
                    "content": target.get("content", ""),
                },
            )
            if self._is_empty_list_target(target):
                reporter("empty_list_target", "暂无消息，本轮不进入聊天")
                self._run_idle_maintenance(reporter)
                self.mark_success()
                return
            self.idle_points_checked = False

            list_content = self._clean_content(target.get("content", ""))
            use_list_content = self._can_reply_from_list(list_content)
            greeting_reply = ""
            if use_list_content and list_content == "你好":
                reporter("greeting_reply_pick", "从 pingtai.txt 选择招呼语")
                greeting_reply = self._pick_greeting_reply()

            point_text = steps.format_point(float(target["x"]), float(target["y"]))
            target_name = str(target.get("name", "") or "").strip()
            reporter(
                "enter_chat",
                "进入聊天界面",
                log_label=f"进入聊天界面：{target_name} {point_text}" if target_name else f"进入聊天界面：{point_text}",
            )
            steps.enter_chat(self.robot, float(target["x"]), float(target["y"]))
            self._mark_activity()
            chat_image = steps.confirm_chat_window(self.robot, self.config, reporter=reporter)
            if steps.send_new_greeting_if_available(self.robot, self.config, image=chat_image, reporter=reporter):
                self._mark_activity()
                reporter("wait_next_round", "等待下一轮扫描")
                self.record_reply_round(time.monotonic() - round_started_at)
                self.mark_success()
                return

            if use_list_content:
                message = steps.build_list_message(target, list_content)
                reporter(
                    "direct_content_reply",
                    "使用列表内容作为对话输入",
                    log_label=f"使用列表内容作为对话输入：{message.get('content', '')}",
                )
            else:
                message = steps.read_message(self.robot, self.config, target, image=chat_image, reporter=reporter)

            self.emit_runtime_log(
                "info",
                "xinyu_message_input",
                {
                    "contact": message.get("contact", ""),
                    "content": message.get("content", ""),
                    "source": "list" if use_list_content else "dialogue",
                },
            )

            input_focused = False
            if greeting_reply:
                reporter("request_ai", "请求 AI 回复")
                steps.focus_input_box(self.robot, self.config, reporter=reporter)
                self._mark_activity()
                input_focused = True
                action = ReplyAction(text=greeting_reply)
            else:
                reporter("request_ai", "请求 AI 回复")
                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(self.reply_provider, message)
                try:
                    steps.focus_input_box(self.robot, self.config, reporter=reporter)
                    self._mark_activity()
                    input_focused = True
                    raw_action = future.result()
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)
                action = raw_action if isinstance(raw_action, ReplyAction) else ReplyAction(text=str(raw_action))

            self.emit_runtime_log(
                "info",
                "xinyu_reply_text",
                {
                    "contact": message.get("contact", ""),
                    "reply_text": action.text,
                    "skip": action.skip,
                    "reason": action.reason,
                },
            )
            if action.skip:
                reporter("skip_reply", "跳过本轮回复")
                steps.go_back(self.robot, self.config, reporter=reporter)
                self._mark_activity()
                self.mark_success()
                return
            if not action.text:
                raise RuntimeError("reply provider returned empty text")

            if not input_focused:
                steps.focus_input_box(self.robot, self.config, reporter=reporter)
                self._mark_activity()
            steps.play_voice_reply(self.robot, action.text, self.config, reporter=reporter)
            self._mark_activity()
            send_clicks = steps.send_reply(self.robot, self.config, reporter=reporter)
            self.emit_runtime_log("info", "xinyu_send_clicks", {"clicks": send_clicks})
            self._mark_activity()
            if self.sent_callback:
                reporter("mark_sent", "记录发送结果")
                self.sent_callback(action, message)
            steps.go_back(self.robot, self.config, double_click_on_first_attempt=True, reporter=reporter)
            self._mark_activity()
            reporter("wait_next_round", "等待下一轮扫描")
            self.record_reply_round(time.monotonic() - round_started_at)
            self.mark_success()
        except steps.NoPhoneScreenError as exc:
            self.set_runtime_step("no_phone_waiting", "无手机异常，等待用户放回手机")
            self.status.last_error = str(exc)
            self.status.is_circuit_open = False
            self.recover_error_count = 0
            self.emit_runtime_log("warning", "no_phone_detected", {"error": str(exc)})
        except steps.BlackScreenError as exc:
            self.set_runtime_step("black_screen_waiting", "黑屏异常，等待用户亮屏")
            self.status.last_error = str(exc)
            self.status.is_circuit_open = False
            self.recover_error_count = 0
            self.emit_runtime_log("warning", "black_screen_detected", {"error": str(exc)})
        except Exception as exc:
            self.on_error(exc)
