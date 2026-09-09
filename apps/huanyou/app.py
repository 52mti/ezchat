from __future__ import annotations

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


class HuanyouApp(BaseApp):
    name = "欢友"
    description = "Huanyou favorability-based chat scan and TTS reply flow"

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
        self._input_box_opened = False
        self.idle_points_checked = False
        self.idle_waiting = False
        self._last_step_detail = ""
        self._startup_back_button_checked = False

    def _reporter(self, step: str, label: str, log_label: str | None = None) -> None:
        self._last_step_detail = str(log_label or label or step)
        self.set_runtime_step(step, label, log_label=log_label)

    def _mark_activity(self) -> None:
        self.last_activity_at = time.monotonic()

    def _mark_recovered_progress(self) -> None:
        self.status.last_error = ""
        self.status.error_count = 0
        self.status.is_circuit_open = False
        self.recover_error_count = 0

    def _ensure_startup_back_button(self, reporter: Callable[..., None]) -> bool:
        if self._startup_back_button_checked:
            return True
        if steps.cache_startup_back_button(self.robot, self.config, reporter=reporter):
            self._startup_back_button_checked = True
            self.emit_runtime_log("info", "huanyou_startup_back_button_cached", {})
            return True
        message = "启动检查未找到返回按钮，停止进程"
        self.set_runtime_step("startup_back_button_missing", message)
        self.status.last_error = message
        self.status.is_circuit_open = True
        self.recover_error_count = 0
        self.emit_runtime_log("warning", "huanyou_startup_back_button_missing", {"error": message})
        return False

    def _reset_round_metrics(self) -> None:
        cache = self.config.setdefault("_runtime_cache", {})
        if isinstance(cache, dict):
            cache["metrics"] = {"ocr_count": 0, "capture_count": 0}

    def _emit_round_metrics(self, started_at: float) -> None:
        cache = self.config.get("_runtime_cache", {})
        metrics = cache.get("metrics", {}) if isinstance(cache, dict) else {}
        if not isinstance(metrics, dict):
            metrics = {}
        self.emit_runtime_log(
            "info",
            "huanyou_round_metrics",
            {
                "ocr_count": int(metrics.get("ocr_count", 0) or 0),
                "ocr_cache_hit_count": int(metrics.get("ocr_cache_hit_count", 0) or 0),
                "detection_cache_hit_count": int(metrics.get("detection_cache_hit_count", 0) or 0),
                "capture_count": int(metrics.get("capture_count", 0) or 0),
                "seconds": round(time.monotonic() - started_at, 3),
                "last_ocr": metrics.get("last_ocr"),
            },
        )

    @staticmethod
    def _balance_from_points(points: Any) -> str:
        try:
            value = Decimal(str(points).strip()) / Decimal("10")
        except (InvalidOperation, ValueError):
            return ""
        return str(int(value))

    def _run_idle_maintenance(self, reporter: Callable[..., None]) -> None:
        keep_awake_interval = float(self.config.get("keep_awake_interval_seconds", 0) or 0)
        if not self.idle_points_checked:
            self.idle_points_checked = True
            reporter("points_check_due", "首次防熄屏读取余额")
            try:
                points = steps.read_my_points(self.robot, self.config, reporter=reporter)
            except Exception as points_exc:
                self.emit_runtime_log("warning", "huanyou_balance_failed", {"error": str(points_exc)})
            else:
                if points:
                    balance = self._balance_from_points(points)
                    self.emit_runtime_log("info", "huanyou_balance", {"balance": balance, "points": points})

        if keep_awake_interval > 0 and time.monotonic() - self.last_activity_at >= keep_awake_interval:
            reporter("keep_awake_due", "执行防熄屏跳转")
            try:
                did_keep_awake = steps.keep_screen_awake(self.robot, self.config, reporter=reporter)
            except Exception as keep_awake_exc:
                self.emit_runtime_log("warning", "keep_awake_failed", {"error": str(keep_awake_exc)})
                return
            if did_keep_awake:
                self._mark_activity()

    def _ensure_list_before_idle_maintenance(self, reporter: Callable[..., None]) -> bool:
        image = steps.ensure_list_screen(self.robot, self.config, reporter=reporter)
        if image is not None:
            return True
        reporter("idle_list_recover_failed", "未确认聊天列表，已尝试返回，本轮不进入防熄屏")
        return False

    def _prepare_chat_window(
        self,
        reporter: Callable[..., None],
        image: Any | None = None,
        expected_name: str | None = None,
    ) -> tuple[Any, bool, bool]:
        chat_image = image if image is not None else steps.confirm_chat_window(self.robot, self.config, reporter=reporter)
        chat_image, quick_greeting_count, stopped_without_next = steps.consume_new_greetings(
            self.robot,
            self.config,
            image=chat_image,
            expected_name=expected_name,
            reporter=reporter,
        )
        if quick_greeting_count:
            self.emit_runtime_log("info", "huanyou_new_greetings", {"count": quick_greeting_count})
            self._mark_activity()
        if stopped_without_next:
            return chat_image, True, bool(quick_greeting_count)

        time.sleep(float(self.config.get("chat_window_scroll_settle_seconds", 0.6) or 0.6))
        swipe_start, swipe_end = steps.scroll_chat_window(self.robot, self.config, reporter=reporter)
        self.emit_runtime_log(
            "info",
            "huanyou_chat_window_swipe",
            {
                "start": {"x": swipe_start[0], "y": swipe_start[1]},
                "end": {"x": swipe_end[0], "y": swipe_end[1]},
            },
        )
        self._mark_activity()
        chat_image = steps.confirm_chat_window(self.robot, self.config, reporter=reporter)
        return chat_image, False, bool(quick_greeting_count)

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

    def _reply_to_message(
        self,
        reporter: Callable[..., None],
        message: dict[str, str],
    ) -> ReplyAction:
        cleaned_content = self._clean_content(message.get("content", ""))
        reporter(
            "dialogue_ocr_last_message",
            "记录对方最后一句",
            log_label=(
                f"对方最后一句：[{message.get('contact', '')}] {message.get('content', '')}"
                if message.get("contact", "") or message.get("content", "")
                else "对方最后一句"
            ),
        )
        self.emit_runtime_log(
            "info",
            "dialogue_ocr_last_message",
            {
                "contact": message.get("contact", ""),
                "content": message.get("content", ""),
                "source": "dialogue",
            },
        )
        self.emit_runtime_log(
            "info",
            "huanyou_message_input",
            {
                "contact": message.get("contact", ""),
                "content": message.get("content", ""),
                "source": "dialogue",
            },
        )

        if cleaned_content == "你好":
            reporter("greeting_reply_pick", "从 pingtai.txt 选择招呼语")
            greeting_reply = self._pick_greeting_reply()
            if greeting_reply:
                action = ReplyAction(text=greeting_reply)
            else:
                raw_action = self.reply_provider(message)
                action = raw_action if isinstance(raw_action, ReplyAction) else ReplyAction(text=str(raw_action))
        else:
            reporter("request_ai", "请求 AI 回复")
            raw_action = self.reply_provider(message)
            action = raw_action if isinstance(raw_action, ReplyAction) else ReplyAction(text=str(raw_action))

        self.emit_runtime_log(
            "info",
            "huanyou_reply_text",
            {
                "contact": message.get("contact", ""),
                "reply_text": action.text,
                "skip": action.skip,
                "reason": action.reason,
            },
        )
        return action

    def _send_action(
        self,
        reporter: Callable[..., None],
        action: ReplyAction,
        message: dict[str, str],
    ) -> bool:
        if action.skip:
            reporter("skip_reply", "跳过本轮回复")
            steps.go_back(self.robot, self.config, reporter=reporter)
            self._mark_activity()
            self.mark_success()
            self._input_box_opened = False
            return False
        if not action.text:
            raise RuntimeError("reply provider returned empty text")
        steps.input_reply(self.robot, action.text, self.config, reporter=reporter)
        self._mark_activity()
        self._input_box_opened = True
        send_clicks = steps.send_reply(self.robot, self.config, reporter=reporter)
        self.emit_runtime_log("info", "huanyou_send_clicks", {"clicks": send_clicks})
        contact = message.get("contact", "")
        send_count = steps.record_contact_send(self.config, contact)
        if send_count:
            self.emit_runtime_log(
                "info",
                "huanyou_contact_send_count",
                {"contact": contact, "count": send_count},
            )
        self._mark_recovered_progress()
        self._mark_activity()
        if self.sent_callback:
            reporter("mark_sent", "记录发送结果")
            self.sent_callback(action, message)
        return True

    def _advance_to_next_chat(
        self,
        reporter: Callable[..., None],
        *,
        current_name: str,
    ) -> tuple[Any, str, int, tuple[float, float] | None]:
        chat_image, next_name, next_clicks, next_name_point = steps.click_next_message_until_changed(
            self.robot,
            self.config,
            expected_name=current_name,
            reporter=reporter,
        )
        self.emit_runtime_log(
            "info",
            "huanyou_next_chat_verified",
            {
                "previous_name": current_name,
                "current_name": next_name,
                "clicks": next_clicks,
            },
        )
        return chat_image, next_name, next_clicks, next_name_point

    def _try_recover_to_list_after_step_unavailable(
        self,
        exc: steps.StepUnavailable,
        step_detail: str,
    ) -> bool:
        try:
            image = steps.ensure_list_screen(self.robot, self.config, reporter=self._reporter)
        except Exception as recover_exc:
            self.emit_runtime_log(
                "warning",
                "欢友兜底恢复失败",
                {
                    "detail": step_detail,
                    "error": str(exc),
                    "recover_error": str(recover_exc),
                },
            )
            return False
        if image is None:
            self.emit_runtime_log(
                "warning",
                "欢友兜底恢复失败",
                {
                    "detail": step_detail,
                    "error": str(exc),
                    "recover_error": "未能恢复到聊天列表",
                },
            )
            return False
        self.emit_runtime_log(
            "warning",
            "欢友步骤不可用，已恢复到聊天列表",
            {
                "detail": step_detail,
                "error": str(exc),
            },
        )
        self._mark_activity()
        self._mark_recovered_progress()
        self.set_runtime_step("step_recovered_to_list", "异常已恢复到聊天列表")
        return True

    def _handle_step_unavailable(self, exc: steps.StepUnavailable) -> None:
        step_detail = self._last_step_detail or self.status.current_step_label or self.status.current_step
        if self._try_recover_to_list_after_step_unavailable(exc, step_detail):
            return

        self.status.error_count += 1
        self.status.last_error = str(exc)
        self._save_error_capture()
        self.recover_error_count = 0
        message = f"欢友步骤不可用：{step_detail} -> {exc}" if step_detail else f"欢友步骤不可用：{exc}"
        will_circuit_open = self.status.error_count >= self.circuit_breaker_threshold
        self.emit_runtime_log(
            "warning",
            message,
            {
                "step": self.status.current_step,
                "label": self.status.current_step_label,
                "detail": step_detail,
                "error": str(exc),
                "error_count": self.status.error_count,
                "circuit_breaker_threshold": self.circuit_breaker_threshold,
                "will_circuit_open": will_circuit_open,
            },
        )
        if will_circuit_open:
            self.status.is_circuit_open = True

    def run_once(self) -> None:
        if self.status.is_circuit_open:
            return
        round_started_at = time.monotonic()
        self._reset_round_metrics()
        try:
            reporter = self._reporter
            if not self._ensure_startup_back_button(reporter):
                self._emit_round_metrics(round_started_at)
                return
            entry_image = self.capture_entry_image()
            if self.idle_waiting:
                target = self.call_step_with_optional_image(
                    steps.rescan_unread_target,
                    self.robot,
                    self.config,
                    reporter=reporter,
                    image=entry_image,
                )
            else:
                reporter("popup_check", "检查并关闭弹窗")
                target = self.call_step_with_optional_image(
                    steps.find_unread_target,
                    self.robot,
                    self.config,
                    reporter=reporter,
                    image=entry_image,
                )
            if not target:
                self.idle_waiting = True
                reporter("empty_list_target", "暂无消息，进入防熄屏等待")
                if not self._ensure_list_before_idle_maintenance(reporter):
                    self._emit_round_metrics(round_started_at)
                    self.mark_success()
                    return
                self._run_idle_maintenance(reporter)
                self._emit_round_metrics(round_started_at)
                self.mark_success()
                return

            self.idle_waiting = False
            self.idle_points_checked = False
            self.emit_runtime_log(
                "info",
                "huanyou_target_selected",
                {
                    "name": target.get("name", ""),
                    "favorability": target.get("favorability"),
                    "unread_count": target.get("unread_count"),
                    "content": target.get("content", ""),
                },
            )

            point_text = steps.format_point(float(target["x"]), float(target["y"]))
            current_name = str(target.get("name", "") or "").strip()
            reporter(
                "enter_chat",
                "点击目标进入聊天",
                log_label=f"点击目标进入聊天：{current_name} {point_text}" if current_name else f"点击目标进入聊天：{point_text}",
            )
            steps.enter_chat(self.robot, float(target["x"]), float(target["y"]))
            self._mark_activity()

            current_target = dict(target)
            chat_image, stopped_without_next, advanced_by_greetings = self._prepare_chat_window(reporter, expected_name=current_name)
            if stopped_without_next:
                steps.go_back(self.robot, self.config, reporter=reporter, force_click=True)
                self._emit_round_metrics(round_started_at)
                self.record_reply_round(time.monotonic() - round_started_at)
                self.mark_success()
                return
            if advanced_by_greetings:
                current_name = steps.read_chat_contact_name(self.robot, self.config, image=chat_image, reporter=reporter) or current_name
                current_target["name"] = current_name

            while True:
                message = steps.read_message(self.robot, self.config, current_target, image=chat_image, reporter=reporter)
                action = self._reply_to_message(reporter, message)
                if action.skip:
                    reporter("skip_reply", "跳过本轮回复")
                    steps.go_back(self.robot, self.config, reporter=reporter, force_click=True)
                    self._mark_activity()
                    self._emit_round_metrics(round_started_at)
                    self.mark_success()
                    return
                input_box_opened = self._send_action(reporter, action, message)

                while True:
                    try:
                        chat_image, next_name, _next_clicks, next_name_point = self._advance_to_next_chat(
                            reporter,
                            current_name=current_name,
                        )
                    except steps.StepUnavailable as exc:
                        if "下一条按钮" not in str(exc):
                            raise
                        self.emit_runtime_log(
                            "info",
                            "huanyou_next_chat_missing",
                            {"previous_name": current_name, "reason": str(exc)},
                        )
                        steps.go_back(self.robot, self.config, reporter=reporter, click_count=2, force_click=True)
                        self._mark_activity()
                        reporter("wait_next_round", "等待下一轮扫描")
                        self._emit_round_metrics(round_started_at)
                        self.record_reply_round(time.monotonic() - round_started_at)
                        self.mark_success()
                        return

                    prepare_image = chat_image
                    normalized_next = re.sub(r"\s+", "", str(next_name or "")).strip()
                    next_name_unconfirmed = not normalized_next
                    advanced_by_greetings = False
                    stopped_without_next = False

                    if next_name_unconfirmed:
                        # The fast path only proves a new-greeting button was seen on the next screenshot.
                        # Handle that screenshot before retracting the old input box; otherwise the button
                        # position can become stale and the same contact may be processed again.
                        chat_image, quick_greeting_count, stopped_without_next = steps.consume_new_greetings(
                            self.robot,
                            self.config,
                            image=chat_image,
                            expected_name=None,
                            reporter=reporter,
                        )
                        if quick_greeting_count:
                            self.emit_runtime_log("info", "huanyou_new_greetings", {"count": quick_greeting_count})
                            self._mark_activity()
                            prepare_image = chat_image
                            advanced_by_greetings = True
                            input_box_opened = False
                        if stopped_without_next:
                            steps.go_back(self.robot, self.config, reporter=reporter)
                            self._mark_activity()
                            reporter("wait_next_round", "等待下一轮扫描")
                            self._emit_round_metrics(round_started_at)
                            self.record_reply_round(time.monotonic() - round_started_at)
                            self.mark_success()
                            return

                    if input_box_opened:
                        retract_x, retract_y = steps.retract_input_box(
                            self.robot,
                            self.config,
                            reporter=reporter,
                            click_point=next_name_point,
                        )
                        self.emit_runtime_log(
                            "info",
                            "huanyou_input_box_retracted",
                            {"x": retract_x, "y": retract_y},
                        )
                        self._mark_activity()
                        prepare_image = None
                    self._input_box_opened = False

                    if next_name_unconfirmed and not advanced_by_greetings:
                        verified_name = steps.read_chat_contact_name(
                            self.robot,
                            self.config,
                            image=prepare_image,
                            reporter=reporter,
                        )
                        normalized_verified = re.sub(r"\s+", "", str(verified_name or "")).strip()
                        normalized_current = re.sub(r"\s+", "", str(current_name or "")).strip()
                        if normalized_current and normalized_verified == normalized_current:
                            reporter(
                                "next_chat_unconfirmed_same_contact",
                                "下一条未确认切人，仍是同一联系人，继续点击下一条",
                                log_label=f"下一条未确认切人，仍是同一联系人：{current_name}",
                            )
                            input_box_opened = False
                            continue
                        if normalized_verified:
                            next_name = verified_name
                            normalized_next = normalized_verified
                            self.emit_runtime_log(
                                "info",
                                "huanyou_next_chat_verified_after_retry",
                                {"previous_name": current_name, "current_name": verified_name},
                            )

                    if normalized_next:
                        self._mark_recovered_progress()
                        current_name = next_name
                    current_target = dict(current_target)
                    current_target["name"] = current_name
                    chat_image, stopped_without_next, prepared_by_greetings = self._prepare_chat_window(
                        reporter,
                        image=prepare_image,
                        expected_name=None,
                    )
                    advanced_by_greetings = advanced_by_greetings or prepared_by_greetings
                    if stopped_without_next:
                        steps.go_back(self.robot, self.config, reporter=reporter)
                        self._mark_activity()
                        reporter("wait_next_round", "等待下一轮扫描")
                        self._emit_round_metrics(round_started_at)
                        self.record_reply_round(time.monotonic() - round_started_at)
                        self.mark_success()
                        return
                    if advanced_by_greetings:
                        current_name = steps.read_chat_contact_name(self.robot, self.config, image=chat_image, reporter=reporter) or current_name
                        current_target["name"] = current_name
                    break
        except steps.NoPhoneScreenError as exc:
            self.set_runtime_step("no_phone_waiting", "无手机异常，停止进程")
            self.status.last_error = str(exc)
            self.status.is_circuit_open = True
            self.recover_error_count = 0
            self.emit_runtime_log("warning", "no_phone_detected", {"error": str(exc)})
        except steps.BlackScreenError as exc:
            self.set_runtime_step("black_screen_waiting", "黑屏异常，等待亮屏")
            self.status.last_error = str(exc)
            self.status.is_circuit_open = False
            self.recover_error_count = 0
            self.emit_runtime_log("warning", "black_screen_detected", {"error": str(exc)})
        except steps.StepUnavailable as exc:
            self._handle_step_unavailable(exc)
        except Exception as exc:
            self.on_error(exc)
