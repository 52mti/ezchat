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


# flow 引擎参数：单轮迁移上限防呆；优雅放弃连续计数（3 次留痕/5 次升级）。
_FLOW_MAX_TICKS = 24
_FLOW_ABANDON_WARN_AT = 3
_FLOW_ABANDON_ESCALATE_AT = 5


class XinyuApp(BaseApp):
    # 能力声明：DeviceWorker 据此注入对应 provider，不再按 App 名字分支。
    supports_dialogue_continuity = True
    supports_greeting = True
    name = "心遇"
    description = "Xinyu unread message scan and TTS reply flow"
    REMOTE_FILTER_RULE_SYNC_INTERVAL_SECONDS = 3600.0

    def __init__(
        self,
        robot: Any,
        config: dict[str, Any] | None = None,
        reply_provider: Callable[[dict[str, str]], str | ReplyAction] | None = None,
        sent_callback: Callable[[ReplyAction, dict[str, str]], None] | None = None,
        continuity_required_provider: Callable[[dict[str, Any]], bool] | None = None,
        greeting_provider: Callable[[str], str] | None = None,
        greeting_sent_notifier: Callable[[str], None] | None = None,
    ) -> None:
        merged_config = _deep_merge(DEFAULT_CONFIG, config or {})
        steps._disable_dialogue_region_filter(merged_config)
        # Keep the worker-owned object so App settings hot refreshes (including
        # an expanded send-button region) are visible to this running app.
        if isinstance(config, dict):
            config.clear()
            config.update(merged_config)
            active_config = config
        else:
            active_config = merged_config
        super().__init__(robot, active_config)
        self.reply_provider = reply_provider or (lambda message: "")
        self.sent_callback = sent_callback
        self.continuity_required_provider = continuity_required_provider
        # 人设 AI 新链路（docs/13 4.4）：招呼改写与平台新招呼占位记录
        self.greeting_provider = greeting_provider
        self.greeting_sent_notifier = greeting_sent_notifier
        self.last_activity_at = time.monotonic()
        self.idle_points_checked = False
        self.idle_online_photo_attempted = False
        self._last_selected_signature: tuple[str, str, int] | None = None
        self._last_remote_filter_rule_sync_at = 0.0
        self._abandon_streak = 0

    def _sync_remote_filter_rules_if_due(self) -> None:
        now = time.monotonic()
        if now - self._last_remote_filter_rule_sync_at < self.REMOTE_FILTER_RULE_SYNC_INTERVAL_SECONDS:
            return
        self._last_remote_filter_rule_sync_at = now
        success, added = steps.sync_remote_filter_rules(self.config)
        if success:
            self.emit_runtime_log("info", "心遇云端过滤词同步完成", {"新增": added, "间隔秒": 3600})
        else:
            self.emit_runtime_log("warning", "心遇云端过滤词同步失败，继续使用本地规则", {})

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
            "xinyu_round_metrics",
            {
                "ocr_count": int(metrics.get("ocr_count", 0) or 0),
                "ocr_cache_hit_count": int(metrics.get("ocr_cache_hit_count", 0) or 0),
                "detection_cache_hit_count": int(metrics.get("detection_cache_hit_count", 0) or 0),
                "capture_count": int(metrics.get("capture_count", 0) or 0),
                "dialogue_compact_fallback_count": int(metrics.get("dialogue_compact_fallback_count", 0) or 0),
                "seconds": round(time.monotonic() - started_at, 3),
                "last_ocr": metrics.get("last_ocr"),
            },
        )

    def _reply_segment_pipeline(
        self,
        message: dict[str, str],
        target_name: str,
        greeting_reply: str,
        round_started_at: float,
        reporter: Callable[..., None],
    ) -> None:
        """原线性流水线回复段（保留为双轨回退路径，行为与历史版本一致）。"""
        input_focused = False
        if greeting_reply:
            reporter("request_ai", "请求 AI 回复")
            if self.greeting_provider:
                # 人设 AI 招呼链路：底稿挑选/AI 改写/防重复/回退全部在 provider 内完成（docs/13 4.4）
                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(self.greeting_provider, str(message.get("contact") or target_name or ""))
                try:
                    steps.focus_input_box(self.robot, self.config, reporter=reporter)
                    self._mark_activity()
                    input_focused = True
                    greeting_text = future.result()
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)
                action = ReplyAction(text=greeting_text or greeting_reply, reason="greeting_reply")
            else:
                steps.focus_input_box(self.robot, self.config, reporter=reporter)
                self._mark_activity()
                input_focused = True
                action = ReplyAction(text=greeting_reply, reason="greeting_reply")
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
            if action.reason == "pending_send":
                self.emit_runtime_log(
                    "warning",
                    "心遇复用待发送回复",
                    {
                        "contact": message.get("contact", ""),
                        "content": message.get("content", ""),
                    },
                )

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
            self._emit_round_metrics(round_started_at)
            self.mark_success()
            return
        if not action.text:
            raise RuntimeError("回复内容为空，未生成可发送文本")

        if not input_focused:
            steps.focus_input_box(self.robot, self.config, reporter=reporter)
            self._mark_activity()
        steps.play_voice_reply(self.robot, action.text, self.config, reporter=reporter)
        self._mark_activity()
        send_clicks = steps.send_reply(self.robot, self.config, reporter=reporter)
        self.emit_runtime_log("info", "xinyu_send_clicks", {"clicks": send_clicks})
        self._mark_recovered_progress()
        if send_clicks > 1:
            self.emit_runtime_log(
                "warning",
                "心遇发送按钮被连续点击",
                {
                    "contact": message.get("contact", ""),
                    "clicks": send_clicks,
                },
            )
        self._mark_activity()
        if self.sent_callback:
            reporter("mark_sent", "记录发送结果")
            self.sent_callback(action, message)
        try:
            return_clicks = steps.go_back(
                self.robot,
                self.config,
                double_click_on_first_attempt=True,
                reporter=reporter,
            )
        except Exception:
            self.emit_runtime_log(
                "warning",
                "心遇发送后未能返回列表",
                {
                    "contact": message.get("contact", ""),
                    "content": message.get("content", ""),
                },
            )
            raise
        self.emit_runtime_log(
            "info",
            "xinyu_send_flow_summary",
            {
                "contact": message.get("contact", ""),
                "input_focused": input_focused,
                "input_box_source": self.config.get("_runtime_cache", {}).get("metrics", {}).get("input_box_source", ""),
                "input_box_template": self.config.get("_runtime_cache", {}).get("metrics", {}).get("input_box_template", ""),
                "microphone_template": self.config.get("_runtime_cache", {}).get("metrics", {}).get("microphone_template", ""),
                "tts_success": self.config.get("_runtime_cache", {}).get("metrics", {}).get("tts_success", False),
                "tts_retry_count": self.config.get("_runtime_cache", {}).get("metrics", {}).get("tts_retry_count", 0),
                "send_clicks": send_clicks,
                "send_clear_extra_clicks": self.config.get("_runtime_cache", {}).get("metrics", {}).get("send_clear_extra_clicks", 0),
                "send_clear_reason": self.config.get("_runtime_cache", {}).get("metrics", {}).get("send_clear_reason", ""),
                "return_clicks": return_clicks,
                "return_reason": self.config.get("_runtime_cache", {}).get("metrics", {}).get("return_reason", ""),
                "after_send_double_back_allowed": True,
            },
        )
        self._mark_activity()
        reporter("wait_next_round", "等待下一轮扫描")
        self.record_reply_round(time.monotonic() - round_started_at)
        self._emit_round_metrics(round_started_at)
        self.mark_success()

    # ------------------------------------------------------------------
    # 流程状态机（v1 接入：回复段 PREPARING→VOICE→SENDING→RETURNING）
    # ------------------------------------------------------------------

    def _flow_engine_enabled(self) -> bool:
        return bool(self.config.get("enable_flow_engine", False))

    def _reply_segment_flow(
        self,
        message: dict[str, str],
        target: dict[str, Any],
        target_name: str,
        greeting_reply: str,
        round_started_at: float,
        reporter: Callable[..., None],
    ) -> None:
        from apps.xinyu.flow import FlowEngine, FlowHardReset, FlowState

        ctx: dict[str, Any] = {
            "message": message,
            "target": target,
            "target_name": target_name,
            "greeting_reply": greeting_reply,
            "round_started_at": round_started_at,
            "reporter": reporter,
            "reenter_attempts": 0,
            "next": None,
        }

        def executor(action: str, *, state: str, trigger: str) -> None:
            ctx["next"] = self._flow_action(action, state, trigger, ctx)

        engine = FlowEngine(state=FlowState.PREPARING_REPLY, executor=executor)
        # 入口动作：ENTERING_CHAT+screen:chat 的迁移动作，由本轮外层流程
        # （find_unread_target/enter_chat/confirm_chat_window）代为完成。
        executor("submit_reply_and_focus_input", state="entering_chat", trigger="screen:chat")
        for _ in range(_FLOW_MAX_TICKS):
            pending = ctx["next"]
            if pending is None:
                return  # 轮次收尾（成功/优雅放弃/中止），外层轮询自然进入下轮
            event, perception = pending
            try:
                engine.tick(event, perception)
            except FlowHardReset as exc:
                raise RuntimeError(f"flow 恢复梯度耗尽：{exc}") from exc
        raise RuntimeError("flow 单轮迁移次数超限")

    def _flow_action(
        self,
        action: str,
        state: str,
        trigger: str,
        ctx: dict[str, Any],
    ) -> tuple[str, Any] | None:
        handler = self._FLOW_ACTION_HANDLERS.get(action)
        if handler is None:
            raise RuntimeError(f"flow 动作未接入：{action}")
        return handler(self, ctx)

    def _flow_submit_reply_and_focus_input(self, ctx: dict[str, Any]) -> tuple[str, Any] | None:
        from apps.xinyu.flow import Perception

        message = ctx["message"]
        reporter = ctx["reporter"]
        reporter("request_ai", "请求 AI 回复")
        executor = ThreadPoolExecutor(max_workers=1)
        if ctx["greeting_reply"]:
            future = executor.submit(
                self.greeting_provider,
                str(message.get("contact") or ctx["target_name"] or ""),
            )
        else:
            future = executor.submit(self.reply_provider, message)
        try:
            try:
                steps.focus_input_box(self.robot, self.config, reporter=reporter)
                self._mark_activity()
            except steps.StepUnavailable:
                # 输入框定位失败：先感知再决定是优雅放弃还是真异常。
                screen = self._flow_perceive_fresh()
                if screen == "list":
                    return "screen:list", Perception.of("list")
                raise
            try:
                raw = future.result()
            except Exception as exc:  # noqa: BLE001 - AI 失败转 ai:failed 事件计一次错误
                from services.cancellation import OperationCancelled

                if isinstance(exc, OperationCancelled):
                    raise
                ctx["ai_error"] = str(exc)
                return "ai:failed", Perception.of("chat")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        if ctx["greeting_reply"]:
            action = ReplyAction(
                text=str(raw or "") or ctx["greeting_reply"],
                reason="greeting_reply",
            )
        else:
            action = raw if isinstance(raw, ReplyAction) else ReplyAction(text=str(raw))
            if action.reason == "pending_send":
                self.emit_runtime_log(
                    "warning",
                    "心遇复用待发送回复",
                    {"contact": message.get("contact", ""), "content": message.get("content", "")},
                )
        ctx["action"] = action
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
            self._flow_best_effort_go_back(ctx)
            self._emit_round_metrics(ctx["round_started_at"])
            self.mark_success()
            return None
        if not action.text:
            return "ai:failed", Perception.of("chat")
        return "ai:ready", Perception.of("chat")

    def _flow_prepare_tts(self, ctx: dict[str, Any]) -> tuple[str, Any] | None:
        """TTS 合成后进入麦克风搜索（带感知门，替代 20 秒干等死局）。"""
        from apps.xinyu.flow import Perception

        action = ctx["action"]
        reporter = ctx["reporter"]
        cfg = self.config
        retry_count = max(0, int(cfg.get("tts_retry_attempts", 1) or 0))
        last_error: Exception | None = None
        prepared_chunks = None
        for attempt in range(retry_count + 1):
            try:
                reporter("tts_prepare", "合成 TTS 音频", log_label="TTS 音频合成中，完成后再点击麦克风")
                prepared_chunks = steps.prepare_tts_input(self.robot, action.text)
                reporter("tts_prepared", "TTS 音频已就绪", log_label="TTS 音频已就绪，准备点击麦克风播放")
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001 - TTS 失败重试语义与流水线一致
                last_error = exc
                if attempt >= retry_count:
                    break
                reporter("tts_retry", "TTS 合成失败，准备重试", log_label=f"TTS 合成失败，准备重试：{exc}")
        if prepared_chunks is None:
            return "mic:exhausted", Perception.of("chat")
        ctx["prepared_chunks"] = prepared_chunks
        return self._flow_mic_search(ctx)

    def _flow_mic_search(self, ctx: dict[str, Any]) -> tuple[str, Any] | None:
        """麦克风搜索 + 播放：找不到先扫弹窗，再感知路由（重进/放弃/耗尽）。"""
        from apps.xinyu.flow import Perception

        reporter = ctx["reporter"]
        cfg = self.config
        deadline = steps.step_deadline(cfg, steps._step_timeout(cfg))
        miss_limit = max(1, int(cfg.get("flow_mic_miss_limit", 6) or 6))
        misses = 0
        popup_sweeps = 0
        microphone = None
        while True:
            image = steps.capture_after_popups(
                self.robot, cfg, label="microphone", reporter=reporter, skip_popup_detection=True
            )
            microphone_match = steps._template_match(image, cfg, steps.MICROPHONE_BUTTON, group="buttons")
            if microphone_match is not None:
                microphone = (
                    microphone_match["x"] + microphone_match["w"] / 2,
                    microphone_match["y"] + microphone_match["h"] / 2,
                )
                break
            misses += 1
            if misses >= miss_limit:
                # 感知门：弹窗遮挡优先清扫（吸收 skip_popup_detection=True 盲区）。
                if popup_sweeps < 2:
                    popup_sweeps += 1
                    steps.capture_after_popups(self.robot, cfg, label="mic_popup_sweep", reporter=reporter)
                    misses = 0
                    continue
                screen = steps.classify_screen(image, cfg)
                if screen == "list":
                    self.emit_runtime_log(
                        "warning",
                        "心遇麦克风缺失且已回列表，准备重进会话",
                        {"contact": ctx["message"].get("contact", "")},
                    )
                    return "screen:list", Perception.of("list")
                if screen == "no_phone":
                    raise steps.NoPhoneScreenError("flow 感知：检测到无手机界面，请先把手机放回支架")
                if screen == "black":
                    raise steps.BlackScreenError("flow 感知：检测到黑屏界面")
                return "mic:exhausted", Perception.of(screen)
            if deadline.expired():
                return "mic:exhausted", Perception.of("chat")
            time.sleep(0.4)

        reporter("microphone_click", "点击麦克风", log_label=f"点击麦克风：{steps.format_point(*microphone)}")
        steps.click_image_point(self.robot, *microphone)
        steps.play_prepared_tts_input(self.robot, ctx["prepared_chunks"])
        time.sleep(steps.TTS_POST_DELAY_SECONDS)
        self._mark_activity()
        return "tts:done", Perception.of("chat")

    def _flow_reenter_conversation(self, ctx: dict[str, Any]) -> tuple[str, Any] | None:
        """掉回列表后重进会话：优先续做本轮，重进两次失败才放弃。"""
        from apps.xinyu.flow import Perception

        reporter = ctx["reporter"]
        target = ctx.get("target") or {}
        limit = max(1, int(self.config.get("flow_reenter_limit", 2) or 2))
        while ctx["reenter_attempts"] < limit:
            ctx["reenter_attempts"] += 1
            try:
                reporter("reenter_chat", "重新进入会话", log_label=f"重新进入会话（第 {ctx['reenter_attempts']} 次）")
                steps.enter_chat(self.robot, float(target["x"]), float(target["y"]))
                self._mark_activity()
                steps.confirm_chat_window(self.robot, self.config, reporter=reporter)
                return "screen:chat", Perception.of("chat")
            except Exception as exc:  # noqa: BLE001 - 重进失败按次数兜底
                self.emit_runtime_log(
                    "warning",
                    "心遇重进会话失败",
                    {"attempt": ctx["reenter_attempts"], "error": str(exc)},
                )
        return "reenter:exhausted", Perception.of("list")

    def _flow_retry_microphone(self, ctx: dict[str, Any]) -> tuple[str, Any] | None:
        """重进成功后的续跑：重点输入框，直接找麦克风（复用已合成 TTS）。"""
        from apps.xinyu.flow import Perception

        reporter = ctx["reporter"]
        if "prepared_chunks" not in ctx:
            return "mic:exhausted", Perception.of("chat")
        try:
            steps.focus_input_box(self.robot, self.config, reporter=reporter)
            self._mark_activity()
        except steps.StepUnavailable:
            screen = self._flow_perceive_fresh()
            if screen == "list":
                return "screen:list", Perception.of("list")
            raise
        return self._flow_mic_search(ctx)

    def _flow_click_send(self, ctx: dict[str, Any]) -> tuple[str, Any] | None:
        from apps.xinyu.flow import Perception

        reporter = ctx["reporter"]
        try:
            send_clicks = steps.send_reply(self.robot, self.config, reporter=reporter)
        except Exception:
            # 发送按钮始终未出现/未消失：与流水线同语义（异常上抛走 on_error）。
            raise
        self.emit_runtime_log("info", "xinyu_send_clicks", {"clicks": send_clicks})
        self._mark_recovered_progress()
        if send_clicks > 1:
            self.emit_runtime_log(
                "warning",
                "心遇发送按钮被连续点击",
                {"contact": ctx["message"].get("contact", ""), "clicks": send_clicks},
            )
        self._mark_activity()
        return "send:sent", Perception.of("chat")

    def _flow_click_back(self, ctx: dict[str, Any]) -> tuple[str, Any] | None:
        from apps.xinyu.flow import Perception

        try:
            steps.go_back(self.robot, self.config, double_click_on_first_attempt=True, reporter=ctx["reporter"])
        except Exception:
            return "back:exhausted", Perception.of("unknown")
        return "back:at_list", Perception.of("list")

    def _flow_mark_round_success(self, ctx: dict[str, Any]) -> None:
        message = ctx["message"]
        action = ctx["action"]
        self._abandon_streak = 0
        if self.sent_callback:
            ctx["reporter"]("mark_sent", "记录发送结果")
            self.sent_callback(action, message)
        self._emit_send_flow_summary(ctx, return_clicks=0, send_clicks=int(
            self.config.get("_runtime_cache", {}).get("metrics", {}).get("send_button_clicks", 0) or 0
        ))
        ctx["reporter"]("wait_next_round", "等待下一轮扫描")
        self.record_reply_round(time.monotonic() - ctx["round_started_at"])
        self._emit_round_metrics(ctx["round_started_at"])
        self.mark_success()
        return None

    def _flow_abandon_round_keep_pending(self, ctx: dict[str, Any]) -> None:
        """优雅放弃：不进熔断计数，但连续放弃要留痕并在阈值升级。"""
        self._abandon_streak += 1
        self.emit_runtime_log(
            "warning" if self._abandon_streak >= _FLOW_ABANDON_WARN_AT else "info",
            "心遇流程优雅放弃本轮",
            {
                "contact": ctx["message"].get("contact", ""),
                "consecutive_abandons": self._abandon_streak,
                "note": "待发回复保留，下轮复用",
            },
        )
        if self._abandon_streak >= _FLOW_ABANDON_ESCALATE_AT:
            raise RuntimeError(
                f"flow 连续优雅放弃 {self._abandon_streak} 次，升级为错误（疑似持久性故障）"
            )
        self._emit_round_metrics(ctx["round_started_at"])
        return None

    def _flow_abort_round_go_back(self, ctx: dict[str, Any]) -> None:
        # 计一次错误（与流水线 ai:failed/发送耗尽口径一致），但不动机械臂。
        reason = str(ctx.get("ai_error", "") or "本轮中止（语音/发送耗尽）")
        self.status.error_count += 1
        self.status.last_error = f"flow: {reason}"
        self.emit_runtime_log(
            "error",
            "app_error",
            {
                "step": self.status.current_step,
                "label": self.status.current_step_label,
                "error": self.status.last_error,
                "error_count": self.status.error_count,
            },
        )
        if self.status.error_count >= self.circuit_breaker_threshold:
            self.status.is_circuit_open = True
        self._flow_best_effort_go_back(ctx)
        return None

    def _flow_finish_round_from_list(self, ctx: dict[str, Any]) -> None:
        return self._flow_mark_round_success(ctx)

    def _flow_recover_navigate(self, ctx: dict[str, Any]) -> tuple[str, Any] | None:
        from apps.xinyu.flow import Perception

        try:
            steps.go_back(self.robot, self.config, reporter=ctx["reporter"])
            return "screen:list", Perception.of("list")
        except Exception:
            return "recover:exhausted", Perception.of("unknown")

    def _flow_relaunch_app(self, ctx: dict[str, Any]) -> tuple[str, Any] | None:
        # v1 未实现软件层重进心遇：直接走硬复位出口（对接现有 on_error）。
        raise RuntimeError("flow: 中恢复(重进App)暂未实现，升级硬复位")

    def _flow_hard_reset(self, ctx: dict[str, Any]) -> tuple[str, Any] | None:
        raise RuntimeError("flow: 恢复梯度耗尽，需要硬复位")

    def _flow_best_effort_go_back(self, ctx: dict[str, Any]) -> None:
        try:
            steps.go_back(self.robot, self.config, reporter=ctx["reporter"])
            self._mark_activity()
        except Exception as exc:
            self.emit_runtime_log("warning", "flow 中止后返回列表失败", {"error": str(exc)})

    def _flow_perceive_fresh(self) -> str:
        image = steps.capture_after_popups(
            self.robot, self.config, label="flow_perceive", skip_popup_detection=True
        )
        return steps.classify_screen(image, self.config)

    def _emit_send_flow_summary(self, ctx: dict[str, Any], *, return_clicks: int, send_clicks: int) -> None:
        metrics = self.config.get("_runtime_cache", {}).get("metrics", {})
        self.emit_runtime_log(
            "info",
            "xinyu_send_flow_summary",
            {
                "contact": ctx["message"].get("contact", ""),
                "input_focused": True,
                "input_box_source": metrics.get("input_box_source", ""),
                "input_box_template": metrics.get("input_box_template", ""),
                "microphone_template": metrics.get("microphone_template", ""),
                "tts_success": metrics.get("tts_success", False),
                "tts_retry_count": metrics.get("tts_retry_count", 0),
                "send_clicks": send_clicks,
                "send_clear_extra_clicks": metrics.get("send_clear_extra_clicks", 0),
                "send_clear_reason": metrics.get("send_clear_reason", ""),
                "return_clicks": return_clicks,
                "return_reason": metrics.get("return_reason", ""),
                "after_send_double_back_allowed": True,
                "flow_engine": True,
            },
        )

    _FLOW_ACTION_HANDLERS: dict[str, Callable[..., Any]] = {
        "submit_reply_and_focus_input": _flow_submit_reply_and_focus_input,
        "prepare_tts": _flow_prepare_tts,
        "click_microphone_and_play": _flow_prepare_tts,
        "reenter_conversation": _flow_reenter_conversation,
        "retry_microphone": _flow_retry_microphone,
        "click_send": _flow_click_send,
        "click_back": _flow_click_back,
        "mark_round_success": _flow_mark_round_success,
        "finish_round_from_list": _flow_finish_round_from_list,
        "abandon_round_keep_pending": _flow_abandon_round_keep_pending,
        "abort_round_go_back": _flow_abort_round_go_back,
        "recover_navigate": _flow_recover_navigate,
        "relaunch_app": _flow_relaunch_app,
        "hard_reset": _flow_hard_reset,
    }

    def _mark_activity(self) -> None:
        self.last_activity_at = time.monotonic()

    def _mark_recovered_progress(self) -> None:
        self.status.last_error = ""
        self.status.error_count = 0
        self.status.is_circuit_open = False
        self.recover_error_count = 0

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
                self.emit_runtime_log("warning", "防熄屏失败", {"error": str(keep_awake_exc)})
            else:
                if did_keep_awake:
                    self._mark_activity()

        if self.idle_points_checked:
            self.emit_runtime_log("info", "心遇本轮空闲已读取余额，跳过重复读取", {})
            return
        self.idle_points_checked = True
        reporter("points_check_due", "首次防熄屏读取余额")
        try:
            points = steps.read_my_points(self.robot, self.config, reporter=reporter)
        except Exception as points_exc:
            self.emit_runtime_log("warning", "心遇余额读取失败", {"error": str(points_exc)})
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
        round_started_at = time.monotonic()
        self._reset_round_metrics()
        try:
            self._sync_remote_filter_rules_if_due()
            reporter = self.set_runtime_step
            reporter("popup_check", "检查并关闭弹窗")
            # 启动首图也必须走完整弹窗处理，不能只写状态日志。
            entry_image = steps.capture_after_popups(
                self.robot,
                self.config,
                label="app_entry_popup_check",
                reporter=reporter,
            )
            target = self.call_step_with_optional_image(
                steps.find_unread_target,
                self.robot,
                self.config,
                reporter=reporter,
                image=entry_image,
            )
            if not target:
                reporter("empty_list_target", "暂无消息，进入防熄屏等待")
                if not self.idle_online_photo_attempted:
                    self.idle_online_photo_attempted = True
                    try:
                        planned_count = max(0, int(self.config.get("online_photo_send_count", 3) or 0))
                    except (TypeError, ValueError):
                        planned_count = 3
                    reporter("online_photo_due", f"无消息，尝试向 {planned_count} 位在线用户发照片")
                    sent_count = 0
                    for index in range(planned_count):
                        try:
                            sent_photo = steps.send_photo_to_online_target(self.robot, self.config, reporter=reporter)
                        except Exception as online_photo_exc:
                            self.emit_runtime_log("warning", "在线用户发照片失败", {"第几位": index + 1, "error": str(online_photo_exc)})
                            break
                        if not sent_photo:
                            break
                        sent_count += 1
                        self._mark_activity()
                    if sent_count:
                        self.emit_runtime_log("info", "在线用户照片已发送", {"人数": sent_count, "计划人数": planned_count})
                self._run_idle_maintenance(reporter)
                self._emit_round_metrics(round_started_at)
                return

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
            target_signature = (
                str(target.get("name", "") or "").strip(),
                self._clean_content(target.get("content", "")),
                int(target.get("unread_count") or 0),
            )
            previous_name = self._last_selected_signature[0] if self._last_selected_signature else ""
            current_name = target_signature[0]
            if previous_name and current_name and previous_name != current_name:
                self._mark_recovered_progress()
            if self._last_selected_signature == target_signature:
                self.emit_runtime_log(
                    "warning",
                    "心遇重复选中同一会话",
                    {
                        "name": target_signature[0],
                        "content": target_signature[1],
                        "unread_count": target_signature[2],
                    },
                )
            self._last_selected_signature = target_signature
            if self._is_empty_list_target(target):
                reporter("empty_list_target", "暂无消息，本轮不进入聊天")
                self._run_idle_maintenance(reporter)
                self._emit_round_metrics(round_started_at)
                self.mark_success()
                return
            if self.idle_points_checked:
                self.emit_runtime_log("info", "心遇检测到真实消息，重置空闲余额读取标记", {})
            self.idle_points_checked = False
            self.idle_online_photo_attempted = False

            list_content = self._clean_content(target.get("content", ""))
            use_list_content = self._can_reply_from_list(list_content)
            greeting_reply = ""
            if use_list_content and list_content in {"你好", "对方发来礼物", "对方发来搭训礼物", "对方发来搭讪礼物", "他在等待来自你的问候，快聊聊~", "发布了新的动态"}:
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
                # 平台代发的新招呼：文本读不到，写占位轮保证人设知道是自己先开的口（docs/13 4.4）
                if self.greeting_sent_notifier:
                    try:
                        self.greeting_sent_notifier(target_name)
                    except Exception as exc:
                        self.emit_runtime_log("warning", "新招呼占位落库失败", {"name": target_name, "error": str(exc)})
                reporter("wait_next_round", "等待下一轮扫描")
                self.record_reply_round(time.monotonic() - round_started_at)
                self._emit_round_metrics(round_started_at)
                self.mark_success()
                return

            if use_list_content:
                message = steps.build_list_message(target, list_content)
                used_dialogue_content = False
                # 列表通常只显示最后一条短消息；除“你好”快速路径外，
                # 补读聊天详情以合并末尾连续的 sender1 消息。
                if list_content not in {"你好", "对方发来礼物", "对方发来搭训礼物", "对方发来搭讪礼物", "他在等待来自你的问候，快聊聊~", "发布了新的动态", "对方发来搭礼物", "发布了新的动态"}:
                    try:
                        dialogue_message = steps.read_message(
                            self.robot,
                            self.config,
                            target,
                            image=chat_image,
                            reporter=reporter,
                        )
                    except Exception as exc:
                        dialogue_message = None
                        self.emit_runtime_log(
                            "warning",
                            "列表内容对应的聊天详情 OCR 失败，继续使用列表内容",
                            {"name": target_name, "error": str(exc)},
                        )
                    dialogue_content = (
                        self._clean_content(dialogue_message.get("content", ""))
                        if isinstance(dialogue_message, dict)
                        else ""
                    )
                    if dialogue_content:
                        message = dialogue_message
                        used_dialogue_content = True
                        reporter(
                            "dialogue_content_merge",
                            "列表内容补充聊天详情",
                            log_label=f"列表内容补充聊天详情：{dialogue_content}",
                        )
                reporter(
                    "direct_content_reply",
                    "使用聊天详情合并内容作为对话输入" if used_dialogue_content else "使用列表内容作为对话输入",
                    log_label=(
                        f"使用聊天详情合并内容作为对话输入：{message.get('content', '')}"
                        if used_dialogue_content
                        else f"使用列表内容作为对话输入：{message.get('content', '')}"
                    ),
                )
            else:
                try:
                    message = steps.read_message(self.robot, self.config, target, image=chat_image, reporter=reporter)
                except steps.DialogueOpponentMessageMissing:
                    # 聊天界面正常但对方新消息是表情/图片等无可读内容：
                    # 按「你好」招呼处理，从 pingtai.txt 挑招呼语回复。
                    message = steps.build_list_message(target, "你好")
                    greeting_reply = self._pick_greeting_reply()
                    reporter(
                        "greeting_reply_pick",
                        "对方消息为表情/图片，按“你好”回招呼语",
                        log_label=f"对方消息为表情/图片，按“你好”回招呼语：{target_name}",
                    )
                    self.emit_runtime_log(
                        "warning",
                        "对方消息无可读内容，按招呼处理",
                        {"name": target_name, "greeting_reply": greeting_reply},
                    )

            if use_list_content and self.continuity_required_provider:
                try:
                    needs_continuity = self.continuity_required_provider(message)
                except Exception as exc:
                    self.emit_runtime_log("warning", "检查聊天记忆失败，跳过连续性匹配", {"error": str(exc)})
                    needs_continuity = False
                if needs_continuity:
                    reporter("dialogue_continuity_read", "昵称未命中，读取近期消息确认会话")
                    message["continuity"] = steps.read_dialogue_continuity(chat_image, self.config)

            self.emit_runtime_log(
                "info",
                "xinyu_message_input",
                {
                    "contact": message.get("contact", ""),
                    "content": message.get("content", ""),
                    "source": "list" if use_list_content else "dialogue",
                },
            )

            if self._flow_engine_enabled():
                self._reply_segment_flow(message, target, target_name, greeting_reply, round_started_at, reporter)
            else:
                self._reply_segment_pipeline(message, target_name, greeting_reply, round_started_at, reporter)
        except steps.NoPhoneScreenError as exc:
            self.set_runtime_step("no_phone_waiting", "无手机异常，等待用户放回手机")
            self.status.last_error = str(exc)
            self.status.is_circuit_open = False
            self.recover_error_count = 0
            self.emit_runtime_log("warning", "检测到无手机", {"error": str(exc)})
        except steps.BlackScreenError as exc:
            self.set_runtime_step("black_screen_waiting", "黑屏异常，等待用户亮屏")
            self.status.last_error = str(exc)
            self.status.is_circuit_open = False
            self.recover_error_count = 0
            self.emit_runtime_log("warning", "检测到黑屏", {"error": str(exc)})
        except Exception as exc:
            self.on_error(exc)
