"""心遇流程状态机（骨架期，未接入 run_once）。

设计口径（docs 讨论定稿）：
- FSM 的状态 = 流程阶段（程序内部记忆）；感知到的屏幕 = 事件（外部输入）。
  输入框焦点/TTS 播放中这类过程状态不可感知，属于阶段本身，不进事件。
- 迁移表是纯数据：故障与正常流程同一套机制表达。麦克风波次撞上
  "screen:list" 不再是异常+复位，而是一条 counts_error=False 的优雅
  放弃迁移（待发回复留在 outbox，下轮复用，不进熔断计数）。
- 恢复梯度三级：RECOVERING_SOFT（软件导航回列表）→ RECOVERING_APP
  （从桌面重进心遇）→ 硬复位（沿用现有 on_error/reset_device/熔断）。

接入方式（第二步，另行提交）：run_once 换成 flow.tick() 循环，动作
handler 调用现有 steps 函数（行为等价），flow_engine 配置开关双轨。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class FlowState(str, Enum):
    IDLE_WAIT = "idle_wait"              # 无消息防熄屏等待
    SCANNING = "scanning"                # 列表页找未读目标
    ENTERING_CHAT = "entering_chat"      # 点进会话、确认到聊天页
    PREPARING_REPLY = "preparing_reply"  # AI 生成(异步) + 聚焦输入框
    VOICE_INPUT = "voice_input"          # 找麦克风 + 点麦克风 + TTS 播放
    SENDING = "sending"                  # 点发送 + 确认发出
    RETURNING = "returning"              # 返回列表 + 确认
    RECOVERING_SOFT = "recovering_soft"  # 软恢复：软件导航回列表
    RECOVERING_APP = "recovering_app"    # 中恢复：从桌面重进心遇


class ScreenEvent(str, Enum):
    """感知层输出（current_screen() + 弹窗标记的归一化）。"""

    LIST = "list"
    CHAT = "chat"
    MY = "my"
    CALL = "call"
    NO_PHONE = "no_phone"
    BLACK = "black"
    UNKNOWN = "unknown"


# 无手机/黑屏维持现有异常语义（等待用户干预、不进熔断，app.py:512-523），
# 不占用迁移行；迁移表只处理可自主恢复的屏幕。
EXTERNAL_EXCEPTION_SCREENS = frozenset({ScreenEvent.NO_PHONE, ScreenEvent.BLACK})


@dataclass(frozen=True)
class Perception:
    """一次感知的结果。popup=True 表示屏幕上有可关闭的弹窗。"""

    screen: ScreenEvent
    popup: bool = False
    detail: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    @classmethod
    def of(cls, screen: ScreenEvent | str, *, popup: bool = False) -> "Perception":
        normalized = screen if isinstance(screen, ScreenEvent) else ScreenEvent(screen)
        return cls(screen=normalized, popup=popup)


@dataclass(frozen=True)
class Transition:
    """一条迁移：源状态 + 触发事件 → 目标状态 + 动作名。

    counts_error=False 表示优雅迁移（放弃/重试），不进熔断计数；
    hard_reset=True 表示耗尽梯度后的硬复位出口（对接现有 on_error）。
    """

    source: FlowState
    trigger: str
    target: FlowState
    action: str
    counts_error: bool = False
    hard_reset: bool = False
    note: str = ""


def _t(
    source: FlowState,
    trigger: str,
    target: FlowState,
    action: str,
    *,
    counts_error: bool = False,
    hard_reset: bool = False,
    note: str = "",
) -> tuple[tuple[FlowState, str], Transition]:
    return (source, trigger), Transition(source, trigger, target, action, counts_error, hard_reset, note)


# ---------------------------------------------------------------------------
# 迁移表。事件命名：屏幕事件 screen:<name>；内部事件 <域>:<名称>。
# 表刻意稀疏：未命中的 (状态, 事件) 走 DEFAULT_POLICY。
# ---------------------------------------------------------------------------
_ROWS: list[tuple[tuple[FlowState, str], Transition]] = [
    # ---- IDLE_WAIT：无消息时的驻留与唤醒 ----
    _t(FlowState.IDLE_WAIT, "idle:due", FlowState.IDLE_WAIT, "keep_awake_jump", note="防熄屏跳转（对应 _run_idle_maintenance）"),
    _t(FlowState.IDLE_WAIT, "photo:due", FlowState.IDLE_WAIT, "send_online_photo", note="空档期在线用户发图（首次进入空档）"),
    _t(FlowState.IDLE_WAIT, "idle:wake", FlowState.SCANNING, "start_scan"),
    # ---- SCANNING ----
    _t(FlowState.SCANNING, "target:none", FlowState.IDLE_WAIT, "enter_idle_wait"),
    _t(FlowState.SCANNING, "target:found", FlowState.ENTERING_CHAT, "click_conversation"),
    _t(FlowState.SCANNING, "screen:call", FlowState.SCANNING, "wait_call_end", note="通话中轮询等待（wait_for_call_screen_to_disappear）"),
    _t(FlowState.SCANNING, "screen:unknown", FlowState.RECOVERING_SOFT, "recover_navigate"),
    # ---- ENTERING_CHAT ----
    _t(FlowState.ENTERING_CHAT, "screen:chat", FlowState.PREPARING_REPLY, "submit_reply_and_focus_input"),
    _t(FlowState.ENTERING_CHAT, "screen:list", FlowState.ENTERING_CHAT, "retry_click_conversation", note="点击未生效仍在列表，重试（次数上限在引擎计数器）"),
    _t(FlowState.ENTERING_CHAT, "timeout", FlowState.SCANNING, "rescan", note="多次未进入会话则放弃该目标重扫"),
    _t(FlowState.ENTERING_CHAT, "screen:my", FlowState.RECOVERING_SOFT, "recover_navigate"),
    _t(FlowState.ENTERING_CHAT, "screen:unknown", FlowState.RECOVERING_SOFT, "recover_navigate"),
    # ---- PREPARING_REPLY：AI 异步 + 输入框 ----
    _t(FlowState.PREPARING_REPLY, "ai:ready", FlowState.VOICE_INPUT, "prepare_tts"),
    _t(FlowState.PREPARING_REPLY, "ai:failed", FlowState.RETURNING, "abort_round_go_back", counts_error=True, note="AI 失败：返回列表，计一次错误"),
    _t(FlowState.PREPARING_REPLY, "screen:list", FlowState.SCANNING, "abandon_round_keep_pending", counts_error=False, note="焦点丢失回列表：优雅放弃，待发回复留 outbox 下轮复用"),
    _t(FlowState.PREPARING_REPLY, "timeout", FlowState.RETURNING, "abort_round_go_back"),
    # ---- VOICE_INPUT：麦克风 + TTS（客户机 20260915 死局的高发区）----
    _t(FlowState.VOICE_INPUT, "mic:visible", FlowState.VOICE_INPUT, "click_microphone_and_play"),
    _t(FlowState.VOICE_INPUT, "tts:done", FlowState.SENDING, "click_send"),
    _t(FlowState.VOICE_INPUT, "mic:missing", FlowState.VOICE_INPUT, "close_popup_and_retry_mic", note="SEARCH 循环内先关弹窗再重找（吸收 skip_popup_detection=True 盲区）"),
    _t(FlowState.VOICE_INPUT, "mic:exhausted", FlowState.RETURNING, "abort_round_go_back", counts_error=True, note="确在聊天页但始终无麦克风：返回重试整轮"),
    _t(FlowState.VOICE_INPUT, "screen:list", FlowState.SCANNING, "abandon_round_keep_pending", counts_error=False, note="死局解药：麦克风找不到+已回列表=优雅放弃，不进熔断"),
    _t(FlowState.VOICE_INPUT, "screen:my", FlowState.RECOVERING_SOFT, "recover_navigate"),
    _t(FlowState.VOICE_INPUT, "screen:unknown", FlowState.RECOVERING_SOFT, "recover_navigate"),
    # ---- SENDING ----
    _t(FlowState.SENDING, "send:sent", FlowState.RETURNING, "click_back"),
    _t(FlowState.SENDING, "send:failed", FlowState.SENDING, "retry_click_send", note="发送重试（上限在引擎计数器）"),
    _t(FlowState.SENDING, "send:exhausted", FlowState.RETURNING, "abort_round_go_back", counts_error=True),
    _t(FlowState.SENDING, "screen:list", FlowState.SCANNING, "finish_round_from_list", counts_error=False, note="发送后自动回列表：当返回完成，正常收账"),
    _t(FlowState.SENDING, "screen:unknown", FlowState.RECOVERING_SOFT, "recover_navigate"),
    # ---- RETURNING ----
    _t(FlowState.RETURNING, "back:at_list", FlowState.SCANNING, "mark_round_success", note="mark_sent + mark_success + 轮次统计"),
    _t(FlowState.RETURNING, "screen:chat", FlowState.RETURNING, "retry_click_back", note="go_back 的多次重试（上限在引擎计数器）"),
    _t(FlowState.RETURNING, "back:exhausted", FlowState.RECOVERING_SOFT, "recover_navigate", counts_error=True),
    _t(FlowState.RETURNING, "screen:unknown", FlowState.RECOVERING_SOFT, "recover_navigate"),
    # ---- RECOVERING_SOFT：软恢复，按感知结果路由出口 ----
    _t(FlowState.RECOVERING_SOFT, "screen:list", FlowState.SCANNING, "resume_scan"),
    _t(FlowState.RECOVERING_SOFT, "screen:chat", FlowState.RETURNING, "click_back", note="恢复后停在聊天页：把返回流程走完"),
    _t(FlowState.RECOVERING_SOFT, "screen:call", FlowState.RECOVERING_SOFT, "wait_call_end"),
    _t(FlowState.RECOVERING_SOFT, "recover:exhausted", FlowState.RECOVERING_APP, "relaunch_app", note="软恢复耗尽→中恢复：从桌面重进心遇"),
    # ---- RECOVERING_APP：中恢复 ----
    _t(FlowState.RECOVERING_APP, "screen:list", FlowState.SCANNING, "resume_scan"),
    _t(FlowState.RECOVERING_APP, "screen:chat", FlowState.RETURNING, "click_back"),
    _t(FlowState.RECOVERING_APP, "recover:exhausted", FlowState.RECOVERING_APP, "hard_reset", counts_error=True, hard_reset=True, note="中恢复仍失败→硬复位（对接现有 on_error/reset_device/熔断）"),
]

TRANSITIONS: dict[tuple[FlowState, str], Transition] = dict(_ROWS)

# 未命中 (状态, 事件) 的兜底：已知工作状态里的意外屏幕一律进软恢复；
# IDLE_WAIT 驻留不动（等待类事件天然多样）。软/中恢复状态里未列出的
# 屏幕保持原状态由 recover:exhausted 计数推进梯度。
DEFAULT_POLICY_TARGET: dict[FlowState, FlowState] = {
    FlowState.IDLE_WAIT: FlowState.IDLE_WAIT,
}
DEFAULT_POLICY_ACTION = "recover_navigate"


def default_transition(source: FlowState, trigger: str) -> Transition:
    target = DEFAULT_POLICY_TARGET.get(source, FlowState.RECOVERING_SOFT)
    action = "keep_idle" if target is FlowState.IDLE_WAIT else DEFAULT_POLICY_ACTION
    return Transition(source, trigger, target, action, note="默认迁移（表未命中）")


def _coerce_state(state: FlowState | str) -> FlowState:
    return state if isinstance(state, FlowState) else FlowState(state)


def resolve(source: FlowState | str, trigger: str) -> Transition:
    source = _coerce_state(source)
    return TRANSITIONS.get((source, trigger), default_transition(source, trigger))


# ---------------------------------------------------------------------------
# 动作注册表。骨架期：handler 由接入方注入，此处只登记合法动作名，
# 迁移表完整性测试据此校验"表里每个动作都有归属"。
# ---------------------------------------------------------------------------
REGISTERED_ACTIONS: frozenset[str] = frozenset(
    {
        "keep_awake_jump",
        "send_online_photo",
        "start_scan",
        "enter_idle_wait",
        "click_conversation",
        "wait_call_end",
        "recover_navigate",
        "retry_click_conversation",
        "rescan",
        "submit_reply_and_focus_input",
        "abort_round_go_back",
        "abandon_round_keep_pending",
        "prepare_tts",
        "click_microphone_and_play",
        "close_popup_and_retry_mic",
        "click_send",
        "retry_click_send",
        "finish_round_from_list",
        "click_back",
        "retry_click_back",
        "mark_round_success",
        "resume_scan",
        "relaunch_app",
        "hard_reset",
        "keep_idle",
    }
)


class FlowHardReset(Exception):
    """中恢复耗尽：引擎抛出，由现有 on_error/reset_device/熔断接管。"""


@dataclass
class FlowEngine:
    """单轮次流程引擎：消费事件、查表迁移、调用注入的动作执行器。

    接入期 run_once 改为循环 tick()；骨架期 executor 注入
    RecordingExecutor 供单测断言动作序列。
    """

    state: FlowState = FlowState.SCANNING
    executor: Callable[..., Any] | None = None
    soft_failures: int = 0
    app_failures: int = 0
    log: list[str] = field(default_factory=list)

    def tick(self, trigger: str, perception: Perception | None = None) -> Transition:
        # 无手机/黑屏维持现有异常语义，不进状态机（等待人工，不熔断）。
        if perception is not None and perception.screen in EXTERNAL_EXCEPTION_SCREENS:
            self.log.append(f"{self.state.value} + screen:{perception.screen.value} -> external_exception")
            raise LookupError(f"screen:{perception.screen.value}")

        # 弹窗在任何工作状态优先吸收：先关弹窗，状态不变，原地重试。
        if (
            perception is not None
            and perception.popup
            and trigger not in {"mic:missing", "screen:call"}
            and self.state not in {FlowState.RECOVERING_SOFT, FlowState.RECOVERING_APP, FlowState.IDLE_WAIT}
        ):
            transition = Transition(self.state, trigger, self.state, "close_popup_and_retry_mic", note="弹窗优先吸收")
            self._apply(transition)
            return transition

        if trigger == "recover:exhausted" and self.state is FlowState.RECOVERING_SOFT:
            self.soft_failures += 1
        elif trigger == "recover:exhausted" and self.state is FlowState.RECOVERING_APP:
            self.app_failures += 1

        transition = resolve(self.state, trigger)
        self._apply(transition)
        return transition

    def _apply(self, transition: Transition) -> None:
        if self.executor is not None:
            self.executor(transition.action, state=transition.source.value, trigger=transition.trigger)
        self.log.append(f"{transition.source.value} + {transition.trigger} -> {transition.target.value}/{transition.action}")
        self.state = transition.target
        if transition.hard_reset:
            raise FlowHardReset(transition.note or "recover gradient exhausted")
