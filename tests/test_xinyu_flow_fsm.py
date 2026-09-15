"""心遇流程状态机（骨架期）全表测试。

锁定三件事：
1. 迁移表本身合法（目标状态/动作名/无重复行，每个状态可达且可达出口）；
2. 关键行为语义（优雅放弃不计错、恢复梯度三级、弹窗优先吸收、
   无手机/黑屏维持外部异常）；
3. 引擎按表驱动（happy path 一轮、故障路由、硬复位出口）。
"""

from __future__ import annotations

from collections import deque

import pytest

from apps.xinyu.flow import (
    REGISTERED_ACTIONS,
    EXTERNAL_EXCEPTION_SCREENS,
    FlowEngine,
    FlowHardReset,
    FlowState,
    Perception,
    ScreenEvent,
    TRANSITIONS,
    _ROWS,
    resolve,
)


# ---------------------------------------------------------------------------
# 迁移表完整性
# ---------------------------------------------------------------------------

def test_transition_table_rows_are_unique():
    assert len(_ROWS) == len(TRANSITIONS)


def test_every_transition_targets_valid_state_and_registered_action():
    for (source, trigger), transition in TRANSITIONS.items():
        assert transition.source is source
        assert transition.trigger == trigger
        assert isinstance(transition.target, FlowState)
        assert transition.action in REGISTERED_ACTIONS, f"{source}+{trigger} -> 未注册动作 {transition.action}"


def test_every_state_reachable_from_scanning():
    edges: dict[FlowState, set[FlowState]] = {}
    for transition in TRANSITIONS.values():
        edges.setdefault(transition.source, set()).add(transition.target)
    seen = {FlowState.SCANNING}
    queue = deque([FlowState.SCANNING])
    while queue:
        for nxt in edges.get(queue.popleft(), ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    assert seen == set(FlowState), f"不可达状态: {set(FlowState) - seen}"


def test_every_state_has_exit():
    sources = {transition.source for transition in TRANSITIONS.values()}
    assert sources == set(FlowState)


# ---------------------------------------------------------------------------
# 关键行为语义
# ---------------------------------------------------------------------------

def test_voice_input_list_event_reenters_conversation_instead_of_abandon():
    """死局正解：掉回列表=会话被退出 → 重进会话续做；放弃只是重进失败的兜底。"""
    transition = resolve(FlowState.VOICE_INPUT, "screen:list")
    assert transition.target is FlowState.REENTERING_CHAT
    assert transition.action == "reenter_conversation"
    assert transition.counts_error is False

    retry = resolve(FlowState.REENTERING_CHAT, "screen:chat")
    assert retry.target is FlowState.VOICE_INPUT
    assert retry.action == "retry_microphone"
    assert retry.counts_error is False

    give_up = resolve(FlowState.REENTERING_CHAT, "reenter:exhausted")
    assert give_up.target is FlowState.SCANNING
    assert give_up.action == "abandon_round_keep_pending"
    assert give_up.counts_error is False


def test_preparing_reply_focus_loss_abandons_without_error():
    transition = resolve(FlowState.PREPARING_REPLY, "screen:list")
    assert transition.target is FlowState.SCANNING
    assert transition.counts_error is False


def test_sending_auto_back_to_list_finishes_round():
    transition = resolve(FlowState.SENDING, "screen:list")
    assert transition.target is FlowState.SCANNING
    assert transition.action == "finish_round_from_list"
    assert transition.counts_error is False


def test_recovery_escalates_soft_then_app_then_hard_reset():
    soft = resolve(FlowState.RECOVERING_SOFT, "recover:exhausted")
    assert soft.target is FlowState.RECOVERING_APP
    assert soft.action == "relaunch_app"
    assert soft.hard_reset is False

    hard = resolve(FlowState.RECOVERING_APP, "recover:exhausted")
    assert hard.target is FlowState.RECOVERING_APP
    assert hard.action == "hard_reset"
    assert hard.hard_reset is True
    assert hard.counts_error is True


def test_ai_failure_counts_error_but_goes_back_gracefully():
    transition = resolve(FlowState.PREPARING_REPLY, "ai:failed")
    assert transition.target is FlowState.RETURNING
    assert transition.counts_error is True


def test_unknown_combination_falls_into_soft_recovery_except_idle():
    falling = resolve(FlowState.RETURNING, "screen:my")
    assert falling.target is FlowState.RECOVERING_SOFT
    assert falling.action == "recover_navigate"

    staying = resolve(FlowState.IDLE_WAIT, "whatever:happens")
    assert staying.target is FlowState.IDLE_WAIT
    assert staying.action == "keep_idle"


# ---------------------------------------------------------------------------
# 引擎行为
# ---------------------------------------------------------------------------

def _recording_executor(record: list[tuple[str, str, str]]):
    def execute(action: str, *, state: str, trigger: str) -> None:
        record.append((state, trigger, action))

    return execute


def test_engine_runs_happy_path_round():
    record: list[tuple[str, str, str]] = []
    engine = FlowEngine(executor=_recording_executor(record))
    assert engine.state is FlowState.SCANNING

    engine.tick("target:found", Perception.of(ScreenEvent.LIST))
    engine.tick("screen:chat", Perception.of(ScreenEvent.CHAT))
    engine.tick("ai:ready", Perception.of(ScreenEvent.CHAT))
    engine.tick("mic:visible", Perception.of(ScreenEvent.CHAT))
    engine.tick("tts:done", Perception.of(ScreenEvent.CHAT))
    engine.tick("send:sent", Perception.of(ScreenEvent.CHAT))
    engine.tick("back:at_list", Perception.of(ScreenEvent.LIST))

    assert engine.state is FlowState.SCANNING
    assert [action for _s, _t, action in record] == [
        "click_conversation",
        "submit_reply_and_focus_input",
        "prepare_tts",
        "click_microphone_and_play",
        "click_send",
        "click_back",
        "mark_round_success",
    ]


def test_engine_absorbs_popup_before_routing():
    engine = FlowEngine()
    engine.tick("target:found", Perception.of(ScreenEvent.LIST))
    assert engine.state is FlowState.ENTERING_CHAT

    transition = engine.tick("screen:unknown", Perception.of(ScreenEvent.CHAT, popup=True))
    # 弹窗优先吸收：状态不变、动作是关弹窗，而不是掉进软恢复。
    assert transition.target is FlowState.ENTERING_CHAT
    assert transition.action == "close_popup_and_retry_mic"
    assert engine.state is FlowState.ENTERING_CHAT


def test_engine_popup_absorption_skipped_for_mic_missing_trigger():
    """mic:missing 触发行自带关弹窗语义，不再被吸收层拦截。"""
    engine = FlowEngine()
    engine.state = FlowState.VOICE_INPUT
    transition = engine.tick("mic:missing", Perception.of(ScreenEvent.CHAT, popup=True))
    assert transition.action == "close_popup_and_retry_mic"
    assert engine.state is FlowState.VOICE_INPUT


def test_engine_external_exception_screens_raise_and_keep_state():
    for screen in EXTERNAL_EXCEPTION_SCREENS:
        engine = FlowEngine()
        engine.state = FlowState.VOICE_INPUT
        with pytest.raises(LookupError):
            engine.tick("mic:visible", Perception.of(screen))
        assert engine.state is FlowState.VOICE_INPUT


def test_engine_hard_reset_raises_flow_exception_and_counts():
    engine = FlowEngine()
    engine.state = FlowState.RECOVERING_APP
    with pytest.raises(FlowHardReset):
        engine.tick("recover:exhausted", Perception.of(ScreenEvent.UNKNOWN))
    assert engine.app_failures == 1


def test_engine_counts_soft_recovery_exhaustions():
    engine = FlowEngine()
    engine.state = FlowState.RECOVERING_SOFT
    engine.tick("recover:exhausted", Perception.of(ScreenEvent.UNKNOWN))
    assert engine.soft_failures == 1
    assert engine.state is FlowState.RECOVERING_APP
