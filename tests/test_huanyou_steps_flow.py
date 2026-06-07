from __future__ import annotations

from pathlib import Path

import pytest

from apps.huanyou import app as huanyou_app


class FakeRobot:
    def __init__(self) -> None:
        self.clicks: list[tuple[float, float]] = []
        self.swipes: list[tuple[float, float, float, float]] = []
        self.reset_count = 0

    def move_click(self, x: float, y: float) -> None:
        self.clicks.append((x, y))

    def swipe(self, start_x: float, start_y: float, end_x: float, end_y: float, wait: bool = True) -> None:
        self.swipes.append((start_x, start_y, end_x, end_y))

    def reset_device(self, wait: bool = True) -> None:
        self.reset_count += 1


def test_huanyou_chat_window_screen_uses_lower_match_threshold(monkeypatch, tmp_path):
    screens_dir = tmp_path / "screens"
    buttons_dir = tmp_path / "buttons"
    screens_dir.mkdir()
    buttons_dir.mkdir()
    (screens_dir / "screen_聊天界面.jpg").write_bytes(b"template")
    (screens_dir / "screen_聊天列表.jpg").write_bytes(b"template")
    config = {
        "templates": {"screens": str(screens_dir), "buttons": str(buttons_dir)},
        "match_threshold": 0.85,
        "template_match_thresholds": {"screen.chat_window": 0.7},
    }
    thresholds: dict[str, float] = {}

    def fake_match_template_with_region(image, template, config, key, *, threshold, match_func):
        thresholds[key] = threshold
        return None

    monkeypatch.setattr(huanyou_app.steps, "match_template_with_region", fake_match_template_with_region)

    huanyou_app.steps._matches_screen("image", config, huanyou_app.steps.CHAT_WINDOW_SCREEN)
    huanyou_app.steps._matches_screen("image", config, huanyou_app.steps.CHAT_LIST_SCREEN)

    assert thresholds[huanyou_app.steps.CHAT_WINDOW_SCREEN] == 0.7
    assert thresholds[huanyou_app.steps.CHAT_LIST_SCREEN] == 0.85


@pytest.fixture(autouse=True)
def _startup_back_button_check_passes(monkeypatch):
    monkeypatch.setattr(
        huanyou_app.steps,
        "cache_startup_back_button",
        lambda robot, config, reporter=None: True,
    )


def test_huanyou_app_scrolls_on_each_chat_without_new_greeting(monkeypatch):
    robot = FakeRobot()
    events: list[dict[str, object]] = []
    logs: list[tuple[str, str, dict[str, object]]] = []

    monkeypatch.setattr(huanyou_app.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps.random, "uniform", lambda a, b: 0.0)

    monkeypatch.setattr(
        huanyou_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: {
            "name": "用户",
            "content": "最近消息",
            "favorability": 9,
            "unread_count": 1,
            "x": 0.2,
            "y": 0.3,
        },
    )
    monkeypatch.setattr(huanyou_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(huanyou_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: b"image")
    monkeypatch.setattr(
        huanyou_app.steps,
        "consume_new_greetings",
        lambda robot, config, image=None, expected_name=None, reporter=None: (image or b"image", 0, False),
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "read_message",
        lambda robot, config, target=None, image=None, reporter=None: {"contact": "用户", "content": "最近在忙什么"},
    )
    monkeypatch.setattr(huanyou_app.steps, "input_reply", lambda robot, text, config, reporter=None: None)
    monkeypatch.setattr(huanyou_app.steps, "send_reply", lambda robot, config, reporter=None: 1)
    monkeypatch.setattr(huanyou_app.steps, "click_next_message", lambda robot, config, image=None, reporter=None: True)
    next_calls: list[int] = []
    go_back_click_counts: list[int] = []

    def fake_next_message(robot, config, expected_name=None, image=None, reporter=None):
        next_calls.append(1)
        if len(next_calls) == 1:
            return image or b"next_image", "下一个", 1, (0.46, 0.072)
        raise huanyou_app.steps.StepUnavailable("查找失败：没有匹配到下一条按钮，已尝试图片：btn_下一条按钮.jpg")

    monkeypatch.setattr(huanyou_app.steps, "click_next_message_until_changed", fake_next_message)
    monkeypatch.setattr(
        huanyou_app.steps,
        "go_back",
        lambda robot, config, reporter=None, **kwargs: go_back_click_counts.append(kwargs.get("click_count", 1)),
    )

    app = huanyou_app.HuanyouApp(robot, {}, reply_provider=lambda message: "回复")
    app.set_status_callback(events.append)
    app.set_log_callback(lambda level, message, payload=None: logs.append((level, message, payload or {})))

    app.run_once()

    assert robot.clicks == [(0.2, 0.3), (0.46, 0.072)]
    assert len(robot.swipes) == 2
    start_x, start_y, end_x, end_y = robot.swipes[0]
    assert start_x == end_x
    assert start_x == 0.5
    assert start_y == 0.6
    assert abs(end_y - 0.3) < 1e-9
    assert any(event.get("current_step") == "enter_chat" for event in events)
    assert any("(0.200, 0.300)" in str(event.get("current_step_log_label", "")) for event in events)
    assert any(
        message == "dialogue_ocr_last_message" and payload.get("content") == "最近在忙什么"
        for _level, message, payload in logs
    )


    assert go_back_click_counts == [2]


def test_huanyou_app_records_contact_send_count_after_successful_send(monkeypatch):
    robot = FakeRobot()
    logs: list[tuple[str, str, dict[str, object]]] = []

    monkeypatch.setattr(huanyou_app.steps, "input_reply", lambda robot, text, config, reporter=None: None)
    monkeypatch.setattr(huanyou_app.steps, "send_reply", lambda robot, config, reporter=None: 1)

    app = huanyou_app.HuanyouApp(robot, {})
    app.emit_runtime_log = lambda level, message, payload=None: logs.append((level, message, payload or {}))
    app.status.error_count = 4
    app.status.is_circuit_open = True
    app.recover_error_count = 2

    sent = app._send_action(
        lambda step, label, log_label=None: None,
        huanyou_app.ReplyAction(text="reply"),
        {"contact": "alice", "content": "hello"},
    )

    assert sent is True
    assert huanyou_app.steps.contact_send_count(app.config, "alice") == 1
    assert app.status.error_count == 0
    assert app.status.is_circuit_open is False
    assert app.recover_error_count == 0
    assert ("info", "huanyou_contact_send_count", {"contact": "alice", "count": 1}) in logs


def test_huanyou_app_keeps_chat_when_next_name_is_unconfirmed(monkeypatch):
    robot = FakeRobot()
    read_targets: list[dict[str, object]] = []
    go_back_calls: list[int] = []

    monkeypatch.setattr(huanyou_app.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps.random, "uniform", lambda a, b: 0.0)
    monkeypatch.setattr(
        huanyou_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: {
            "name": "same-user",
            "content": "latest",
            "favorability": 9,
            "unread_count": 1,
            "x": 0.2,
            "y": 0.3,
        },
    )
    monkeypatch.setattr(huanyou_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(huanyou_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: b"image")
    monkeypatch.setattr(
        huanyou_app.steps,
        "consume_new_greetings",
        lambda robot, config, image=None, expected_name=None, reporter=None: (image or b"image", 0, False),
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "scroll_chat_window",
        lambda robot, config, reporter=None: ((0.5, 0.6), (0.5, 0.3)),
    )

    def fake_read_message(robot, config, target=None, image=None, reporter=None):
        read_targets.append(dict(target or {}))
        if len(read_targets) == 1:
            return {"contact": "same-user", "content": "hello"}
        raise huanyou_app.steps.StepUnavailable("stop after verifying continued chat")

    monkeypatch.setattr(huanyou_app.steps, "read_message", fake_read_message)
    monkeypatch.setattr(huanyou_app.steps, "input_reply", lambda robot, text, config, reporter=None: None)
    monkeypatch.setattr(huanyou_app.steps, "send_reply", lambda robot, config, reporter=None: 1)
    monkeypatch.setattr(
        huanyou_app.steps,
        "click_next_message_until_changed",
        lambda robot, config, expected_name=None, image=None, reporter=None: (image or b"next_image", "", 1, None),
    )
    monkeypatch.setattr(huanyou_app.steps, "go_back", lambda robot, config, reporter=None, **kwargs: go_back_calls.append(1))

    app = huanyou_app.HuanyouApp(robot, {}, reply_provider=lambda message: "reply")

    app.run_once()

    assert len(read_targets) == 2
    assert go_back_calls == []
    assert app.status.last_error == "stop after verifying continued chat"
    assert robot.clicks == [(0.2, 0.3), (0.5, 0.3)]


def test_huanyou_app_resets_error_count_when_next_contact_confirmed(monkeypatch):
    robot = FakeRobot()
    read_calls: list[str] = []

    monkeypatch.setattr(huanyou_app.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps.random, "uniform", lambda a, b: 0.0)
    monkeypatch.setattr(
        huanyou_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: {
            "name": "old-user",
            "content": "latest",
            "favorability": 9,
            "unread_count": 1,
            "x": 0.2,
            "y": 0.3,
        },
    )
    monkeypatch.setattr(huanyou_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(huanyou_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: b"image")
    monkeypatch.setattr(
        huanyou_app.steps,
        "consume_new_greetings",
        lambda robot, config, image=None, expected_name=None, reporter=None: (image or b"image", 0, False),
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "scroll_chat_window",
        lambda robot, config, reporter=None: ((0.5, 0.6), (0.5, 0.3)),
    )

    def fake_read_message(robot, config, target=None, image=None, reporter=None):
        name = str((target or {}).get("name", ""))
        read_calls.append(name)
        if len(read_calls) == 1:
            return {"contact": name, "content": "hello"}
        raise huanyou_app.steps.StepUnavailable("stop after next contact")

    monkeypatch.setattr(huanyou_app.steps, "read_message", fake_read_message)
    monkeypatch.setattr(huanyou_app.steps, "input_reply", lambda robot, text, config, reporter=None: None)
    monkeypatch.setattr(huanyou_app.steps, "send_reply", lambda robot, config, reporter=None: 1)

    app = huanyou_app.HuanyouApp(robot, {}, reply_provider=lambda message: "reply")

    def fake_next_message(robot, config, expected_name=None, image=None, reporter=None):
        app.status.error_count = 4
        app.recover_error_count = 2
        return image or b"next_image", "new-user", 1, None

    monkeypatch.setattr(huanyou_app.steps, "click_next_message_until_changed", fake_next_message)

    app.run_once()

    assert read_calls == ["old-user", "new-user"]
    assert app.status.error_count == 1
    assert app.recover_error_count == 0
    assert app.status.is_circuit_open is False
    assert app.status.last_error == "stop after next contact"


def test_huanyou_app_empty_target_enters_idle_wait_without_keep_awake(monkeypatch):
    robot = FakeRobot()
    events: list[dict[str, object]] = []
    calls: list[str] = []

    monkeypatch.setattr(huanyou_app.steps, "find_unread_target", lambda robot, config, reporter=None: None)
    monkeypatch.setattr(
        huanyou_app.steps,
        "read_my_points",
        lambda robot, config, reporter=None: calls.append("points") or "120",
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "keep_screen_awake",
        lambda robot, config, reporter=None: calls.append("keep_awake") or True,
    )

    app = huanyou_app.HuanyouApp(robot, {"keep_awake_interval_seconds": 30})
    app.set_status_callback(events.append)

    app.run_once()

    assert calls == ["points"]
    assert any(event["current_step"] == "empty_list_target" for event in events)


def test_huanyou_app_checks_and_caches_back_button_on_first_run(monkeypatch):
    robot = FakeRobot()
    calls: list[str] = []

    monkeypatch.setattr(
        huanyou_app.steps,
        "cache_startup_back_button",
        lambda robot, config, reporter=None: calls.append("startup_back") or True,
    )
    monkeypatch.setattr(huanyou_app.steps, "find_unread_target", lambda robot, config, reporter=None: calls.append("find") or None)
    monkeypatch.setattr(huanyou_app.steps, "rescan_unread_target", lambda robot, config, reporter=None: calls.append("rescan") or None)
    monkeypatch.setattr(huanyou_app.steps, "read_my_points", lambda robot, config, reporter=None: None)
    monkeypatch.setattr(huanyou_app.steps, "keep_screen_awake", lambda robot, config, reporter=None: False)

    app = huanyou_app.HuanyouApp(robot, {"keep_awake_interval_seconds": 30})

    app.run_once()
    app.run_once()

    assert calls == ["startup_back", "find", "rescan"]
    assert app.status.is_circuit_open is False


def test_huanyou_app_stops_when_startup_back_button_missing(monkeypatch):
    robot = FakeRobot()
    calls: list[str] = []
    logs: list[tuple[str, str, dict[str, object]]] = []

    monkeypatch.setattr(
        huanyou_app.steps,
        "cache_startup_back_button",
        lambda robot, config, reporter=None: calls.append("startup_back") or False,
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "find_unread_target",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("normal flow should not run")),
    )

    app = huanyou_app.HuanyouApp(robot, {})
    app.set_log_callback(lambda level, message, payload=None: logs.append((level, message, payload or {})))

    app.run_once()

    assert calls == ["startup_back"]
    assert app.status.current_step == "startup_back_button_missing"
    assert app.status.last_error == "启动检查未找到返回按钮，停止进程"
    assert app.status.is_circuit_open is True
    assert any(message == "huanyou_startup_back_button_missing" for _level, message, _payload in logs)


def test_huanyou_app_emits_round_metrics_for_empty_target(monkeypatch):
    robot = FakeRobot()
    logs: list[tuple[str, dict[str, object]]] = []

    def fake_find_unread_target(robot, config, reporter=None):
        config.setdefault("_runtime_cache", {}).setdefault("metrics", {}).update(
            {"ocr_count": 1, "capture_count": 2, "last_ocr": {"mode": "list"}}
        )
        return None

    monkeypatch.setattr(huanyou_app.steps, "find_unread_target", fake_find_unread_target)
    monkeypatch.setattr(huanyou_app.steps, "read_my_points", lambda robot, config, reporter=None: None)
    monkeypatch.setattr(huanyou_app.steps, "keep_screen_awake", lambda robot, config, reporter=None: False)

    app = huanyou_app.HuanyouApp(robot, {"keep_awake_interval_seconds": 30})
    app.set_log_callback(lambda level, message, payload=None: logs.append((message, payload or {})))

    app.run_once()

    metrics = [payload for message, payload in logs if message == "huanyou_round_metrics"]
    assert metrics
    assert metrics[-1]["ocr_count"] == 1
    assert metrics[-1]["capture_count"] == 2
    assert metrics[-1]["last_ocr"] == {"mode": "list"}
    assert "seconds" in metrics[-1]


def test_huanyou_app_empty_target_runs_keep_awake_after_idle_interval(monkeypatch):
    robot = FakeRobot()
    calls: list[str] = []

    monkeypatch.setattr(huanyou_app.steps, "find_unread_target", lambda robot, config, reporter=None: None)
    monkeypatch.setattr(
        huanyou_app.steps,
        "read_my_points",
        lambda robot, config, reporter=None: calls.append("points") or "120",
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "keep_screen_awake",
        lambda robot, config, reporter=None: calls.append("keep_awake") or True,
    )

    app = huanyou_app.HuanyouApp(robot, {"keep_awake_interval_seconds": 30})
    app.last_activity_at -= 31

    app.run_once()

    assert calls == ["points", "keep_awake"]


def test_huanyou_app_recaptures_before_new_greeting_after_retract(monkeypatch):
    robot = FakeRobot()
    consume_images: list[bytes] = []
    confirm_images = iter([b"initial_chat", b"initial_ready", b"fresh_after_retract", b"fresh_ready"])

    monkeypatch.setattr(huanyou_app.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps.random, "uniform", lambda a, b: 0.0)
    monkeypatch.setattr(
        huanyou_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: {
            "name": "first",
            "content": "latest",
            "favorability": 9,
            "unread_count": 1,
            "x": 0.2,
            "y": 0.3,
        },
    )
    monkeypatch.setattr(huanyou_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(
        huanyou_app.steps,
        "confirm_chat_window",
        lambda robot, config, reporter=None: next(confirm_images),
    )

    def fake_consume_new_greetings(robot, config, image=None, expected_name=None, reporter=None):
        consume_images.append(image)
        return image, 0, False

    monkeypatch.setattr(huanyou_app.steps, "consume_new_greetings", fake_consume_new_greetings)
    monkeypatch.setattr(
        huanyou_app.steps,
        "scroll_chat_window",
        lambda robot, config, reporter=None: ((0.5, 0.6), (0.5, 0.3)),
    )

    read_calls: list[bytes] = []

    def fake_read_message(robot, config, target=None, image=None, reporter=None):
        read_calls.append(image)
        if len(read_calls) == 1:
            return {"contact": "first", "content": "hello"}
        raise huanyou_app.steps.StepUnavailable("stop after fresh capture")

    monkeypatch.setattr(huanyou_app.steps, "read_message", fake_read_message)
    monkeypatch.setattr(huanyou_app.steps, "input_reply", lambda robot, text, config, reporter=None: None)
    monkeypatch.setattr(huanyou_app.steps, "send_reply", lambda robot, config, reporter=None: 1)
    monkeypatch.setattr(
        huanyou_app.steps,
        "click_next_message_until_changed",
        lambda robot, config, expected_name=None, image=None, reporter=None: (b"stale_before_retract", "second", 1, (0.47, 0.068)),
    )

    app = huanyou_app.HuanyouApp(robot, {}, reply_provider=lambda message: "reply")

    app.run_once()

    assert robot.clicks == [(0.2, 0.3), (0.47, 0.068)]
    assert consume_images == [b"initial_chat", b"fresh_after_retract"]
    assert read_calls == [b"initial_ready", b"fresh_ready"]
    assert app.status.last_error == "stop after fresh capture"


def test_huanyou_app_does_not_retract_when_reply_was_skipped(monkeypatch):
    robot = FakeRobot()
    go_back_calls: list[int] = []

    monkeypatch.setattr(huanyou_app.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps.random, "uniform", lambda a, b: 0.0)
    monkeypatch.setattr(
        huanyou_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: {
            "name": "same-user",
            "content": "latest",
            "favorability": 9,
            "unread_count": 1,
            "x": 0.2,
            "y": 0.3,
        },
    )
    monkeypatch.setattr(huanyou_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(huanyou_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: b"image")
    monkeypatch.setattr(
        huanyou_app.steps,
        "consume_new_greetings",
        lambda robot, config, image=None, expected_name=None, reporter=None: (image or b"image", 0, False),
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "read_message",
        lambda robot, config, target=None, image=None, reporter=None: {"contact": "same-user", "content": "hello"},
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "click_next_message_until_changed",
        lambda robot, config, expected_name=None, image=None, reporter=None: (image or b"next_image", "other", 1, None),
    )
    monkeypatch.setattr(huanyou_app.steps, "go_back", lambda robot, config, reporter=None, **kwargs: go_back_calls.append(kwargs.get("click_count", 1)))

    app = huanyou_app.HuanyouApp(robot, {}, reply_provider=lambda message: huanyou_app.ReplyAction(skip=True))

    app.run_once()

    assert go_back_calls == [1]
    assert robot.clicks == [(0.2, 0.3)]


def test_click_next_message_offsets_x(monkeypatch):
    robot = FakeRobot()

    monkeypatch.setattr(huanyou_app.steps, "_template_center", lambda image, config, key, group: (0.4, 0.5))

    assert huanyou_app.steps.click_next_message(robot, {}, image=b"image") is True
    assert robot.clicks == [(0.4, 0.5)]


def test_click_new_greeting_uses_configured_offsets(monkeypatch):
    robot = FakeRobot()
    regions: list[tuple[float, float, float, float]] = []

    def fake_template_center_in_region(image, config, key, *, group, region):
        regions.append(region)
        return (0.4, 0.5)

    monkeypatch.setattr(huanyou_app.steps, "_template_center_in_region", fake_template_center_in_region)
    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)

    assert (
        huanyou_app.steps.click_new_greeting(
            robot,
            {"new_greeting_click_x_offset": 0.03, "new_greeting_click_y_offset": -0.02},
            image=b"image",
        )
        is True
    )
    assert robot.clicks == [(0.43, 0.48)]
    assert regions == [huanyou_app.steps.NEW_GREETING_SEARCH_REGION]


def test_click_next_message_uses_configured_offsets(monkeypatch):
    robot = FakeRobot()

    monkeypatch.setattr(huanyou_app.steps, "_template_center", lambda image, config, key, group: (0.4, 0.5))

    assert (
        huanyou_app.steps.click_next_message(
            robot,
            {"next_message_click_x_offset": 0.05, "next_message_click_y_offset": -0.03},
            image=b"image",
        )
        is True
    )
    assert robot.clicks == [(0.45, 0.47)]


def test_retract_input_box_clicks_center_with_jitter(monkeypatch):
    robot = FakeRobot()
    offsets = iter([0.02, -0.01])

    monkeypatch.setattr(huanyou_app.steps.random, "uniform", lambda a, b: next(offsets))
    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)

    x, y = huanyou_app.steps.retract_input_box(robot, {})

    assert x == 0.52
    assert y == 0.29
    assert robot.clicks == [(0.52, 0.29)]


def test_retract_input_box_clicks_name_point_without_jitter(monkeypatch):
    robot = FakeRobot()

    monkeypatch.setattr(huanyou_app.steps.random, "uniform", lambda a, b: 0.02)
    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)

    x, y = huanyou_app.steps.retract_input_box(robot, {}, click_point=(0.48, 0.072))

    assert x == 0.48
    assert y == 0.072
    assert robot.clicks == [(0.48, 0.072)]


def test_read_chat_contact_name_point_uses_top_center_name_ocr(monkeypatch):
    monkeypatch.setattr(huanyou_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: b"image")
    monkeypatch.setattr(
        huanyou_app.steps,
        "_run_ocr_mode",
        lambda image, config, mode, region=None: {
            "result": {"user_name": "火龙果", "dialogues": []},
            "ocr_items": [
                {"text": "12km", "x": 0.48, "y": 0.02, "w": 0.05, "h": 0.02},
                {"text": "火龙果", "x": 0.44, "y": 0.052, "w": 0.12, "h": 0.028},
                {"text": "你好", "x": 0.2, "y": 0.3, "w": 0.1, "h": 0.04},
            ],
        },
    )

    point = huanyou_app.steps.read_chat_contact_name_point(None, {})

    assert point == (0.5, 0.066)


def test_focus_input_box_waits_after_click(monkeypatch):
    robot = FakeRobot()
    sleeps: list[float] = []

    monkeypatch.setattr(huanyou_app.steps, "capture_after_popups", lambda *args, **kwargs: b"image")
    monkeypatch.setattr(huanyou_app.steps, "_template_center", lambda image, config, key, group: (0.9, 0.854))
    monkeypatch.setattr(huanyou_app.steps.time, "sleep", sleeps.append)

    huanyou_app.steps.focus_input_box(robot, {"input_box_focus_settle_seconds": 1.2})

    assert robot.clicks == [(0.4, 0.854)]
    assert sleeps == [1.2]


def test_play_voice_reply_reports_when_microphone_missing(monkeypatch):
    robot = FakeRobot()
    events: list[tuple[str, str, str | None]] = []
    now = [0.0]

    monkeypatch.setattr(huanyou_app.steps, "capture_after_popups", lambda *args, **kwargs: b"image")
    monkeypatch.setattr(huanyou_app.steps, "_template_center", lambda image, config, key, group: None)
    monkeypatch.setattr(huanyou_app.steps.time, "monotonic", lambda: now[0])

    def fake_sleep(seconds: float) -> None:
        now[0] += seconds

    monkeypatch.setattr(huanyou_app.steps.time, "sleep", fake_sleep)

    try:
        huanyou_app.steps.play_voice_reply(
            robot,
            "reply",
            {"step_timeout": 1, "tts_retry_attempts": 0},
            reporter=lambda step, label, log_label=None: events.append((step, label, log_label)),
        )
    except huanyou_app.steps.StepUnavailable as exc:
        assert "查找失败：没有匹配到麦克风按钮" in str(exc)
        assert "btn_麦克风按钮.jpg" in str(exc)
    else:
        raise AssertionError("expected microphone lookup failure")

    assert any(step == "microphone_not_found" and "麦克风按钮未找到" in str(log_label) for step, _label, log_label in events)


def test_go_back_clicks_first_back_template_once_by_default(monkeypatch):
    robot = FakeRobot()

    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps, "ensure_send_cleared", lambda robot, config, reporter=None: 0)
    monkeypatch.setattr(
        huanyou_app.steps,
        "capture_after_popups",
        lambda robot, config, label="screen", max_dismiss=2, reporter=None: label,
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "_matches_screen",
        lambda image, config, key: image == "after_go_back",
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "_template_center_with_path",
        lambda image, config, key, group: (0.3, 0.4, Path("btn_返回按钮.jpg")),
    )

    huanyou_app.steps.go_back(robot, {})

    assert robot.clicks == [(0.3, 0.4)]


def test_go_back_clicks_second_back_template_once(monkeypatch):
    robot = FakeRobot()

    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps, "ensure_send_cleared", lambda robot, config, reporter=None: 0)
    monkeypatch.setattr(
        huanyou_app.steps,
        "capture_after_popups",
        lambda robot, config, label="screen", max_dismiss=2, reporter=None: label,
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "_matches_screen",
        lambda image, config, key: image == "after_go_back",
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "_template_center_with_path",
        lambda image, config, key, group: (0.3, 0.4, Path("btn_返回按钮_2.jpg")),
    )

    huanyou_app.steps.go_back(robot, {})

    assert robot.clicks == [(0.3, 0.4)]


def test_go_back_clicks_twice_when_requested(monkeypatch):
    robot = FakeRobot()

    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps, "ensure_send_cleared", lambda robot, config, reporter=None: 0)
    monkeypatch.setattr(
        huanyou_app.steps,
        "capture_after_popups",
        lambda robot, config, label="screen", max_dismiss=2, reporter=None: label,
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "_matches_screen",
        lambda image, config, key: image == "after_go_back",
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "_template_center_with_path",
        lambda image, config, key, group: (0.3, 0.4, Path("btn_杩斿洖鎸夐挳.jpg")),
    )

    huanyou_app.steps.go_back(robot, {}, click_count=2)

    assert robot.clicks == [(0.3, 0.4), (0.3, 0.4)]


def test_click_next_message_until_changed_retries_same_contact(monkeypatch):
    robot = FakeRobot()
    confirmed_images = iter([b"same_image", b"next_image"])
    names = iter(["用户", "另一个"])

    monkeypatch.setattr(huanyou_app.steps, "_template_center", lambda image, config, key, group: (0.4, 0.5))
    monkeypatch.setattr(
        huanyou_app.steps,
        "confirm_chat_window",
        lambda robot, config, reporter=None: next(confirmed_images),
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "read_chat_contact_identity",
        lambda robot, config, image=None, reporter=None: (next(names), None),
    )

    image, name, clicks, name_point = huanyou_app.steps.click_next_message_until_changed(
        robot,
        {},
        expected_name="用户",
        image=b"start_image",
        retry_limit=2,
    )

    assert image == b"next_image"
    assert name == "另一个"
    assert clicks == 2
    assert name_point is None
    assert robot.clicks == [(0.4, 0.5), (0.4, 0.5)]


def test_consume_new_greetings_stops_without_next_when_expected_name(monkeypatch):
    robot = FakeRobot()
    reports: list[tuple[str, str]] = []

    monkeypatch.setattr(huanyou_app.steps, "click_new_greeting", lambda robot, config, image=None, reporter=None: True)
    monkeypatch.setattr(
        huanyou_app.steps,
        "capture_after_popups",
        lambda robot, config, label="", reporter=None: b"after_greeting",
    )

    def fake_next_message(robot, config, expected_name=None, image=None, reporter=None):
        raise huanyou_app.steps.StepUnavailable("查找失败：没有匹配到下一条按钮，已尝试图片：btn_下一条按钮.jpg")

    monkeypatch.setattr(huanyou_app.steps, "click_next_message_until_changed", fake_next_message)

    image, handled, stopped_without_next = huanyou_app.steps.consume_new_greetings(
        robot,
        {},
        image=b"chat",
        expected_name="余额",
        reporter=lambda key, label, log_label=None: reports.append((key, label)),
    )

    assert image == b"after_greeting"
    assert handled == 1
    assert stopped_without_next is True
    assert ("new_greeting_no_next", "新招呼后未找到下一条，返回聊天列表") in reports


def test_huanyou_keep_screen_awake_clicks_message_once(monkeypatch):
    robot = FakeRobot()

    monkeypatch.setattr(huanyou_app.steps, "ensure_list_screen", lambda robot, config, reporter=None: b"list")
    monkeypatch.setattr(
        huanyou_app.steps,
        "_run_ocr_mode",
        lambda image, config, mode, region=None: {
            "result": {
                "bottom_nav": [{"name": "消息", "bounds": [0.663, 0.926, 0.73, 0.946]}]
            }
        },
    )
    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)

    assert huanyou_app.steps.keep_screen_awake(robot, {}) is True
    assert robot.clicks == [(0.6965, 0.936)]


def test_huanyou_app_skips_scroll_when_new_greeting_exists(monkeypatch):
    robot = FakeRobot()

    monkeypatch.setattr(
        huanyou_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: {
            "name": "用户",
            "content": "新招呼",
            "favorability": 9,
            "unread_count": 1,
            "x": 0.2,
            "y": 0.3,
        },
    )
    monkeypatch.setattr(huanyou_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(huanyou_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: b"image")
    monkeypatch.setattr(
        huanyou_app.steps,
        "consume_new_greetings",
        lambda robot, config, image=None, expected_name=None, reporter=None: (image or b"image", 1, True),
    )
    monkeypatch.setattr(huanyou_app.steps, "go_back", lambda robot, config, reporter=None, **kwargs: None)

    app = huanyou_app.HuanyouApp(robot, {}, reply_provider=lambda message: "回复")

    app.run_once()

    assert robot.clicks == [(0.2, 0.3)]
    assert robot.swipes == []


def test_huanyou_app_reads_new_contact_after_consuming_greeting_and_switching(monkeypatch):
    robot = FakeRobot()

    monkeypatch.setattr(huanyou_app.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps.random, "uniform", lambda a, b: 0.0)

    monkeypatch.setattr(
        huanyou_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: {
            "name": "user",
            "content": "new greeting",
            "favorability": 9,
            "unread_count": 1,
            "x": 0.2,
            "y": 0.3,
        },
    )
    monkeypatch.setattr(huanyou_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(huanyou_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: b"image")
    call_order: list[str] = []

    def fake_scroll_chat_window(robot, config, reporter=None):
        call_order.append("scroll")
        robot.swipe(0.5, 0.6, 0.5, 0.3)
        return (0.5, 0.6), (0.5, 0.3)

    monkeypatch.setattr(huanyou_app.steps, "scroll_chat_window", fake_scroll_chat_window)
    monkeypatch.setattr(
        huanyou_app.steps,
        "consume_new_greetings",
        lambda robot, config, image=None, expected_name=None, reporter=None: (
            call_order.append("consume_greeting") or (image or b"image", 1, False)
        ),
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "read_chat_contact_name",
        lambda robot, config, image=None, reporter=None: call_order.append("read_name") or "next-user",
    )
    read_targets: list[str] = []

    def fake_read_message(robot, config, target=None, image=None, reporter=None):
        call_order.append("read_message")
        read_targets.append(str((target or {}).get("name", "")))
        return {"contact": str((target or {}).get("name", "")), "content": "new contact message"}

    monkeypatch.setattr(huanyou_app.steps, "read_message", fake_read_message)
    monkeypatch.setattr(huanyou_app.steps, "input_reply", lambda robot, text, config, reporter=None: None)
    monkeypatch.setattr(huanyou_app.steps, "send_reply", lambda robot, config, reporter=None: 1)
    monkeypatch.setattr(huanyou_app.steps, "click_next_message", lambda robot, config, image=None, reporter=None: True)
    next_calls: list[int] = []

    def fake_next_message(robot, config, expected_name=None, image=None, reporter=None):
        call_order.append("next_message")
        next_calls.append(1)
        raise huanyou_app.steps.StepUnavailable("查找失败：没有匹配到下一条按钮，已尝试图片：btn_下一条按钮.jpg")

    monkeypatch.setattr(huanyou_app.steps, "click_next_message_until_changed", fake_next_message)
    go_back_click_counts: list[int] = []
    monkeypatch.setattr(
        huanyou_app.steps,
        "go_back",
        lambda robot, config, reporter=None, **kwargs: go_back_click_counts.append(kwargs.get("click_count", 1)),
    )

    app = huanyou_app.HuanyouApp(robot, {}, reply_provider=lambda message: "reply")
    app.run_once()

    assert robot.clicks == [(0.2, 0.3)]
    assert len(robot.swipes) == 1
    assert call_order == ["consume_greeting", "scroll", "read_name", "read_message", "next_message"]
    assert read_targets == ["next-user"]
    assert len(next_calls) == 1
    assert go_back_click_counts == [2]


def test_huanyou_app_black_screen_reports_status_without_stopping(monkeypatch):
    robot = FakeRobot()
    events: list[dict[str, object]] = []
    logs: list[tuple[str, str, dict[str, object]]] = []

    monkeypatch.setattr(
        huanyou_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: (_ for _ in ()).throw(
            huanyou_app.steps.BlackScreenError("black screen detected")
        ),
    )

    app = huanyou_app.HuanyouApp(robot, {})
    app.set_status_callback(events.append)
    app.set_log_callback(lambda level, message, payload=None: logs.append((level, message, payload or {})))
    app.run_once()

    assert app.status.current_step == "black_screen_waiting"
    assert app.status.current_step_label == "黑屏异常，等待亮屏"
    assert app.status.is_circuit_open is False
    assert app.status.last_error == "black screen detected"
    assert logs[-1][1] == "black_screen_detected"


def test_huanyou_app_no_phone_reports_status_and_stops(monkeypatch):
    robot = FakeRobot()
    logs: list[tuple[str, str, dict[str, object]]] = []

    monkeypatch.setattr(
        huanyou_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: (_ for _ in ()).throw(
            huanyou_app.steps.NoPhoneScreenError("no phone detected")
        ),
    )

    app = huanyou_app.HuanyouApp(robot, {})
    app.set_log_callback(lambda level, message, payload=None: logs.append((level, message, payload or {})))
    app.run_once()

    assert app.status.current_step == "no_phone_waiting"
    assert app.status.current_step_label == "无手机异常，停止进程"
    assert app.status.is_circuit_open is True
    assert app.status.last_error == "no phone detected"
    assert logs[-1][1] == "no_phone_detected"


def test_huanyou_step_unavailable_does_not_reset_robot(monkeypatch):
    robot = FakeRobot()
    logs: list[tuple[str, str, dict[str, object]]] = []

    monkeypatch.setattr(
        huanyou_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: {
            "name": "user",
            "content": "latest message",
            "favorability": 9,
            "unread_count": 1,
            "x": 0.2,
            "y": 0.3,
        },
    )
    monkeypatch.setattr(huanyou_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(huanyou_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: b"image")
    monkeypatch.setattr(
        huanyou_app.steps,
        "consume_new_greetings",
        lambda robot, config, image=None, expected_name=None, reporter=None: (image or b"image", 0, False),
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "read_message",
        lambda robot, config, target=None, image=None, reporter=None: (
            reporter("dialogue_ocr_read", "聊天详情 OCR") if reporter else None,
            (_ for _ in ()).throw(
                huanyou_app.steps.StepUnavailable("聊天详情 OCR 没有识别到对方消息：目标 sender=sender1")
            ),
        )[1],
    )

    app = huanyou_app.HuanyouApp(robot, {}, reply_provider=lambda message: "reply")
    app.set_log_callback(lambda level, message, payload=None: logs.append((level, message, payload or {})))

    app.run_once()

    assert robot.reset_count == 0
    assert app.status.last_error == "聊天详情 OCR 没有识别到对方消息：目标 sender=sender1"
    assert any(
        message.startswith("欢友步骤不可用：")
        and "聊天详情 OCR" in message
        and "聊天详情 OCR 没有识别到对方消息：目标 sender=sender1" in message
        and payload.get("detail") == "聊天详情 OCR"
        and payload.get("error") == "聊天详情 OCR 没有识别到对方消息：目标 sender=sender1"
        and payload.get("error_count") == 1
        and payload.get("circuit_breaker_threshold") == 5
        and payload.get("will_circuit_open") is False
        for _level, message, payload in logs
    )


def test_huanyou_step_unavailable_recovers_to_list_without_error_count(monkeypatch):
    robot = FakeRobot()
    logs: list[tuple[str, str, dict[str, object]]] = []

    monkeypatch.setattr(
        huanyou_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: {
            "name": "user",
            "content": "latest message",
            "favorability": 9,
            "unread_count": 1,
            "x": 0.2,
            "y": 0.3,
        },
    )
    monkeypatch.setattr(huanyou_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(huanyou_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: b"image")
    monkeypatch.setattr(
        huanyou_app.steps,
        "consume_new_greetings",
        lambda robot, config, image=None, expected_name=None, reporter=None: (image or b"image", 0, False),
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "read_message",
        lambda robot, config, target=None, image=None, reporter=None: (
            reporter("dialogue_ocr_read", "聊天详情 OCR") if reporter else None,
            (_ for _ in ()).throw(huanyou_app.steps.StepUnavailable("聊天详情 OCR 未识别到任何文字")),
        )[1],
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "ensure_list_screen",
        lambda robot, config, reporter=None: reporter("screen_detect", "确认欢友聊天列表") or b"list",
    )

    app = huanyou_app.HuanyouApp(robot, {}, reply_provider=lambda message: "reply")
    app.status.error_count = 4
    app.status.last_error = "old error"
    app.recover_error_count = 2
    app.set_log_callback(lambda level, message, payload=None: logs.append((level, message, payload or {})))

    app.run_once()

    assert app.status.error_count == 0
    assert app.status.last_error == ""
    assert app.status.is_circuit_open is False
    assert app.recover_error_count == 0
    assert app.status.current_step == "step_recovered_to_list"
    assert app.status.current_step_label == "异常已恢复到聊天列表"
    assert any(message == "欢友步骤不可用，已恢复到聊天列表" for _level, message, _payload in logs)


def test_find_unread_target_selects_highest_favorability_among_unread(monkeypatch):
    robot = FakeRobot()

    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps, "ensure_list_screen", lambda robot, config, reporter=None: b"image")
    monkeypatch.setattr(
        huanyou_app.steps,
        "capture_after_popups",
        lambda robot, config, label="screen", max_dismiss=2, reporter=None: b"image2",
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "run_ocr_mode",
        lambda image, config, mode, region=None: {
            "result": {
                "bottom_nav": [
                    {"name": "交友", "bounds": [0.0, 0.9, 0.2, 1.0]},
                    {"name": "消息", "bounds": [0.6, 0.9, 0.8, 1.0]},
                ],
                "message_list": [
                    {
                        "name": "低好感",
                        "favorability": 3,
                        "unread_count": 1,
                        "content": "低",
                        "bounds": [0.1, 0.2, 0.5, 0.3],
                    },
                    {
                        "name": "高好感",
                        "favorability": 18,
                        "unread_count": 1,
                        "content": "高",
                        "bounds": [0.2, 0.5, 0.8, 0.7],
                    },
                ]
            }
        },
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "capture_after_popups",
        lambda robot, config, label="screen", max_dismiss=2, reporter=None: b"image2",
    )

    target = huanyou_app.steps.find_unread_target(robot, {})

    assert target is not None
    assert target["name"] == "高好感"
    assert target["x"] == 0.5
    assert target["y"] == 0.6


def test_find_unread_target_returns_none_when_no_unread_rows(monkeypatch):
    robot = FakeRobot()

    monkeypatch.setattr(huanyou_app.steps, "ensure_list_screen", lambda robot, config, reporter=None: b"image")
    monkeypatch.setattr(
        huanyou_app.steps,
        "capture_after_popups",
        lambda robot, config, label="screen", max_dismiss=2, reporter=None: b"image2",
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "run_ocr_mode",
        lambda image, config, mode, region=None: {
            "result": {
                "bottom_nav": [
                    {"name": "交友", "bounds": [0.0, 0.9, 0.2, 1.0]},
                    {"name": "消息", "bounds": [0.6, 0.9, 0.8, 1.0]},
                ],
                "message_list": [
                    {
                        "name": "娇艳的冲锋衣",
                        "favorability": 19,
                        "unread_count": None,
                        "content": "人家在等哥哥教我怎么用呢？",
                        "bounds": [0.1, 0.2, 0.5, 0.3],
                    }
                ],
            }
        },
    )

    target = huanyou_app.steps.find_unread_target(robot, {})

    assert target is None


def test_select_unread_target_filters_contact_sent_three_times_in_current_run():
    reports: list[tuple[str, str, str | None]] = []
    config = {"_runtime_cache": {"contact_send_counts": {"alice": 3}}}
    ocr_result = {
        "result": {
            "message_list": [
                {
                    "name": "alice",
                    "favorability": 99,
                    "unread_count": 1,
                    "content": "high",
                    "bounds": [0.1, 0.2, 0.5, 0.3],
                },
                {
                    "name": "bob",
                    "favorability": 1,
                    "unread_count": 1,
                    "content": "low",
                    "bounds": [0.2, 0.4, 0.8, 0.6],
                },
            ]
        }
    }

    target = huanyou_app.steps._select_unread_target_from_ocr_result(
        ocr_result,
        config,
        reporter=lambda step, label, log_label=None: reports.append((step, label, log_label)),
    )

    assert target is not None
    assert target["name"] == "bob"
    assert huanyou_app.steps.contact_send_count(config, "alice") == 3
    assert any(step == "target_send_limit_filtered" for step, _label, _log_label in reports)


def test_contact_send_count_is_case_insensitive_for_ascii_names():
    config: dict = {}

    assert huanyou_app.steps.record_contact_send(config, "z") == 1
    assert huanyou_app.steps.record_contact_send(config, "Z") == 2
    assert huanyou_app.steps.contact_send_count(config, "ｚ") == 2


def test_find_unread_target_prefers_unread_when_favorability_missing(monkeypatch):
    robot = FakeRobot()

    monkeypatch.setattr(huanyou_app.steps, "ensure_list_screen", lambda robot, config, reporter=None: b"image")
    monkeypatch.setattr(
        huanyou_app.steps,
        "run_ocr_mode",
        lambda image, config, mode, region=None: {
            "result": {
                "message_list": [
                    {
                        "name": "已读",
                        "favorability": None,
                        "unread_count": None,
                        "content": "今天忙不忙",
                        "bounds": [0.1, 0.2, 0.5, 0.3],
                    },
                    {
                        "name": "未读",
                        "favorability": None,
                        "unread_count": 1,
                        "content": "对方发来礼物",
                        "bounds": [0.2, 0.5, 0.8, 0.7],
                    },
                ]
            }
        },
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "capture_after_popups",
        lambda robot, config, label="screen", max_dismiss=2, reporter=None: b"image2",
    )

    target = huanyou_app.steps.find_unread_target(robot, {})

    assert target is not None
    assert target["name"] == "未读"


def test_find_unread_target_clicks_bottom_nav_message_twice(monkeypatch):
    robot = FakeRobot()

    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps, "ensure_list_screen", lambda robot, config, reporter=None: b"image")
    monkeypatch.setattr(
        huanyou_app.steps,
        "capture_after_popups",
        lambda robot, config, label="screen", max_dismiss=2, reporter=None: b"image2",
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "run_ocr_mode",
        lambda image, config, mode, region=None: {
            "result": {
                "bottom_nav": [
                    {"name": "广场", "bounds": [0.0, 0.9, 0.2, 1.0]},
                    {"name": "消息", "bounds": [0.6, 0.9, 0.8, 1.0]},
                ],
                "message_list": [
                    {
                        "name": "用户",
                        "favorability": 10,
                        "unread_count": 1,
                        "content": "你好",
                        "bounds": [0.1, 0.2, 0.5, 0.3],
                    }
                ],
            }
        },
    )

    target = huanyou_app.steps.find_unread_target(robot, {})

    assert target is not None
    assert robot.clicks == [(0.7, 0.95), (0.7, 0.95)]


def test_find_unread_target_ignores_missing_bottom_nav_message(monkeypatch):
    robot = FakeRobot()

    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps, "ensure_list_screen", lambda robot, config, reporter=None: b"image")
    monkeypatch.setattr(
        huanyou_app.steps,
        "run_ocr_mode",
        lambda image, config, mode, region=None: {
            "result": {
                "bottom_nav": [{"name": "我的", "bounds": [0.8, 0.9, 1.0, 1.0]}],
                "message_list": [
                    {
                        "name": "用户",
                        "favorability": 10,
                        "unread_count": 1,
                        "content": "你好",
                        "bounds": [0.1, 0.2, 0.5, 0.3],
                    }
                ],
            }
        },
    )

    target = huanyou_app.steps.find_unread_target(robot, {})

    assert target is not None
    assert robot.clicks == []


def test_huanyou_ensure_list_screen_clicks_back_on_unknown_screen(monkeypatch):
    robot = FakeRobot()
    captures = iter([b"unknown", b"list"])
    back_calls: list[bytes] = []

    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        huanyou_app.steps,
        "capture_after_popups",
        lambda robot, config, label="screen", max_dismiss=2, reporter=None: next(captures),
    )
    monkeypatch.setattr(huanyou_app.steps, "_raise_if_abnormal_screen", lambda image, config: None)

    def fake_matches_screen(image, config, key):
        return image == b"list" and key == huanyou_app.steps.CHAT_LIST_SCREEN

    def fake_click_back_once(robot, config, image=None, reporter=None):
        back_calls.append(image)
        return True

    monkeypatch.setattr(huanyou_app.steps, "_matches_screen", fake_matches_screen)
    monkeypatch.setattr(huanyou_app.steps, "_click_back_once", fake_click_back_once)

    image = huanyou_app.steps.ensure_list_screen(robot, {})

    assert image == b"list"
    assert back_calls == [b"unknown"]


def test_find_unread_target_rescans_after_bottom_nav_message_click(monkeypatch):
    robot = FakeRobot()
    ocr_calls: list[bytes] = []

    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps, "ensure_list_screen", lambda robot, config, reporter=None: b"before_click")
    monkeypatch.setattr(
        huanyou_app.steps,
        "capture_after_popups",
        lambda robot, config, label="screen", max_dismiss=2, reporter=None: b"after_click",
    )

    def fake_run_ocr_mode(image, config, mode, region=None):
        ocr_calls.append(image)
        if len(ocr_calls) == 1:
            return {
                "result": {
                    "bottom_nav": [{"name": "消息", "bounds": [0.6, 0.9, 0.8, 1.0]}],
                    "message_list": [
                        {
                            "name": "旧坐标用户",
                            "favorability": 99,
                            "unread_count": 1,
                            "content": "旧消息",
                            "bounds": [0.1, 0.2, 0.3, 0.3],
                        }
                    ],
                }
            }
        return {
            "result": {
                "bottom_nav": [{"name": "消息", "bounds": [0.6, 0.9, 0.8, 1.0]}],
                "message_list": [
                    {
                        "name": "重扫用户",
                        "favorability": 1,
                        "unread_count": 1,
                        "content": "新消息",
                        "bounds": [0.4, 0.5, 0.8, 0.7],
                    }
                ],
            }
        }

    monkeypatch.setattr(huanyou_app.steps, "run_ocr_mode", fake_run_ocr_mode)

    target = huanyou_app.steps.find_unread_target(robot, {})

    assert target is not None
    assert target["name"] == "重扫用户"
    assert abs(target["x"] - 0.6) < 1e-9
    assert abs(target["y"] - 0.6) < 1e-9
    assert ocr_calls == [b"before_click", b"after_click"]
    assert robot.clicks == [(0.7, 0.95), (0.7, 0.95)]


def test_huanyou_read_my_points_uses_list_bottom_nav_positions(monkeypatch):
    robot = FakeRobot()
    seen_modes: list[tuple[object, str]] = []

    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps, "ensure_list_screen", lambda robot, config, reporter=None: b"list_image")
    monkeypatch.setattr(
        huanyou_app.steps,
        "_template_center",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("template fallback must not be used")),
    )

    def fake_capture_after_popups(robot, config, label="screen", max_dismiss=2, reporter=None):
        if label == "my_screen":
            return b"my_image"
        return b"list_after_points"

    def fake_run_ocr_mode(image, config, mode, region=None):
        seen_modes.append((image, mode))
        if image == b"list_image" and mode == "list":
            return {
                "result": {
                    "bottom_nav": [
                        {"name": "消息", "bounds": [0.663, 0.926, 0.73, 0.946]},
                        {"name": "我的", "bounds": [0.862, 0.926, 0.929, 0.944]},
                    ]
                }
            }
        if image == b"my_image" and mode == "points":
            return {"result": {"points": "124.15", "balance": "124.15"}}
        if image == b"my_image" and mode == "list":
            return {
                "result": {
                    "bottom_nav": [
                        {"name": "消息", "bounds": [0.663, 0.926, 0.73, 0.946]},
                        {"name": "我的", "bounds": [0.862, 0.926, 0.929, 0.944]},
                    ]
                }
            }
        return {"result": {}}

    monkeypatch.setattr(huanyou_app.steps, "capture_after_popups", fake_capture_after_popups)
    monkeypatch.setattr(huanyou_app.steps, "run_ocr_mode", fake_run_ocr_mode)

    assert huanyou_app.steps.read_my_points(robot, {}) == "124.15"
    assert seen_modes == [(b"list_image", "list"), (b"my_image", "points")]
    assert robot.clicks == [(0.8955, 0.935), (0.6965, 0.936)]


def test_huanyou_app_reads_points_once_per_idle_wait_and_reports_balance(monkeypatch):
    robot = FakeRobot()
    calls: list[str] = []
    logs: list[tuple[str, dict[str, object] | None]] = []

    monkeypatch.setattr(huanyou_app.steps, "find_unread_target", lambda robot, config, reporter=None: None)
    monkeypatch.setattr(huanyou_app.steps, "rescan_unread_target", lambda robot, config, reporter=None: calls.append("rescan") or None)
    monkeypatch.setattr(huanyou_app.steps, "read_my_points", lambda robot, config, reporter=None: calls.append("points") or "124.15")
    monkeypatch.setattr(huanyou_app.steps, "keep_screen_awake", lambda robot, config, reporter=None: calls.append("keep_awake") or True)

    app = huanyou_app.HuanyouApp(robot, {"keep_awake_interval_seconds": 30})
    app.set_log_callback(lambda level, message, payload=None: logs.append((message, payload)))

    app.run_once()
    app.run_once()

    assert calls == ["points", "rescan"]
    assert ("huanyou_balance", {"balance": "12", "points": "124.15"}) in logs


def test_huanyou_idle_wait_uses_list_rescan_instead_of_full_list_confirm(monkeypatch):
    robot = FakeRobot()
    calls: list[str] = []

    monkeypatch.setattr(huanyou_app.steps, "find_unread_target", lambda robot, config, reporter=None: calls.append("find") or None)
    monkeypatch.setattr(huanyou_app.steps, "rescan_unread_target", lambda robot, config, reporter=None: calls.append("rescan") or None)
    monkeypatch.setattr(huanyou_app.steps, "read_my_points", lambda robot, config, reporter=None: calls.append("points") or None)
    monkeypatch.setattr(huanyou_app.steps, "keep_screen_awake", lambda robot, config, reporter=None: calls.append("keep_awake") or True)

    app = huanyou_app.HuanyouApp(robot, {"keep_awake_interval_seconds": 30})

    app.run_once()
    app.run_once()

    assert calls == ["find", "points", "rescan"]


def test_rescan_unread_target_dismisses_popups_before_list_ocr(monkeypatch):
    robot = FakeRobot()
    calls: list[str] = []

    monkeypatch.setattr(
        huanyou_app.steps,
        "ensure_list_screen",
        lambda robot, config, reporter=None: calls.append("ensure_list_screen") or b"list_screen",
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "_run_ocr_mode",
        lambda image, config, mode, region=None: calls.append(f"ocr:{image!r}") or {
            "result": {
                "message_list": [
                    {
                        "name": "未读用户",
                        "favorability": 1,
                        "unread_count": 1,
                        "content": "你好",
                        "bounds": [0.2, 0.4, 0.6, 0.5],
                    }
                ]
            }
        },
    )

    target = huanyou_app.steps.rescan_unread_target(robot, {})

    assert target is not None
    assert target["name"] == "未读用户"
    assert calls == ["ensure_list_screen", "ocr:b'list_screen'"]


def test_huanyou_balance_display_truncates_decimals_after_dividing_by_ten():
    assert huanyou_app.HuanyouApp._balance_from_points("124.15") == "12"
    assert huanyou_app.HuanyouApp._balance_from_points("125.29") == "12"


def test_read_message_uses_configured_dialogue_target_sender(monkeypatch):
    robot = FakeRobot()

    monkeypatch.setattr(
        huanyou_app.steps,
        "run_ocr_mode",
        lambda image, config, mode, region=None: {
            "ocr_items": [{"text": "???"}],
            "result": {
                "user_name": "???",
                "dialogues": [
                    {"sender": "sender1", "content": "旧消息"},
                    {"sender": "sender2", "content": "自己回复"},
                    {"sender": "sender1", "content": "第一句"},
                    {"sender": "sender1", "content": "第二句"},
                ],
            },
        },
    )

    message = huanyou_app.steps.read_message(
        robot,
        {
            "ocr_modes": {
                "dialogue": {
                    "rules": {
                        "target_sender": "sender1",
                    },
                },
            },
        },
        {"name": "???"},
        image=b"image",
    )

    assert message["contact"] == "???"
    assert message["content"] == "第一句第二句"


def test_read_message_defaults_to_hello_when_target_sender_missing(monkeypatch):
    robot = FakeRobot()
    reports: list[tuple[str, str, str | None]] = []

    monkeypatch.setattr(
        huanyou_app.steps,
        "run_ocr_mode",
        lambda image, config, mode, region=None: {
            "ocr_items": [{"text": "聊天标题"}],
            "result": {
                "user_name": "用户",
                "dialogues": [
                    {"sender": "sender2", "content": "自己回复"},
                    {"sender": "sender2", "content": "继续回复"},
                ],
            },
        },
    )

    message = huanyou_app.steps.read_message(
        robot,
        {
            "ocr_modes": {
                "dialogue": {
                    "rules": {
                        "target_sender": "sender1",
                    },
                },
            },
        },
        {"name": "列表用户"},
        image=b"image",
        reporter=lambda step, label, log_label=None: reports.append((step, label, log_label)),
    )

    assert message["contact"] == "列表用户"
    assert message["content"] == "你好"
    assert message["sender"] == "sender1"
    assert any(step == "dialogue_ocr_fallback" for step, _label, _log_label in reports)


def test_find_unread_target_uses_cached_message_then_single_list_ocr(monkeypatch):
    robot = FakeRobot()
    ocr_calls: list[object] = []
    config = {"_runtime_cache": {"screen_anchors": {"bottom_message": (0.66, 0.94)}}}

    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps, "ensure_list_screen", lambda robot, config, reporter=None: b"before")
    monkeypatch.setattr(
        huanyou_app.steps,
        "capture_after_popups",
        lambda robot, config, label="screen", max_dismiss=2, reporter=None: b"after_cached_click",
    )
    monkeypatch.setattr(huanyou_app.steps, "_matches_chat_list_screen", lambda image, config, reporter=None: True)

    def fake_run_ocr_mode(image, config, mode, region=None):
        ocr_calls.append(image)
        return {
            "result": {
                "message_list": [
                    {
                        "name": "缓存用户",
                        "favorability": 8,
                        "unread_count": 1,
                        "content": "你好",
                        "bounds": [0.2, 0.3, 0.8, 0.4],
                    }
                ]
            }
        }

    monkeypatch.setattr(huanyou_app.steps, "run_ocr_mode", fake_run_ocr_mode)

    target = huanyou_app.steps.find_unread_target(robot, config)

    assert target is not None
    assert target["name"] == "缓存用户"
    assert ocr_calls == [b"after_cached_click"]
    assert robot.clicks == [(0.66, 0.94), (0.66, 0.94)]


def test_recover_by_cached_bottom_message_prefers_message_tab(monkeypatch):
    robot = FakeRobot()
    config = {"_runtime_cache": {"ocr_regions": {"bottom_nav_region": [0.0, 0.9, 1.0, 0.1]}}}

    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps, "_raise_if_abnormal_screen", lambda image, config: None)
    monkeypatch.setattr(huanyou_app.steps, "_matches_chat_list_screen", lambda image, config, reporter=None: True)
    monkeypatch.setattr(
        huanyou_app.steps,
        "capture_after_popups",
        lambda robot, config, label="screen", max_dismiss=2, reporter=None: b"list",
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "run_ocr_mode",
        lambda image, config, mode, region=None: {
            "result": {"bottom_nav": [{"name": "消息", "bounds": [0.6, 0.9, 0.8, 1.0]}]}
        },
    )

    image = huanyou_app.steps._recover_by_cached_bottom_message(robot, config, b"unknown")

    assert image == b"list"
    assert robot.clicks == [(0.7, 0.95)]


def test_next_message_with_new_greeting_skips_name_ocr(monkeypatch):
    robot = FakeRobot()
    calls: list[str] = []
    reports: list[str] = []

    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps, "click_next_message", lambda *args, **kwargs: calls.append("next") or True)
    monkeypatch.setattr(huanyou_app.steps, "confirm_chat_window", lambda *args, **kwargs: b"chat")
    monkeypatch.setattr(huanyou_app.steps, "_template_center_in_region", lambda *args, **kwargs: (0.2, 0.8))
    monkeypatch.setattr(
        huanyou_app.steps,
        "read_chat_contact_identity",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("name OCR should be skipped")),
    )

    image, name, clicks, name_point = huanyou_app.steps.click_next_message_until_changed(
        robot,
        {},
        expected_name="旧用户",
        image=b"start",
        reporter=lambda step, label, log_label=None: reports.append(step),
    )

    assert image == b"chat"
    assert name == ""
    assert clicks == 1
    assert name_point is None
    assert calls == ["next"]
    assert "next_chat_new_greeting_check" in reports
    assert "next_chat_new_greeting_fast_path" in reports


def test_next_message_button_uses_dedicated_low_match_threshold(monkeypatch, tmp_path):
    robot = FakeRobot()
    thresholds: list[float] = []
    template = tmp_path / "btn_下一条按钮.jpg"
    template.write_bytes(b"template")

    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps, "_template_candidates", lambda config, key, group: [template])

    def fake_match_template_with_region(image, template, config, key, *, threshold, match_func):
        thresholds.append(float(threshold))
        return {"x": 0.8, "y": 0.4, "w": 0.1, "h": 0.1}

    monkeypatch.setattr(huanyou_app.steps, "match_template_with_region", fake_match_template_with_region)

    clicked = huanyou_app.steps.click_next_message(robot, {}, image=b"chat")

    assert clicked is True
    assert thresholds == [0.7]
    assert robot.clicks == [(0.85, 0.45)]


def test_next_message_checks_new_greeting_before_name_ocr(monkeypatch):
    robot = FakeRobot()
    reports: list[str] = []

    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps, "click_next_message", lambda *args, **kwargs: True)
    monkeypatch.setattr(huanyou_app.steps, "confirm_chat_window", lambda *args, **kwargs: b"chat")
    monkeypatch.setattr(huanyou_app.steps, "_template_center_in_region", lambda *args, **kwargs: None)

    def fake_read_chat_contact_identity(robot, config, image=None, reporter=None):
        reports.append("name_ocr")
        return "新用户", (0.5, 0.06)

    monkeypatch.setattr(huanyou_app.steps, "read_chat_contact_identity", fake_read_chat_contact_identity)

    image, name, clicks, name_point = huanyou_app.steps.click_next_message_until_changed(
        robot,
        {},
        expected_name="旧用户",
        image=b"start",
        reporter=lambda step, label, log_label=None: reports.append(step),
    )

    assert image == b"chat"
    assert name == "新用户"
    assert clicks == 1
    assert name_point == (0.5, 0.06)
    assert reports.index("next_chat_new_greeting_check") < reports.index("name_ocr")
    assert reports.index("next_chat_new_greeting_check_not_found") < reports.index("name_ocr")


def test_read_chat_contact_identity_falls_back_from_small_name_region(monkeypatch):
    robot = FakeRobot()
    regions: list[object] = []

    def fake_run_ocr_mode(image, config, mode, region=None):
        regions.append(region)
        if len(regions) == 1:
            return {"ocr_items": [], "result": {"user_name": "", "dialogues": []}}
        return {
            "ocr_items": [{"text": "青松", "x": 0.45, "y": 0.04, "w": 0.1, "h": 0.03}],
            "result": {"user_name": "青松", "dialogues": []},
        }

    monkeypatch.setattr(huanyou_app.steps, "run_ocr_mode", fake_run_ocr_mode)

    name, point = huanyou_app.steps.read_chat_contact_identity(robot, {}, image=b"chat")

    assert name == "青松"
    assert point == (0.5, 0.055)
    assert len(regions) == 2
    assert regions[0] != regions[1]


def test_read_message_falls_back_from_small_message_region(monkeypatch):
    robot = FakeRobot()
    regions: list[object] = []
    config = {"_runtime_cache": {"current_chat": {"input_box_y": 0.85}}}

    def fake_run_ocr_mode(image, config, mode, region=None):
        regions.append(region)
        if len(regions) == 1:
            return {
                "ocr_items": [{"text": "自己"}],
                "result": {"user_name": "用户", "dialogues": [{"sender": "sender2", "content": "自己回复"}]},
            }
        return {
            "ocr_items": [{"text": "对方"}],
            "result": {"user_name": "用户", "dialogues": [{"sender": "sender1", "content": "对方消息"}]},
        }

    monkeypatch.setattr(huanyou_app.steps, "run_ocr_mode", fake_run_ocr_mode)

    message = huanyou_app.steps.read_message(robot, config, {"name": "用户"}, image=b"chat")

    assert message["content"] == "对方消息"
    assert len(regions) == 2
    assert regions[0] != regions[1]
