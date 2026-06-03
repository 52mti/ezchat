from __future__ import annotations

from pathlib import Path

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
            return image or b"next_image", "下一个", 1
        raise huanyou_app.steps.StepUnavailable("next message button was not found")

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

    assert robot.clicks == [(0.2, 0.3), (0.5, 0.3)]
    assert len(robot.swipes) == 2
    start_x, start_y, end_x, end_y = robot.swipes[0]
    assert start_x == end_x
    assert start_x == 0.5
    assert start_y == 0.8
    assert abs(end_y - 0.2) < 1e-9
    assert any(event.get("current_step") == "enter_chat" for event in events)
    assert any("(0.200, 0.300)" in str(event.get("current_step_log_label", "")) for event in events)
    assert any(
        message == "dialogue_ocr_last_message" and payload.get("content") == "最近在忙什么"
        for _level, message, payload in logs
    )


    assert go_back_click_counts == [2]


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
        lambda robot, config, reporter=None: ((0.5, 0.8), (0.5, 0.2)),
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
        lambda robot, config, expected_name=None, image=None, reporter=None: (image or b"next_image", "", 1),
    )
    monkeypatch.setattr(huanyou_app.steps, "go_back", lambda robot, config, reporter=None, **kwargs: go_back_calls.append(1))

    app = huanyou_app.HuanyouApp(robot, {}, reply_provider=lambda message: "reply")

    app.run_once()

    assert len(read_targets) == 2
    assert go_back_calls == []
    assert app.status.last_error == "stop after verifying continued chat"
    assert robot.clicks == [(0.2, 0.3), (0.5, 0.3)]


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
        lambda robot, config, reporter=None: ((0.5, 0.8), (0.5, 0.2)),
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
        lambda robot, config, expected_name=None, image=None, reporter=None: (b"stale_before_retract", "second", 1),
    )

    app = huanyou_app.HuanyouApp(robot, {}, reply_provider=lambda message: "reply")

    app.run_once()

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
        lambda robot, config, expected_name=None, image=None, reporter=None: (image or b"next_image", "other", 1),
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

    monkeypatch.setattr(huanyou_app.steps, "_template_center", lambda image, config, key, group: (0.4, 0.5))
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
        assert str(exc) == "microphone button was not found"
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
        "read_chat_contact_name",
        lambda robot, config, image=None, reporter=None: next(names),
    )

    image, name, clicks = huanyou_app.steps.click_next_message_until_changed(
        robot,
        {},
        expected_name="用户",
        image=b"start_image",
        retry_limit=2,
    )

    assert image == b"next_image"
    assert name == "另一个"
    assert clicks == 2
    assert robot.clicks == [(0.4, 0.5), (0.4, 0.5)]


def test_huanyou_keep_screen_awake_clicks_message_twice(monkeypatch):
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
    assert robot.clicks == [(0.6965, 0.936), (0.6965, 0.936)]


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


def test_huanyou_app_scrolls_after_consuming_new_greetings(monkeypatch):
    robot = FakeRobot()

    monkeypatch.setattr(huanyou_app.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(huanyou_app.steps.random, "uniform", lambda a, b: 0.0)

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
        lambda robot, config, image=None, expected_name=None, reporter=None: (image or b"image", 1, False),
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

    def fake_next_message(robot, config, expected_name=None, image=None, reporter=None):
        next_calls.append(1)
        if len(next_calls) == 1:
            return image or b"next_image", "下一个", 1
        raise huanyou_app.steps.StepUnavailable("next message button was not found")

    monkeypatch.setattr(huanyou_app.steps, "click_next_message_until_changed", fake_next_message)
    monkeypatch.setattr(huanyou_app.steps, "go_back", lambda robot, config, reporter=None, **kwargs: None)

    app = huanyou_app.HuanyouApp(robot, {}, reply_provider=lambda message: "回复")
    app.run_once()

    assert robot.clicks == [(0.2, 0.3), (0.5, 0.3)]
    assert len(robot.swipes) == 2


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
        lambda robot, config, target=None, image=None, reporter=None: (_ for _ in ()).throw(
            huanyou_app.steps.StepUnavailable("dialogue OCR did not return a sender1 message")
        ),
    )

    app = huanyou_app.HuanyouApp(robot, {}, reply_provider=lambda message: "reply")
    app.set_log_callback(lambda level, message, payload=None: logs.append((level, message, payload or {})))

    app.run_once()

    assert robot.reset_count == 0
    assert app.status.last_error == "dialogue OCR did not return a sender1 message"
    assert any(
        message.startswith("huanyou_step_unavailable:")
        and payload.get("error") == "dialogue OCR did not return a sender1 message"
        for _level, message, payload in logs
    )


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
    assert seen_modes == [(b"list_image", "list"), (b"my_image", "points"), (b"my_image", "list")]
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
        "capture_after_popups",
        lambda robot, config, label="screen", max_dismiss=2, reporter=None: calls.append(label) or b"after_popup",
    )
    monkeypatch.setattr(
        huanyou_app.steps,
        "run_ocr_mode",
        lambda image, config, mode, region=None: {
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
    assert calls == ["idle_list_rescan"]


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
                    {"sender": "sender1", "content": "????????????"},
                    {"sender": "sender2", "content": "????"},
                    {"sender": "sender1", "content": "??"},
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
    assert message["content"] == "??"
