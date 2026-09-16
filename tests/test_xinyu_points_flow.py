from __future__ import annotations

from pathlib import Path

import pytest

from apps.xinyu import app as xinyu_app
from apps.xinyu import steps


CHAT_LIST_SCREEN_KEY = "screen.chat_list"
CHAT_LIST_TITLE_REGION = (0.0, 0.023, 0.489, 0.127)


SCREENSHOT_MARKERS = (
    "list", "chat", "my", "unknown", "no_phone", "microphone", "black",
    "after_greeting", "after", "still_chat", "popup", "list_confirm",
    "list_by_ocr", "call", "android_home", "after_unread",
)


@pytest.fixture(autouse=True)
def _chat_list_title_ocr_stub(monkeypatch):
    """聊天列表判定 stub（对齐 ce27647 后的产品行为）。

    产品代码（_matches_chat_list_screen）已改为"标题区域内 OCR 识别
    消息/通话"：region 由 template_match_region 解析、OCR 走
    _run_ocr_mode。测试 config 的 template_match_regions=None 不能动
    （模板匹配路径依赖它为 None 直通 mock 的 match），因此这里单独
    stub 三个符号：

    - template_match_region：仅聊天列表键返回生产默认标题区域，其余 None；
    - _run_ocr_mode：标题判定场景下，截图标记 "list" 返回"消息/通话"
      命中（等价旧 fixture 的 screen_聊天列表.jpg 模板命中语义），其余
      截图返回空结果；非标题场景在运行时解析 steps.run_ocr_mode，
      让测试体内的显式 monkeypatch 优先生效；
    - _ocr_reuse_signature：签名改为基于截图标记的稳定值。原实现对
      任意截图都会真实解码图片（ensure_portrait_image），字符串截图
      fixture 过不了解码这一步。

    测试体内的显式 monkeypatch 优先于本 autouse stub（fixture 先应用，
    测试体后覆盖）。
    """
    original_region = steps.template_match_region

    def fake_region(config, key):
        if key == CHAT_LIST_SCREEN_KEY:
            return CHAT_LIST_TITLE_REGION
        return original_region(config, key)

    monkeypatch.setattr(steps, "template_match_region", fake_region)

    # 默认模板匹配 mock：无命中。产品路径里弹窗扫描/屏幕判定都会真调
    # match，真 matcher 会解码字符串截图 fixture 报"图片读取失败"，并被
    # run_once 吞成静默提前退出；默认拦截为"全不命中"，测试体内显式的
    # fake_match monkeypatch 后应用、优先于本默认。
    monkeypatch.setattr(steps, "match", lambda image, template, threshold=0.85: None)

    def fake_signature(image, region=None):
        import numpy as np

        return (f"{image}|{region}", np.zeros((4, 4), dtype=np.uint8))

    monkeypatch.setattr(steps, "_ocr_reuse_signature", fake_signature)

    original_run_ocr_mode = steps._run_ocr_mode

    def fake_run_ocr_mode(image, config, mode, region=None, **kwargs):
        if mode == "list" and region == CHAT_LIST_TITLE_REGION:
            if str(image) == "list":
                return {
                    "mode": mode,
                    "items": [],
                    "ocr_items": [
                        {"text": "消息", "x": 0.10, "y": 0.03, "w": 0.10, "h": 0.04},
                        {"text": "通话", "x": 0.30, "y": 0.03, "w": 0.10, "h": 0.04},
                    ],
                    "result": {},
                }
            return {"mode": mode, "items": [], "ocr_items": [], "result": {}}
        try:
            # 委托原实现：保留其 OCR 结果复用逻辑（_reused_ocr_result 等）。
            # 原实现内部经模块命名空间调 run_ocr_mode，测试体的 mock 优先生效；
            # 签名计算已被 fake_signature 拦截，字符串截图不会再解码失败，
            # 兜底捕获以防万一。
            return original_run_ocr_mode(image, config, mode, region=region)
        except (RuntimeError, OSError):
            return {"mode": mode, "items": [], "ocr_items": [], "result": {}}

    monkeypatch.setattr(steps, "_run_ocr_mode", fake_run_ocr_mode)
    yield


class FakeRobot:
    def __init__(self, images: list[object]) -> None:
        self.images = list(images)
        self.clicks: list[tuple[float, float]] = []

    def get_transform_image(self, pre_move: bool = True, save_path=None):
        return self.images.pop(0) if self.images else "list"

    def move_click(self, x: float, y: float) -> None:
        self.clicks.append((x, y))


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"template")


def _config(tmp_path: Path) -> dict[str, object]:
    buttons_dir = tmp_path / "buttons"
    screens_dir = tmp_path / "screens"
    for path in [
        buttons_dir / "btn_返回按钮.jpg",
        screens_dir / "screen_聊天列表.jpg",
        screens_dir / "screen_聊天界面.jpg",
        screens_dir / "screen_我的界面.jpg",
        screens_dir / "screen_黑屏异常.jpg",
    ]:
        _touch(path)
    return {
        "templates": {"buttons": str(buttons_dir), "screens": str(screens_dir)},
        "step_timeout": 0.2,
        "template_match_regions": None,
    }


def test_xinyu_read_my_points_clicks_my_reads_points_and_returns_message(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list", "my", "list"])
    seen_modes: list[str] = []

    def fake_match(image, template, threshold=0.85):
        name = Path(template).name
        if image == "list" and name == "screen_聊天列表.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        if image == "my" and name == "screen_我的界面.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    def fake_ocr(image, cfg, mode, region=None):
        seen_modes.append(mode)
        if mode == "list":
            return {
                "result": {
                    "bottom_nav": [
                        {"name": "消息", "bounds": [0.56, 0.86, 0.68, 0.94]},
                        {"name": "我的", "bounds": [0.78, 0.86, 0.90, 0.94]},
                    ]
                },
                "ocr_items": [],
            }
        return {"result": {"points": "1288", "balance": "1288"}, "ocr_items": []}

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps, "run_ocr_mode", fake_ocr)
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)

    assert steps.read_my_points(robot, config) == "1288"
    assert seen_modes == ["list", "points"]
    assert robot.clicks == [pytest.approx((0.84, 0.88)), pytest.approx((0.62, 0.88))]


def test_xinyu_read_my_points_reuses_cached_my_coordinate(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["_runtime_cache"] = {"screen_anchors": {"bottom_my": (0.91, 0.93), "bottom_message": (0.62, 0.9)}}
    robot = FakeRobot(["list", "list", "my", "list"])

    def fake_match(image, template, threshold=0.85):
        name = Path(template).name
        if image == "list" and name == "screen_聊天列表.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        if image == "my" and name == "screen_我的界面.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps, "run_ocr_mode", lambda image, cfg, mode, region=None: {"result": {"points": "100"}, "ocr_items": []})
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)

    assert steps.read_my_points(robot, config) == "100"
    assert config["_runtime_cache"]["screen_anchors"]["bottom_my"] == (0.91, 0.93)
    assert robot.clicks[0] == pytest.approx((0.91, 0.93))


def test_xinyu_read_my_points_uses_cached_my_without_list_ocr(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["_runtime_cache"] = {"screen_anchors": {"bottom_my": (0.91, 0.93)}}
    robot = FakeRobot(["list", "my", "list"])
    seen_modes: list[str] = []

    def fake_match(image, template, threshold=0.85):
        name = Path(template).name
        if image == "list" and name == "screen_聊天列表.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        if image == "my" and name == "screen_我的界面.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    def fake_ocr(image, cfg, mode, region=None):
        seen_modes.append(mode)
        if mode == "list":
            raise AssertionError("cached bottom_my should avoid list OCR before reading points")
        return {"result": {"points": "100"}, "ocr_items": []}

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps, "run_ocr_mode", fake_ocr)
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)

    assert steps.read_my_points(robot, config) == "100"
    assert seen_modes == ["points"]
    assert robot.clicks == [pytest.approx((0.91, 0.93)), pytest.approx((0.71, 0.93))]


def test_xinyu_read_my_points_skips_when_bottom_nav_has_no_my(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list", "list"])
    reports: list[tuple[str, str, str | None]] = []

    def fake_match(image, template, threshold=0.85):
        if image == "list" and Path(template).name == "screen_聊天列表.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {"bottom_nav": [{"name": "主页", "bounds": [0.56, 0.86, 0.68, 0.94]}]},
            "ocr_items": [],
        },
    )

    assert steps.read_my_points(
        robot,
        config,
        reporter=lambda step, label, log_label=None: reports.append((step, label, log_label)),
    ) is None
    assert robot.clicks == []
    assert any(label == "底部导航未识别到我的，跳过积分读取" for _step, label, _log_label in reports)


def test_xinyu_read_my_points_uses_cached_my_without_message_ocr(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list", "my", "list"])

    def fake_match(image, template, threshold=0.85):
        name = Path(template).name
        if image == "list" and name == "screen_聊天列表.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        if image == "my" and name == "screen_我的界面.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: (
            {
                "result": {"bottom_nav": [{"name": "我的", "bounds": [0.78, 0.86, 0.90, 0.94]}]},
                "ocr_items": [],
            }
            if mode == "list"
            else {"result": {"points": "100"}, "ocr_items": []}
        ),
    )
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)

    assert steps.read_my_points(robot, config) == "100"
    assert robot.clicks == [pytest.approx((0.84, 0.88)), pytest.approx((0.64, 0.88))]


def test_xinyu_app_reads_points_once_per_idle_wait_and_reports_balance(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["keep_awake_interval_seconds"] = 30
    robot = FakeRobot(["list"])
    calls: list[str] = []
    logs: list[tuple[str, dict[str, object] | None]] = []

    monkeypatch.setattr(xinyu_app.steps, "find_unread_target", lambda robot, config, reporter=None: None)
    monkeypatch.setattr(
        xinyu_app.steps,
        "keep_screen_awake",
        lambda robot, config, reporter=None: calls.append("keep_awake") or True,
    )
    monkeypatch.setattr(
        xinyu_app.steps,
        "read_my_points",
        lambda robot, config, reporter=None: calls.append("points") or "1288",
    )

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI")
    app.set_log_callback(lambda level, message, payload=None: logs.append((message, payload)))
    app.last_activity_at -= 31

    app.run_once()
    app.run_once()

    assert calls == ["keep_awake", "points"]
    assert ("xinyu_balance", {"balance": "128", "points": "1288"}) in logs
    assert ("心遇本轮空闲已读取余额，跳过重复读取", {}) in logs


def test_xinyu_balance_display_truncates_decimals_after_dividing_by_ten():
    assert xinyu_app.XinyuApp._balance_from_points("1013.25") == "101"


def test_xinyu_app_reads_points_again_after_real_message_resets_idle_wait(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["keep_awake_interval_seconds"] = 0
    robot = FakeRobot(["list"])
    calls: list[str] = []
    targets = [
        None,
        {"name": "用户", "content": "哦哦", "favorability": 1, "unread_count": 1, "bounds": [0.1, 0.2, 0.3, 0.4], "x": 0.2, "y": 0.3},
        None,
    ]

    monkeypatch.setattr(xinyu_app.steps, "find_unread_target", lambda robot, config, reporter=None: targets.pop(0))
    monkeypatch.setattr(xinyu_app.steps, "read_my_points", lambda robot, config, reporter=None: calls.append("points") or "100")
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: calls.append("enter_chat"))
    monkeypatch.setattr(xinyu_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: "chat")
    monkeypatch.setattr(xinyu_app.steps, "build_list_message", lambda target, content: {"contact": target["name"], "content": content})
    monkeypatch.setattr(xinyu_app.steps, "focus_input_box", lambda robot, config, reporter=None: None)
    monkeypatch.setattr(xinyu_app.steps, "play_voice_reply", lambda robot, text, config, reporter=None: None)
    monkeypatch.setattr(xinyu_app.steps, "send_reply", lambda robot, config, reporter=None: 1)
    monkeypatch.setattr(xinyu_app.steps, "go_back", lambda robot, config, reporter=None, double_click_on_first_attempt=False: 0)

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI")
    app.run_once()
    app.run_once()
    app.run_once()

    assert calls == ["points", "enter_chat", "points"]
