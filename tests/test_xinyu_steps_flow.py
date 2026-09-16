from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from apps.xinyu import app as xinyu_app
from apps.xinyu import steps
from services.rule_learning import RuleLearningStore


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
        self.played: list[str] = []

    def get_transform_image(self, pre_move: bool = True, save_path=None):
        return self.images.pop(0) if self.images else "list"

    def move_click(self, x: float, y: float) -> None:
        self.clicks.append((x, y))

    def play_audio(self, chunk: str, wait: bool = True) -> None:
        self.played.append(chunk)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"template")


def _config(tmp_path: Path) -> dict[str, object]:
    buttons_dir = tmp_path / "buttons"
    screens_dir = tmp_path / "screens"
    for path in [
        buttons_dir / "btn_返回按钮.jpg",
        buttons_dir / "btn_返回按钮_2.jpg",
        buttons_dir / "btn_未读消息按钮.jpg",
        buttons_dir / "btn_输入框.jpg",
        buttons_dir / "btn_麦克风按钮.jpg",
        buttons_dir / "btn_发送按钮.jpg",
        buttons_dir / "btn_新招呼按钮.jpg",
        buttons_dir / "popup_custom_1.jpg",
        screens_dir / "screen_聊天列表.jpg",
        screens_dir / "screen_聊天界面.jpg",
        screens_dir / "screen_我的界面.jpg",
        screens_dir / "screen_黑屏异常.jpg",
        screens_dir / "screen_无手机界面.jpg",
    ]:
        _touch(path)
    return {
        "templates": {"buttons": str(buttons_dir), "screens": str(screens_dir)},
        "step_timeout": 0.2,
        "greetings_dir": str(tmp_path / "greetings"),
        "template_match_regions": None,
    }


def test_xinyu_learned_system_rules_skip_covered_text_and_shorten_new_text(monkeypatch, tmp_path):
    monkeypatch.setattr(steps, "RuleLearningStore", lambda: RuleLearningStore(tmp_path / "rule-learning"))
    rules = {"line_excluded_keywords": ["金币", "亲密度"]}
    config = {"ocr_modes": {"dialogue": {"rules": rules}}}
    payload = {
        "excluded_messages": [
            {
                "content": "为表诚意，对方发消息需消耗金币。你回复越快获得积分越多。",
                "reason": "平台提示",
            },
            {
                "content": "平台温馨提示请完善资料后继续使用。",
                "reason": "平台提示",
            },
            {"content": "亲密度37", "reason": "等级"},
        ]
    }

    learned = steps._learn_llm_system_messages(config, payload)

    assert learned == 1
    assert rules["line_excluded_keywords"] == ["金币", "亲密度", "平台温馨提示"]


def test_xinyu_chat_window_screen_uses_lower_match_threshold(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["match_threshold"] = 0.85
    config["template_match_thresholds"] = {"screen.chat_window": 0.7}
    thresholds: dict[str, float] = {}

    def fake_match_template_with_region(image, template, config, key, *, threshold, match_func):
        thresholds[key] = threshold
        return None

    monkeypatch.setattr(steps, "match_template_with_region", fake_match_template_with_region)

    steps._matches_screen("image", config, steps.CHAT_WINDOW_SCREEN)
    steps._matches_screen("image", config, steps.CHAT_LIST_SCREEN)

    assert thresholds[steps.CHAT_WINDOW_SCREEN] == 0.7
    assert thresholds[steps.CHAT_LIST_SCREEN] == 0.85


def test_xinyu_complete_button_ocr_fallback_uses_configured_range(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["template_match_regions"] = {"button.complete": [0.0, 0.0, 1.0, 0.35]}
    seen: dict[str, object] = {}

    def fake_run_ocr_mode(image, cfg, mode, region=None):
        seen["image"] = image
        seen["mode"] = mode
        seen["region"] = region
        return {
            "ocr_items": [
                {"text": "完成", "x": 0.76, "y": 0.08, "w": 0.12, "h": 0.06},
            ]
        }

    monkeypatch.setattr(steps, "_run_ocr_mode", fake_run_ocr_mode)

    assert steps._complete_button_ocr_center("album", config) == pytest.approx((0.82, 0.11))
    assert seen == {"image": "album", "mode": "list", "region": [0.0, 0.0, 1.0, 0.35]}


def test_xinyu_online_photo_uses_complete_button_ocr_after_template_retries(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config.update(
        {
            "template_match_regions": {"button.complete": [0.0, 0.0, 1.0, 0.35]},
            "photo_select_max_attempts": 2,
            "photo_select_random_bounds": [0.3, 0.2, 0.9, 0.5],
        }
    )
    robot = FakeRobot([])
    reports: list[str] = []
    list_ocr_results = iter(
        [
            {"result": {"bottom_nav": [{"name": "在线", "bounds": [0.5, 0.1, 0.6, 0.2]}], "message_list": []}},
            {"result": {"bottom_nav": [{"name": "在线", "bounds": [0.5, 0.1, 0.6, 0.2]}], "message_list": [{"name": "测试用户", "bounds": [0.2, 0.3, 0.8, 0.4]}]}},
        ]
    )

    monkeypatch.setattr(steps, "ensure_list_screen", lambda robot, config, reporter=None: "list")
    monkeypatch.setattr(steps, "_run_list_ocr", lambda image, config, reporter=None: next(list_ocr_results))
    monkeypatch.setattr(steps, "_red_pixel_stats", lambda image, result: {"has_red": True})
    monkeypatch.setattr(steps, "confirm_chat_window", lambda robot, config, reporter=None: "chat")
    monkeypatch.setattr(
        steps,
        "_template_center",
        lambda image, config, key, group: (0.1, 0.9) if key == steps.PHOTO_BUTTON else None,
    )
    monkeypatch.setattr(steps, "capture_after_popups", lambda robot, config, label="screen", reporter=None, **kwargs: "album")
    monkeypatch.setattr(
        steps,
        "_run_ocr_mode",
        lambda image, config, mode, region=None: {
            "ocr_items": [{"text": "发送", "x": 0.75, "y": 0.08, "w": 0.1, "h": 0.05}]
        },
    )
    monkeypatch.setattr(steps, "go_back", lambda *args, **kwargs: 1)
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)

    assert steps.send_photo_to_online_target(robot, config, reporter=lambda _step, _label, log_label=None: reports.append(log_label or ""))
    assert robot.clicks[-1] == pytest.approx((0.8, 0.105))
    assert any("OCR 识别到发送或完成" in report for report in reports)


def test_xinyu_list_ocr_treats_left_badge_as_default_unread(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])

    def fake_match(image, template, threshold=0.85):
        if image == "list" and Path(template).name == "screen_聊天列表.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps, "_red_pixel_stats", lambda image, match_result: {"has_red": True, "red_pixels": 8, "total_pixels": 100, "threshold": 3, "region": [1, 2, 3, 4]})
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "bottom_nav": [{"name": "未回复", "bounds": [0.329, 0.125, 0.47, 0.153]}],
                "message_list": [
                    {"name": "白嫖", "favorability": 44, "unread_count": 1, "content": "", "bounds": [0.1, 0.2, 0.4, 0.3]},
                    {"name": "张三", "favorability": 12, "unread_count": 1, "content": "你好", "bounds": [0.2, 0.4, 0.5, 0.5]},
                ]
            },
            "ocr_items": [],
        },
    )

    target = steps.find_unread_target(robot, config)

    assert target["name"] == "白嫖"
    assert target["favorability"] == 44
    assert target["unread_count"] == 1


def test_xinyu_reuses_same_screenshot_ocr_and_template_detection(monkeypatch, tmp_path):
    config = _config(tmp_path)
    image = Image.new("RGB", (120, 240), "white")
    ocr_calls = 0
    match_calls = 0
    reports: list[tuple[str, str]] = []

    def fake_ocr(image, cfg, mode, region=None):
        nonlocal ocr_calls
        ocr_calls += 1
        return {"result": {"message_list": [], "bottom_nav": []}, "ocr_items": []}

    def fake_match_template_with_region(image, template, config, key, *, threshold, match_func):
        nonlocal match_calls
        match_calls += 1
        return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}

    monkeypatch.setattr(steps, "run_ocr_mode", fake_ocr)
    monkeypatch.setattr(steps, "match_template_with_region", fake_match_template_with_region)

    reporter = lambda step, label, log_label=None: reports.append((step, label))
    steps._run_list_ocr(image, config, reporter=reporter)
    steps._run_list_ocr(image, config, reporter=reporter)
    assert ocr_calls == 1
    assert ("list_ocr_scan", "复用上一轮 list OCR 结果") in reports

    assert steps._matches_screen(image, config, steps.CHAT_LIST_SCREEN)
    assert steps._matches_screen(image, config, steps.CHAT_LIST_SCREEN)
    # 首轮 = warm 探测（2 个返回按钮候选）+ 真匹配 1 次；第二轮全复用不再增长。
    assert match_calls == 3


def test_xinyu_dialogue_region_filter_is_forced_off(tmp_path):
    config = _config(tmp_path)
    config["ocr_modes"] = {
        "dialogue": {
            "rules": {
                "basic_filter": {
                    "enable_center_y_region_filter": True,
                    "center_y_min_ratio": 0.18,
                    "center_y_max_ratio": 0.88,
                }
            }
        }
    }

    resolved = steps._config(config)

    assert resolved["ocr_modes"]["dialogue"]["rules"]["basic_filter"]["enable_center_y_region_filter"] is False


def test_xinyu_app_dialogue_region_filter_is_forced_off(tmp_path):
    config = _config(tmp_path)
    config["ocr_modes"] = {
        "dialogue": {
            "rules": {
                "basic_filter": {
                    "enable_center_y_region_filter": True,
                    "center_y_min_ratio": 0.18,
                    "center_y_max_ratio": 0.88,
                }
            }
        }
    }

    app = xinyu_app.XinyuApp(FakeRobot([]), config, reply_provider=lambda message: "AI回复")

    assert app.config["ocr_modes"]["dialogue"]["rules"]["basic_filter"]["enable_center_y_region_filter"] is False


def test_xinyu_ensure_list_screen_raises_on_black_screen(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["enable_black_screen_detection"] = True
    robot = FakeRobot(["black"])

    def fake_match(image, template, threshold=0.85):
        if image == "black" and Path(template).name == "screen_黑屏异常.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)

    with pytest.raises(steps.BlackScreenError):
        steps.ensure_list_screen(robot, config)


def test_xinyu_ensure_list_screen_raises_on_pixel_black_screen(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["enable_black_screen_detection"] = True
    black_image = Image.new("RGB", (200, 400), (3, 3, 3))
    robot = FakeRobot([black_image])

    monkeypatch.setattr(steps, "match", lambda image, template, threshold=0.85: None)

    with pytest.raises(steps.BlackScreenError):
        steps.ensure_list_screen(robot, config)
    assert robot.clicks == []


def test_xinyu_ensure_list_screen_raises_on_no_phone_screen(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["no_phone"])

    def fake_match(image, template, threshold=0.85):
        if image == "no_phone" and Path(template).name == "screen_无手机界面.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)

    with pytest.raises(steps.NoPhoneScreenError):
        steps.ensure_list_screen(robot, config)
    assert robot.clicks == []


def test_xinyu_ensure_list_screen_prioritizes_returning_to_list_before_popup_scan(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["chat", "list"])
    capture_labels: list[str] = []

    def fake_capture(robot, config, pre_move=True, label="screen", **kwargs):
        capture_labels.append(label)
        return robot.get_transform_image()

    monkeypatch.setattr(steps, "capture_transform_image", fake_capture)
    monkeypatch.setattr(
        steps,
        "capture_after_popups",
        lambda robot, config, label="screen", max_dismiss=2, reporter=None: (
            capture_labels.append(label) or robot.get_transform_image()
        ),
    )
    monkeypatch.setattr(
        steps,
        "_matches_screen",
        lambda image, config, key: (image == "chat" and key == steps.CHAT_WINDOW_SCREEN)
        or (image == "list" and key == steps.CHAT_LIST_SCREEN),
    )
    monkeypatch.setattr(
        steps,
        "_template_center",
        lambda image, config, key, group="buttons": (0.15, 0.15)
        if image == "chat" and key == steps.BACK_BUTTON and group == "buttons"
        else None,
    )

    image = steps.ensure_list_screen(robot, config)

    assert image == "list"
    assert capture_labels[0] == "startup_screen_detect"
    assert "confirm_list" not in capture_labels
    assert robot.clicks == [pytest.approx((0.15, 0.15))]


def test_xinyu_ensure_list_screen_accepts_title_ocr_when_template_misses(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["template_match_regions"] = {
        steps.CHAT_LIST_SCREEN: [0.0, 0.0, 1.0, 0.2],
    }
    robot = FakeRobot(["list", "list_confirm"])

    monkeypatch.setattr(
        steps,
        "capture_after_popups",
        lambda robot, cfg, label="screen", max_dismiss=2, image=None, reporter=None: (
            image if image is not None else robot.get_transform_image()
        ),
    )
    monkeypatch.setattr(
        steps,
        "match",
        lambda image, template, threshold=0.85: None,
    )
    monkeypatch.setattr(steps, "_matches_screen", lambda image, cfg, key: False)
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {"message_list": [], "bottom_nav": []},
            "ocr_items": [{"text": "消息"}, {"text": "通话"}],
        },
    )

    image = steps.ensure_list_screen(robot, config)

    assert image == "list"
    assert robot.clicks == []


def test_xinyu_ensure_list_screen_enters_app_from_android_home(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["android_home", "list"])

    monkeypatch.setattr(
        steps,
        "_template_center",
        lambda image, _config, key, group="buttons": (
            (0.5, 0.4)
            if image == "android_home" and key == steps.XINYU_APP_ENTRY_BUTTON
            else None
        ),
    )
    monkeypatch.setattr(steps, "_matches_chat_list_screen", lambda image, *_args, **_kwargs: image == "list")
    monkeypatch.setattr(steps, "_raise_if_black_screen", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(steps.time, "sleep", lambda _seconds: None)

    image = steps.ensure_list_screen(robot, config)

    assert image == "list"
    assert robot.clicks == [pytest.approx((0.5, 0.4))]


def test_xinyu_android_home_entry_uses_strict_match_settings():
    from apps.xinyu.config import DEFAULT_CONFIG

    key = "button.custom.主界面心遇按钮"
    assert DEFAULT_CONFIG["template_match_thresholds"][key] == pytest.approx(0.88)
    assert DEFAULT_CONFIG["template_match_channels_by_key"][key] == ["bgr"]
    assert DEFAULT_CONFIG["template_match_scales_by_key"][key] == [1.0]


def test_xinyu_ensure_list_screen_reports_back_clicked_but_not_returned(monkeypatch, tmp_path):
    config = _config(tmp_path)
    # 取图始终是聊天界面（含确认重试），保证"点了返回但从未回到列表"。
    robot = FakeRobot(["chat"] * 12)

    monkeypatch.setattr(steps, "capture_transform_image", lambda robot, cfg, pre_move=True, label="screen": robot.get_transform_image())
    monkeypatch.setattr(steps, "capture_after_popups", lambda robot, cfg, label="screen", reporter=None, **kwargs: robot.get_transform_image())
    monkeypatch.setattr(
        steps,
        "_matches_screen",
        lambda image, config, key: (image == "chat" and key == steps.CHAT_WINDOW_SCREEN)
        or (image == "still_chat" and key == steps.CHAT_WINDOW_SCREEN),
    )
    monkeypatch.setattr(
        steps,
        "_template_center",
        lambda image, config, key, group="buttons": (0.15, 0.15)
        if image == "chat" and key == steps.BACK_BUTTON and group == "buttons"
        else None,
    )
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)

    with pytest.raises(steps.StepUnavailable, match="已点击返回按钮，但未回到聊天列表"):
        steps.ensure_list_screen(robot, config)


def test_xinyu_ensure_list_screen_reports_unknown_screen_back_failure(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["unknown"])

    monkeypatch.setattr(
        steps,
        "capture_transform_image",
        lambda robot, cfg, pre_move=True, label="screen", **kwargs: "unknown",
    )
    monkeypatch.setattr(steps, "match", lambda image, template, threshold=0.85: None)

    assert steps.ensure_list_screen(robot, config) is None


def test_xinyu_unknown_screen_does_not_use_cached_message_anchor(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["_runtime_cache"] = {"screen_anchors": {"bottom_message": (0.7, 0.9)}}
    robot = FakeRobot(["unknown"])
    message_clicks: list[object] = []

    monkeypatch.setattr(steps, "capture_transform_image", lambda robot, cfg, pre_move=True, label="screen": robot.get_transform_image())
    monkeypatch.setattr(steps, "match", lambda image, template, threshold=0.85: None)
    monkeypatch.setattr(steps, "_run_ocr_mode", lambda image, cfg, mode, region=None: {"ocr_items": [], "result": {}})
    monkeypatch.setattr(steps, "_try_open_message_list", lambda *args, **kwargs: message_clicks.append(True) or False)

    assert steps.ensure_list_screen(robot, config) is None
    assert message_clicks == []


def test_xinyu_screen_detection_rules_recognize_main_tab_by_own_ocr_region(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["screen_detection_rules"] = [
        {
            "screen": "xinyu-缘分",
            "match_region": [0.0, 0.1, 0.5, 0.2],
            "condition": {
                "type": "ocr_raw",
                "region": [0.0, 0.1, 0.5, 0.2],
                "text": "推荐在线",
                "match_mode": "contains",
                "min_confidence": 0.75,
            },
        }
    ]
    seen: list[object] = []

    monkeypatch.setattr(
        steps,
        "_run_ocr_mode",
        lambda image, cfg, mode, region=None: seen.append(region)
        or {"ocr_items": [{"text": "推荐在线", "confidence": 0.9}]},
    )

    result = steps._screen_detection_rule_ocr_result("main", config)

    assert result is not None
    assert result["_xinyu_main_screen_rule"] == "xinyu-缘分"
    assert seen == [[0.0, 0.1, 0.5, 0.2]]


def test_xinyu_screen_detection_rule_combines_split_raw_ocr_text(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["screen_detection_rules"] = [
        {
            "screen": "xinyu-娱乐交友",
            "condition": {
                "type": "ocr_raw",
                "region": [0.0, 0.0, 0.5, 0.2],
                "text": "娱乐交友",
                "match_mode": "contains",
                "min_confidence": 0.75,
            },
        }
    ]
    monkeypatch.setattr(
        steps,
        "_run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "ocr_items": [
                {"text": "娱乐", "confidence": 0.91},
                {"text": "交友", "confidence": 0.92},
            ]
        },
    )

    result = steps._screen_detection_rule_ocr_result("main", config)

    assert result is not None
    assert result["_xinyu_main_screen_rule"] == "xinyu-娱乐交友"


def test_xinyu_waits_until_call_screen_variant_disappears(monkeypatch, tmp_path):
    config = _config(tmp_path)
    screens_dir = Path(config["templates"]["screens"])
    _touch(screens_dir / "screen_通话界面_2.jpg")
    config["template_match_regions"] = {
        "screen.call_window": [0.0, 0.7, 0.5, 0.3],
        "screen.chat_list": [0.0, 0.0, 1.0, 0.2],
    }
    config["call_screen_wait_poll_seconds"] = 0
    robot = FakeRobot(["call", "list", "list"])
    reports: list[tuple[str, str, str | None]] = []

    def fake_match_template_with_region(image, template, config, key, *, threshold, match_func):
        name = Path(template).name
        if image == "call" and name == "screen_通话界面_2.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        if image == "list" and name == "screen_聊天列表.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match_template_with_region", fake_match_template_with_region)
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {"ocr_items": [], "result": {"message_list": [], "bottom_nav": []}},
    )

    assert steps.ensure_list_screen(
        robot,
        config,
        reporter=lambda step, label, log_label=None: reports.append((step, label, log_label)),
    ) == "list"
    assert any(label == "检测到通话界面，等待通话结束" for _step, label, _log_label in reports)
    assert any(label == "通话界面已消失，继续流程" for _step, label, _log_label in reports)


def test_xinyu_app_black_screen_reports_waiting_without_circuit(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["black"])
    logs: list[tuple[str, str, dict | None]] = []

    monkeypatch.setattr(
        xinyu_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: (_ for _ in ()).throw(steps.BlackScreenError("black screen detected")),
    )

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI回复")
    app.set_log_callback(lambda level, message, payload=None: logs.append((level, message, payload)))
    app.run_once()

    assert app.status.is_circuit_open is False
    assert app.status.error_count == 0
    assert "black screen" in app.status.last_error
    assert app.status.current_step == "black_screen_waiting"
    assert app.status.current_step_label == "黑屏异常，等待用户亮屏"
    assert logs[-1][0] == "warning"
    assert logs[-1][1] == "检测到黑屏"


def test_xinyu_app_no_phone_reports_waiting_without_circuit(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["no_phone"])
    logs: list[tuple[str, str, dict | None]] = []

    monkeypatch.setattr(
        xinyu_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: (_ for _ in ()).throw(
            steps.NoPhoneScreenError("no phone detected")
        ),
    )

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI回复")
    app.set_log_callback(lambda level, message, payload=None: logs.append((level, message, payload)))
    app.run_once()

    assert app.status.is_circuit_open is False
    assert app.status.error_count == 0
    assert "no phone" in app.status.last_error
    assert app.status.current_step == "no_phone_waiting"
    assert app.status.current_step_label == "无手机异常，等待用户放回手机"
    assert logs[-1][0] == "warning"
    assert logs[-1][1] == "检测到无手机"


def test_xinyu_unread_red_pixel_detection_accepts_pink_red_badge():
    red_badge = Image.new("RGB", (28, 27), (90, 128, 165))
    for x in range(9, 19):
        for y in range(8, 18):
            red_badge.putpixel((x, y), (104, 54, 91))
    no_red = Image.new("RGB", (28, 27), (90, 128, 165))
    full_match = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}

    assert steps._has_red_pixels(red_badge, full_match) is True
    assert steps._has_red_pixels(no_red, full_match) is False


def test_xinyu_unread_red_pixel_detection_accepts_muted_screen_red_text():
    muted_red_text = Image.new("RGB", (80, 60), (130, 143, 151))
    for x in range(20, 60):
        for y in range(22, 34):
            muted_red_text.putpixel((x, y), (104, 86, 94))
    gray_text = Image.new("RGB", (80, 60), (130, 143, 151))
    for x in range(20, 60):
        for y in range(22, 34):
            gray_text.putpixel((x, y), (95, 100, 107))
    full_match = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}

    assert steps._has_red_pixels(muted_red_text, full_match) is True
    assert steps._has_red_pixels(gray_text, full_match) is False


def test_xinyu_unreplied_filter_clicks_when_no_red(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["after"])

    ocr_calls: list[object] = []

    def fake_ocr(image, cfg, mode, region=None):
        ocr_calls.append(image)
        return {
            "result": {
                "bottom_nav": [{"name": "未回复", "bounds": [0.2, 0.1, 0.4, 0.2]}],
                "message_list": [],
            },
            "ocr_items": [],
        }

    monkeypatch.setattr(steps, "run_ocr_mode", fake_ocr)
    monkeypatch.setattr(
        steps,
        "_red_pixel_stats",
        lambda image, match_result: {
            "has_red": image == "after_unread",
            "red_pixels": 8 if image == "after_unread" else 0,
            "total_pixels": 100,
            "threshold": 3,
            "region": [1, 2, 3, 4],
        },
    )
    monkeypatch.setattr(steps, "capture_after_popups", lambda robot, cfg, label="screen", reporter=None, **kwargs: "after")
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)

    image, _ocr_result = steps.check_unread_button(robot, config, image="list")

    assert image == "after"
    assert robot.clicks == [pytest.approx((0.3, 0.15))]
    assert ocr_calls == ["list", "after"]


def test_xinyu_unreplied_filter_does_not_click_when_red(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["after"])

    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "bottom_nav": [{"name": "未回复", "bounds": [0.2, 0.1, 0.4, 0.2]}],
                "message_list": [],
            },
            "ocr_items": [],
        },
    )
    monkeypatch.setattr(
        steps,
        "_red_pixel_stats",
        lambda image, match_result: {"has_red": True, "red_pixels": 8, "total_pixels": 100, "threshold": 3, "region": [1, 2, 3, 4]},
    )

    image, _ocr_result = steps.check_unread_button(robot, config, image="list")

    assert image == "list"
    assert robot.clicks == []
    assert config["_runtime_cache"]["ocr_regions"]["unread_tab_band"][0] == pytest.approx(0.0)
    assert config["_runtime_cache"]["ocr_regions"]["unread_tab_band"][2] == pytest.approx(1.0)


def test_xinyu_popup_dismiss_ignores_after_two_clicks(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["popup_1", "popup_2", "popup_3"])

    def fake_match(image, template, threshold=0.85):
        if Path(template).name == "popup_custom_1.jpg":
            return {"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.3, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)

    image = steps.capture_after_popups(robot, config)

    assert image == "popup_3"
    assert robot.clicks == [pytest.approx((0.46, 0.23)), pytest.approx((0.46, 0.23))]


def test_xinyu_popup_detection_skips_when_chat_window(monkeypatch, tmp_path):
    """聊天界面不做弹窗检测：即使弹窗模板在聊天内容上模糊命中也不点击。

    回归背景：弹窗模板误命中聊天区导致“关闭弹窗”点进聊天内容里；
    真弹窗浮在聊天界面上时先回列表、下一轮入口再关。
    """
    config = _config(tmp_path)
    robot = FakeRobot(["chat"])
    checked_templates: list[str] = []

    def fake_match(image, template, threshold=0.85):
        template_name = Path(template).name
        checked_templates.append(template_name)
        if image == "chat" and template_name == "screen_聊天界面.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        if image == "chat" and template_name == "popup_custom_1.jpg":
            return {"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.3, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)

    image = steps.capture_after_popups(robot, config)

    assert image == "chat"
    assert robot.clicks == []
    assert "screen_聊天界面.jpg" in checked_templates
    assert "popup_custom_1.jpg" not in checked_templates


def test_xinyu_popup_detection_stops_when_after_popup_image_is_chat(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["popup", "chat"])
    checked_templates: list[tuple[str, str]] = []

    def fake_match(image, template, threshold=0.85):
        template_name = Path(template).name
        checked_templates.append((str(image), template_name))
        if image == "popup" and template_name == "popup_custom_1.jpg":
            return {"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.3, "score": 0.99}
        if image == "chat" and template_name == "screen_聊天界面.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)

    image = steps.capture_after_popups(robot, config)

    assert image == "chat"
    assert robot.clicks == [pytest.approx((0.46, 0.23))]
    assert ("chat", "screen_聊天界面.jpg") in checked_templates
    assert ("chat", "popup_custom_1.jpg") not in checked_templates


def test_xinyu_unknown_screen_does_not_click_default_back(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["unknown"])

    monkeypatch.setattr(
        steps,
        "capture_transform_image",
        lambda robot, cfg, pre_move=True, label="screen", **kwargs: "unknown",
    )
    monkeypatch.setattr(steps, "match", lambda image, template, threshold=0.85: None)

    # 未知界面且无返回按钮：软等待（返回 None、不点击、不抛），不执行复位。
    assert steps.ensure_list_screen(robot, config) is None
    assert robot.clicks == []


def test_xinyu_go_back_requires_matched_back_button(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["chat"])

    monkeypatch.setattr(steps, "ensure_send_cleared", lambda robot, config, reporter=None: 0)
    monkeypatch.setattr(steps, "match", lambda image, template, threshold=0.85: None)

    with pytest.raises(steps.StepUnavailable, match="未找到返回按钮模板"):
        steps.go_back(robot, config)
    assert robot.clicks == []


def test_xinyu_input_box_error_is_reported_in_chinese(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["chat"])

    monkeypatch.setattr(steps, "capture_after_popups", lambda robot, cfg, label="screen", reporter=None, **kwargs: "chat")
    monkeypatch.setattr(steps, "match", lambda image, template, threshold=0.85: None)

    with pytest.raises(steps.StepUnavailable, match="未找到输入框模板"):
        steps.focus_input_box(robot, config)


def test_xinyu_focus_input_box_clamps_negative_offset(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["chat"])

    monkeypatch.setattr(steps, "capture_after_popups", lambda robot, cfg, label="screen", reporter=None, **kwargs: "chat")

    def fake_match(image, template, threshold=0.85):
        if Path(template).name == "btn_输入框.jpg":
            return {"x": 0.02, "y": 0.80, "w": 0.11, "h": 0.08, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)

    steps.focus_input_box(robot, config)

    assert robot.clicks == [(0.12, 0.84)]
    assert config["_runtime_cache"]["metrics"]["input_box_source"] == "template"
    assert config["_runtime_cache"]["metrics"]["input_box_template"] == "btn_输入框.jpg"


def test_xinyu_focus_input_box_reuses_cached_position(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["chat", "chat"])
    match_calls: list[str] = []

    monkeypatch.setattr(steps, "capture_after_popups", lambda robot, cfg, label="screen", reporter=None, **kwargs: "chat")

    def fake_match(image, template, threshold=0.85):
        match_calls.append(Path(template).name)
        if Path(template).name == "btn_输入框.jpg":
            return {"x": 0.82, "y": 0.80, "w": 0.1, "h": 0.08, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)

    steps.focus_input_box(robot, config)
    steps.focus_input_box(robot, config)

    assert robot.clicks == [(0.37, 0.84), (0.37, 0.84)]
    assert match_calls == ["btn_输入框.jpg"]
    assert config["_runtime_cache"]["metrics"]["input_box_source"] == "cache"


def test_xinyu_chat_window_error_is_reported_in_chinese(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])

    monkeypatch.setattr(steps, "capture_after_popups", lambda robot, cfg, label="screen", reporter=None: "list")
    monkeypatch.setattr(steps, "_matches_screen", lambda image, config, key: False)

    with pytest.raises(steps.StepUnavailable, match="未匹配到聊天界面模板"):
        steps.confirm_chat_window(robot, config)


def test_xinyu_list_selects_highest_favorability_target(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])

    def fake_match(image, template, threshold=0.85):
        if image == "list" and Path(template).name == "screen_聊天列表.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps, "_red_pixel_stats", lambda image, match_result: {"has_red": True, "red_pixels": 8, "total_pixels": 100, "threshold": 3, "region": [1, 2, 3, 4]})
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "bottom_nav": [{"name": "未回复", "bounds": [0.329, 0.125, 0.47, 0.153]}],
                "message_list": [
                    {"name": "甲", "favorability": 4, "unread_count": 1, "content": "短消息", "bounds": [0.1, 0.2, 0.3, 0.3]},
                    {"name": "乙", "favorability": 9, "unread_count": 1, "content": "更高", "bounds": [0.2, 0.4, 0.5, 0.5]},
                ]
            },
            "ocr_items": [],
        },
    )

    target = steps.find_unread_target(robot, config)

    assert target["name"] == "乙"


def test_xinyu_confirmed_unread_tab_does_not_skip_row_with_missing_unread_badge():
    result = {
        "result": {
            "message_list": [
                {"name": "青山如黛", "favorability": 38, "unread_count": None, "content": "没有", "bounds": [0.177, 0.297, 0.973, 0.369]},
                {"name": "清风徐来", "favorability": None, "unread_count": 1, "content": "你好", "bounds": [0.177, 0.395, 0.97, 0.469]},
            ]
        }
    }

    target = steps._select_list_target_from_ocr_result(
        result,
        prefer_detected_unread=False,
    )

    assert target is not None
    assert target["name"] == "青山如黛"


def test_xinyu_list_uses_compact_region_after_unread_tab_click(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["after_unread"])
    regions: list[object] = []

    monkeypatch.setattr(steps, "ensure_list_screen", lambda robot, config, reporter=None, image=None: "list")
    monkeypatch.setattr(
        steps,
        "_red_pixel_stats",
        lambda image, match_result: {
            "has_red": image == "after_unread",
            "red_pixels": 8 if image == "after_unread" else 0,
            "total_pixels": 100,
            "threshold": 3,
            "region": [1, 2, 3, 4],
        },
    )
    monkeypatch.setattr(steps, "capture_after_popups", lambda robot, cfg, label="screen", reporter=None: "after_unread")
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)

    def fake_ocr(image, cfg, mode, region=None):
        regions.append(region)
        if region is None:
            return {
                "result": {
                    "bottom_nav": [{"name": "未回复", "bounds": [0.2, 0.1, 0.4, 0.2]}, {"name": "消息", "bounds": [0.6, 0.8, 0.8, 1.0]}],
                    "message_list": [{"name": "旧目标", "favorability": 1, "unread_count": 0, "content": "", "bounds": [0.1, 0.24, 0.8, 0.32]}],
                },
                "ocr_items": [],
            }
        return {
            "result": {
                "bottom_nav": [{"name": "未回复", "bounds": [0.2, 0.1, 0.4, 0.2]}],
                "message_list": [{"name": "新目标", "favorability": 9, "unread_count": 1, "content": "在吗", "bounds": [0.1, 0.3, 0.8, 0.4]}],
            },
            "ocr_items": [],
        }

    monkeypatch.setattr(steps, "run_ocr_mode", fake_ocr)

    target = steps.find_unread_target(robot, config)

    assert target["name"] == "新目标"
    assert robot.clicks == [pytest.approx((0.3, 0.15))]
    assert regions[0] is None
    assert isinstance(regions[1], list)
    assert regions[1] != []


def test_xinyu_list_compact_region_falls_back_to_full_ocr(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["_runtime_cache"] = {"ocr_regions": {"list_compact": [0.0, 0.2, 1.0, 0.5]}}
    regions: list[object] = []

    def fake_ocr(image, cfg, mode, region=None):
        regions.append(region)
        if region is not None:
            return {"result": {"message_list": []}, "ocr_items": []}
        return {
            "result": {
                "bottom_nav": [{"name": "消息", "bounds": [0.6, 0.8, 0.8, 1.0]}],
                "message_list": [{"name": "完整结果", "favorability": 3, "unread_count": 1, "content": "", "bounds": [0.1, 0.3, 0.8, 0.4]}],
            },
            "ocr_items": [],
        }

    monkeypatch.setattr(steps, "run_ocr_mode", fake_ocr)

    result = steps._run_list_ocr("list", config, prefer_compact=True)

    assert result["result"]["message_list"][0]["name"] == "完整结果"
    assert regions == [[0.0, 0.2, 1.0, 0.5], None]


def test_xinyu_list_falls_back_to_content_when_unread_is_missing(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["enable_screenshot_ocr_reuse"] = False
    robot = FakeRobot(["list"])
    monkeypatch.setattr(steps, "ensure_list_screen", lambda robot, config, reporter=None: "list")
    monkeypatch.setattr(steps, "check_unread_button", lambda robot, config, image, reporter=None, ocr_result=None: (image, ocr_result))
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "bottom_nav": [{"name": "未回复", "bounds": [0.329, 0.125, 0.47, 0.153]}],
                "message_list": [
                    {"name": "user_a", "favorability": 2, "unread_count": 0, "content": "hello", "bounds": [0.1, 0.2, 0.3, 0.3]},
                    {"name": "user_b", "favorability": 7, "unread_count": 0, "content": "reply me", "bounds": [0.2, 0.4, 0.5, 0.5]},
                ]
            },
            "ocr_items": [],
        },
    )

    target = steps.find_unread_target(robot, config)

    assert target["name"] == "user_b"
    assert target["unread_count"] == 0


def test_xinyu_list_keeps_initial_content_when_unread_switch_is_empty(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])
    reports: list[str] = []

    monkeypatch.setattr(steps, "ensure_list_screen", lambda robot, config, reporter=None: "list")
    monkeypatch.setattr(steps, "capture_after_popups", lambda robot, config, label="screen", reporter=None: "after_unread")
    monkeypatch.setattr(steps, "_red_pixel_stats", lambda image, match_result: {"has_red": False, "red_pixels": 0, "total_pixels": 100, "threshold": 1, "region": [1, 2, 3, 4]})
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)

    def fake_ocr(image, cfg, mode, region=None):
        if image == "after_unread":
            return {"result": {"message_list": [], "bottom_nav": []}, "ocr_items": []}
        return {
            "result": {
                "message_list": [
                    {"name": "薄荷", "favorability": None, "unread_count": None, "content": "看看你的诚意", "bounds": [0.208, 0.187, 0.943, 0.262]},
                ],
                "bottom_nav": [
                    {"name": "未回复", "bounds": [0.329, 0.125, 0.47, 0.153]},
                    {"name": "消息", "bounds": [0.664, 0.919, 0.726, 0.936]},
                ],
            },
            "ocr_items": [],
        }

    monkeypatch.setattr(steps, "run_ocr_mode", fake_ocr)

    target = steps.find_unread_target(
        robot,
        config,
        reporter=lambda _step, label, log_label=None: reports.append(log_label or label),
    )

    assert target is None
    assert robot.clicks == [pytest.approx(((0.329 + 0.47) / 2, (0.125 + 0.153) / 2))]
    assert "切换后未找到未回复入口，且无未读消息，不读取列表目标" in reports
    assert "未回复入口不是红色，不读取列表目标" in reports


def test_xinyu_list_uses_unread_rows_when_switched_tab_text_is_missed(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])

    monkeypatch.setattr(steps, "ensure_list_screen", lambda robot, config, reporter=None, image=None: "list")
    monkeypatch.setattr(steps, "capture_after_popups", lambda robot, config, label="screen", reporter=None: "after_unread")
    monkeypatch.setattr(
        steps,
        "_red_pixel_stats",
        lambda image, match_result: {"has_red": False, "red_pixels": 0, "total_pixels": 100, "threshold": 1, "region": [1, 2, 3, 4]},
    )
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)

    def fake_ocr(image, cfg, mode, region=None):
        if image == "after_unread":
            return {
                "result": {
                    "bottom_nav": [],
                    "message_list": [
                        {"name": "心想事成", "favorability": 63, "unread_count": 1, "content": "你好", "bounds": [0.184, 0.177, 0.977, 0.247]},
                        {"name": "风回路转", "favorability": 87, "unread_count": 1, "content": "在吗", "bounds": [0.184, 0.374, 0.977, 0.445]},
                    ],
                },
                "ocr_items": [],
            }
        return {
            "result": {"bottom_nav": [{"name": "未回复", "bounds": [0.329, 0.125, 0.47, 0.153]}], "message_list": []},
            "ocr_items": [],
        }

    monkeypatch.setattr(steps, "run_ocr_mode", fake_ocr)

    target = steps.find_unread_target(robot, config)

    assert target is not None
    assert target["name"] == "风回路转"


def test_xinyu_list_ignores_empty_placeholder_when_falling_back_to_content(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["enable_screenshot_ocr_reuse"] = False
    robot = FakeRobot(["list"])
    monkeypatch.setattr(steps, "ensure_list_screen", lambda robot, config, reporter=None: "list")
    monkeypatch.setattr(steps, "check_unread_button", lambda robot, config, image, reporter=None, ocr_result=None: (image, ocr_result))
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "bottom_nav": [{"name": "未回复", "bounds": [0.329, 0.125, 0.47, 0.153]}],
                "message_list": [
                    {"name": "\u6682\u65e0\u6d88\u606f", "favorability": 99, "unread_count": 0, "content": "", "bounds": [0.1, 0.2, 0.3, 0.3]},
                    {"name": "user_b", "favorability": 3, "unread_count": 0, "content": "available", "bounds": [0.2, 0.4, 0.5, 0.5]},
                ]
            },
            "ocr_items": [],
        },
    )

    target = steps.find_unread_target(robot, config)

    assert target["name"] == "user_b"


def test_xinyu_list_still_prefers_unread_target_over_content_fallback(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["enable_screenshot_ocr_reuse"] = False
    robot = FakeRobot(["list"])
    monkeypatch.setattr(steps, "ensure_list_screen", lambda robot, config, reporter=None: "list")
    monkeypatch.setattr(steps, "check_unread_button", lambda robot, config, image, reporter=None, ocr_result=None: (image, ocr_result))
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "bottom_nav": [{"name": "未回复", "bounds": [0.329, 0.125, 0.47, 0.153]}],
                "message_list": [
                    {"name": "content_only", "favorability": 50, "unread_count": 0, "content": "reply me", "bounds": [0.1, 0.2, 0.3, 0.3]},
                    {"name": "unread_user", "favorability": 4, "unread_count": 1, "content": "", "bounds": [0.2, 0.4, 0.5, 0.5]},
                ]
            },
            "ocr_items": [],
        },
    )

    target = steps.find_unread_target(robot, config)

    # 9/2 起未回复视图内改为内容优先（prefer_detected_unread=False，
    # 提交 125676e）：视图内行本就全有未读，角标缺失不该被跳过。
    assert target["name"] == "content_only"
    assert target["unread_count"] == 0


def test_xinyu_list_falls_back_to_content_when_unread_is_missing(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])
    monkeypatch.setattr(steps, "ensure_list_screen", lambda robot, config, reporter=None, image=None: "list")
    def fake_check_unread_button(robot, config, image, reporter=None, ocr_result=None):
        ocr_result["_xinyu_unread_tab_has_red"] = True
        return image, ocr_result
    monkeypatch.setattr(steps, "check_unread_button", fake_check_unread_button)

    def fake_match(image, template, threshold=0.85):
        if image == "list" and Path(template).name == "screen_鑱婂ぉ鍒楄〃.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(
        steps,
        "_red_pixel_stats",
        lambda image, match_result: {"has_red": True, "red_pixels": 8, "total_pixels": 100, "threshold": 3, "region": [1, 2, 3, 4]},
    )
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "bottom_nav": [{"name": "未回复", "bounds": [0.329, 0.125, 0.47, 0.153]}],
                "message_list": [
                    {"name": "user_a", "favorability": 2, "unread_count": 0, "content": "hello", "bounds": [0.1, 0.2, 0.3, 0.3]},
                    {"name": "user_b", "favorability": 7, "unread_count": 0, "content": "reply me", "bounds": [0.2, 0.4, 0.5, 0.5]},
                ]
            },
            "ocr_items": [],
        },
    )

    target = steps.find_unread_target(robot, config)

    assert target["name"] == "user_b"
    assert target["unread_count"] == 0


def test_xinyu_list_ignores_empty_placeholder_when_falling_back_to_content(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])
    monkeypatch.setattr(steps, "ensure_list_screen", lambda robot, config, reporter=None, image=None: "list")
    def fake_check_unread_button(robot, config, image, reporter=None, ocr_result=None):
        ocr_result["_xinyu_unread_tab_has_red"] = True
        return image, ocr_result
    monkeypatch.setattr(steps, "check_unread_button", fake_check_unread_button)

    def fake_match(image, template, threshold=0.85):
        if image == "list" and Path(template).name == "screen_鑱婂ぉ鍒楄〃.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(
        steps,
        "_red_pixel_stats",
        lambda image, match_result: {"has_red": True, "red_pixels": 8, "total_pixels": 100, "threshold": 3, "region": [1, 2, 3, 4]},
    )
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "bottom_nav": [{"name": "未回复", "bounds": [0.329, 0.125, 0.47, 0.153]}],
                "message_list": [
                    {"name": "\u6682\u65e0\u6d88\u606f", "favorability": 99, "unread_count": 0, "content": "", "bounds": [0.1, 0.2, 0.3, 0.3]},
                    {"name": "user_b", "favorability": 3, "unread_count": 0, "content": "available", "bounds": [0.2, 0.4, 0.5, 0.5]},
                ]
            },
            "ocr_items": [],
        },
    )

    target = steps.find_unread_target(robot, config)

    assert target["name"] == "user_b"


def test_xinyu_list_still_prefers_unread_target_over_content_fallback(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])
    monkeypatch.setattr(steps, "ensure_list_screen", lambda robot, config, reporter=None, image=None: "list")
    def fake_check_unread_button(robot, config, image, reporter=None, ocr_result=None):
        ocr_result["_xinyu_unread_tab_has_red"] = True
        return image, ocr_result
    monkeypatch.setattr(steps, "check_unread_button", fake_check_unread_button)

    def fake_match(image, template, threshold=0.85):
        if image == "list" and Path(template).name == "screen_鑱婂ぉ鍒楄〃.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(
        steps,
        "_red_pixel_stats",
        lambda image, match_result: {"has_red": True, "red_pixels": 8, "total_pixels": 100, "threshold": 3, "region": [1, 2, 3, 4]},
    )
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "bottom_nav": [{"name": "未回复", "bounds": [0.329, 0.125, 0.47, 0.153]}],
                "message_list": [
                    {"name": "content_only", "favorability": 50, "unread_count": 0, "content": "reply me", "bounds": [0.1, 0.2, 0.3, 0.3]},
                    {"name": "unread_user", "favorability": 4, "unread_count": 1, "content": "", "bounds": [0.2, 0.4, 0.5, 0.5]},
                ]
            },
            "ocr_items": [],
        },
    )

    target = steps.find_unread_target(robot, config)

    # 9/2 起未回复视图内改为内容优先（prefer_detected_unread=False，提交 125676e）。
    assert target["name"] == "content_only"
    assert target["unread_count"] == 0


def test_xinyu_app_uses_short_list_content_directly(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list", "chat", "chat", "chat", "chat", "list"])
    sent: list[tuple[str, str]] = []
    replies: list[dict[str, str]] = []

    monkeypatch.setattr(xinyu_app.steps, "find_unread_target", lambda robot, config, reporter=None: {"name": "白嫖", "content": "在吗", "favorability": 44, "unread_count": 1, "bounds": [0.1, 0.2, 0.3, 0.4], "x": 0.2, "y": 0.3})
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(xinyu_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: "chat")
    monkeypatch.setattr(xinyu_app.steps, "send_new_greeting_if_available", lambda robot, config, image=None, reporter=None: False)
    monkeypatch.setattr(xinyu_app.steps, "build_list_message", lambda target, content: {"contact": target["name"], "content": content})
    monkeypatch.setattr(xinyu_app.steps, "focus_input_box", lambda robot, config, reporter=None: robot.move_click(0.3, 0.8))
    monkeypatch.setattr(xinyu_app.steps, "play_voice_reply", lambda robot, text, config, reporter=None: replies.append({"text": text}))
    monkeypatch.setattr(xinyu_app.steps, "send_reply", lambda robot, config, reporter=None: 1)
    monkeypatch.setattr(xinyu_app.steps, "go_back", lambda robot, config, reporter=None: robot.move_click(0.08, 0.06))

    events: list[dict[str, object]] = []
    app = xinyu_app.XinyuApp(
        robot,
        config,
        reply_provider=lambda message: "AI回复",
        sent_callback=lambda action, message: sent.append((action.text, message["content"])),
    )
    app.set_status_callback(events.append)

    app.run_once()

    assert replies == [{"text": "AI回复"}]
    assert sent == [("AI回复", "在吗")]
    assert (0.3, 0.8) in robot.clicks
    assert any(event.get("current_step_label") == "进入聊天界面" for event in events)
    assert any(event.get("current_step_log_label") == "进入聊天界面：白嫖 (0.200, 0.300)" for event in events)


def test_xinyu_app_logs_list_content_used_for_reply(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list", "chat", "chat", "chat", "list"])
    statuses: list[dict[str, object]] = []

    monkeypatch.setattr(
        xinyu_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: {"name": "白嫖", "content": "在吗", "favorability": 44, "unread_count": 1, "bounds": [0.1, 0.2, 0.3, 0.4], "x": 0.2, "y": 0.3},
    )
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(xinyu_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: "chat")
    monkeypatch.setattr(xinyu_app.steps, "send_new_greeting_if_available", lambda robot, config, image=None, reporter=None: False)
    monkeypatch.setattr(xinyu_app.steps, "build_list_message", lambda target, content: {"contact": target["name"], "content": content})
    monkeypatch.setattr(xinyu_app.steps, "focus_input_box", lambda robot, config, reporter=None: None)
    monkeypatch.setattr(xinyu_app.steps, "play_voice_reply", lambda robot, text, config, reporter=None: None)
    monkeypatch.setattr(xinyu_app.steps, "send_reply", lambda robot, config, reporter=None: 1)
    monkeypatch.setattr(xinyu_app.steps, "go_back", lambda robot, config, reporter=None: None)
    monkeypatch.setattr(xinyu_app.random, "choice", lambda values: values[0])

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI回复")
    app.set_status_callback(statuses.append)
    app.run_once()

    assert any(
        status.get("current_step") == "direct_content_reply"
        and status.get("current_step_label") == "使用列表内容作为对话输入"
        and status.get("current_step_log_label") == "使用列表内容作为对话输入：在吗"
        for status in statuses
    )


def test_xinyu_app_logs_duplicate_selection(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list", "chat", "chat", "chat", "list", "list", "chat", "chat", "chat", "list"])
    target = {"name": "白嫖", "content": "在吗", "favorability": 44, "unread_count": 1, "bounds": [0.1, 0.2, 0.3, 0.4], "x": 0.2, "y": 0.3}
    logs: list[tuple[str, str, dict | None]] = []

    monkeypatch.setattr(xinyu_app.steps, "find_unread_target", lambda robot, config, reporter=None: dict(target))
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(xinyu_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: "chat")
    monkeypatch.setattr(xinyu_app.steps, "send_new_greeting_if_available", lambda robot, config, image=None, reporter=None: False)
    monkeypatch.setattr(xinyu_app.steps, "build_list_message", lambda target, content: {"contact": target["name"], "content": content})
    monkeypatch.setattr(xinyu_app.steps, "focus_input_box", lambda robot, config, reporter=None: None)
    monkeypatch.setattr(xinyu_app.steps, "play_voice_reply", lambda robot, text, config, reporter=None: None)
    monkeypatch.setattr(xinyu_app.steps, "send_reply", lambda robot, config, reporter=None: 1)
    monkeypatch.setattr(xinyu_app.steps, "go_back", lambda robot, config, reporter=None, double_click_on_first_attempt=False: None)

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI回复")
    app.set_log_callback(lambda level, message, payload=None: logs.append((level, message, payload)))
    app.run_once()
    app.run_once()

    assert any(message == "心遇重复选中同一会话" for _, message, _ in logs)


def test_xinyu_app_resets_error_count_when_selected_contact_changes(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list", "chat"])

    monkeypatch.setattr(
        xinyu_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: {
            "name": "new-user",
            "content": "在吗",
            "favorability": 44,
            "unread_count": 1,
            "bounds": [0.1, 0.2, 0.3, 0.4],
            "x": 0.2,
            "y": 0.3,
        },
    )
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(
        xinyu_app.steps,
        "confirm_chat_window",
        lambda robot, config, reporter=None: (_ for _ in ()).throw(xinyu_app.steps.StepUnavailable("stop after changed contact")),
    )

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI回复")
    app._last_selected_signature = ("old-user", "旧消息", 1)
    app.status.error_count = 4
    app.status.is_circuit_open = False
    app.recover_error_count = 2

    app.run_once()

    assert app.status.last_error == "stop after changed contact"
    assert app.status.error_count == 1
    assert app.recover_error_count == 0
    assert app.status.is_circuit_open is False
    assert app._last_selected_signature == ("new-user", "在吗", 1)


def test_xinyu_app_logs_multi_send_clicks(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list", "chat", "chat", "chat", "list"])
    logs: list[tuple[str, str, dict | None]] = []

    monkeypatch.setattr(
        xinyu_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: {"name": "白嫖", "content": "在吗", "favorability": 44, "unread_count": 1, "bounds": [0.1, 0.2, 0.3, 0.4], "x": 0.2, "y": 0.3},
    )
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(xinyu_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: "chat")
    monkeypatch.setattr(xinyu_app.steps, "send_new_greeting_if_available", lambda robot, config, image=None, reporter=None: False)
    monkeypatch.setattr(xinyu_app.steps, "build_list_message", lambda target, content: {"contact": target["name"], "content": content})
    def fake_focus_input_box(robot, config, reporter=None):
        config["_runtime_cache"]["metrics"].update({"input_box_source": "cache", "input_box_template": ""})

    def fake_play_voice_reply(robot, text, config, reporter=None):
        config["_runtime_cache"]["metrics"].update(
            {"microphone_template": "btn_麦克风按钮.jpg", "tts_success": True, "tts_retry_count": 1}
        )

    def fake_send_reply(robot, config, reporter=None):
        config["_runtime_cache"]["metrics"].update(
            {"send_clear_extra_clicks": 1, "send_clear_reason": "send_button_still_visible"}
        )
        return 3

    def fake_go_back(robot, config, reporter=None, double_click_on_first_attempt=False):
        config["_runtime_cache"]["metrics"].update({"return_clicks": 2, "return_reason": "after_send_double_back"})
        return 2

    monkeypatch.setattr(xinyu_app.steps, "focus_input_box", fake_focus_input_box)
    monkeypatch.setattr(xinyu_app.steps, "play_voice_reply", fake_play_voice_reply)
    monkeypatch.setattr(xinyu_app.steps, "send_reply", fake_send_reply)
    monkeypatch.setattr(xinyu_app.steps, "go_back", fake_go_back)

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI回复")
    app.status.error_count = 4
    app.status.is_circuit_open = False
    app.recover_error_count = 2
    app.set_log_callback(lambda level, message, payload=None: logs.append((level, message, payload)))
    app.run_once()

    assert any(message == "心遇发送按钮被连续点击" for _, message, _ in logs)
    assert app.status.error_count == 0
    assert app.status.is_circuit_open is False
    assert app.recover_error_count == 0
    summaries = [payload for _, message, payload in logs if message == "xinyu_send_flow_summary"]
    assert summaries
    assert summaries[-1]["input_box_source"] == "cache"
    assert summaries[-1]["microphone_template"] == "btn_麦克风按钮.jpg"
    assert summaries[-1]["tts_success"] is True
    assert summaries[-1]["tts_retry_count"] == 1
    assert summaries[-1]["send_clear_extra_clicks"] == 1
    assert summaries[-1]["return_reason"] == "after_send_double_back"


def test_xinyu_app_skip_reply_still_focuses_input_box_before_back(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list", "chat", "chat", "chat", "list"])
    clicks: list[tuple[float, float]] = []

    monkeypatch.setattr(
        xinyu_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: {"name": "白嫖", "content": "在吗", "favorability": 44, "unread_count": 1, "bounds": [0.1, 0.2, 0.3, 0.4], "x": 0.2, "y": 0.3},
    )
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(xinyu_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: "chat")
    monkeypatch.setattr(xinyu_app.steps, "send_new_greeting_if_available", lambda robot, config, image=None, reporter=None: False)
    monkeypatch.setattr(xinyu_app.steps, "build_list_message", lambda target, content: {"contact": target["name"], "content": content})
    monkeypatch.setattr(xinyu_app.steps, "focus_input_box", lambda robot, config, reporter=None: clicks.append((0.31, 0.79)))
    monkeypatch.setattr(xinyu_app.steps, "play_voice_reply", lambda robot, text, config, reporter=None: None)
    monkeypatch.setattr(xinyu_app.steps, "go_back", lambda robot, config, reporter=None: clicks.append((0.08, 0.06)))
    monkeypatch.setattr(xinyu_app.steps, "send_reply", lambda robot, config, reporter=None: 1)

    app = xinyu_app.XinyuApp(
        robot,
        config,
        reply_provider=lambda message: xinyu_app.ReplyAction(skip=True, reason="already_processed"),
    )
    app.run_once()

    assert clicks[0] == (0.31, 0.79)
    assert clicks[-1] == (0.08, 0.06)


def test_xinyu_app_empty_target_does_not_keep_awake_before_idle_interval(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["keep_awake_interval_seconds"] = 30
    robot = FakeRobot(["list"])
    calls: list[str] = []

    monkeypatch.setattr(
        xinyu_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: {
            "name": "暂无消息",
            "content": "",
            "favorability": None,
            "unread_count": 1,
            "bounds": [0.1, 0.2, 0.3, 0.4],
            "x": 0.2,
            "y": 0.3,
        },
    )
    monkeypatch.setattr(xinyu_app.steps, "keep_screen_awake", lambda robot, config, reporter=None: calls.append("keep_awake") or True)
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: calls.append("enter_chat"))

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI回复")
    app.run_once()

    assert calls == []
    assert app.status.last_error == ""


def test_xinyu_app_empty_target_keep_awake_only_after_idle_interval(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["keep_awake_interval_seconds"] = 30
    robot = FakeRobot(["list"])
    calls: list[str] = []

    monkeypatch.setattr(
        xinyu_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: {
            "name": "暂无消息",
            "content": "",
            "favorability": None,
            "unread_count": 1,
            "bounds": [0.1, 0.2, 0.3, 0.4],
            "x": 0.2,
            "y": 0.3,
        },
    )
    monkeypatch.setattr(xinyu_app.steps, "keep_screen_awake", lambda robot, config, reporter=None: calls.append("keep_awake") or True)
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: calls.append("enter_chat"))

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI回复")
    app.last_activity_at -= 31
    app.run_once()

    assert calls == ["keep_awake"]
    assert app.status.last_error == ""


def test_xinyu_app_uses_greeting_file_for_nihao(monkeypatch, tmp_path):
    config = _config(tmp_path)
    greetings = Path(config["greetings_dir"])
    greetings.mkdir(parents=True, exist_ok=True)
    (greetings / "pingtai.txt").write_text("您好\n你好呀\n", encoding="utf-8")
    robot = FakeRobot(["list", "chat", "chat", "chat", "list"])
    sent: list[str] = []

    monkeypatch.setattr(xinyu_app.steps, "find_unread_target", lambda robot, config, reporter=None: {"name": "白嫖", "content": "你好", "favorability": 44, "unread_count": 1, "bounds": [0.1, 0.2, 0.3, 0.4], "x": 0.2, "y": 0.3})
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(xinyu_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: "chat")
    monkeypatch.setattr(xinyu_app.steps, "send_new_greeting_if_available", lambda robot, config, image=None, reporter=None: False)
    monkeypatch.setattr(xinyu_app.steps, "build_list_message", lambda target, content: {"contact": target["name"], "content": content})
    monkeypatch.setattr(xinyu_app.steps, "focus_input_box", lambda robot, config, reporter=None: robot.move_click(0.31, 0.79))
    monkeypatch.setattr(xinyu_app.steps, "play_voice_reply", lambda robot, text, config, reporter=None: sent.append(text))
    monkeypatch.setattr(xinyu_app.steps, "send_reply", lambda robot, config, reporter=None: 1)
    monkeypatch.setattr(xinyu_app.steps, "go_back", lambda robot, config, reporter=None: robot.move_click(0.08, 0.06))
    monkeypatch.setattr(xinyu_app.random, "choice", lambda values: values[0])

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI回复")
    app.run_once()

    assert sent == ["您好"]
    assert (0.31, 0.79) in robot.clicks


def test_xinyu_sends_new_greeting_and_returns(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["after_greeting"])

    def fake_match(image, template, threshold=0.85):
        template_name = Path(template).name
        if image == "chat" and template_name == "btn_新招呼按钮.jpg":
            return {"x": 0.3, "y": 0.4, "w": 0.1, "h": 0.1, "score": 0.99}
        if image == "after_greeting" and template_name == "btn_返回按钮_2.jpg":
            return {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)

    assert steps.send_new_greeting_if_available(robot, config, image="chat") is True
    assert robot.clicks == [pytest.approx((0.55, 0.45)), pytest.approx((0.15, 0.15))]


def test_xinyu_new_greeting_back_secondary_reuses_back_region(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["template_match_regions"] = {
        "button.new_greeting": [0.0, 0.0, 1.0, 1.0],
        "button.back": [0.0, 0.9, 1.0, 0.1],
    }
    robot = FakeRobot(["after_greeting"])
    used_keys: list[str] = []

    def fake_match_template_with_region(image, template, config, key, *, threshold, match_func):
        used_keys.append(key)
        template_name = Path(template).name
        if image == "chat" and template_name == "btn_新招呼按钮.jpg":
            return {"x": 0.3, "y": 0.4, "w": 0.1, "h": 0.1, "score": 0.99}
        if image == "after_greeting" and template_name == "btn_返回按钮_2.jpg":
            return {"x": 0.1, "y": 0.91, "w": 0.1, "h": 0.08, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match_template_with_region", fake_match_template_with_region)
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)
    # capture_after_popups 的聊天界面豁免会做列表标题 OCR + 模板匹配：
    # 本测试聚焦返回按钮区域复用，界面判定一律返回“不是列表、不是聊天、无弹窗”。
    monkeypatch.setattr(steps, "_matches_chat_list_screen", lambda image, config, reporter=None: False)
    monkeypatch.setattr(steps, "match", lambda image, template, threshold=0.85: None)

    assert steps.send_new_greeting_if_available(robot, config, image="chat") is True
    assert "button.back.secondary" not in used_keys
    assert used_keys[-1] == "button.back"
    assert robot.clicks[-1] == pytest.approx((0.15, 0.95))


def test_xinyu_new_greeting_missing_continues_normal_flow(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot([])

    monkeypatch.setattr(steps, "match", lambda image, template, threshold=0.85: None)

    assert steps.send_new_greeting_if_available(robot, config, image="chat") is False
    assert robot.clicks == []


def test_xinyu_app_new_greeting_skips_ai_flow(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list", "chat", "chat", "list"])
    calls: list[str] = []

    monkeypatch.setattr(
        xinyu_app.steps,
        "find_unread_target",
        lambda robot, config, reporter=None: {"name": "白嫖", "content": "在吗", "favorability": 44, "unread_count": 1, "bounds": [0.1, 0.2, 0.3, 0.4], "x": 0.2, "y": 0.3},
    )
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(xinyu_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: "chat")
    monkeypatch.setattr(xinyu_app.steps, "send_new_greeting_if_available", lambda robot, config, image=None, reporter=None: calls.append("new_greeting") or True)
    monkeypatch.setattr(xinyu_app.steps, "read_message", lambda *args, **kwargs: calls.append("read_message") or {"contact": "白嫖", "content": "在吗"})
    monkeypatch.setattr(xinyu_app.steps, "focus_input_box", lambda *args, **kwargs: calls.append("focus_input_box"))
    monkeypatch.setattr(xinyu_app.steps, "play_voice_reply", lambda *args, **kwargs: calls.append("play_voice_reply"))
    monkeypatch.setattr(xinyu_app.steps, "send_reply", lambda *args, **kwargs: calls.append("send_reply") or 1)

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: calls.append("reply_provider") or "AI回复")
    app.run_once()

    assert calls == ["new_greeting"]
    assert app.status.last_error == ""


def test_xinyu_app_reads_dialogue_sender1_last_message(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list", "chat", "chat", "chat", "chat", "list"])
    seen: list[dict[str, str]] = []

    monkeypatch.setattr(xinyu_app.steps, "find_unread_target", lambda robot, config, reporter=None: {"name": "白嫖", "content": "这是一个很长的消息内容，需要进详情", "favorability": 44, "unread_count": 1, "bounds": [0.1, 0.2, 0.3, 0.4], "x": 0.2, "y": 0.3})
    monkeypatch.setattr(xinyu_app.steps, "enter_chat", lambda robot, x, y: robot.move_click(x, y))
    monkeypatch.setattr(xinyu_app.steps, "confirm_chat_window", lambda robot, config, reporter=None: "chat")
    monkeypatch.setattr(xinyu_app.steps, "send_new_greeting_if_available", lambda robot, config, image=None, reporter=None: False)
    monkeypatch.setattr(
        xinyu_app.steps,
        "read_message",
        lambda robot, config, target, image=None, reporter=None: {"contact": target["name"], "content": "最后一条", "message_time": ""},
    )
    monkeypatch.setattr(xinyu_app.steps, "focus_input_box", lambda robot, config, reporter=None: robot.move_click(0.31, 0.79))
    monkeypatch.setattr(xinyu_app.steps, "play_voice_reply", lambda robot, text, config, reporter=None: seen.append({"text": text}))
    monkeypatch.setattr(xinyu_app.steps, "send_reply", lambda robot, config, reporter=None: 1)
    monkeypatch.setattr(xinyu_app.steps, "go_back", lambda robot, config, reporter=None: robot.move_click(0.08, 0.06))

    app = xinyu_app.XinyuApp(robot, config, reply_provider=lambda message: "AI回复")
    app.run_once()

    assert seen == [{"text": "AI回复"}]


def test_xinyu_read_message_uses_dialogue_target_sender(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["ocr_modes"] = {
        **config.get("ocr_modes", {}),
        "dialogue": {
            "preprocess": {"mode": "original", "scale": 2, "rotate_to_portrait": True},
            "rules": {
                "min_confidence": 0.3,
                "min_box_height_ratio": 0.02,
                "line_gap": 24,
                "name_gap_ratio": 0.037,
                "sender_detection": {"method": "bubble_color", "sender2_min_x_ratio": 0.52},
                "target_sender": "sender1",
                "ocr_result_excluded_keywords": [],
                "line_excluded_keywords": [],
                "line_and_below_keywords": [],
                "basic_filter": {
                    "enable_time_text_filter": True,
                    "enable_debug_text_filter": True,
                    "enable_reward_text_filter": True,
                    "enable_center_y_region_filter": False,
                    "center_y_min_ratio": 0.18,
                    "center_y_max_ratio": 0.88,
                    "enable_max_x_filter": True,
                    "max_x_ratio": 0.90,
                    "enable_max_width_filter": True,
                    "max_width_ratio": 0.80,
                    "enable_low_conf_single_char_filter": True,
                    "single_char_confidence_threshold": 0.90,
                    "enable_right_narrow_non_cjk_filter": True,
                    "right_narrow_min_x_ratio": 0.88,
                    "right_narrow_max_width_ratio": 0.12,
                    "enable_non_cjk_non_emoji_filter": True,
                },
            },
        },
    }
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "ocr_items": [{"text": "顶部名字"}],
            "result": {
                "user_name": "",
                "dialogues": [
                    {"sender": "sender1", "content": "更早的对方消息"},
                    {"sender": "sender2", "content": "右侧自己"},
                    {"sender": "sender1", "content": "第一句"},
                    {"sender": "sender1", "content": "第二句"},
                ],
            },
        },
    )

    result = steps.read_message(
        FakeRobot([]),
        config,
        target={"name": "白嫖", "unread_count": 1, "favorability": 12},
        image="chat",
    )

    assert result["content"] == "第一句第二句"
    assert result["sender"] == "sender1"


def test_xinyu_read_message_uses_dialogue_compact_region(monkeypatch, tmp_path):
    config = _config(tmp_path)
    regions: list[object] = []

    def fake_match(image, template, threshold=0.85):
        if Path(template).name == "btn_输入框.jpg":
            return {"x": 0.8, "y": 0.82, "w": 0.1, "h": 0.08, "score": 0.99}
        return None

    def fake_ocr(image, cfg, mode, region=None):
        regions.append(region)
        return {
            "ocr_items": [{"text": "顶部名字"}],
            "result": {
                "user_name": "",
                "dialogues": [
                    {"sender": "sender2", "content": "自己"},
                    {"sender": "sender1", "content": "第一句"},
                    {"sender": "sender1", "content": "第二句"},
                ],
            },
        }

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps, "run_ocr_mode", fake_ocr)

    result = steps.read_message(
        FakeRobot([]),
        config,
        target={"name": "白嫖", "unread_count": 1, "favorability": 12},
        image="chat",
    )

    assert result["content"] == "第一句第二句"
    assert len(regions) == 1
    assert isinstance(regions[0], list)
    assert regions[0][1] == pytest.approx(0.1)
    assert regions[0][3] < 0.8


def test_xinyu_read_message_dialogue_compact_falls_back_to_full(monkeypatch, tmp_path):
    config = _config(tmp_path)
    regions: list[object] = []

    def fake_match(image, template, threshold=0.85):
        if Path(template).name == "btn_输入框.jpg":
            return {"x": 0.8, "y": 0.82, "w": 0.1, "h": 0.08, "score": 0.99}
        return None

    def fake_ocr(image, cfg, mode, region=None):
        regions.append(region)
        if region is not None:
            return {
                "ocr_items": [{"text": "顶部名字"}],
                "result": {"user_name": "", "dialogues": [{"sender": "sender2", "content": "自己"}]},
            }
        return {
            "ocr_items": [{"text": "顶部名字"}],
            "result": {
                "user_name": "",
                "dialogues": [
                    {"sender": "sender2", "content": "自己"},
                    {"sender": "sender1", "content": "完整消息"},
                ],
            },
        }

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps, "run_ocr_mode", fake_ocr)

    result = steps.read_message(
        FakeRobot([]),
        config,
        target={"name": "白嫖", "unread_count": 1, "favorability": 12},
        image="chat",
    )

    assert result["content"] == "完整消息"
    assert isinstance(regions[0], list)
    assert regions[1] is None
    assert config["_runtime_cache"]["metrics"]["dialogue_compact_fallback_count"] == 1


def test_xinyu_go_back_clicks_back_twice_after_send(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["chat", "chat", "list"])

    def fake_match(image, template, threshold=0.85):
        if image == "chat" and Path(template).name in {"btn_返回按钮_2.jpg", "btn_返回按钮.jpg"}:
            return {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1, "score": 0.99}
        if image == "list" and Path(template).name == "screen_聊天列表.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        if str(image) == "chat" and Path(template).name.startswith("btn_麦克风按钮"):
            # 8/25 后双击返回的前提是麦克风模板命中（退出语音输入态）。
            return {"x": 0.5, "y": 0.9, "w": 0.1, "h": 0.05, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps, "ensure_send_cleared", lambda robot, config, reporter=None: 0)
    monkeypatch.setattr(
        steps,
        "capture_after_popups",
        lambda robot, config, label="screen", max_dismiss=3, reporter=None, **kwargs: robot.get_transform_image(),
    )

    back_clicks = steps.go_back(robot, config, double_click_on_first_attempt=True)

    assert robot.clicks == [pytest.approx((0.15, 0.15)), pytest.approx((0.15, 0.15))]
    assert back_clicks == 2


def test_xinyu_go_back_reuses_cached_back_position_after_send(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["_runtime_cache"] = {"template_points": {"back": (0.23, 0.07)}}
    robot = FakeRobot(["chat", "chat", "list"])

    def fake_match(image, template, threshold=0.85):
        if Path(template).name in {"btn_返回按钮_2.jpg", "btn_返回按钮.jpg"}:
            raise AssertionError("cached back position should avoid template matching")
        if image == "list" and Path(template).name == "screen_聊天列表.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        if str(image) == "chat" and Path(template).name.startswith("btn_麦克风按钮"):
            # 8/25 后双击返回的前提是麦克风模板命中（退出语音输入态）。
            return {"x": 0.5, "y": 0.9, "w": 0.1, "h": 0.05, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps, "ensure_send_cleared", lambda robot, config, reporter=None: 0)
    monkeypatch.setattr(
        steps,
        "capture_after_popups",
        lambda robot, config, label="screen", max_dismiss=3, reporter=None, **kwargs: robot.get_transform_image(),
    )

    back_clicks = steps.go_back(robot, config, double_click_on_first_attempt=True)

    assert robot.clicks == [pytest.approx((0.23, 0.07)), pytest.approx((0.23, 0.07))]
    assert back_clicks == 2


def test_xinyu_go_back_keeps_cached_back_position_until_next_start(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["_runtime_cache"] = {"template_points": {"back": (0.23, 0.07)}}
    # 8 张 chat：入口 1 + attempt0 点击前 1 + 确认 3 + attempt1 点击前 1 + 确认 2，
    # 第 9 次取图耗尽兜底返回 "list" → attempt1 确认成功（retry_back_success）。
    robot = FakeRobot(["chat"] * 8)
    reports: list[tuple[str, str, str | None]] = []

    def fake_match(image, template, threshold=0.85):
        if image == "chat" and Path(template).name in {"btn_返回按钮_2.jpg", "btn_返回按钮.jpg"}:
            raise AssertionError("cached back position should not be rematched in the same startup")
        if str(image) == "chat" and Path(template).name.startswith("btn_麦克风按钮"):
            return {"x": 0.5, "y": 0.9, "w": 0.1, "h": 0.05, "score": 0.99}
        if image == "list" and Path(template).name == "screen_聊天列表.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        if str(image) == "chat" and Path(template).name.startswith("btn_麦克风按钮"):
            # 8/25 后双击返回的前提是麦克风模板命中（退出语音输入态）。
            return {"x": 0.5, "y": 0.9, "w": 0.1, "h": 0.05, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps, "ensure_send_cleared", lambda robot, config, reporter=None: 0)
    monkeypatch.setattr(
        steps,
        "capture_after_popups",
        lambda robot, config, label="screen", max_dismiss=3, reporter=None, **kwargs: robot.get_transform_image(),
    )

    back_clicks = steps.go_back(robot, config, reporter=lambda step, label, log_label=None: reports.append((step, label, log_label)))

    assert robot.clicks == [pytest.approx((0.23, 0.07)), pytest.approx((0.23, 0.07))]
    assert back_clicks == 2
    assert config["_runtime_cache"]["metrics"]["return_reason"] == "retry_back_success"
    assert config["_runtime_cache"]["template_points"]["back"] == (0.23, 0.07)
    assert any(label == "点击返回后未到聊天列表，重新截图后再尝试" for _step, label, _log_label in reports)


def test_xinyu_go_back_stops_when_chat_list_ocr_matches(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["_runtime_cache"] = {"template_points": {"back": (0.23, 0.07)}}
    config["template_match_regions"] = {"screen.chat_list": [0.0, 0.0, 1.0, 0.2]}
    robot = FakeRobot(["chat", "chat", "list_by_ocr"])

    def fake_match_template_with_region(image, template, config, key, *, threshold, match_func):
        if Path(template).name in {"btn_返回按钮_2.jpg", "btn_返回按钮.jpg"}:
            raise AssertionError("cached back position should avoid template matching")
        if Path(template).name == "screen_聊天列表.jpg":
            return None
        return None

    monkeypatch.setattr(steps, "match_template_with_region", fake_match_template_with_region)
    monkeypatch.setattr(steps, "ensure_send_cleared", lambda robot, config, reporter=None: 0)
    monkeypatch.setattr(
        steps,
        "capture_after_popups",
        lambda robot, config, label="screen", max_dismiss=3, reporter=None, **kwargs: robot.get_transform_image(),
    )
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: (
            {"ocr_items": [{"text": "消息通话"}], "result": {"message_list": [], "bottom_nav": []}}
            if image == "list_by_ocr"
            else {"ocr_items": [], "result": {"message_list": [], "bottom_nav": []}}
        ),
    )

    back_clicks = steps.go_back(robot, config)

    assert robot.clicks == [pytest.approx((0.23, 0.07))]
    assert back_clicks == 1
    assert config["_runtime_cache"]["metrics"]["return_reason"] == "single_back_success"


def test_xinyu_go_back_single_clicks_when_not_after_send(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["chat", "chat", "chat", "chat", "list"])

    def fake_match(image, template, threshold=0.85):
        if image == "chat" and Path(template).name in {"btn_返回按钮_2.jpg", "btn_返回按钮.jpg"}:
            return {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1, "score": 0.99}
        if image == "list" and Path(template).name == "screen_聊天列表.jpg":
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "score": 0.99}
        return None

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps, "ensure_send_cleared", lambda robot, config, reporter=None: 0)
    monkeypatch.setattr(
        steps,
        "capture_after_popups",
        lambda robot, config, label="screen", max_dismiss=3, reporter=None, **kwargs: robot.get_transform_image(),
    )

    back_clicks = steps.go_back(robot, config)

    # 无 double 参数：单次点击后确认循环兜底图 "list" 判定成功即返回。
    assert robot.clicks == [pytest.approx((0.15, 0.15))]
    assert back_clicks == 1


def test_xinyu_keep_screen_awake_clicks_message_bottom_nav(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])

    monkeypatch.setattr(steps, "ensure_list_screen", lambda robot, config, reporter=None: "list")
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "bottom_nav": [
                    {"name": "全部消息", "bounds": [0.0, 0.1, 0.2, 0.2]},
                    {"name": "喜欢", "bounds": [0.0, 0.8, 0.2, 1.0]},
                    {"name": "消息", "bounds": [0.6, 0.8, 0.8, 1.0]},
                ]
            },
            "ocr_items": [],
        },
    )
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)

    assert steps.keep_screen_awake(robot, config)
    assert robot.clicks == [pytest.approx((0.7, 0.88))]


def test_xinyu_keep_screen_awake_reuses_cached_message_bottom_nav(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["_runtime_cache"] = {"screen_anchors": {"bottom_message": (0.72, 0.91)}}
    robot = FakeRobot(["list"])

    monkeypatch.setattr(steps, "ensure_list_screen", lambda robot, config, reporter=None: "list")
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: (_ for _ in ()).throw(AssertionError("cached message should avoid OCR")),
    )
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)

    assert steps.keep_screen_awake(robot, config)
    assert robot.clicks == [pytest.approx((0.72, 0.91))]


def test_xinyu_bottom_nav_coordinates_are_cached_once(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["_runtime_cache"] = {
        "screen_anchors": {
            "bottom_message": (0.72, 0.91),
            "bottom_my": (0.88, 0.91),
        }
    }
    robot = FakeRobot(["list"])

    monkeypatch.setattr(steps, "ensure_list_screen", lambda robot, config, reporter=None: "list")
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "bottom_nav": [
                    {"name": "消息", "bounds": [0.1, 0.8, 0.2, 1.0]},
                    {"name": "我的", "bounds": [0.2, 0.8, 0.3, 1.0]},
                ]
            },
            "ocr_items": [],
        },
    )
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)

    assert steps.keep_screen_awake(robot, config)

    anchors = config["_runtime_cache"]["screen_anchors"]
    assert anchors["bottom_message"] == (0.72, 0.91)
    assert anchors["bottom_my"] == (0.88, 0.91)
    assert robot.clicks == [pytest.approx((0.72, 0.91))]


def test_xinyu_caches_all_five_bottom_navigation_positions(tmp_path):
    config = _config(tmp_path)
    ocr_result = {
        "result": {
            "bottom_nav": [
                {"name": "缘分", "bounds": [0.02, 0.86, 0.14, 0.94]},
                {"name": "动态", "bounds": [0.22, 0.86, 0.34, 0.94]},
                {"name": "娱乐交友", "bounds": [0.42, 0.86, 0.58, 0.94]},
                {"name": "消息", "bounds": [0.62, 0.86, 0.74, 0.94]},
                {"name": "我的", "bounds": [0.82, 0.86, 0.94, 0.94]},
            ]
        }
    }

    steps._cache_bottom_nav_from_ocr_result(config, ocr_result)

    anchors = config["_runtime_cache"]["screen_anchors"]
    assert anchors["bottom_fate"] == pytest.approx((0.08, 0.88))
    assert anchors["bottom_feed"] == pytest.approx((0.28, 0.88))
    assert anchors["bottom_entertainment"] == pytest.approx((0.5, 0.88))
    assert anchors["bottom_message"] == pytest.approx((0.68, 0.88))
    assert anchors["bottom_my"] == pytest.approx((0.88, 0.88))


def test_xinyu_infers_all_bottom_navigation_positions_from_one_ocr_item(tmp_path):
    config = _config(tmp_path)
    ocr_result = {
        "result": {
            "bottom_nav": [
                {"name": "消息", "bounds": [0.62, 0.86, 0.74, 0.94]},
            ]
        }
    }

    steps._cache_bottom_nav_from_ocr_result(config, ocr_result)

    anchors = config["_runtime_cache"]["screen_anchors"]
    assert anchors["bottom_fate"] == pytest.approx((0.08, 0.88))
    assert anchors["bottom_feed"] == pytest.approx((0.28, 0.88))
    assert anchors["bottom_entertainment"] == pytest.approx((0.48, 0.88))
    assert anchors["bottom_message"] == pytest.approx((0.68, 0.88))
    assert anchors["bottom_my"] == pytest.approx((0.88, 0.88))


def test_xinyu_caches_my_from_message_when_my_bottom_nav_missing(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])

    monkeypatch.setattr(steps, "ensure_list_screen", lambda robot, config, reporter=None: "list")
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {
                "bottom_nav": [
                    {"name": "消息", "bounds": [0.56, 0.86, 0.68, 0.94]},
                ]
            },
            "ocr_items": [],
        },
    )
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)

    assert steps.keep_screen_awake(robot, config)

    anchors = config["_runtime_cache"]["screen_anchors"]
    assert anchors["bottom_message"] == pytest.approx((0.62, 0.88))
    assert anchors["bottom_my"] == pytest.approx((0.82, 0.88))


def test_xinyu_keep_screen_awake_does_not_click_without_message_bottom_nav(monkeypatch, tmp_path):
    config = _config(tmp_path)
    robot = FakeRobot(["list"])

    monkeypatch.setattr(steps, "ensure_list_screen", lambda robot, config, reporter=None: "list")
    monkeypatch.setattr(
        steps,
        "run_ocr_mode",
        lambda image, cfg, mode, region=None: {
            "result": {"bottom_nav": [{"name": "主页", "bounds": [0.8, 0.8, 1.0, 1.0]}]},
            "ocr_items": [],
        },
    )

    assert steps.keep_screen_awake(robot, config) is False
    assert robot.clicks == []


def test_xinyu_play_voice_reply_reclicks_microphone_when_tts_fails(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["tts_retry_attempts"] = 1
    robot = FakeRobot(["microphone", "microphone"])
    played: list[str] = []

    def fake_match(image, template, threshold=0.85):
        if image == "microphone" and Path(template).name == "btn_麦克风按钮.jpg":
            return {"x": 0.4, "y": 0.5, "w": 0.2, "h": 0.2, "score": 0.99}
        return None

    def fake_play(robot, chunks):
        played.append("".join(chunks))
        if len(played) == 1:
            raise RuntimeError("speaker timeout")

    monkeypatch.setattr(steps, "match", fake_match)
    monkeypatch.setattr(steps, "prepare_tts_input", lambda robot, text: [text])
    monkeypatch.setattr(steps, "play_prepared_tts_input", fake_play)
    monkeypatch.setattr(steps.time, "sleep", lambda seconds: None)

    steps.play_voice_reply(robot, "你好", config)

    assert played == ["你好", "你好"]
    assert robot.clicks == [pytest.approx((0.5, 0.6)), pytest.approx((0.5, 0.6))]
    assert config["_runtime_cache"]["metrics"]["microphone_template"] == "btn_麦克风按钮.jpg"
    assert config["_runtime_cache"]["metrics"]["tts_success"] is True
    assert config["_runtime_cache"]["metrics"]["tts_retry_count"] == 1
