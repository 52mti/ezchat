from __future__ import annotations

from apps.xinyu import steps
from apps.xinyu.config import DEFAULT_CONFIG
from apps.xinyu.ocr_config import LIST_OCR_MODE
from apps.xinyu.ocr_rules import build_list_result


def test_unread_tab_falls_back_to_raw_ocr_when_banner_pushes_filter_down(monkeypatch):
    ocr_result = {
        "result": {"bottom_nav": []},
        "ocr_items": [{"text": "未回复", "x": 0.294, "y": 0.240, "w": 0.132, "h": 0.024}],
    }
    item = steps._unread_tab_item(ocr_result)

    assert item is not None
    nav_item, center = item
    assert nav_item["bounds"] == [0.294, 0.240, 0.426, 0.264]
    assert center == (0.36, 0.252)

    monkeypatch.setattr(
        steps,
        "_red_pixel_stats",
        lambda _image, _match: {"has_red": True, "red_pixels": 473, "total_pixels": 6732, "threshold": 13, "region": [149, 299, 281, 350]},
    )
    image, checked = steps.check_unread_button(object(), dict(DEFAULT_CONFIG), image="screen", ocr_result=ocr_result)

    assert image == "screen"
    assert checked["_xinyu_unread_tab_has_red"] is True


def test_xinyu_list_starts_after_detected_top_filter_strip():
    rules = LIST_OCR_MODE["rules"]
    def item(text: str, left: int, right: int, top: int, bottom: int) -> dict:
        return {
            "text": text, "raw_text": text, "confidence": 0.99,
            "min_x": left, "max_x": right, "min_y": top, "max_y": bottom,
            "center_x": (left + right) / 2, "center_y": (top + bottom) / 2,
            "width": right - left, "height": bottom - top,
        }

    items = [
        item("全部消息", 30, 120, 240, 265), item("未回复", 175, 255, 240, 265),
        item("在线", 295, 350, 240, 265), item("距离", 390, 445, 240, 265),
        item("亲密度", 485, 560, 240, 265), item("薄荷", 120, 180, 310, 335),
        item("看看你的诚意", 120, 300, 340, 365),
    ]

    result = build_list_result(items, (600, 1000), rules)

    assert {item["name"] for item in result["bottom_nav"]} >= {"全部消息", "未回复", "在线", "距离", "亲密度"}
    assert [item["name"] for item in result["message_list"]] == ["薄荷"]


def test_xinyu_list_confirmation_wins_over_false_chat_template(monkeypatch):
    config = dict(DEFAULT_CONFIG)
    reports: list[str] = []
    monkeypatch.setattr(steps, "_matches_chat_list_screen", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        steps,
        "_matches_screen",
        lambda _image, _config, key: key == steps.CHAT_WINDOW_SCREEN,
    )
    monkeypatch.setattr(steps, "wait_for_call_screen_to_disappear", lambda _robot, _config, image, **_kwargs: image)

    image = steps.ensure_list_screen(
        object(), config, image="list", reporter=lambda _step, label, log_label=None: reports.append(log_label or label)
    )

    assert image == "list"
    assert "已确认聊天列表（第 1 轮）" in reports
    assert "启动时在聊天界面，匹配返回按钮回到列表" not in reports


def test_xinyu_list_confirmation_never_falls_back_to_template(monkeypatch):
    config = dict(DEFAULT_CONFIG)
    monkeypatch.setattr(steps, "_run_ocr_mode", lambda *_args, **_kwargs: {"ocr_items": []})
    monkeypatch.setattr(
        steps,
        "_matches_screen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("不应匹配聊天列表模板")),
    )

    assert steps._matches_chat_list_screen("list", config) is False


def test_check_unread_button_full_fallback_has_no_orphan_calls(monkeypatch):
    """条带未见未回复入口 → 回退完整 OCR 的路径不得调用已删除的导航缓存函数。

    回归背景：导航合并下线时漏删了回退路径里的调用点，全屏 OCR 跑完后
    NameError → app_error → 设备复位（2026-09-13 20:05 事故）。
    """
    import apps.xinyu.steps as steps_module

    ocr_result = {"result": {"bottom_nav": []}, "ocr_items": []}  # 无未回复入口
    full_result = {
        "result": {"bottom_nav": []},
        "ocr_items": [{"text": "未回复", "x": 0.29, "y": 0.115, "w": 0.13, "h": 0.025}],
    }
    monkeypatch.setattr(steps_module, "_run_ocr_mode", lambda *_a, **_k: full_result)
    monkeypatch.setattr(steps_module, "click_image_point", lambda *a, **k: None)
    monkeypatch.setattr(steps_module, "capture_after_popups", lambda *a, **k: "screen")
    monkeypatch.setattr(steps_module.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        steps_module,
        "_red_pixel_stats",
        lambda _i, _m: {"has_red": True, "red_pixels": 400, "total_pixels": 7000, "threshold": 14, "region": [0, 0, 1, 1]},
    )
    for gone in ("_cache_list_navigation_result", "_merge_list_navigation_result"):
        assert not hasattr(steps_module, gone)

    image, checked = steps_module.check_unread_button(object(), dict(DEFAULT_CONFIG), image="screen", ocr_result=ocr_result)

    assert checked["_xinyu_unread_tab_has_red"] is True  # 全屏结果里找到入口并确认红色
