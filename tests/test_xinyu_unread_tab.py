from __future__ import annotations

from apps.xinyu import steps
from apps.xinyu.config import DEFAULT_CONFIG


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
