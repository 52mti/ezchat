from __future__ import annotations

from apps.xinyu.config import DEFAULT_CONFIG as XINYU_CONFIG


def test_xinyu_owns_points_ocr_mode():
    assert "points" in XINYU_CONFIG["ocr_modes"]
    assert XINYU_CONFIG["ocr_modes"]["points"]["preprocess"]["mode"] == "original"
    assert "\u6211\u7684\u79ef\u5206" in XINYU_CONFIG["ocr_modes"]["points"]["rules"]["keywords"]
    assert "points_check_interval_seconds" not in XINYU_CONFIG
