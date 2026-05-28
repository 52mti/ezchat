from __future__ import annotations

from apps.wechat.config import DEFAULT_CONFIG


def test_wechat_runtime_ocr_enabled_by_default():
    assert DEFAULT_CONFIG["ocr"]["runtime_enabled"] is True


def test_wechat_poll_interval_defaults_to_one_second():
    assert DEFAULT_CONFIG["poll_interval"] == 1


def test_wechat_keep_awake_defaults_to_200_seconds():
    assert DEFAULT_CONFIG["keep_awake_interval_seconds"] == 200
