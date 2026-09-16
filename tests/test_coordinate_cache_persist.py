from __future__ import annotations

import json
from pathlib import Path

import pytest

from services import coordinate_cache
from services.coordinate_cache import (
    flush_persist_debounce,
    load_into_runtime,
    persist_from_runtime,
)


@pytest.fixture(autouse=True)
def _isolated_root(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinate_cache, "DEFAULT_ROOT", tmp_path / "coordinate_cache")
    monkeypatch.setattr(coordinate_cache, "_DIRTY", {})
    yield
    flush_persist_debounce()


def _make_config(*, signature: str = "sig-1") -> dict:
    return {
        "_runtime": {
            "device_ip": "1.1.1.1",
            "app_name": "xinyu",
            "perspective_signature": signature,
        },
        "_runtime_cache": {},
    }


def test_persist_debounce_coalesces_writes_and_flush_writes_final_value():
    config = _make_config()
    cache_file = coordinate_cache._path(config)

    for i in range(5):
        config["_runtime_cache"].setdefault("ocr_regions", {})[f"key_{i}"] = [0.1 * i, 0.2, 0.3, 0.4]
        persist_from_runtime(config)
    # 防抖窗口内：尚未落盘。
    assert not cache_file.exists()

    config["_runtime_cache"]["ocr_regions"]["final"] = [0.9, 0.8, 0.7, 0.6]
    persist_from_runtime(config)
    flush_persist_debounce()

    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    assert set(payload["ocr_regions"]) == {"key_0", "key_1", "key_2", "key_3", "key_4", "final"}


def test_persist_immediate_bypasses_debounce():
    config = _make_config(signature="sig-2")
    cache_file = coordinate_cache._path(config)

    config["_runtime_cache"]["template_points"] = {"btn_back": [0.5, 0.5]}
    persist_from_runtime(config, immediate=True)

    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    assert payload["template_points"]["btn_back"] == [0.5, 0.5]


def test_load_into_runtime_skips_unchanged_file():
    config = _make_config(signature="sig-3")
    config["_runtime_cache"]["template_points"] = {"btn_send": [0.25, 0.75]}
    persist_from_runtime(config, immediate=True)

    other = _make_config(signature="sig-3")
    assert load_into_runtime(other) is True
    assert other["_runtime_cache"]["template_points"]["btn_send"] == [0.25, 0.75]

    # 文件未变化：第二次加载凭文件戳短路，不重读盘——已被 pop 的键
    # 不会从文件恢复（若未短路，btn_send 会被重新加载回来）。
    other["_runtime_cache"]["template_points"].pop("btn_send")
    assert load_into_runtime(other) is True
    assert "btn_send" not in other["_runtime_cache"]["template_points"]

    # 文件变化后短路失效，重新读到新值。
    payload = json.loads(coordinate_cache._path(other).read_text(encoding="utf-8"))
    payload["template_points"]["btn_send"] = [0.3, 0.8]
    coordinate_cache._atomic_write(coordinate_cache._path(other), payload)
    assert load_into_runtime(other) is True
    assert other["_runtime_cache"]["template_points"]["btn_send"] == [0.3, 0.8]
