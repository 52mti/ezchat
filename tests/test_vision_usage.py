from __future__ import annotations

from services import vision_usage


def test_vision_usage_survives_process_cache_reset(tmp_path, monkeypatch):
    path = tmp_path / "vision_usage.json"
    monkeypatch.setattr(vision_usage, "DEFAULT_PATH", path)
    config = {"_runtime": {"device_ip": "192.168.1.1", "app_name": "xinyu", "perspective_signature": "same"}}
    assert vision_usage.acquire(config, limit=2) == (True, 1)
    assert vision_usage.acquire(config, limit=2) == (True, 2)
    assert vision_usage.acquire(config, limit=2) == (False, 2)
    # New config object simulates a process restart.
    restarted = {"_runtime": dict(config["_runtime"])}
    assert vision_usage.acquire(restarted, limit=2) == (False, 2)


def test_vision_usage_prefers_package_root_when_loaded_from_zip(tmp_path, monkeypatch):
    package_root = tmp_path / "package"
    package_root.mkdir()
    monkeypatch.setenv("JIXIEBI_PACKAGE_ROOT", str(package_root))
    monkeypatch.setattr(vision_usage, "DEFAULT_PATH", package_root / "app_modules.zip" / "runtime" / "vision_usage.json")

    assert vision_usage.acquire({}, limit=1) == (True, 1)
    assert (package_root / "runtime" / "vision_usage.json").is_file()
    assert not (package_root / "app_modules.zip").exists()


def test_vision_usage_isolated_by_perspective(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_usage, "DEFAULT_PATH", tmp_path / "vision_usage.json")
    first = {"_runtime": {"device_ip": "1", "app_name": "xinyu", "perspective_signature": "a"}}
    second = {"_runtime": {"device_ip": "1", "app_name": "xinyu", "perspective_signature": "b"}}
    assert vision_usage.acquire(first, limit=1) == (True, 1)
    assert vision_usage.acquire(second, limit=1) == (True, 1)


def test_rule_learning_disables_after_consecutive_zero_results(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_usage, "DEFAULT_PATH", tmp_path / "vision_usage.json")
    config = {"_runtime": {"device_ip": "1", "app_name": "xinyu", "perspective_signature": "same"}}

    assert vision_usage.rule_learning_status(config) == {}
    first = vision_usage.record_rule_learning(config, added_count=0, disable_after_zero=3)
    second = vision_usage.record_rule_learning(config, added_count=0, disable_after_zero=3)
    third = vision_usage.record_rule_learning(config, added_count=0, disable_after_zero=3)

    assert first["zero_count"] == 1 and not first["disabled"]
    assert second["zero_count"] == 2 and not second["disabled"]
    assert third["zero_count"] == 3 and third["disabled"]

    reset = vision_usage.record_rule_learning(config, added_count=1, disable_after_zero=3)
    assert reset == {"zero_count": 0, "disabled": False, "threshold": 3}


def test_rule_learning_isolated_by_perspective(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_usage, "DEFAULT_PATH", tmp_path / "vision_usage.json")
    first = {"_runtime": {"device_ip": "1", "app_name": "xinyu", "perspective_signature": "a"}}
    second = {"_runtime": {"device_ip": "1", "app_name": "xinyu", "perspective_signature": "b"}}

    vision_usage.record_rule_learning(first, added_count=0, disable_after_zero=1)

    assert vision_usage.rule_learning_status(first)["disabled"] is True
    assert vision_usage.rule_learning_status(second) == {}


def test_reset_clears_usage_and_rule_learning_for_all_device_app_perspectives(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_usage, "DEFAULT_PATH", tmp_path / "vision_usage.json")
    first = {"_runtime": {"device_ip": "1", "app_name": "xinyu", "perspective_signature": "a"}}
    second = {"_runtime": {"device_ip": "1", "app_name": "xinyu", "perspective_signature": "b"}}
    other = {"_runtime": {"device_ip": "2", "app_name": "xinyu", "perspective_signature": "a"}}

    assert vision_usage.acquire(first, limit=1) == (True, 1)
    assert vision_usage.acquire(second, limit=1) == (True, 1)
    assert vision_usage.acquire(other, limit=1) == (True, 1)
    vision_usage.record_rule_learning(first, added_count=0, disable_after_zero=1)

    vision_usage.reset({"_runtime": {"device_ip": "1", "app_name": "xinyu"}})

    assert vision_usage.acquire(first, limit=1) == (True, 1)
    assert vision_usage.acquire(second, limit=1) == (True, 1)
    assert vision_usage.rule_learning_status(first) == {}
    assert vision_usage.acquire(other, limit=1) == (False, 1)
