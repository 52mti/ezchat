"""数据分层规范测试：outbox 迁入 runtime、photo 迁入 runtime、事故归档清理。"""

from __future__ import annotations

import json
import os
import time

import services.photo_capture as photo_capture_module
from services.config_manager import ConfigManager
from services.photo_capture import ensure_photo_root
from services.retention import cleanup_old_error_incidents


def test_config_manager_outbox_defaults_to_runtime(tmp_path):
    manager = ConfigManager(config_path=tmp_path / "config.json")

    assert manager.outbox_path == tmp_path / "runtime" / "outbox.json"


def test_legacy_root_outbox_migrates_into_runtime(tmp_path):
    legacy = tmp_path / "outbox.json"
    legacy.write_text(json.dumps({"schema_version": 1, "messages": [{"message_fingerprint": "fp-1"}]}), encoding="utf-8")

    manager = ConfigManager(config_path=tmp_path / "config.json")

    assert not legacy.exists()
    assert (tmp_path / "runtime" / "outbox.json").is_file()
    payload = json.loads((tmp_path / "runtime" / "outbox.json").read_text(encoding="utf-8"))
    assert payload["messages"][0]["message_fingerprint"] == "fp-1"
    assert manager.load_outbox()[0]["message_fingerprint"] == "fp-1"


def test_existing_runtime_outbox_wins_over_stale_legacy(tmp_path):
    """runtime 副本与旧文件都非空且更新时，runtime 副本胜出。"""
    (tmp_path / "runtime").mkdir()
    runtime_copy = tmp_path / "runtime" / "outbox.json"
    runtime_copy.write_text(json.dumps({"schema_version": 1, "messages": [{"message_fingerprint": "current"}]}), encoding="utf-8")
    newer = time.time() + 60
    os.utime(runtime_copy, (newer, newer))
    legacy = tmp_path / "outbox.json"
    legacy.write_text(json.dumps({"schema_version": 1, "messages": [{"message_fingerprint": "stale"}]}), encoding="utf-8")

    manager = ConfigManager(config_path=tmp_path / "config.json")

    assert legacy.exists()  # 更旧的旧文件保留不搬
    assert manager.load_outbox()[0]["message_fingerprint"] == "current"


def test_nonempty_legacy_outbox_beats_newer_empty_runtime_copy(tmp_path):
    """真实事故形态：后写入的空壳（如测试产物）不得盖掉非空的旧账本。"""
    (tmp_path / "runtime").mkdir()
    runtime_copy = tmp_path / "runtime" / "outbox.json"
    runtime_copy.write_text(json.dumps({"schema_version": 1, "messages": []}), encoding="utf-8")
    newer = time.time() + 60
    os.utime(runtime_copy, (newer, newer))
    legacy = tmp_path / "outbox.json"
    legacy.write_text(json.dumps({"schema_version": 1, "messages": [{"message_fingerprint": "real-1"}]}), encoding="utf-8")

    manager = ConfigManager(config_path=tmp_path / "config.json")

    assert not legacy.exists()
    assert manager.load_outbox()[0]["message_fingerprint"] == "real-1"


def test_fresher_legacy_outbox_replaces_stale_runtime_copy(tmp_path):
    """旧版程序最后写下的根文件比 runtime 副本新时必须以它为准（已发送标记不回退）。"""
    import os
    import time

    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "outbox.json").write_text(json.dumps({"schema_version": 1, "messages": []}), encoding="utf-8")
    legacy = tmp_path / "outbox.json"
    legacy.write_text(json.dumps({"schema_version": 1, "messages": [{"message_fingerprint": "fresh-1"}]}), encoding="utf-8")
    newer = time.time() + 60
    os.utime(legacy, (newer, newer))

    manager = ConfigManager(config_path=tmp_path / "config.json")

    assert not legacy.exists()
    assert manager.load_outbox()[0]["message_fingerprint"] == "fresh-1"


def test_legacy_photo_root_migrates_into_runtime(tmp_path, monkeypatch):
    legacy = tmp_path / "photo"
    target = tmp_path / "runtime" / "photo"
    (legacy / "errors").mkdir(parents=True)
    (legacy / "errors" / "a.jpg").write_bytes(b"jpg")
    monkeypatch.setattr(photo_capture_module, "_LEGACY_PHOTO_ROOT", legacy)
    monkeypatch.setattr(photo_capture_module, "PHOTO_ROOT", target)

    assert ensure_photo_root() == target
    assert (target / "errors" / "a.jpg").read_bytes() == b"jpg"
    assert not legacy.exists()


def test_photo_root_kept_when_target_exists(tmp_path, monkeypatch):
    legacy = tmp_path / "photo"
    target = tmp_path / "runtime" / "photo"
    target.mkdir(parents=True)
    (target / "keep.txt").write_text("kept", encoding="utf-8")
    legacy.mkdir()  # 空旧目录：合并后直接清掉
    monkeypatch.setattr(photo_capture_module, "_LEGACY_PHOTO_ROOT", legacy)
    monkeypatch.setattr(photo_capture_module, "PHOTO_ROOT", target)

    ensure_photo_root()

    assert (target / "keep.txt").exists()
    assert not legacy.exists()


def test_photo_root_merges_leftovers_when_target_exists(tmp_path, monkeypatch):
    legacy = tmp_path / "photo"
    target = tmp_path / "runtime" / "photo"
    target.mkdir(parents=True)
    (target / "keep.txt").write_text("kept", encoding="utf-8")
    (legacy / "errors").mkdir(parents=True)
    (legacy / "errors" / "new.jpg").write_bytes(b"jpg")
    (legacy / "keep.txt").write_text("legacy", encoding="utf-8")  # 目标同名已存在，不覆盖
    monkeypatch.setattr(photo_capture_module, "_LEGACY_PHOTO_ROOT", legacy)
    monkeypatch.setattr(photo_capture_module, "PHOTO_ROOT", target)

    ensure_photo_root()

    assert (target / "keep.txt").read_text(encoding="utf-8") == "kept"
    assert (target / "errors" / "new.jpg").read_bytes() == b"jpg"
    # 与目标同名冲突的旧文件不覆盖、留在原地，避免丢数据
    assert legacy.exists() and (legacy / "keep.txt").is_file()
    assert not (legacy / "errors").exists()


def test_cleanup_old_error_incidents_prunes_by_date(tmp_path):
    import datetime

    root = tmp_path / "error_incidents"
    old_date = (datetime.date.today() - datetime.timedelta(days=60)).strftime("%Y%m%d")
    recent_date = datetime.date.today().strftime("%Y%m%d")
    (root / old_date).mkdir(parents=True)
    (root / old_date / "incident.json").write_text("{}", encoding="utf-8")
    (root / recent_date).mkdir()
    (root / recent_date / "incident.json").write_text("{}", encoding="utf-8")

    removed = cleanup_old_error_incidents(root, retention_days=30)

    assert removed == 1
    assert not (root / old_date).exists()
    assert (root / recent_date / "incident.json").is_file()
