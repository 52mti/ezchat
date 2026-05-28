from __future__ import annotations

import json

from services.config_manager import ConfigManager


def test_config_manager_get_set_and_atomic_write(tmp_path):
    config_path = tmp_path / "config.json"
    outbox_path = tmp_path / "outbox.json"
    manager = ConfigManager(config_path, outbox_path)

    manager.set("global", "apikey", value="abc123")

    assert manager.get("global", "apikey") == "abc123"
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["global"]["apikey"] == "abc123"


def test_config_manager_has_global_greetings_dir_default(tmp_path):
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")

    assert manager.get("global", "greetings_dir") == "runtime/greetings"
    assert "greetings_dir" not in manager.get_app_config("1.1.1.1", "wechat", {})


def test_config_manager_environment_override_is_not_written(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    outbox_path = tmp_path / "outbox.json"
    config_path.write_text(
        json.dumps({"global": {"apikey": "file-key"}, "devices": []}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CHATBOT_APIKEY", "env-key")

    manager = ConfigManager(config_path, outbox_path)

    assert manager.get("global", "apikey") == "env-key"
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["global"]["apikey"] == "file-key"


def test_config_manager_upserts_outbox_entry(tmp_path):
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")

    manager.save_outbox_entry({"message_fingerprint": "fp1", "pending_send": True, "conversation_id": "cid"})
    manager.save_outbox_entry({"message_fingerprint": "fp1", "pending_send": False})

    assert manager.load_outbox() == [
        {"message_fingerprint": "fp1", "pending_send": False, "conversation_id": "cid"}
    ]


def test_config_manager_device_helpers_and_app_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "devices": [
                    {
                        "ip": "1.1.1.1",
                        "app_overrides": {"wechat": {"poll_interval": 9}},
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manager = ConfigManager(config_path, tmp_path / "outbox.json")

    app_config = manager.get_app_config("1.1.1.1", "wechat", {"poll_interval": 5})
    manager.update_device("1.1.1.1", {"license_expires": "2099-01-01"})

    assert app_config["poll_interval"] == 9
    assert manager.find_device("1.1.1.1")["license_expires"] == "2099-01-01"


def test_config_manager_normalizes_device_template_paths_to_app_scope(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "devices": [
                    {
                        "ip": "192.168.1.108",
                        "selected_app": "wechat",
                        "app_overrides": {
                            "wechat": {
                                "templates": {
                                    "buttons": "runtime\\templates\\192_168_1_108\\wechat\\buttons",
                                    "screens": "runtime\\templates\\192_168_1_108\\wechat\\screens",
                                }
                            }
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manager = ConfigManager(config_path, tmp_path / "outbox.json")
    app_config = manager.get_app_config("192.168.1.108", "wechat", {})

    assert app_config["templates"]["buttons"] == "runtime\\templates\\wechat\\buttons"
    assert app_config["templates"]["screens"] == "runtime\\templates\\wechat\\screens"


def test_config_manager_migrates_and_backs_up_old_schema(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"schema_version": 0, "devices": [{"ip": "1.1.1.1"}]}),
        encoding="utf-8",
    )

    manager = ConfigManager(config_path, tmp_path / "outbox.json")
    device = manager.find_device("1.1.1.1")

    assert manager.get("schema_version") == 1
    assert device["selected_app"] == "wechat"
    assert (tmp_path / "config.json.bak").exists()


def test_config_manager_add_device_sorts_and_rejects_duplicates(tmp_path):
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")

    assert manager.add_device({"ip": "192.168.1.20", "selected_app": "wechat"}) is True
    assert manager.add_device({"ip": "192.168.1.3", "selected_app": "wechat"}) is True
    assert manager.add_device({"ip": "192.168.1.20", "selected_app": "wechat"}) is False

    devices = manager.devices()
    assert [device["ip"] for device in devices] == ["192.168.1.3", "192.168.1.20"]


def test_config_manager_remove_device(tmp_path):
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")
    manager.add_device({"ip": "192.168.1.20", "selected_app": "wechat"})
    manager.add_device({"ip": "192.168.1.3", "selected_app": "wechat"})

    assert manager.remove_device("192.168.1.20") is True
    assert manager.remove_device("192.168.1.20") is False
    assert [device["ip"] for device in manager.devices()] == ["192.168.1.3"]
