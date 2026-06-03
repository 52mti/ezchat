from __future__ import annotations

import json

from apps.wechat.config import DEFAULT_CONFIG as WECHAT_DEFAULT_CONFIG
from services.config_manager import ConfigManager


def _read_app_config(tmp_path, app_name):
    path = tmp_path / "apps" / app_name / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_config_manager_get_set_and_atomic_write(tmp_path):
    config_path = tmp_path / "config.json"
    outbox_path = tmp_path / "outbox.json"
    manager = ConfigManager(config_path, outbox_path)

    manager.set("global", "apikey", value="abc123")

    assert manager.get("global", "apikey") == "abc123"
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert "apikey" not in saved["global"]
    assert (tmp_path / ".env" / "apikey.txt").read_text(encoding="utf-8") == "CHATBOT_APIKEY=abc123\n"


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
    assert "apikey" not in saved["global"]


def test_config_manager_local_env_apikey_override_is_not_written(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    outbox_path = tmp_path / "outbox.json"
    config_path.write_text(
        json.dumps({"global": {"apikey": "file-key"}, "devices": []}),
        encoding="utf-8",
    )
    env_dir = tmp_path / ".env"
    env_dir.mkdir()
    (env_dir / "apikey.txt").write_text("# local only\nCHATBOT_APIKEY=local-key\n", encoding="utf-8")
    monkeypatch.delenv("CHATBOT_APIKEY", raising=False)

    manager = ConfigManager(config_path, outbox_path)

    assert manager.get("global", "apikey") == "local-key"
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert "apikey" not in saved["global"]


def test_config_manager_reads_local_env_apikey_with_utf8_bom(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    outbox_path = tmp_path / "outbox.json"
    env_dir = tmp_path / ".env"
    env_dir.mkdir()
    (env_dir / "apikey.txt").write_text("\ufeff41d83b9a-525b-424b-bbea-c5857de9d21c\n", encoding="utf-8")
    monkeypatch.delenv("CHATBOT_APIKEY", raising=False)

    manager = ConfigManager(config_path, outbox_path)

    assert manager.get("global", "apikey") == "41d83b9a-525b-424b-bbea-c5857de9d21c"


def test_config_manager_reads_key_value_local_env_apikey_with_utf8_bom(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    outbox_path = tmp_path / "outbox.json"
    env_dir = tmp_path / ".env"
    env_dir.mkdir()
    (env_dir / "apikey.txt").write_text("\ufeffCHATBOT_APIKEY=bom-key\n", encoding="utf-8")
    monkeypatch.delenv("CHATBOT_APIKEY", raising=False)

    manager = ConfigManager(config_path, outbox_path)

    assert manager.get("global", "apikey") == "bom-key"


def test_config_manager_environment_apikey_wins_over_local_env_file(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    outbox_path = tmp_path / "outbox.json"
    config_path.write_text(
        json.dumps({"global": {"apikey": "file-key"}, "devices": []}),
        encoding="utf-8",
    )
    env_dir = tmp_path / ".env"
    env_dir.mkdir()
    (env_dir / "apikey.txt").write_text("local-key\n", encoding="utf-8")
    monkeypatch.setenv("CHATBOT_APIKEY", "real-env-key")

    manager = ConfigManager(config_path, outbox_path)

    assert manager.get("global", "apikey") == "real-env-key"
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert "apikey" not in saved["global"]


def test_config_manager_migrates_file_apikey_to_local_env(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    outbox_path = tmp_path / "outbox.json"
    config_path.write_text(
        json.dumps({"global": {"apikey": "file-key"}, "devices": []}),
        encoding="utf-8",
    )
    monkeypatch.delenv("CHATBOT_APIKEY", raising=False)

    manager = ConfigManager(config_path, outbox_path)

    assert manager.get("global", "apikey") == "file-key"
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert "apikey" not in saved["global"]
    assert (tmp_path / ".env" / "apikey.txt").read_text(encoding="utf-8") == "CHATBOT_APIKEY=file-key\n"


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
                        "app_overrides": {
                            "wechat": {
                                "poll_interval": 9,
                                "ocr_modes": {"dialogue": {"rules": {"line_gap": 1}}},
                            }
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manager = ConfigManager(config_path, tmp_path / "outbox.json")

    app_config = manager.get_app_config("1.1.1.1", "wechat", WECHAT_DEFAULT_CONFIG)
    manager.update_device("1.1.1.1", {"license_expires": "2099-01-01"})
    device = manager.find_device("1.1.1.1") or {}

    assert app_config["poll_interval"] == 9
    assert app_config["ocr_modes"]["dialogue"]["rules"]["line_gap"] == 1
    assert device["runtime_overrides"]["poll_interval"] == 9
    assert device["license_expires"] == "2099-01-01"

def test_config_manager_migrates_legacy_app_override_scopes_on_update(tmp_path):
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")

    manager.update_device(
        "1.1.1.1",
        {
            "app_overrides": {
                "wechat": {
                    "ocr_modes": {"dialogue": {"rules": {"line_gap": 1}}},
                    "poll_interval": 9,
                }
            }
        },
    )

    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    app_saved = _read_app_config(tmp_path, "wechat")
    assert saved["devices"][0]["runtime_overrides"]["poll_interval"] == 9
    assert "apps" not in saved
    assert app_saved["ocr_modes"]["dialogue"]["rules"]["line_gap"] == 1
    assert saved["devices"][0]["app_overrides"] == {}
    app_config = manager.get_app_config("1.1.1.1", "wechat", WECHAT_DEFAULT_CONFIG)
    assert app_config["poll_interval"] == 9
    assert app_config["ocr_modes"]["dialogue"]["rules"]["line_gap"] == 1


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
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    app_saved = _read_app_config(tmp_path, "wechat")

    assert app_config["templates"]["buttons"] == "apps\\wechat\\templates\\buttons"
    assert app_config["templates"]["screens"] == "apps\\wechat\\templates\\screens"
    assert "apps" not in saved
    assert app_saved["templates"]["buttons"] == "apps\\wechat\\templates\\buttons"
    assert app_saved["templates"]["screens"] == "apps\\wechat\\templates\\screens"


def test_config_manager_migrates_and_backs_up_old_schema(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"schema_version": 0, "devices": [{"ip": "1.1.1.1"}]}),
        encoding="utf-8",
    )

    manager = ConfigManager(config_path, tmp_path / "outbox.json")
    device = manager.find_device("1.1.1.1")

    assert manager.get("schema_version") == 3
    assert device["selected_app"] == "wechat"
    assert (tmp_path / "config.json.bak").exists()


def test_config_manager_updates_app_runtime_and_persona_scopes_independently(tmp_path):
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")
    manager.add_device({"ip": "1.1.1.1", "selected_app": "wechat", "app_overrides": {"wechat": {"persona": {}}}})

    manager.update_app_config("wechat", {"match_threshold": 0.91, "ocr": {"det_db_thresh": 0.21}})
    manager.update_device_runtime(
        "1.1.1.1",
        {
            "poll_interval": 6,
            "keep_awake_interval_seconds": 300,
            "exclude_ocr_wait_from_step_timeout": False,
        },
    )
    manager.update_device_app_overrides("1.1.1.1", "wechat", {"persona": {"name": "bot"}})

    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    app_saved = _read_app_config(tmp_path, "wechat")
    device = manager.find_device("1.1.1.1") or {}
    merged = manager.get_app_config("1.1.1.1", "wechat", WECHAT_DEFAULT_CONFIG)

    assert "apps" not in saved
    assert app_saved["match_threshold"] == 0.91
    assert app_saved["ocr"]["det_db_thresh"] == 0.21
    assert saved["devices"][0]["runtime_overrides"]["poll_interval"] == 6
    assert saved["devices"][0]["runtime_overrides"]["keep_awake_interval_seconds"] == 300
    assert saved["devices"][0]["runtime_overrides"]["exclude_ocr_wait_from_step_timeout"] is False
    assert saved["devices"][0]["app_overrides"]["wechat"]["persona"]["name"] == "bot"
    assert device["runtime_overrides"]["poll_interval"] == 6
    assert merged["match_threshold"] == 0.91
    assert merged["persona"]["name"] == "bot"


def test_config_manager_updates_device_app_offsets_independently(tmp_path):
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")
    manager.add_device({"ip": "1.1.1.1", "selected_app": "huanyou", "app_overrides": {"huanyou": {"persona": {}}}})

    manager.update_device_app_overrides(
        "1.1.1.1",
        "huanyou",
        {
            "new_greeting_click_x_offset": 0.02,
            "new_greeting_click_y_offset": -0.03,
            "next_message_click_x_offset": 0.05,
            "next_message_click_y_offset": -0.02,
        },
    )

    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    merged = manager.get_app_config("1.1.1.1", "huanyou", {})

    assert saved["devices"][0]["app_overrides"]["huanyou"]["new_greeting_click_x_offset"] == 0.02
    assert saved["devices"][0]["app_overrides"]["huanyou"]["new_greeting_click_y_offset"] == -0.03
    assert saved["devices"][0]["app_overrides"]["huanyou"]["next_message_click_x_offset"] == 0.05
    assert saved["devices"][0]["app_overrides"]["huanyou"]["next_message_click_y_offset"] == -0.02
    assert not (tmp_path / "apps" / "huanyou" / "config.json").exists()
    assert merged["new_greeting_click_x_offset"] == 0.02
    assert merged["new_greeting_click_y_offset"] == -0.03
    assert merged["next_message_click_x_offset"] == 0.05
    assert merged["next_message_click_y_offset"] == -0.02


def test_config_manager_migrates_legacy_app_level_offsets_to_device_app_scope(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "apps": {
                    "huanyou": {
                        "new_greeting_click_x_offset": 0.03,
                        "new_greeting_click_y_offset": -0.02,
                        "next_message_click_x_offset": 0.04,
                        "next_message_click_y_offset": -0.01,
                    }
                },
                "devices": [
                    {
                        "ip": "1.1.1.1",
                        "selected_app": "huanyou",
                        "app_overrides": {"huanyou": {"persona": {"name": "tester"}}},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manager = ConfigManager(config_path, tmp_path / "outbox.json")
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    merged = manager.get_app_config("1.1.1.1", "huanyou", {})

    assert "apps" not in saved
    assert not (tmp_path / "apps" / "huanyou" / "config.json").exists()
    assert saved["devices"][0]["app_overrides"]["huanyou"]["new_greeting_click_x_offset"] == 0.03
    assert saved["devices"][0]["app_overrides"]["huanyou"]["new_greeting_click_y_offset"] == -0.02
    assert saved["devices"][0]["app_overrides"]["huanyou"]["next_message_click_x_offset"] == 0.04
    assert saved["devices"][0]["app_overrides"]["huanyou"]["next_message_click_y_offset"] == -0.01
    assert merged["new_greeting_click_x_offset"] == 0.03
    assert merged["new_greeting_click_y_offset"] == -0.02
    assert merged["next_message_click_x_offset"] == 0.04
    assert merged["next_message_click_y_offset"] == -0.01


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
