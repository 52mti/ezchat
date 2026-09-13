from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from apps.wechat.config import DEFAULT_CONFIG as WECHAT_DEFAULT_CONFIG
from apps.xinyu.config import DEFAULT_CONFIG as XINYU_DEFAULT_CONFIG
from services.config_manager import ConfigManager


def _read_app_config(tmp_path, app_name):
    # 双层布局：UI 保存只写客户层 runtime/configs/<app>/。
    path = tmp_path / "runtime" / "configs" / app_name / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _read_app_ocr_mode_config(tmp_path, app_name, filename):
    path = tmp_path / "runtime" / "configs" / app_name / filename
    return json.loads(path.read_text(encoding="utf-8"))


def test_config_manager_get_set_and_atomic_write(tmp_path):
    config_path = tmp_path / "config.json"
    outbox_path = tmp_path / "outbox.json"
    manager = ConfigManager(config_path, outbox_path)

    manager.update_global("persona_chat_model", "deepseek-flash")

    assert manager.get("global", "persona_chat_model") == "deepseek-flash"
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["global"]["persona_chat_model"] == "deepseek-flash"


def test_config_manager_has_global_greetings_dir_default(tmp_path):
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")

    assert manager.get("global", "greetings_dir") == "runtime/greetings"
    assert manager.get("global", "ocr_max_concurrency") == 1
    assert "greetings_dir" not in manager.get_app_config("1.1.1.1", "wechat", {})


def test_config_manager_strips_stale_global_apikey_on_write(tmp_path):
    """旧链路 global.apikey 已无消费方：写盘时剥离（docs/13 第七节 C5 清理）。"""
    config_path = tmp_path / "config.json"
    outbox_path = tmp_path / "outbox.json"
    config_path.write_text(
        json.dumps({"global": {"apikey": "legacy-key"}, "devices": []}),
        encoding="utf-8",
    )

    manager = ConfigManager(config_path, outbox_path)
    manager.update_global("persona_chat_model", "deepseek-flash")

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert "apikey" not in saved["global"]


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
    dialogue_saved = _read_app_ocr_mode_config(tmp_path, "wechat", "dialogue_ocr_config.json")
    assert saved["devices"][0]["runtime_overrides"]["poll_interval"] == 9
    assert "apps" not in saved
    assert "dialogue" not in app_saved.get("ocr_modes", {})
    assert dialogue_saved["rules"]["line_gap"] == 1
    assert saved["devices"][0]["app_overrides"] == {}
    app_config = manager.get_app_config("1.1.1.1", "wechat", WECHAT_DEFAULT_CONFIG)
    assert app_config["poll_interval"] == 9
    assert app_config["ocr_modes"]["dialogue"]["rules"]["line_gap"] == 1


def test_config_manager_splits_app_level_dialogue_and_list_ocr_files_on_save(tmp_path):
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")

    manager.update_app_config(
        "xinyu",
        {
            "ocr_modes": {
                "dialogue": {"rules": {"line_gap": 31}},
                "list": {"rules": {"main_top_ratio": 0.18}},
                "points": {"rules": {"neighbor_x_ratio": 0.66}},
            }
        },
    )

    base_saved = _read_app_config(tmp_path, "xinyu")
    dialogue_saved = _read_app_ocr_mode_config(tmp_path, "xinyu", "dialogue_ocr_config.json")
    list_saved = _read_app_ocr_mode_config(tmp_path, "xinyu", "list_ocr_config.json")

    assert "dialogue" not in base_saved.get("ocr_modes", {})
    assert "list" not in base_saved.get("ocr_modes", {})
    # points 等其余模式属开发域（随包默认下发），不落客户层。
    assert "ocr_modes" not in base_saved
    assert dialogue_saved["rules"]["line_gap"] == 31
    assert list_saved["rules"]["main_top_ratio"] == 0.18


def test_config_manager_reads_split_app_level_dialogue_and_list_ocr_files(tmp_path):
    apps_dir = tmp_path / "apps" / "xinyu"
    apps_dir.mkdir(parents=True)
    (apps_dir / "config.json").write_text(
        json.dumps({"ocr_modes": {"points": {"rules": {"neighbor_x_ratio": 0.45}}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (apps_dir / "dialogue_ocr_config.json").write_text(
        json.dumps({"rules": {"line_gap": 35}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (apps_dir / "list_ocr_config.json").write_text(
        json.dumps({"rules": {"main_top_ratio": 0.16}}, ensure_ascii=False),
        encoding="utf-8",
    )

    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")
    app_config = manager.get_app_config("1.1.1.1", "xinyu", {})

    assert app_config["ocr_modes"]["dialogue"]["rules"]["line_gap"] == 35
    assert app_config["ocr_modes"]["list"]["rules"]["main_top_ratio"] == 0.16
    assert app_config["ocr_modes"]["points"]["rules"]["neighbor_x_ratio"] == 0.45


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
    app_config = manager.get_app_config("192.168.1.108", "wechat", WECHAT_DEFAULT_CONFIG)
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    app_saved = _read_app_config(tmp_path, "wechat")

    # 模板路径属开发域：历史设备层残留的 runtime\templates 路径被清除，
    # 生效值回落到随包/代码默认（本就是规范化目标位置）。
    assert app_config["templates"]["buttons"] == WECHAT_DEFAULT_CONFIG["templates"]["buttons"]
    assert app_config["templates"]["screens"] == WECHAT_DEFAULT_CONFIG["templates"]["screens"]
    assert "apps" not in saved
    assert "templates" not in app_saved


def test_config_manager_migrates_and_backs_up_old_schema(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"schema_version": 0, "devices": [{"ip": "1.1.1.1"}]}),
        encoding="utf-8",
    )

    manager = ConfigManager(config_path, tmp_path / "outbox.json")
    device = manager.find_device("1.1.1.1")

    assert manager.get("schema_version") == 3
    assert device["selected_app"] == "xinyu"
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
    # 视觉调参属开发域：客户层与存盘都不保留，生效值回到随包/代码默认。
    assert "match_threshold" not in app_saved
    assert "ocr" not in app_saved
    assert saved["devices"][0]["runtime_overrides"]["poll_interval"] == 6
    assert saved["devices"][0]["runtime_overrides"]["keep_awake_interval_seconds"] == 300
    assert saved["devices"][0]["runtime_overrides"]["exclude_ocr_wait_from_step_timeout"] is False
    assert saved["devices"][0]["app_overrides"]["wechat"]["persona"]["name"] == "bot"
    assert device["runtime_overrides"]["poll_interval"] == 6
    assert merged["match_threshold"] == WECHAT_DEFAULT_CONFIG["match_threshold"]
    assert merged["persona"]["name"] == "bot"


def test_config_manager_keeps_saved_match_threshold(tmp_path):
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")
    app_dir = tmp_path / "apps" / "xinyu"
    app_dir.mkdir(parents=True)
    (app_dir / "config.json").write_text('{"match_threshold": 0.7}', encoding="utf-8")

    config = manager.get_app_config("1.1.1.1", "xinyu", {})

    assert config["match_threshold"] == 0.7


def test_config_manager_defaults_invalid_match_threshold(tmp_path):
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")
    app_dir = tmp_path / "apps" / "xinyu"
    app_dir.mkdir(parents=True)
    (app_dir / "config.json").write_text('{"match_threshold": "bad"}', encoding="utf-8")

    config = manager.get_app_config("1.1.1.1", "xinyu", {})

    assert config["match_threshold"] == 0.7


def test_config_manager_keeps_template_match_regions_in_app_scope(tmp_path):
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")
    manager.add_device({"ip": "1.1.1.1", "selected_app": "xinyu", "app_overrides": {"xinyu": {"persona": {}}}})

    manager.update_app_config(
        "xinyu",
        {
            "template_match_regions": {
                "button.send": [0.1, 0.7, 0.8, 0.2],
                "popup.custom.popup_custom_1": [0.2, 0.3, 0.6, 0.3],
            }
        },
    )

    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    app_saved = _read_app_config(tmp_path, "xinyu")
    merged = manager.get_app_config("1.1.1.1", "xinyu", {})

    # 客户层只保留 popup.* 模板规则；button/screen 区域属开发域随包默认。
    assert list(app_saved["template_match_regions"]) == ["popup.custom.popup_custom_1"]
    assert "template_match_regions" not in saved["devices"][0].get("app_overrides", {}).get("xinyu", {})
    assert merged["template_match_regions"]["popup.custom.popup_custom_1"] == [0.2, 0.3, 0.6, 0.3]


def test_config_manager_keeps_template_match_thresholds_in_app_scope(tmp_path):
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")

    manager.update_app_config(
        "xinyu",
        {
            "template_match_thresholds": {
                "screen.chat_window": 0.7,
                "popup.custom.popup_custom_2": 0.85,
            }
        },
    )

    app_saved = _read_app_config(tmp_path, "xinyu")
    merged = manager.get_app_config("1.1.1.1", "xinyu", XINYU_DEFAULT_CONFIG)

    assert list(app_saved["template_match_thresholds"]) == ["popup.custom.popup_custom_2"]
    assert merged["template_match_thresholds"]["popup.custom.popup_custom_2"] == 0.85


def test_config_manager_keeps_xinyu_photo_settings_in_device_app_scope(tmp_path):
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")
    manager.add_device({"ip": "1.1.1.1", "selected_app": "xinyu"})

    manager.update_device_app_overrides(
        "1.1.1.1",
        "xinyu",
        {
            "online_photo_send_count": 0,
            "photo_sent_cooldown_seconds": 3600,
        },
    )

    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    merged = manager.get_app_config("1.1.1.1", "xinyu", XINYU_DEFAULT_CONFIG)

    device_app = saved["devices"][0]["app_overrides"]["xinyu"]
    assert device_app["online_photo_send_count"] == 0
    assert device_app["photo_sent_cooldown_seconds"] == 3600
    assert not (tmp_path / "apps" / "xinyu" / "config.json").exists()
    assert merged["online_photo_send_count"] == 0
    assert merged["photo_sent_cooldown_seconds"] == 3600


def test_xinyu_partial_app_config_keeps_default_template_regions(tmp_path):
    app_dir = tmp_path / "apps" / "xinyu"
    app_dir.mkdir(parents=True)
    (app_dir / "config.json").write_text(
        json.dumps(
            {
                "template_match_regions": {
                    "popup.custom.popup_custom_4": [0.037, 0.282, 0.954, 0.443]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")

    merged = manager.get_app_config("1.1.1.1", "xinyu", XINYU_DEFAULT_CONFIG)

    assert merged["template_match_regions"]["screen.chat_window"] == [0.007, 0.476, 0.261, 0.523]
    assert merged["template_match_regions"]["popup.custom.popup_custom_4"] == [0.037, 0.282, 0.954, 0.443]


def test_config_manager_add_device_sorts_and_rejects_duplicates(tmp_path):
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")

    assert manager.add_device({"ip": "192.168.1.20", "selected_app": "wechat"}) is True
    assert manager.add_device({"ip": "192.168.1.3", "selected_app": "wechat"}) is True
    assert manager.add_device({"ip": "192.168.1.20", "selected_app": "wechat"}) is False

    devices = manager.devices()
    assert [device["ip"] for device in devices] == ["192.168.1.3", "192.168.1.20"]


def test_config_manager_concurrent_device_updates_do_not_overwrite_each_other(tmp_path):
    config_path = tmp_path / "config.json"
    outbox_path = tmp_path / "outbox.json"
    manager = ConfigManager(config_path, outbox_path)
    for index in range(20):
        manager.add_device({"ip": f"192.168.1.{index + 1}", "selected_app": "wechat"})

    def update(index: int) -> None:
        local_manager = ConfigManager(config_path, outbox_path)
        local_manager.update_device(
            f"192.168.1.{index + 1}",
            {"license_expires": f"2099-01-{index + 1:02d}"},
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(update, range(20)))

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    expires_by_ip = {device["ip"]: device.get("license_expires") for device in saved["devices"]}

    assert len(expires_by_ip) == 20
    for index in range(20):
        assert expires_by_ip[f"192.168.1.{index + 1}"] == f"2099-01-{index + 1:02d}"


def test_config_manager_remove_device(tmp_path):
    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")
    manager.add_device({"ip": "192.168.1.20", "selected_app": "wechat"})
    manager.add_device({"ip": "192.168.1.3", "selected_app": "wechat"})

    assert manager.remove_device("192.168.1.20") is True
    assert manager.remove_device("192.168.1.20") is False
    assert [device["ip"] for device in manager.devices()] == ["192.168.1.3"]


def test_config_manager_preserves_app_screen_detection_rules(tmp_path):
    app_dir = tmp_path / "apps" / "xinyu"
    app_dir.mkdir(parents=True)
    rules = [{"screen": "xinyu-缘分", "condition": {"type": "ocr_raw", "text": "推荐在线"}}]
    (app_dir / "config.json").write_text(json.dumps({"screen_detection_rules": rules}), encoding="utf-8")

    manager = ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json")
    config = manager.get_app_config("192.168.1.1", "xinyu", {})

    assert config["screen_detection_rules"] == rules


def test_packaged_migration_moves_legacy_configs_to_customer_layer(tmp_path, monkeypatch):
    """客户机迁移：复制到 runtime/configs 并消费源文件，防止旧 json 遮蔽新默认。"""
    monkeypatch.setenv("JIXIEBI_PACKAGE_ROOT", str(tmp_path))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"schema_version": 3, "devices": []}), encoding="utf-8")
    for filename in ("config.json", "dialogue_ocr_config.json", "list_ocr_config.json"):
        app_dir = tmp_path / "apps" / "xinyu"
        app_dir.mkdir(parents=True, exist_ok=True)
        (app_dir / filename).write_text('{"legacy": true}', encoding="utf-8")

    ConfigManager(config_path, tmp_path / "outbox.json")

    for filename in ("config.json", "dialogue_ocr_config.json", "list_ocr_config.json"):
        customer = tmp_path / "runtime" / "configs" / "xinyu" / filename
        assert customer.is_file(), filename
        assert json.loads(customer.read_text(encoding="utf-8")) == {"legacy": True}
        assert not (tmp_path / "apps" / "xinyu" / filename).exists(), filename
    assert (tmp_path / "runtime" / "configs" / ".layout_v2").is_file()


def test_dev_environment_keeps_apps_configs_as_defaults_layer(tmp_path, monkeypatch):
    """开发机（无打包环境变量）：apps json 原地保留，只作 defaults 层。"""
    monkeypatch.delenv("JIXIEBI_PACKAGE_ROOT", raising=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"schema_version": 3, "devices": []}), encoding="utf-8")
    (tmp_path / "apps" / "xinyu").mkdir(parents=True)
    (tmp_path / "apps" / "xinyu" / "config.json").write_text(
        json.dumps({"match_threshold": 0.75}), encoding="utf-8"
    )

    manager = ConfigManager(config_path, tmp_path / "outbox.json")

    assert (tmp_path / "apps" / "xinyu" / "config.json").is_file()
    assert manager.get_app_overrides("xinyu")["match_threshold"] == 0.75


def test_fresh_install_starts_customer_layer_empty(tmp_path):
    """全新安装：客户层从空开始，绝不复制随包默认值。"""
    config_path = tmp_path / "config.json"
    (tmp_path / "apps" / "xinyu").mkdir(parents=True)
    (tmp_path / "apps" / "xinyu" / "config.json").write_text(
        json.dumps({"match_threshold": 0.75}), encoding="utf-8"
    )

    manager = ConfigManager(config_path, tmp_path / "outbox.json")

    # 客户层从空开始（不复制默认值文件）；读取时 defaults 层兜底，
    # 客户机上 apps json 不随包分发，因此 defaults 层同样为空。
    assert not (tmp_path / "runtime" / "configs" / "xinyu" / "config.json").exists()
    assert manager.get_app_overrides("xinyu")["match_threshold"] == 0.75
    assert (tmp_path / "runtime" / "configs" / ".layout_v2").is_file()
