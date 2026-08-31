from __future__ import annotations

import json

import update_apply


def test_update_copy_merges_app_config_and_keeps_local_popup_regions(tmp_path):
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_app = source_root / "apps" / "xinyu"
    target_app = target_root / "apps" / "xinyu"
    source_app.mkdir(parents=True)
    target_app.mkdir(parents=True)
    (source_app / "config.json").write_text(
        json.dumps(
            {
                "match_threshold": 0.7,
                "template_match_region_expand_ratio": 0.03,
                "template_match_regions": {
                    "popup.custom.popup_custom_1": [0.1, 0.1, 0.2, 0.2],
                },
                "persona": {"requestPersona": "new-default"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (target_app / "config.json").write_text(
        json.dumps(
            {
                "match_threshold": 0.82,
                "template_match_regions": {
                    "popup.custom.popup_custom_5": [0.5, 0.2, 0.3, 0.4],
                },
                "persona": {"requestPersona": "customer-persona"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    update_apply._copy_update_tree(source_root, target_root)

    merged = json.loads((target_app / "config.json").read_text(encoding="utf-8"))
    assert merged["match_threshold"] == 0.82
    assert merged["template_match_region_expand_ratio"] == 0.03
    assert merged["template_match_regions"]["popup.custom.popup_custom_1"] == [0.1, 0.1, 0.2, 0.2]
    assert merged["template_match_regions"]["popup.custom.popup_custom_5"] == [0.5, 0.2, 0.3, 0.4]
    assert merged["persona"]["requestPersona"] == "customer-persona"


def test_update_remove_obsolete_files_keeps_local_app_config(tmp_path):
    target_root = tmp_path / "target"
    app_config = target_root / "apps" / "huanyou" / "config.json"
    app_config.parent.mkdir(parents=True)
    app_config.write_text('{"template_match_regions": {"popup.custom.popup_custom_5": [0, 0, 1, 1]}}', encoding="utf-8")
    old_paths = {"apps/huanyou/config.json", "apps/huanyou/old.py"}
    new_paths = {"apps/huanyou/new.py"}
    (target_root / "apps" / "huanyou" / "old.py").write_text("old", encoding="utf-8")

    update_apply._remove_obsolete_files(target_root, old_paths, new_paths)

    assert app_config.exists()
    assert not (target_root / "apps" / "huanyou" / "old.py").exists()


def test_update_preserves_customer_templates_and_does_not_overwrite_them(tmp_path):
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_template = source_root / "apps" / "xinyu" / "templates" / "screens"
    target_template = target_root / "apps" / "xinyu" / "templates" / "screens"
    source_template.mkdir(parents=True)
    target_template.mkdir(parents=True)
    customer_file = target_template / "screen_聊天界面__customer_1.jpg"
    customer_file.write_bytes(b"customer")

    old_paths = {"apps/xinyu/templates/screens/screen_聊天界面__customer_1.jpg"}
    new_paths: set[str] = set()
    update_apply._remove_obsolete_files(target_root, old_paths, new_paths)
    assert customer_file.read_bytes() == b"customer"

    (source_template / customer_file.name).write_bytes(b"update-package")
    update_apply._copy_update_tree(source_root, target_root)
    assert customer_file.read_bytes() == b"customer"


def test_update_replaces_existing_app_modules_archive_atomically(tmp_path):
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "app_modules.zip").write_bytes(b"new-modules")
    (target_root / "app_modules.zip").mkdir()
    (target_root / "app_modules.zip" / "stale.txt").write_text("stale", encoding="utf-8")

    update_apply._copy_update_tree(source_root, target_root)

    assert (target_root / "app_modules.zip").is_file()
    assert (target_root / "app_modules.zip").read_bytes() == b"new-modules"


def test_update_preserves_running_launcher_files(tmp_path):
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "机械臂脚本.exe").write_bytes(b"new")
    (target_root / "机械臂脚本.exe").write_bytes(b"running")

    update_apply._copy_update_tree(source_root, target_root)

    assert (target_root / "机械臂脚本.exe").read_bytes() == b"running"
