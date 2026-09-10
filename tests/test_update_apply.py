from __future__ import annotations

import json

import update_apply


def test_update_keeps_app_config_assets_from_copy_and_removal(tmp_path):
    """App 级配置三件套是客户界面数据：不覆盖、不删除（与 build 排除互为镜像）。"""
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    for root in (source_root, target_root):
        (root / "apps" / "xinyu").mkdir(parents=True)
    (source_root / "apps" / "xinyu" / "config.json").write_text('{"match_threshold": 0.7}', encoding="utf-8")
    (target_root / "apps" / "xinyu" / "config.json").write_text('{"match_threshold": 0.82}', encoding="utf-8")
    (target_root / "apps" / "xinyu" / "dialogue_ocr_config.json").write_text("{}", encoding="utf-8")

    update_apply._copy_update_tree(source_root, target_root, full_payload=False)
    merged = json.loads((target_root / "apps" / "xinyu" / "config.json").read_text(encoding="utf-8"))
    assert merged["match_threshold"] == 0.82

    update_apply._remove_obsolete_files(
        target_root,
        {"apps/xinyu/config.json", "apps/xinyu/dialogue_ocr_config.json", "apps/xinyu/old.py"},
        set(),
        full_payload=False,
    )
    assert (target_root / "apps" / "xinyu" / "config.json").exists()
    assert (target_root / "apps" / "xinyu" / "dialogue_ocr_config.json").exists()
    assert not (target_root / "apps" / "xinyu" / "old.py").exists()


def test_update_preserves_customer_templates_and_does_not_overwrite_them(tmp_path):
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_template = source_root / "apps" / "xinyu" / "templates" / "screens"
    target_template = target_root / "apps" / "xinyu" / "templates" / "screens"
    source_template.mkdir(parents=True)
    target_template.mkdir(parents=True)
    customer_file = target_template / "screen_聊天界面__customer_1.jpg"
    customer_file.write_bytes(b"customer")

    update_apply._remove_obsolete_files(
        target_root,
        {"apps/xinyu/templates/screens/screen_聊天界面__customer_1.jpg"},
        set(),
        full_payload=False,
    )
    assert customer_file.read_bytes() == b"customer"

    (source_template / customer_file.name).write_bytes(b"update-package")
    update_apply._copy_update_tree(source_root, target_root, full_payload=False)
    assert customer_file.read_bytes() == b"customer"


def test_update_replaces_existing_app_modules_archive_atomically(tmp_path):
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "app_modules.zip").write_bytes(b"new-modules")
    (target_root / "app_modules.zip").mkdir()
    (target_root / "app_modules.zip" / "stale.txt").write_text("stale", encoding="utf-8")

    update_apply._copy_update_tree(source_root, target_root, full_payload=False)

    assert (target_root / "app_modules.zip").is_file()
    assert (target_root / "app_modules.zip").read_bytes() == b"new-modules"


def test_update_preserves_running_launcher_files(tmp_path):
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "机械臂脚本.exe").write_bytes(b"new")
    (target_root / "机械臂脚本.exe").write_bytes(b"running")

    update_apply._copy_update_tree(source_root, target_root, full_payload=False)

    assert (target_root / "机械臂脚本.exe").read_bytes() == b"running"


def test_incremental_payload_never_touches_structurally_isolated_roots(tmp_path):
    """增量包：engine/、models/、runtime/、tts/piper 不删、不覆盖。"""
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    protected_files = {
        "engine/app/python.exe": b"engine-python",
        "engine/ocr_cpu/python.exe": b"engine-ocr",
        "models/ocr/model.yml": b"model",
        "runtime/app/python.exe": b"legacy-python",
        "runtime/tts/piper/zh_CN.onnx": b"piper-model",
        "runtime/logs/run.log": b"customer-log",
    }
    for relative, payload in protected_files.items():
        path = target_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        (source_root / relative).parent.mkdir(parents=True, exist_ok=True)
        (source_root / relative).write_bytes(b"update-payload")

    update_apply._remove_obsolete_files(target_root, set(protected_files), set(), full_payload=False)
    update_apply._copy_update_tree(source_root, target_root, full_payload=False)

    for relative, payload in protected_files.items():
        assert (target_root / relative).read_bytes() == payload, relative


def test_full_payload_removes_legacy_engine_but_copies_new_one(tmp_path):
    """全量包：清单差集允许清理旧布局引擎，并复制新 engine/ 内容。"""
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    legacy = target_root / "runtime" / "app" / "python.exe"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy-python")
    new_engine = source_root / "engine" / "app" / "python.exe"
    new_engine.parent.mkdir(parents=True)
    new_engine.write_bytes(b"new-engine")

    update_apply._remove_obsolete_files(
        target_root,
        {"runtime/app/python.exe"},
        {"engine/app/python.exe"},
        full_payload=True,
    )
    assert not legacy.exists()

    update_apply._copy_update_tree(source_root, target_root, full_payload=True)
    assert new_engine.read_bytes() == b"new-engine"
    assert (target_root / "engine" / "app" / "python.exe").read_bytes() == b"new-engine"


def test_is_full_payload_detects_engine_or_legacy_layout(tmp_path):
    engine_root = tmp_path / "a"
    (engine_root / "engine" / "app").mkdir(parents=True)
    legacy_root = tmp_path / "b"
    (legacy_root / "runtime" / "app").mkdir(parents=True)
    incremental_root = tmp_path / "c"
    incremental_root.mkdir()

    assert update_apply._is_full_payload(engine_root)
    assert update_apply._is_full_payload(legacy_root)
    assert not update_apply._is_full_payload(incremental_root)


def test_preserved_scopes_unit_assertions():
    assert update_apply._is_preserved("apps/xinyu/config.json")
    assert update_apply._is_preserved("apps/huanyou/dialogue_ocr_config.json")
    assert update_apply._is_preserved("apps/huanyou/list_ocr_config.json")
    assert update_apply._is_preserved("apps/xinyu/templates/buttons/btn_test__customer_1.jpg")
    assert update_apply._is_preserved("config.json")
    assert update_apply._is_preserved("photo/1.jpg")
    assert not update_apply._is_preserved("apps/xinyu/templates/buttons/btn_send.jpg")
    assert not update_apply._is_preserved("apps/xinyu/config.py")


def test_quiet_flag_is_accepted(tmp_path, monkeypatch):
    """编排层以 --quiet 调用：成功时不弹任何消息框。"""
    archive = tmp_path / "update.zip"
    source = tmp_path / "payload"
    payload_root = source / "jixiebijiaoben"
    payload_root.mkdir(parents=True)
    (payload_root / "app_modules.zip").write_bytes(b"modules")
    import zipfile

    with zipfile.ZipFile(archive, "w") as z:
        z.write(payload_root / "app_modules.zip", "jixiebijiaoben/app_modules.zip")

    install_root = tmp_path / "install"
    install_root.mkdir()
    shown: list[str] = []
    monkeypatch.setattr(update_apply, "_show_message", lambda *args, **kwargs: shown.append("msg"))

    code = update_apply.main(
        ["--package-root", str(install_root), "--archive", str(archive), "--pid", "0", "--quiet"]
    )
    assert code == 0
    assert shown == []
    assert (install_root / "app_modules.zip").read_bytes() == b"modules"


def test_orchestration_scripts_are_generated_from_templates(tmp_path):
    import update_orchestration

    ps1 = update_orchestration.write_orchestration_script(tmp_path / "scripts")
    bat = update_orchestration.write_manual_update_bat(tmp_path / "scripts")

    ps1_text = ps1.read_text(encoding="utf-8-sig")
    assert ps1.read_bytes().startswith(b"\xef\xbb\xbf")
    assert "engine" + chr(92) + "app" + chr(92) + "python.exe" in ps1_text
    assert "\\\\" not in ps1_text
    bat_bytes = bat.read_bytes()
    bat_bytes.decode("ascii")
    assert b"\r\n" in bat_bytes


def test_orchestration_validator_rejects_double_backslash():
    import update_orchestration

    update_orchestration.validate_ps1_text(update_orchestration.WINDOWS_UPDATE_PS1)
    corrupted = "Write-Host 'a" + "\\\\" + "b'"
    try:
        update_orchestration.validate_ps1_text(corrupted)
    except ValueError:
        pass
    else:
        raise AssertionError("double backslash must be rejected")
