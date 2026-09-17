from __future__ import annotations

import json
import shutil
import zipfile

import pytest

import build


def test_requirements_specs_only_include_real_package_lines(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "\n".join(
            [
                "opencv-python",
                "",
                "# comment",
                "paddleocr==3.4.0",
                'pywin32; platform_system == "Windows"',
            ]
        ),
        encoding="utf-8",
    )

    assert build._requirements_specs(tmp_path) == [
        "opencv-python",
        "paddleocr==3.4.0",
        'pywin32; platform_system == "Windows"',
    ]


def test_bundled_dependency_specs_exclude_shared_runtime_packages(tmp_path):
    (tmp_path / "requirements.txt").write_text(
        "\n".join(
            [
                "opencv-python",
                "paddleocr==3.4.0",
                "paddlex[ocr-core]==3.4.3",
                "PyQt6",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "runtime_manifest.json").write_text(
        """
{
  "sharedPackages": [
    {"spec": "paddleocr==3.4.0"},
    {"spec": "paddlex[ocr-core]==3.4.3"}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    assert build._bundled_dependency_specs(tmp_path) == ["opencv-python", "PyQt6"]


def test_copy_sources_does_not_package_release_manifest(tmp_path):
    root = tmp_path / "repo"
    package_root = tmp_path / "package"
    root.mkdir()
    (root / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "release.json").write_text('{"app_version": "remote"}', encoding="utf-8")

    build._copy_sources(root, package_root)

    assert (package_root / "main.py").is_file()
    assert not (package_root / "release.json").exists()


def test_copy_sources_keeps_app_config_json_but_excludes_root_config_json(tmp_path):
    root = tmp_path / "repo"
    package_root = tmp_path / "package"
    (root / "apps" / "xinyu").mkdir(parents=True)
    (root / "apps" / "xinyu" / "config.json").write_text('{"app": "xinyu"}', encoding="utf-8")
    (root / "config.json").write_text('{"secret": true}', encoding="utf-8")

    build._copy_sources(root, package_root)

    assert (package_root / "apps" / "xinyu" / "config.json").read_text(encoding="utf-8") == '{"app": "xinyu"}'
    assert not (package_root / "config.json").exists()


def test_copy_sources_excludes_deprecated_wechat_app(tmp_path):
    root = tmp_path / "repo"
    package_root = tmp_path / "package"
    (root / "apps" / "wechat").mkdir(parents=True)
    (root / "apps" / "wechat" / "app.py").write_text("legacy\n", encoding="utf-8")
    (root / "apps" / "xinyu").mkdir(parents=True)
    (root / "apps" / "xinyu" / "app.py").write_text("supported\n", encoding="utf-8")

    build._copy_sources(root, package_root)

    assert not (package_root / "apps" / "wechat").exists()
    assert (package_root / "apps" / "xinyu" / "app.py").is_file()


def test_copy_sources_ships_customer_templates_and_strips_developer_calibration(tmp_path):
    """客户模板（开发机本机 MAC 采集 + 外来 MAC 收集 + 旧格式共享资产）随包
    分发，__developer_ 开发标定不进包；目标机侧保护由 update_apply 精确执行。"""
    from apps.visual_assets import machine_template_id

    root = tmp_path / "repo"
    package_root = tmp_path / "package"
    buttons = root / "apps" / "xinyu" / "templates" / "buttons"
    buttons.mkdir(parents=True)
    machine = machine_template_id()
    own = buttons / f"btn_主界面心遇按钮__customer_{machine}_1.jpg"
    own.write_bytes(b"own-machine")
    foreign = buttons / "btn_主界面心遇按钮__customer_aaaaaaaaaaaa_1.jpg"
    foreign.write_bytes(b"collected")
    legacy = buttons / "popup_custom__customer_1.jpg"
    legacy.write_bytes(b"legacy")
    dev = buttons / "btn_发送按钮__developer_1.jpg"
    dev.write_bytes(b"calibration")

    build._copy_sources(root, package_root)

    target = package_root / "apps" / "xinyu" / "templates" / "buttons"
    assert (target / own.name).read_bytes() == b"own-machine"
    assert (target / foreign.name).read_bytes() == b"collected"
    assert (target / legacy.name).read_bytes() == b"legacy"
    assert not (target / dev.name).exists()


def _write_vision_key(root: Path, key: str) -> None:
    (root / ".env").mkdir(exist_ok=True)
    (root / ".env" / "vision_dialogue_keys.json").write_text(
        json.dumps({"openrouter_key": key}, ensure_ascii=False), encoding="utf-8"
    )


def test_copy_bundled_vision_key_injects_into_package(tmp_path):
    root = tmp_path / "repo"
    package_root = tmp_path / "package"
    root.mkdir()
    _write_vision_key(root, "sk-or-v1-test-key")

    build._copy_bundled_vision_key(root, package_root)

    injected = package_root / ".env" / "vision_dialogue_keys.json"
    assert json.loads(injected.read_text(encoding="utf-8"))["openrouter_key"] == "sk-or-v1-test-key"


def test_copy_bundled_vision_key_requires_valid_key(tmp_path):
    root = tmp_path / "repo"
    package_root = tmp_path / "package"
    root.mkdir()

    with pytest.raises(SystemExit):
        build._copy_bundled_vision_key(root, package_root)

    _write_vision_key(root, "请在这里填写你自己的 OpenRouter API Key")
    with pytest.raises(SystemExit):
        build._copy_bundled_vision_key(root, package_root)

    assert not (package_root / ".env").exists()


def test_package_cache_key_changes_with_vision_key(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _write_vision_key(root, "sk-or-v1-key-a")
    first = build._package_cache_key(
        root, model_tier="lite", nvidia_cuda_lane="cpu", include_full_payload=False
    )
    _write_vision_key(root, "sk-or-v1-key-b")
    second = build._package_cache_key(
        root, model_tier="lite", nvidia_cuda_lane="cpu", include_full_payload=False
    )

    assert first != second


def test_write_customer_update_archive_excludes_full_runtimes_and_models(tmp_path):
    package_root = tmp_path / build.PACKAGE_NAME
    dist_root = tmp_path / "dist"
    (package_root / "apps").mkdir(parents=True)
    (package_root / "runtime" / "app").mkdir(parents=True)
    (package_root / "runtime" / "ocr_cpu").mkdir(parents=True)
    (package_root / "runtime" / "tts" / "piper").mkdir(parents=True)
    (package_root / "models" / "ocr" / "paddlex_named_models").mkdir(parents=True)
    (package_root / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (package_root / build.LAUNCHER_EXE).write_bytes(b"binary")
    (package_root / "apps" / "config.json").write_text("{}", encoding="utf-8")
    (package_root / "runtime" / "app" / "python.exe").write_bytes(b"python")
    (package_root / "runtime" / "ocr_cpu" / "python.exe").write_bytes(b"python")
    (package_root / "runtime" / "tts" / "piper" / "zh_CN-huayan-medium.onnx").write_bytes(b"piper")
    (package_root / "models" / "ocr" / "paddlex_named_models" / "demo.bin").write_bytes(b"model")

    archive_path = build._write_customer_update_archive(package_root, dist_root)

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())

    assert f"{build.PACKAGE_NAME}/main.py" in names
    assert f"{build.PACKAGE_NAME}/{build.LAUNCHER_EXE}" not in names
    assert f"{build.PACKAGE_NAME}/{build.LAUNCHER_ALIAS_EXE}" not in names
    assert all(not name.startswith(f"{build.PACKAGE_NAME}/engine/") for name in names)
    assert all(not name.startswith(f"{build.PACKAGE_NAME}/runtime/app/") for name in names)
    assert all(not name.startswith(f"{build.PACKAGE_NAME}/runtime/ocr_cpu/") for name in names)
    assert all(not name.startswith(f"{build.PACKAGE_NAME}/runtime/tts/piper/") for name in names)
    assert all(not name.startswith(f"{build.PACKAGE_NAME}/models/ocr/") for name in names)


def test_write_archives_can_include_timestamp_suffix(tmp_path):
    package_root = tmp_path / build.PACKAGE_NAME
    dist_root = tmp_path / "dist"
    package_root.mkdir()
    (package_root / "main.py").write_text("print('ok')\n", encoding="utf-8")

    archive_path = build._write_customer_update_archive(package_root, dist_root, timestamp="20260604_1825")

    assert archive_path.name == f"{build.PACKAGE_NAME}_update_20260604_1825.zip"
    assert archive_path.is_file()


def test_write_customer_update_archive_includes_app_json_config_defaults(tmp_path):
    package_root = tmp_path / build.PACKAGE_NAME
    dist_root = tmp_path / "dist"
    app_root = package_root / "apps" / "xinyu"
    (app_root / "templates" / "buttons").mkdir(parents=True)
    (app_root / "app.py").write_text("APP = 'xinyu'\n", encoding="utf-8")
    (app_root / "steps.py").write_text("STEPS = []\n", encoding="utf-8")
    (app_root / "config.py").write_text("NAME = 'xinyu'\n", encoding="utf-8")
    (app_root / "config.json").write_text('{"persona": "custom"}', encoding="utf-8")
    (app_root / "ocr_config.py").write_text("RULES = {}\n", encoding="utf-8")
    (app_root / "dialogue_ocr_config.json").write_text('{"mode": "dialogue"}', encoding="utf-8")
    (app_root / "list_ocr_config.json").write_text('{"mode": "list"}', encoding="utf-8")
    (app_root / "templates" / "buttons" / "btn_返回按钮.jpg").write_bytes(b"image")

    archive_path = build._write_customer_update_archive(package_root, dist_root)

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())

    assert f"{build.PACKAGE_NAME}/apps/xinyu/app.py" in names
    assert f"{build.PACKAGE_NAME}/apps/xinyu/steps.py" in names
    assert f"{build.PACKAGE_NAME}/apps/xinyu/config.py" in names
    # App 级配置三件套是开发端默认值的下发通道（docs/11 L1），必须进归档，
    # 客户机经更新正常落地；客户自己的设定在 runtime/configs/<app>/（更新保护）。
    # .py 不由归档排除；真实构建中源码在编译阶段已被移除进 app_modules.zip。
    assert f"{build.PACKAGE_NAME}/apps/xinyu/config.json" in names
    assert f"{build.PACKAGE_NAME}/apps/xinyu/ocr_config.py" in names
    assert f"{build.PACKAGE_NAME}/apps/xinyu/dialogue_ocr_config.json" in names
    assert f"{build.PACKAGE_NAME}/apps/xinyu/list_ocr_config.json" in names
    assert f"{build.PACKAGE_NAME}/apps/xinyu/templates/buttons/btn_返回按钮.jpg" in names


def test_write_full_package_archive_includes_runtime_state_and_photo_dir(tmp_path):
    package_root = tmp_path / build.PACKAGE_NAME
    dist_root = tmp_path / "dist"
    (package_root / "apps").mkdir(parents=True)
    (package_root / "photo").mkdir(parents=True)
    (package_root / "runtime" / "install_state.json").parent.mkdir(parents=True, exist_ok=True)
    (package_root / "runtime" / "install_state.json").write_text("{}", encoding="utf-8")
    (package_root / "main.py").write_text("print('ok')\n", encoding="utf-8")

    archive_path = build._write_full_package_archive(package_root, dist_root)

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())

    assert f"{build.PACKAGE_NAME}/main.py" in names
    assert f"{build.PACKAGE_NAME}/photo/" in names
    assert f"{build.PACKAGE_NAME}/runtime/install_state.json" in names


def test_write_full_package_archive_includes_app_json_config_defaults(tmp_path):
    package_root = tmp_path / build.PACKAGE_NAME
    dist_root = tmp_path / "dist"
    app_root = package_root / "apps" / "xinyu"
    (app_root / "templates" / "buttons").mkdir(parents=True)
    (app_root / "config.json").write_text('{"persona": "default"}', encoding="utf-8")
    (app_root / "ocr_config.py").write_text("RULES = {}\n", encoding="utf-8")
    (app_root / "ocr_defaults.py").write_text("DIALOGUE_OCR_DEFAULT = {}\n", encoding="utf-8")
    (app_root / "dialogue_ocr_config.json").write_text('{"mode": "dialogue"}', encoding="utf-8")
    (app_root / "list_ocr_config.json").write_text('{"mode": "list"}', encoding="utf-8")
    (app_root / "templates" / "buttons" / "btn_返回按钮.jpg").write_bytes(b"image")

    archive_path = build._write_full_package_archive(package_root, dist_root)

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())

    # 三件套是开发端默认值下发通道（docs/11 L1），全量包同样收录；
    # 代码层默认(ocr_defaults.py)继续随包分发作为兜底。
    assert f"{build.PACKAGE_NAME}/apps/xinyu/config.json" in names
    assert f"{build.PACKAGE_NAME}/apps/xinyu/ocr_config.py" in names
    assert f"{build.PACKAGE_NAME}/apps/xinyu/dialogue_ocr_config.json" in names
    assert f"{build.PACKAGE_NAME}/apps/xinyu/list_ocr_config.json" in names
    assert f"{build.PACKAGE_NAME}/apps/xinyu/ocr_defaults.py" in names
    assert f"{build.PACKAGE_NAME}/apps/xinyu/templates/buttons/btn_返回按钮.jpg" in names


def test_copy_executable_aliases_creates_ascii_alias(tmp_path):
    package_root = tmp_path / build.PACKAGE_NAME
    package_root.mkdir()
    launcher_bytes = b"launcher"
    (package_root / build.LAUNCHER_EXE).write_bytes(launcher_bytes)

    build._copy_executable_aliases(package_root)

    assert (package_root / build.LAUNCHER_ALIAS_EXE).read_bytes() == launcher_bytes


def test_write_cmd_launchers_emit_ascii_crlf_scripts(tmp_path):
    package_root = tmp_path / build.PACKAGE_NAME
    package_root.mkdir()

    build._write_cmd_launchers(package_root)

    launcher_bytes = (package_root / build.START_CMD).read_bytes()
    assert b"\r\n" in launcher_bytes
    assert b"\n" not in launcher_bytes.replace(b"\r\n", b"")
    assert all(byte < 128 for byte in launcher_bytes)

    launcher_text = launcher_bytes.decode("ascii")
    assert build.LAUNCHER_ALIAS_EXE in launcher_text
    assert "install_runtime" not in launcher_text
    assert ".runtime_env" not in launcher_text


def test_write_package_release_metadata_updates_runtime_manifest_versions(tmp_path):
    package_root = tmp_path / build.PACKAGE_NAME
    package_root.mkdir()
    (package_root / "runtime_manifest.json").write_text(
        json.dumps({"version": "2026.06.02-full-package"}, ensure_ascii=False),
        encoding="utf-8",
    )

    build._write_package_release_metadata(
        package_root,
        app_version="app-deadbeefcafebabe",
        runtime_version="2026.06.02-full-package",
        model_version="2026.06.02-lite",
    )

    payload = json.loads((package_root / "runtime_manifest.json").read_text(encoding="utf-8"))
    assert payload["appVersion"] == "app-deadbeefcafebabe"
    assert payload["runtimeVersion"] == "2026.06.02-full-package"
    assert payload["modelVersion"] == "2026.06.02-lite"


def test_write_release_manifest_writes_uploadable_dist_payload(tmp_path):
    dist_root = tmp_path / "dist"

    manifest_path = build._write_release_manifest(
        dist_root,
        app_version="app-deadbeefcafebabe",
        runtime_version="2026.06.02-full-package",
        model_version="2026.06.02-lite",
        update_file_name="jixiebijiaoben_update_20260607_0904.zip",
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload == {
        "app_version": "app-deadbeefcafebabe",
        "runtime_version": "2026.06.02-full-package",
        "model_version": "2026.06.02-lite",
        "requires_full_package": False,
        "update_share_url": build.DEFAULT_UPDATE_SHARE_URL,
        "update_file_name": "jixiebijiaoben_update_20260607_0904.zip",
        "update_url": (
            "https://jixiebi-1311918180.cos.ap-shanghai.myqcloud.com/packages/"
            "jixiebijiaoben_update_20260607_0904.zip"
        ),
        "full_package_url": build.DEFAULT_FULL_PACKAGE_PAGE_URL,
        "release_notes": "",
        "release_manifest_url": build.DEFAULT_RELEASE_MANIFEST_URL,
    }


class _FakeCosResponse:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise build.requests.HTTPError(f"{self.status_code} error")


class _FakeCosSession:
    def __init__(self) -> None:
        self.put_calls: list[dict[str, object]] = []
        self.get_calls: list[dict[str, object]] = []

    def put(self, url, data=None, headers=None, timeout=None):  # noqa: ANN001
        self.put_calls.append({"url": url, "data": data, "headers": headers, "timeout": timeout})
        return _FakeCosResponse(200)

    def get(self, url, headers=None, timeout=None):  # noqa: ANN001
        self.get_calls.append({"url": url, "headers": headers, "timeout": timeout})
        return _FakeCosResponse(200)


def test_upload_release_manifest_puts_public_json_and_verifies_readback():
    payload = build._release_manifest_payload(
        app_version="app-deadbeefcafebabe",
        runtime_version="2026.06.02-full-package",
        model_version="2026.06.02-lite",
        update_file_name="jixiebijiaoben_update_20260607_0904.zip",
    )
    session = _FakeCosSession()

    build._upload_release_manifest(
        payload=payload,
        secret_id="test-secret-id",
        secret_key="test-secret-key",
        session=session,
    )

    assert len(session.put_calls) == 1
    put_call = session.put_calls[0]
    assert put_call["url"] == build.DEFAULT_RELEASE_MANIFEST_URL
    assert put_call["data"] == build._release_manifest_bytes(payload)
    assert put_call["headers"]["Host"] == "jixiebi-1311918180.cos.ap-shanghai.myqcloud.com"
    assert put_call["headers"]["Content-Type"] == "application/json; charset=utf-8"
    assert put_call["headers"]["x-cos-acl"] == "public-read"
    assert "Authorization" in put_call["headers"]
    assert len(session.get_calls) == 1
    assert session.get_calls[0]["url"] == build.DEFAULT_RELEASE_MANIFEST_URL


def test_apply_local_env_files_loads_nested_env_dir_txt_file(tmp_path, monkeypatch):
    env_dir = tmp_path / ".env"
    env_dir.mkdir()
    (env_dir / "cos.txt").write_text(
        "\n".join(
            [
                "TENCENT_COS_SECRET_ID=test-secret-id",
                "TENCENT_COS_SECRET_KEY=test-secret-key",
                "TENCENT_COS_SESSION_TOKEN=test-session-token",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TENCENT_COS_SECRET_ID", raising=False)
    monkeypatch.delenv("TENCENT_COS_SECRET_KEY", raising=False)
    monkeypatch.delenv("TENCENT_COS_SESSION_TOKEN", raising=False)

    build._apply_local_env_files(tmp_path)

    assert build.os.environ["TENCENT_COS_SECRET_ID"] == "test-secret-id"
    assert build.os.environ["TENCENT_COS_SECRET_KEY"] == "test-secret-key"
    assert build.os.environ["TENCENT_COS_SESSION_TOKEN"] == "test-session-token"


def test_lanzou_find_folder_id_matches_exact_name():
    folders = [
        {"fol_id": "1", "name": "测试"},
        {"fol_id": "13580356", "name": "机械臂更新"},
    ]

    assert build._lanzou_find_folder_id(folders, "机械臂更新") == 13580356
    assert build._lanzou_find_folder_id(folders, "不存在") is None


def test_publish_update_package_to_lanzou_uploads_then_deletes_old_files(monkeypatch, tmp_path):
    update_zip = tmp_path / "jixiebijiaoben_update_20260607_1743.zip"
    update_zip.write_bytes(b"zip")
    fake_session = object()
    deleted_ids: list[int] = []

    monkeypatch.setattr(
        build,
        "_lanzou_credentials_from_env",
        lambda: ("user", "pass", 13580356, "机械臂更新"),
    )
    monkeypatch.setattr(
        build,
        "_lanzou_login_session",
        lambda **kwargs: (fake_session, 2680763),
    )
    monkeypatch.setattr(
        build,
        "_lanzou_resolve_update_folder_id",
        lambda *args, **kwargs: 13580356,
    )
    states = [
        [{"id": "289469534", "name_all": "jixiebijiaoben_update_20260607_1706.zip"}],
        [
            {"id": "289469534", "name_all": "jixiebijiaoben_update_20260607_1706.zip"},
            {"id": "289500000", "name_all": update_zip.name},
        ],
        [{"id": "289500000", "name_all": update_zip.name}],
    ]
    monkeypatch.setattr(build, "_lanzou_list_files", lambda *args, **kwargs: states.pop(0))
    uploaded: list[str] = []
    monkeypatch.setattr(
        build,
        "_lanzou_upload_file",
        lambda session, *, file_path, folder_id, timeout_seconds=3600: uploaded.append(f"{folder_id}:{file_path.name}"),
    )
    monkeypatch.setattr(
        build,
        "_lanzou_delete_file",
        lambda session, *, file_id, timeout_seconds=build.DEFAULT_LANZOU_TIMEOUT_SECONDS: deleted_ids.append(file_id),
    )

    build._publish_update_package_to_lanzou(update_zip)

    assert uploaded == [f"13580356:{update_zip.name}"]
    assert deleted_ids == [289469534]


def test_publish_release_manifest_requires_package_in_lanzou(monkeypatch):
    monkeypatch.setattr(
        "services.update_sources.resolve_lanzou_update_package",
        lambda **kwargs: (_ for _ in ()).throw(
            __import__("services.update_sources", fromlist=["UpdateSourceError"]).UpdateSourceError("missing")
        ),
    )

    with __import__("pytest").raises(RuntimeError, match="not available in the configured Lanzou share yet"):
        build._publish_release_manifest(
            app_version="app-deadbeefcafebabe",
            runtime_version="2026.06.02-full-package",
            model_version="2026.06.02-lite",
            update_file_name="jixiebijiaoben_update_20260607_0904.zip",
        )


def test_install_runtime_into_package_emits_expected_command(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    package_root = tmp_path / "package"
    captured: dict[str, object] = {}

    def fake_run_build_step(command, *, cwd):
        captured["command"] = list(command)
        captured["cwd"] = cwd

    monkeypatch.setattr(build, "_run_build_step", fake_run_build_step)
    monkeypatch.setattr(build.sys, "executable", "C:/Python/python.exe")

    build._install_runtime_into_package(
        root,
        package_root,
        runtime_name="ocr_nvidia",
        install_scope="backend",
        backend="paddle_cuda",
        model_tier="lite",
        cuda_lane="cu118",
        skip_validation=True,
        allow_missing_hardware=True,
    )

    assert captured["cwd"] == root
    assert captured["command"] == [
        "C:/Python/python.exe",
        str(root / "runtime_builder.py"),
        "--package-root",
        str(package_root),
        "--runtime-name",
        "ocr_nvidia",
        "--install-scope",
        "backend",
        "--model-tier",
        "lite",
        "--skip-state-write",
        "--backend",
        "paddle_cuda",
        "--cuda-lane",
        "cu118",
        "--skip-validation",
        "--allow-missing-hardware",
    ]


def test_write_full_package_runtime_state_uses_auto_backend(tmp_path):
    package_root = tmp_path / build.PACKAGE_NAME
    (package_root / "runtime" / "app").mkdir(parents=True)
    (package_root / "runtime_manifest.json").write_text(
        """
{
  "models": {
    "tiers": {
      "lite": {
        "backends": {
          "cpu": {
            "provider": "paddlex_named_models",
            "detModelName": "PP-OCRv5_mobile_det",
            "recModelName": "PP-OCRv5_mobile_rec",
            "detVersion": "PP-OCRv5_mobile_det",
            "recVersion": "PP-OCRv5_mobile_rec"
          }
        }
      }
    }
  }
}
""".strip(),
        encoding="utf-8",
    )

    build._write_full_package_runtime_state(package_root, model_tier="lite", nvidia_cuda_lane="cu118")

    payload = json.loads((package_root / "runtime" / "install_state.json").read_text(encoding="utf-8"))
    assert payload["backend"] == "auto"
    assert payload["execution_mode"] == "subprocess"
    assert payload["package_mode"] == "full"
    assert payload["cache_home"] == "runtime/.paddlex"
    assert payload["python_executable"] == "engine/app/python.exe"
    assert payload["runtime_path"] == "engine/app"
    assert payload["runtime_inventory"]["app"] == "engine/app"
    assert payload["runtime_inventory"]["cpu"] == "engine/ocr_cpu"
    assert payload["installed_backends"] == ["cpu", "paddle_cuda", "onnx_directml"]


def test_runtime_cache_key_changes_with_cuda_lane(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for relative, content in (
        ("runtime_builder.py", "print('builder')\n"),
        ("runtime_manifest.json", "{}\n"),
        ("requirements.txt", "opencv-python\n"),
        ("runtime_layout.py", "print('layout')\n"),
        ("ocr_subprocess_entry.py", "print('ocr')\n"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    lane_a = build._runtime_cache_key(
        root,
        runtime_name="ocr_nvidia",
        install_scope="backend",
        backend="paddle_cuda",
        model_tier="lite",
        cuda_lane="cu118",
        skip_validation=True,
        bundle_models=False,
    )
    lane_b = build._runtime_cache_key(
        root,
        runtime_name="ocr_nvidia",
        install_scope="backend",
        backend="paddle_cuda",
        model_tier="lite",
        cuda_lane="cu126",
        skip_validation=True,
        bundle_models=False,
    )

    assert lane_a != lane_b


def test_runtime_cache_key_ignores_local_release_metadata_fields(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for relative, content in (
        ("runtime_builder.py", "print('builder')\n"),
        ("runtime_manifest.json", '{\n  "version": "2026.06.02-full-package"\n}\n'),
        ("requirements.txt", "opencv-python\n"),
        ("runtime_layout.py", "print('layout')\n"),
        ("ocr_subprocess_entry.py", "print('ocr')\n"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    key_a = build._runtime_cache_key(
        root,
        runtime_name="ocr_nvidia",
        install_scope="backend",
        backend="paddle_cuda",
        model_tier="lite",
        cuda_lane="cu118",
        skip_validation=True,
        bundle_models=False,
    )
    (root / "runtime_manifest.json").write_text(
        json.dumps(
            {
                "version": "2026.06.02-full-package",
                "appVersion": "app-deadbeefcafebabe",
                "runtimeVersion": "2026.06.02-full-package",
                "modelVersion": "2026.06.02-lite",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    key_b = build._runtime_cache_key(
        root,
        runtime_name="ocr_nvidia",
        install_scope="backend",
        backend="paddle_cuda",
        model_tier="lite",
        cuda_lane="cu118",
        skip_validation=True,
        bundle_models=False,
    )

    assert key_a == key_b


def test_package_cache_key_changes_with_source_update(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "apps" / "xinyu").mkdir(parents=True)
    (root / "apps" / "xinyu" / "app.py").write_text("print('a')\n", encoding="utf-8")
    (root / "vision").mkdir()
    (root / "vision" / "ocr.py").write_text("print('ocr')\n", encoding="utf-8")
    (root / "services").mkdir()
    (root / "ui").mkdir()
    (root / "workers").mkdir()
    (root / "lgb").mkdir()
    for relative, content in (
        ("main.py", "print('main')\n"),
        ("launcher_entry.py", "print('launcher')\n"),
        ("ocr_subprocess_entry.py", "print('ocr_sub')\n"),
        ("runtime_builder.py", "print('builder')\n"),
        ("runtime_layout.py", "print('layout')\n"),
        ("requirements.txt", "opencv-python\n"),
        ("pyproject.toml", "[build-system]\n"),
        ("config.json.example", "{}\n"),
        ("runtime_manifest.json", "{}\n"),
        ("build.py", "print('build')\n"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    key_a = build._package_cache_key(root, model_tier="lite", nvidia_cuda_lane="cu118", include_full_payload=False)
    (root / "apps" / "xinyu" / "app.py").write_text("print('b')\n", encoding="utf-8")
    key_b = build._package_cache_key(root, model_tier="lite", nvidia_cuda_lane="cu118", include_full_payload=False)

    assert key_a != key_b


def test_package_cache_key_ignores_local_release_metadata_fields(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "apps" / "xinyu").mkdir(parents=True)
    (root / "apps" / "xinyu" / "app.py").write_text("print('a')\n", encoding="utf-8")
    (root / "vision").mkdir()
    (root / "vision" / "ocr.py").write_text("print('ocr')\n", encoding="utf-8")
    (root / "services").mkdir()
    (root / "ui").mkdir()
    (root / "workers").mkdir()
    (root / "lgb").mkdir()
    for relative, content in (
        ("main.py", "print('main')\n"),
        ("launcher_entry.py", "print('launcher')\n"),
        ("ocr_subprocess_entry.py", "print('ocr_sub')\n"),
        ("runtime_builder.py", "print('builder')\n"),
        ("runtime_layout.py", "print('layout')\n"),
        ("requirements.txt", "opencv-python\n"),
        ("pyproject.toml", "[build-system]\n"),
        ("config.json.example", "{}\n"),
        ("runtime_manifest.json", '{\n  "version": "2026.06.02-full-package"\n}\n'),
        ("build.py", "print('build')\n"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    key_a = build._package_cache_key(root, model_tier="lite", nvidia_cuda_lane="cu118", include_full_payload=False)
    (root / "runtime_manifest.json").write_text(
        json.dumps(
            {
                "version": "2026.06.02-full-package",
                "appVersion": "app-deadbeefcafebabe",
                "runtimeVersion": "2026.06.02-full-package",
                "modelVersion": "2026.06.02-lite",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    key_b = build._package_cache_key(root, model_tier="lite", nvidia_cuda_lane="cu118", include_full_payload=False)

    assert key_a == key_b


def test_restore_package_from_cache_restores_zip_and_package_dir(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    dist_root = tmp_path / "dist"
    cache_dir = tmp_path / "pkg_cache" / "cache123"
    cached_package = cache_dir / build.PACKAGE_NAME
    cached_package.mkdir(parents=True)
    (cached_package / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (cache_dir / f"{build.PACKAGE_NAME}_full.zip").write_bytes(b"full")
    (cache_dir / f"{build.PACKAGE_NAME}_update.zip").write_bytes(b"update")

    monkeypatch.setattr(build, "_junction_or_copy_dir", lambda src, dst: shutil.copytree(src, dst))

    published_root, full_zip, update_zip = build._restore_package_from_cache(root, dist_root, cache_dir)

    assert published_root == dist_root / build.PACKAGE_NAME
    assert (published_root / "main.py").read_text(encoding="utf-8") == "print('ok')\n"
    assert full_zip.read_bytes() == b"full"
    assert update_zip.read_bytes() == b"update"


def test_restore_archives_from_cache_restores_only_zip_files(tmp_path):
    dist_root = tmp_path / "dist"
    cache_dir = tmp_path / "pkg_cache" / "cache123"
    cache_dir.mkdir(parents=True)
    (cache_dir / f"{build.PACKAGE_NAME}_full.zip").write_bytes(b"full")
    (cache_dir / f"{build.PACKAGE_NAME}_update.zip").write_bytes(b"update")

    full_zip, update_zip = build._restore_archives_from_cache(dist_root, cache_dir, include_full=False)

    assert full_zip is None
    assert update_zip.read_bytes() == b"update"
    assert not (dist_root / build.PACKAGE_NAME).exists()


def test_restore_archives_from_cache_can_restore_full_zip(tmp_path):
    dist_root = tmp_path / "dist"
    cache_dir = tmp_path / "pkg_cache" / "cache123"
    cache_dir.mkdir(parents=True)
    (cache_dir / f"{build.PACKAGE_NAME}_full.zip").write_bytes(b"full")
    (cache_dir / f"{build.PACKAGE_NAME}_update.zip").write_bytes(b"update")

    full_zip, update_zip = build._restore_archives_from_cache(dist_root, cache_dir, include_full=True)

    assert full_zip is not None
    assert full_zip.read_bytes() == b"full"
    assert update_zip.read_bytes() == b"update"


def test_restore_archives_from_cache_writes_only_timestamped_names(tmp_path):
    dist_root = tmp_path / "dist"
    cache_dir = tmp_path / "pkg_cache" / "cache123"
    cache_dir.mkdir(parents=True)
    (cache_dir / f"{build.PACKAGE_NAME}_full.zip").write_bytes(b"full")
    (cache_dir / f"{build.PACKAGE_NAME}_update.zip").write_bytes(b"update")

    full_zip, update_zip = build._restore_archives_from_cache(
        dist_root,
        cache_dir,
        include_full=True,
        timestamp="20260604_1825",
    )

    assert full_zip is not None
    assert full_zip.name == f"{build.PACKAGE_NAME}_full_20260604_1825.zip"
    assert update_zip.name == f"{build.PACKAGE_NAME}_update_20260604_1825.zip"
    assert not (dist_root / f"{build.PACKAGE_NAME}_full.zip").exists()
    assert not (dist_root / f"{build.PACKAGE_NAME}_update.zip").exists()


def test_remove_dist_archives_removes_legacy_and_timestamped_zip_names(tmp_path):
    dist_root = tmp_path / "dist"
    dist_root.mkdir()
    (dist_root / f"{build.PACKAGE_NAME}_update.zip").write_bytes(b"legacy")
    (dist_root / f"{build.PACKAGE_NAME}_update_20260604_1825.zip").write_bytes(b"new")
    (dist_root / "other.zip").write_bytes(b"other")

    build._remove_dist_archives(dist_root)

    assert not (dist_root / f"{build.PACKAGE_NAME}_update.zip").exists()
    assert not (dist_root / f"{build.PACKAGE_NAME}_update_20260604_1825.zip").exists()
    assert (dist_root / "other.zip").is_file()


def test_ensure_runtime_available_reuses_cached_runtime(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    package_root = tmp_path / "package"
    runtime_cache = tmp_path / "runtime_cache" / "app" / "cache123" / "engine" / "app"
    (runtime_cache / "Lib").mkdir(parents=True)
    (runtime_cache / "python.exe").write_bytes(b"python")

    monkeypatch.setattr(build, "_runtime_cache_key", lambda *args, **kwargs: "cache123")
    monkeypatch.setattr(build, "_runtime_cache_root", lambda _root: tmp_path / "runtime_cache")
    install_calls: list[tuple] = []
    monkeypatch.setattr(build, "_install_runtime_into_package", lambda *args, **kwargs: install_calls.append((args, kwargs)))

    build._ensure_runtime_available(
        root,
        package_root,
        runtime_name="app",
        install_scope="app",
        model_tier="lite",
        use_cache=True,
    )

    assert (package_root / "engine" / "app" / "python.exe").read_bytes() == b"python"
    assert install_calls == []


def test_main_default_builds_update_only(tmp_path, monkeypatch):
    dist_root = tmp_path / "dist"
    package_root = tmp_path / "staging" / build.PACKAGE_NAME
    package_root.mkdir(parents=True)
    (tmp_path / "runtime_manifest.json").write_text(
        json.dumps(
            {
                "appVersion": "20260917-05",
                "runtimeVersion": "2026.09.10-engine",
                "modelVersion": "2026.06.02-lite",
            }
        ),
        encoding="utf-8",
    )
    update_zip = dist_root / f"{build.PACKAGE_NAME}_update.zip"
    update_zip.parent.mkdir(parents=True)
    update_zip.write_bytes(b"update")

    monkeypatch.setattr(build, "__file__", str(tmp_path / "build.py"))
    monkeypatch.setattr(build, "_prepare_staging_package_root", lambda _dist: package_root)
    monkeypatch.setattr(build, "_copy_sources", lambda *args, **kwargs: None)
    monkeypatch.setattr(build, "_copy_bundled_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(build, "_copy_bundled_vision_key", lambda *args, **kwargs: None)
    runtime_calls: list[str] = []
    full_calls: list[str] = []
    monkeypatch.setattr(build, "_build_full_package_runtimes", lambda *args, **kwargs: runtime_calls.append("runtime"))
    monkeypatch.setattr(build, "_cleanup_packaged_transients", lambda *args, **kwargs: None)
    monkeypatch.setattr(build, "_cleanup_python_caches", lambda *args, **kwargs: None)
    monkeypatch.setattr(build, "_write_full_package_runtime_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(build, "_freeze_executables", lambda *args, **kwargs: None)
    monkeypatch.setattr(build, "_copy_executable_aliases", lambda *args, **kwargs: None)
    monkeypatch.setattr(build, "_write_cmd_launchers", lambda *args, **kwargs: None)
    monkeypatch.setattr(build, "_write_package_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(build, "_write_full_package_archive", lambda *args, **kwargs: full_calls.append("full") or dist_root / "full.zip")
    monkeypatch.setattr(build, "_write_customer_update_archive", lambda *args, **kwargs: update_zip)
    monkeypatch.setattr(build, "_store_package_cache", lambda *args, **kwargs: tmp_path / "cache")
    monkeypatch.setattr(build, "_cleanup_staging", lambda *args, **kwargs: None)
    monkeypatch.setattr(build, "_package_cache_complete", lambda *args, **kwargs: False)

    assert build.main(["--skip-release-upload"]) == 0
    assert runtime_calls == []
    assert full_calls == []


def test_require_release_versions_rejects_empty_fields():
    """回归（20260917-02 P0）：空版本字段的清单随更新包下发会卡死客户机
    更新通道，必须在打包期直接失败。"""
    build._require_release_versions("20260917-05", "2026.09.10-engine", "2026.06.02-lite")

    with pytest.raises(SystemExit, match="runtimeVersion"):
        build._require_release_versions("20260917-05", "", "2026.06.02-lite")
    with pytest.raises(SystemExit, match="modelVersion"):
        build._require_release_versions("20260917-05", "2026.09.10-engine", "")
    with pytest.raises(SystemExit, match="appVersion"):
        build._require_release_versions("", "2026.09.10-engine", "2026.06.02-lite")
