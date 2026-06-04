from __future__ import annotations

import json
import shutil
import zipfile

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


def test_write_customer_update_archive_excludes_full_runtimes_and_models(tmp_path):
    package_root = tmp_path / build.PACKAGE_NAME
    dist_root = tmp_path / "dist"
    (package_root / "apps").mkdir(parents=True)
    (package_root / "runtime" / "app").mkdir(parents=True)
    (package_root / "runtime" / "ocr_cpu").mkdir(parents=True)
    (package_root / "models" / "ocr" / "paddlex_named_models").mkdir(parents=True)
    (package_root / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (package_root / build.LAUNCHER_EXE).write_bytes(b"binary")
    (package_root / "apps" / "config.json").write_text("{}", encoding="utf-8")
    (package_root / "runtime" / "app" / "python.exe").write_bytes(b"python")
    (package_root / "runtime" / "ocr_cpu" / "python.exe").write_bytes(b"python")
    (package_root / "models" / "ocr" / "paddlex_named_models" / "demo.bin").write_bytes(b"model")

    archive_path = build._write_customer_update_archive(package_root, dist_root)

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())

    assert f"{build.PACKAGE_NAME}/main.py" in names
    assert f"{build.PACKAGE_NAME}/{build.LAUNCHER_EXE}" in names
    assert all(not name.startswith(f"{build.PACKAGE_NAME}/runtime/app/") for name in names)
    assert all(not name.startswith(f"{build.PACKAGE_NAME}/runtime/ocr_cpu/") for name in names)
    assert all(not name.startswith(f"{build.PACKAGE_NAME}/models/ocr/") for name in names)


def test_write_customer_update_archive_excludes_app_customer_config_assets(tmp_path):
    package_root = tmp_path / build.PACKAGE_NAME
    dist_root = tmp_path / "dist"
    app_root = package_root / "apps" / "xinyu"
    (app_root / "templates" / "buttons").mkdir(parents=True)
    (app_root / "app.py").write_text("APP = 'xinyu'\n", encoding="utf-8")
    (app_root / "steps.py").write_text("STEPS = []\n", encoding="utf-8")
    (app_root / "config.py").write_text("NAME = 'xinyu'\n", encoding="utf-8")
    (app_root / "config.json").write_text('{"persona": "custom"}', encoding="utf-8")
    (app_root / "ocr_config.py").write_text("RULES = {}\n", encoding="utf-8")
    (app_root / "templates" / "buttons" / "btn_返回按钮.jpg").write_bytes(b"image")

    archive_path = build._write_customer_update_archive(package_root, dist_root)

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())

    assert f"{build.PACKAGE_NAME}/apps/xinyu/app.py" in names
    assert f"{build.PACKAGE_NAME}/apps/xinyu/steps.py" in names
    assert f"{build.PACKAGE_NAME}/apps/xinyu/config.py" in names
    assert f"{build.PACKAGE_NAME}/apps/xinyu/config.json" not in names
    assert f"{build.PACKAGE_NAME}/apps/xinyu/ocr_config.py" not in names
    assert all(not name.startswith(f"{build.PACKAGE_NAME}/apps/xinyu/templates/") for name in names)


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


def test_write_full_package_archive_includes_app_customer_config_assets(tmp_path):
    package_root = tmp_path / build.PACKAGE_NAME
    dist_root = tmp_path / "dist"
    app_root = package_root / "apps" / "xinyu"
    (app_root / "templates" / "buttons").mkdir(parents=True)
    (app_root / "config.json").write_text('{"persona": "default"}', encoding="utf-8")
    (app_root / "ocr_config.py").write_text("RULES = {}\n", encoding="utf-8")
    (app_root / "templates" / "buttons" / "btn_返回按钮.jpg").write_bytes(b"image")

    archive_path = build._write_full_package_archive(package_root, dist_root)

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())

    assert f"{build.PACKAGE_NAME}/apps/xinyu/config.json" in names
    assert f"{build.PACKAGE_NAME}/apps/xinyu/ocr_config.py" in names
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


def test_ensure_runtime_available_reuses_cached_runtime(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    package_root = tmp_path / "package"
    runtime_cache = tmp_path / "runtime_cache" / "app" / "cache123" / "runtime" / "app"
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

    assert (package_root / "runtime" / "app" / "python.exe").read_bytes() == b"python"
    assert install_calls == []


def test_main_default_builds_update_only(tmp_path, monkeypatch):
    dist_root = tmp_path / "dist"
    package_root = tmp_path / "staging" / build.PACKAGE_NAME
    package_root.mkdir(parents=True)
    update_zip = dist_root / f"{build.PACKAGE_NAME}_update.zip"
    update_zip.parent.mkdir(parents=True)
    update_zip.write_bytes(b"update")

    monkeypatch.setattr(build, "__file__", str(tmp_path / "build.py"))
    monkeypatch.setattr(build, "_prepare_staging_package_root", lambda _dist: package_root)
    monkeypatch.setattr(build, "_copy_sources", lambda *args, **kwargs: None)
    monkeypatch.setattr(build, "_copy_bundled_data", lambda *args, **kwargs: None)
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

    assert build.main([]) == 0
    assert runtime_calls == []
    assert full_calls == []
