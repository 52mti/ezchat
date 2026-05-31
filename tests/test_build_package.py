from __future__ import annotations

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
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
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
    manifest_path = tmp_path / "runtime_manifest.json"
    manifest_path.write_text(
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


def test_runtime_manifest_payload_reads_shared_and_backend_packages(tmp_path):
    manifest_path = tmp_path / "runtime_manifest.json"
    manifest_path.write_text(
        """
{
  "sharedPackages": [{"spec": "paddleocr==3.4.0"}],
  "backends": {
    "cpu": {
      "packages": [{"spec": "paddlepaddle==3.2.2"}]
    }
  }
}
""".strip(),
        encoding="utf-8",
    )

    payload = build._runtime_manifest_payload(tmp_path)

    assert payload["sharedPackages"][0]["spec"] == "paddleocr==3.4.0"
    assert payload["backends"]["cpu"]["packages"][0]["spec"] == "paddlepaddle==3.2.2"


def test_write_customer_update_archive_excludes_bootstrap(tmp_path):
    package_root = tmp_path / build.PACKAGE_NAME
    dist_root = tmp_path / "dist"
    (package_root / "apps").mkdir(parents=True)
    (package_root / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (package_root / build.INSTALLER_EXE).write_bytes(b"binary")
    (package_root / "apps" / "config.json").write_text("{}", encoding="utf-8")

    archive_path = build._write_customer_update_archive(package_root, dist_root)

    assert archive_path.is_file()
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())

    assert f"{build.PACKAGE_NAME}/main.py" in names
    assert f"{build.PACKAGE_NAME}/apps/config.json" in names
    assert f"{build.PACKAGE_NAME}/{build.INSTALLER_EXE}" in names
    assert all(not name.startswith(f"{build.PACKAGE_NAME}/_bootstrap/") for name in names)


def test_write_full_package_archive_excludes_bootstrap(tmp_path):
    package_root = tmp_path / build.PACKAGE_NAME
    dist_root = tmp_path / "dist"
    (package_root / "apps").mkdir(parents=True)
    (package_root / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (package_root / "apps" / "config.json").write_text("{}", encoding="utf-8")
    (package_root / build.INSTALLER_EXE).write_bytes(b"binary")
    (package_root / "runtime" / "install_state.json").parent.mkdir(parents=True, exist_ok=True)
    (package_root / "runtime" / "install_state.json").write_text("{}", encoding="utf-8")

    archive_path = build._write_full_package_archive(package_root, dist_root)

    assert archive_path.is_file()
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())

    assert f"{build.PACKAGE_NAME}/main.py" in names
    assert f"{build.PACKAGE_NAME}/apps/config.json" in names
    assert f"{build.PACKAGE_NAME}/{build.INSTALLER_EXE}" in names
    assert all(not name.startswith(f"{build.PACKAGE_NAME}/_bootstrap/") for name in names)
    assert all("runtime/install_state.json" not in name for name in names)


def test_copy_executable_aliases_creates_ascii_aliases(tmp_path):
    package_root = tmp_path / build.PACKAGE_NAME
    package_root.mkdir()
    launcher_bytes = b"launcher"
    installer_bytes = b"installer"
    (package_root / build.LAUNCHER_EXE).write_bytes(launcher_bytes)
    (package_root / build.INSTALLER_EXE).write_bytes(installer_bytes)

    build._copy_executable_aliases(package_root)

    assert (package_root / build.LAUNCHER_ALIAS_EXE).read_bytes() == launcher_bytes
    assert (package_root / build.INSTALLER_ALIAS_EXE).read_bytes() == installer_bytes


def test_write_cmd_launchers_emit_ascii_crlf_scripts(tmp_path):
    package_root = tmp_path / build.PACKAGE_NAME
    package_root.mkdir()

    build._write_cmd_launchers(package_root)

    installer_bytes = (package_root / "安装环境.cmd").read_bytes()
    launcher_bytes = (package_root / "启动程序.cmd").read_bytes()

    for payload in (installer_bytes, launcher_bytes):
        assert b"\r\n" in payload
        assert b"\n" not in payload.replace(b"\r\n", b"")
        assert all(byte < 128 for byte in payload)

    installer_text = installer_bytes.decode("ascii")
    launcher_text = launcher_bytes.decode("ascii")

    assert build.INSTALLER_ALIAS_EXE in installer_text
    assert build.LAUNCHER_ALIAS_EXE in launcher_text
    assert ":run_installer" in installer_text
    assert ":run_runtime" in launcher_text
    assert build.INSTALLER_EXE not in installer_text
    assert build.LAUNCHER_EXE not in launcher_text
