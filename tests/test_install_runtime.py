from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import install_runtime
from install_runtime import _prune_installed_docs


def test_prune_installed_docs_removes_docs_and_test_dirs(tmp_path: Path):
    site_packages = tmp_path / "Lib" / "site-packages" / "demo_pkg"
    tests_dir = site_packages / "tests"
    docs_file = site_packages / "README.md"
    kept_file = site_packages / "runtime.py"

    tests_dir.mkdir(parents=True)
    (tests_dir / "test_sample.py").write_text("pass\n", encoding="utf-8")
    docs_file.write_text("doc\n", encoding="utf-8")
    kept_file.write_text("print('ok')\n", encoding="utf-8")

    _prune_installed_docs(tmp_path / "Lib" / "site-packages")

    assert not tests_dir.exists()
    assert not docs_file.exists()
    assert kept_file.exists()


def test_install_runtime_import_does_not_load_cv2():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import install_runtime; print('cv2' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.stdout.strip() == "False"


def test_install_runtime_import_sets_package_local_paddlex_cache():
    env = dict(os.environ)
    env["PUBLIC"] = str(Path.cwd() / "Public")
    env.pop("PADDLE_PDX_CACHE_HOME", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, install_runtime; print(os.environ.get('PADDLE_PDX_CACHE_HOME', ''))",
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.stdout.strip().endswith(r"jixiebijiaoben_runtime\.paddlex")


def test_package_root_uses_frozen_executable_directory(monkeypatch, tmp_path: Path):
    exe_path = tmp_path / "package" / "安装环境.exe"
    exe_path.parent.mkdir(parents=True)
    exe_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(install_runtime.sys, "frozen", True, raising=False)
    monkeypatch.setattr(install_runtime.sys, "executable", str(exe_path))

    assert install_runtime._package_root() == exe_path.parent.resolve()


def test_private_site_packages_supports_bootstrap_layout(tmp_path: Path, monkeypatch):
    package_root = tmp_path / "package"
    root_site = package_root / ".runtime_env" / "Lib" / "site-packages"
    root_site.mkdir(parents=True)

    monkeypatch.setattr(install_runtime, "_package_root", lambda: package_root)

    assert install_runtime._private_site_packages() == root_site


def test_create_runtime_venv_copies_bootstrap_python_on_windows(tmp_path: Path, monkeypatch):
    package_root = tmp_path / "package"
    bootstrap_python = package_root / "_bootstrap" / "python"
    bootstrap_python.mkdir(parents=True)
    (bootstrap_python / "python.exe").write_bytes(b"python")
    (bootstrap_python / "pythonw.exe").write_bytes(b"pythonw")
    (bootstrap_python / "python313._pth").write_text("python313.zip\n.\nimport site\n", encoding="utf-8")

    captured: list[tuple[Path, bool]] = []

    def fake_bootstrap_pip(python_exe: Path, *, dry_run: bool) -> None:
        captured.append((python_exe, dry_run))

    monkeypatch.setattr(install_runtime, "_package_root", lambda: package_root)
    monkeypatch.setattr(install_runtime, "_bootstrap_pip_into_venv", fake_bootstrap_pip)

    python_exe = install_runtime._create_runtime_venv(dry_run=False)

    assert python_exe == package_root / ".runtime_env" / "python.exe"
    assert python_exe.read_bytes() == b"python"
    assert (package_root / ".runtime_env" / "pythonw.exe").read_bytes() == b"pythonw"
    assert (package_root / ".runtime_env" / "Lib" / "site-packages").is_dir()
    assert ".." in (package_root / ".runtime_env" / "python313._pth").read_text(encoding="utf-8").splitlines()
    assert captured == [(python_exe, False)]


def test_pip_install_specs_suppresses_script_location_warning(tmp_path: Path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_check_call(command, cwd=None, env=None):
        captured["command"] = list(command)
        captured["cwd"] = cwd
        captured["env"] = env

    monkeypatch.setattr(install_runtime, "_package_root", lambda: tmp_path)
    monkeypatch.setattr(install_runtime, "_pip_env", lambda _python_exe=None: {})
    monkeypatch.setattr(install_runtime.subprocess, "check_call", fake_check_call)

    install_runtime._pip_install_specs(
        tmp_path / ".runtime_env" / "python.exe",
        ["demo-package"],
        "https://example.com/simple",
        dry_run=False,
    )

    assert "--no-warn-script-location" in captured["command"]
