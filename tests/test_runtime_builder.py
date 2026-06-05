from __future__ import annotations

from pathlib import Path

import runtime_builder


def test_copy_python_tree_creates_portable_runtime_layout(tmp_path: Path):
    source = tmp_path / "source_python"
    target = tmp_path / "runtime"
    (source / "Lib" / "site-packages").mkdir(parents=True)
    (source / "DLLs").mkdir()
    (source / "libs").mkdir()
    (source / "Scripts").mkdir()

    for file_name in (
        "python.exe",
        "pythonw.exe",
        "python3.dll",
        f"python{runtime_builder.sys.version_info.major}{runtime_builder.sys.version_info.minor}.dll",
    ):
        (source / file_name).write_bytes(b"bin")
    (source / "Lib" / "os.py").write_text("print('ok')\n", encoding="utf-8")
    (source / "Lib" / "site-packages" / "should_not_copy.txt").write_text("x", encoding="utf-8")

    runtime_builder._copy_python_tree(source, target)

    assert (target / "python.exe").is_file()
    assert (target / "pythonw.exe").is_file()
    assert (target / "Lib" / "os.py").is_file()
    assert (target / "Lib" / "site-packages").is_dir()
    assert not (target / "Lib" / "site-packages" / "should_not_copy.txt").exists()


def test_copy_python_tree_bundles_visual_cpp_runtime_dlls(monkeypatch, tmp_path: Path):
    source = tmp_path / "source_python"
    target = tmp_path / "runtime"
    system_root = tmp_path / "Windows"
    (source / "Lib").mkdir(parents=True)
    (source / "DLLs").mkdir()
    (source / "libs").mkdir()
    (system_root / "System32").mkdir(parents=True)
    (source / "python.exe").write_bytes(b"python")
    (source / "pythonw.exe").write_bytes(b"pythonw")
    (source / "python3.dll").write_bytes(b"python3")
    (source / f"python{runtime_builder.sys.version_info.major}{runtime_builder.sys.version_info.minor}.dll").write_bytes(b"python-version")
    (system_root / "System32" / "msvcp140.dll").write_bytes(b"msvcp")
    (system_root / "System32" / "vcruntime140.dll").write_bytes(b"vcruntime")

    monkeypatch.setattr(runtime_builder.os, "name", "nt")
    monkeypatch.setenv("SystemRoot", str(system_root))

    runtime_builder._copy_python_tree(source, target)

    assert (target / "msvcp140.dll").read_bytes() == b"msvcp"
    assert (target / "vcruntime140.dll").read_bytes() == b"vcruntime"


def test_uninstall_conflicting_paddle_backend_for_cuda(monkeypatch, tmp_path: Path):
    python_exe = tmp_path / "runtime" / "python.exe"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_bytes(b"python")
    captured: dict[str, object] = {}

    class Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["env"] = dict(kwargs["env"])
        return Completed()

    monkeypatch.setattr(runtime_builder.subprocess, "run", fake_run)

    runtime_builder._uninstall_conflicting_paddle_backend(python_exe, "paddle_cuda", dry_run=False)

    assert captured["command"] == [str(python_exe), "-m", "pip", "uninstall", "-y", "paddlepaddle"]
    assert captured["env"]["PYTHONNOUSERSITE"] == "1"


def test_uninstall_conflicting_paddle_backend_for_cpu(monkeypatch, tmp_path: Path):
    python_exe = tmp_path / "runtime" / "python.exe"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_bytes(b"python")
    captured: dict[str, object] = {}

    class Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        return Completed()

    monkeypatch.setattr(runtime_builder.subprocess, "run", fake_run)

    runtime_builder._uninstall_conflicting_paddle_backend(python_exe, "cpu", dry_run=False)

    assert captured["command"] == [str(python_exe), "-m", "pip", "uninstall", "-y", "paddlepaddle-gpu"]
