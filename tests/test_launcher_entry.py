from __future__ import annotations

from pathlib import Path

import launcher_entry


def test_spawn_packaged_app_uses_inline_bootstrap(monkeypatch, tmp_path: Path):
    root = tmp_path / "package"
    python_exe = root / "runtime" / "app" / "python.exe"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("", encoding="utf-8")

    captured: dict[str, object] = {}

    class FakeProcess:
        def wait(self) -> int:
            return 0

    def fake_popen(command, cwd=None, env=None, creationflags=0):
        captured["command"] = list(command)
        captured["cwd"] = cwd
        captured["env"] = dict(env or {})
        captured["creationflags"] = creationflags
        return FakeProcess()

    monkeypatch.setattr(
        launcher_entry,
        "_load_runtime_state",
        lambda _root: {"status": "installed", "validation": "ok", "cache_home": str(root / "cache")},
    )
    monkeypatch.setattr(launcher_entry, "_runtime_python", lambda _root: python_exe)
    monkeypatch.setattr(launcher_entry, "_runtime_python_fallback", lambda _root: python_exe)
    monkeypatch.setattr(launcher_entry.subprocess, "Popen", fake_popen)

    assert launcher_entry._spawn_packaged_app(root) == 0
    assert captured["cwd"] == root
    assert captured["command"][:4] == [str(python_exe), "-s", "-u", "-c"]
    assert "sys.path.insert(0, str(root))" in captured["command"][4]
    assert "from main import main as _entry" in captured["command"][4]
    assert captured["env"]["PADDLEX_HOME"] == str(root / "cache")


def test_spawn_packaged_app_ignores_staging_cache_home(monkeypatch, tmp_path: Path):
    root = tmp_path / "package"
    python_exe = root / "runtime" / "app" / "python.exe"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("", encoding="utf-8")
    public_root = tmp_path / "Public"
    captured: dict[str, object] = {}

    class FakeProcess:
        def wait(self) -> int:
            return 0

    def fake_popen(command, cwd=None, env=None, creationflags=0):
        captured["env"] = dict(env or {})
        return FakeProcess()

    monkeypatch.setenv("PUBLIC", str(public_root))
    monkeypatch.setattr(
        launcher_entry,
        "_load_runtime_state",
        lambda _root: {
            "status": "installed",
            "validation": "ok",
            "cache_home": "D:/Program Files (x86)/appkaifa/jixiebijiaoben/dist/_staging/build_001/jixiebijiaoben/runtime/.paddlex",
        },
    )
    monkeypatch.setattr(launcher_entry, "_runtime_python", lambda _root: python_exe)
    monkeypatch.setattr(launcher_entry, "_runtime_python_fallback", lambda _root: python_exe)
    monkeypatch.setattr(launcher_entry.subprocess, "Popen", fake_popen)

    assert launcher_entry._spawn_packaged_app(root) == 0
    expected = public_root / "jixiebijiaoben_runtime" / ".paddlex"
    assert captured["env"]["PADDLEX_HOME"] == str(expected)
    assert captured["env"]["PADDLE_PDX_CACHE_HOME"] == str(expected)


def test_spawn_packaged_app_logs_nonzero_exit(monkeypatch, tmp_path: Path):
    root = tmp_path / "package"
    python_exe = root / "runtime" / "app" / "python.exe"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("", encoding="utf-8")

    logged: list[tuple[str, str]] = []

    class FakeProcess:
        def wait(self) -> int:
            return 3

    monkeypatch.setattr(
        launcher_entry,
        "_load_runtime_state",
        lambda _root: {"status": "installed", "validation": "ok", "cache_home": str(root / "cache")},
    )
    monkeypatch.setattr(launcher_entry, "_runtime_python", lambda _root: python_exe)
    monkeypatch.setattr(launcher_entry, "_runtime_python_fallback", lambda _root: python_exe)
    monkeypatch.setattr(launcher_entry.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(
        launcher_entry,
        "_write_launcher_error",
        lambda _root, name, details: logged.append((name, details)),
    )

    assert launcher_entry._spawn_packaged_app(root) == 3
    assert logged
    assert logged[-1][0] == "launcher_exit_status.txt"
    assert "exit_code=3" in logged[-1][1]
