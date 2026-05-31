from __future__ import annotations

import install_launcher_entry as launcher


def test_install_launcher_delegates_to_install_runtime(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(
        launcher.install_runtime,
        "main",
        lambda: calls.append("install_runtime_main") or 0,
    )

    assert launcher.main() == 0
    assert calls == ["install_runtime_main"]
