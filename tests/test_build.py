from __future__ import annotations

from pathlib import Path

from build import _ignore


def test_build_ignore_excludes_generated_and_private_files():
    ignored = _ignore(
        "x",
        ["__pycache__", ".pytest_cache", "config.json", "outbox.json", "a.pyc", "keep.py"],
    )

    assert "__pycache__" in ignored
    assert "config.json" in ignored
    assert "outbox.json" in ignored
    assert "a.pyc" in ignored
    assert "keep.py" not in ignored
