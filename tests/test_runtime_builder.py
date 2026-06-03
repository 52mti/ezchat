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
