from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path


INCLUDE_PATHS = [
    "apps",
    "vision",
    "services",
    "ui",
    "workers",
    "lgb",
    "jiekou",
    "main.py",
    "launcher_entry.py",
    "install_runtime.py",
    "runtime_manifest.json",
    "requirements.txt",
    "config.json.example",
    "pyproject.toml",
]


def _ignore(dir_path: str, names: list[str]) -> set[str]:
    ignored = {".git", "__pycache__", ".pytest_cache", "dist", "config.json", "outbox.json"}
    return {
        name
        for name in names
        if name in ignored
        or name.endswith((".pyc", ".pyo", ".tmp"))
        or name == "install_state.json"
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_item(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, ignore=_ignore, dirs_exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _handle_remove_readonly(function, path, excinfo) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


def main() -> int:
    root = Path.cwd()
    dist = root / "dist"
    package_root = dist / "机械臂脚本"
    if package_root.exists():
        shutil.rmtree(package_root, onexc=_handle_remove_readonly)
    package_root.mkdir(parents=True, exist_ok=True)

    for relative in INCLUDE_PATHS:
        source = root / relative
        if source.exists():
            _copy_item(source, package_root / relative)

    for relative in ["runtime/logs", "runtime/memory", "photo", "photo/errors"]:
        (package_root / relative).mkdir(parents=True, exist_ok=True)
    for cache in package_root.rglob("__pycache__"):
        if cache.is_dir():
            shutil.rmtree(cache)
    for compiled in list(package_root.rglob("*.pyc")) + list(package_root.rglob("*.pyo")):
        compiled.unlink(missing_ok=True)

    files = []
    for path in sorted(package_root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(package_root).as_posix()
            files.append({"path": relative, "sha256": _sha256(path), "size": path.stat().st_size})

    manifest = {
        "schema_version": 1,
        "package_name": "机械臂脚本",
        "contains_private_config": False,
        "files": files,
    }
    (package_root / "package_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote package directory: {package_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
