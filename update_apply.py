from __future__ import annotations

import argparse
import ctypes
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path


PACKAGE_NAME = "jixiebijiaoben"
WAIT_TIMEOUT_SECONDS = 120
PRESERVE_PREFIXES = (
    "config.json",
    "outbox.json",
    "runtime/install_state.json",
    "runtime/logs/",
    "runtime/errors/",
    "runtime/memory/",
    "runtime/reports/",
    "runtime/tmp/",
    "runtime/.paddlex/",
    "photo/",
    "jiekou/",
)
PRESERVED_LAUNCHERS = {"机械臂脚本.exe", "jixiebijiaoben_launcher.exe"}


def _is_app_config_relative(relative: str) -> bool:
    parts = relative.replace("\\", "/").lstrip("/").split("/")
    return len(parts) == 3 and parts[0] == "apps" and parts[2] == "config.json" and bool(parts[1])


def _is_customer_template_relative(relative: str) -> bool:
    """客户新增模板跨版本保留，更新包不能覆盖或清理。"""
    parts = relative.replace("\\", "/").lstrip("/").split("/")
    if len(parts) < 4 or parts[0] != "apps" or parts[2] != "templates":
        return False
    return any("__customer_" in part for part in parts[3:])


def _is_windows() -> bool:
    return os.name == "nt"


def _show_message(title: str, message: str, *, error: bool = False) -> None:
    if _is_windows():
        style = 0x10 if error else 0x40
        ctypes.windll.user32.MessageBoxW(None, message, title, style)
        return
    stream = sys.stderr if error else sys.stdout
    print(f"{title}: {message}", file=stream)


def _wait_for_process_exit(pid: int, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    if _is_windows():
        synchronize = 0x00100000
        wait_timeout = 0x00000102
        kernel32 = ctypes.windll.kernel32
        while time.monotonic() < deadline:
            handle = kernel32.OpenProcess(synchronize, False, pid)
            if not handle:
                return
            try:
                result = kernel32.WaitForSingleObject(handle, 250)
            finally:
                kernel32.CloseHandle(handle)
            if result != wait_timeout:
                return
        raise RuntimeError("Timed out waiting for the running program to exit.")
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.25)
    raise RuntimeError("Timed out waiting for the running program to exit.")


def _extract_archive(archive_path: Path) -> Path:
    temp_root = Path(tempfile.mkdtemp(prefix="jxb_update_"))
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(temp_root)
    except zipfile.BadZipFile as exc:
        raise RuntimeError("Downloaded update package is not a valid zip archive.") from exc
    package_root = temp_root / PACKAGE_NAME
    if package_root.is_dir():
        return package_root
    children = [path for path in temp_root.iterdir() if path.is_dir()]
    if len(children) == 1:
        return children[0]
    raise RuntimeError("Could not locate the extracted update package root.")


def _read_manifest_paths(package_root: Path) -> set[str]:
    manifest_path = package_root / "package_manifest.json"
    if not manifest_path.is_file():
        return set()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    files = payload.get("files", [])
    if not isinstance(files, list):
        return set()
    paths: set[str] = set()
    for item in files:
        if isinstance(item, dict):
            relative = str(item.get("path", "") or "").strip().replace("\\", "/")
            if relative:
                paths.add(relative)
    return paths


def _is_preserved(relative: str) -> bool:
    normalized = relative.replace("\\", "/").lstrip("/")
    if normalized in PRESERVED_LAUNCHERS:
        return True
    if _is_app_config_relative(normalized) or _is_customer_template_relative(normalized):
        return True
    return any(
        normalized == prefix.rstrip("/") or normalized.startswith(prefix)
        for prefix in PRESERVE_PREFIXES
    )


def _read_json_object(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _copy_or_merge_update_file(source: Path, target: Path, relative: str) -> None:
    if _is_app_config_relative(relative) and target.is_file():
        update_config = _read_json_object(source)
        local_config = _read_json_object(target)
        if update_config is not None and local_config is not None:
            merged = _deep_merge(update_config, local_config)
            target.write_text(
                json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return
    # 程序模块 ZIP 可能仍被旧启动器占用。先在同目录写临时文件，再用
    # Windows 原子替换；避免直接创建已存在目标时触发 WinError 183。
    if relative.replace("\\", "/") == "app_modules.zip":
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            shutil.copy2(source, temporary)
            # 某些旧安装可能把同名路径留下为目录；清理这个精确目标后再替换。
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            os.replace(temporary, target)
        finally:
            with contextlib.suppress(OSError):
                temporary.unlink()
        return
    shutil.copy2(source, target)


def _remove_obsolete_files(target_root: Path, old_paths: set[str], new_paths: set[str]) -> None:
    obsolete = sorted(path for path in old_paths - new_paths if not _is_preserved(path))
    for relative in obsolete:
        target = target_root / relative
        if not target.exists():
            continue
        if target.is_file() or target.is_symlink():
            target.unlink(missing_ok=True)


def _copy_update_tree(source_root: Path, target_root: Path) -> None:
    for source in sorted(source_root.rglob("*")):
        relative = source.relative_to(source_root).as_posix()
        is_app_config = _is_app_config_relative(relative)
        if _is_preserved(relative) and not is_app_config:
            continue
        target = target_root / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_or_merge_update_file(source, target, relative)


def _write_error_log(target_root: Path, message: str) -> None:
    error_dir = target_root / "runtime" / "errors"
    try:
        error_dir.mkdir(parents=True, exist_ok=True)
        (error_dir / "update_apply_error.txt").write_text(message, encoding="utf-8")
    except OSError:
        pass


def apply_update(*, package_root: Path, archive_path: Path, pid: int) -> None:
    _wait_for_process_exit(pid, WAIT_TIMEOUT_SECONDS)
    extracted_root = _extract_archive(archive_path)
    existing_paths = _read_manifest_paths(package_root)
    new_paths = _read_manifest_paths(extracted_root)
    _remove_obsolete_files(package_root, existing_paths, new_paths)
    _copy_update_tree(extracted_root, package_root)
    archive_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply a downloaded jixiebijiaoben update package.")
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--pid", required=True, type=int)
    args = parser.parse_args(argv)

    package_root = Path(args.package_root).resolve()
    archive_path = Path(args.archive).resolve()

    try:
        apply_update(package_root=package_root, archive_path=archive_path, pid=args.pid)
    except Exception as exc:
        _write_error_log(package_root, str(exc))
        _show_message("更新失败", f"应用更新失败：{exc}", error=True)
        return 1

    _show_message("更新完成", "更新已完成，请重新打开主程序。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
