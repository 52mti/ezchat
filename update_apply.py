from __future__ import annotations

import argparse
import contextlib
import ctypes
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
LOCK_RETRY_SECONDS = 30.0

# 更新器结构隔离规则（与 build.py 的更新包排除规则互为镜像）：
# 1) engine/、models/ 只随全量包变化，runtime/ 是客户运行状态，tts/piper
#    是顶层 TTS 引擎——四者任何情况下都不删除、不覆盖，不依赖清单判断；
# 2) 以下前缀是客户数据或本机文件，同样保留。
UNTOUCHABLE_PREFIXES = ("engine/", "models/", "runtime/", "tts/piper/")
PRESERVE_PREFIXES = (
    "config.json",
    "outbox.json",
    "photo/",
    "jiekou/",
)
# 客户自己的 App 设置在 runtime/configs/<app>/ 下，已随 runtime/ 不可触碰
# 规则整体保护；apps/<app>/ 下的三个配置文件是开发端随包默认值（docs/11
# L1 下发通道），随更新包正常落地、正常覆盖，客户层按键遮蔽它。
PRESERVED_LAUNCHERS = {"机械臂脚本.exe", "jixiebijiaoben_launcher.exe"}
# 模块 ZIP 可能被残留子进程的 zipimport 句柄占用，替换需要带重试。
ATOMIC_REPLACE_RELATIVES = {"app_modules.zip", "ocr_bridge_modules.zip"}


def _normalize(relative: str) -> str:
    return relative.replace("\\", "/").lstrip("/")


def _is_untouchable(relative: str) -> bool:
    normalized = _normalize(relative)
    return any(normalized.startswith(prefix) for prefix in UNTOUCHABLE_PREFIXES)


def _is_customer_template(relative: str) -> bool:
    parts = _normalize(relative).split("/")
    if len(parts) < 4 or parts[0] != "apps" or parts[2] != "templates":
        return False
    return any("__customer_" in part for part in parts[3:])


def _is_preserved(relative: str) -> bool:
    normalized = _normalize(relative)
    if normalized in PRESERVED_LAUNCHERS:
        return True
    if _is_customer_template(normalized):
        return True
    return any(
        normalized == prefix.rstrip("/") or normalized.startswith(prefix)
        for prefix in PRESERVE_PREFIXES
    )


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
    if pid <= 0:
        return
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
        shutil.rmtree(temp_root, ignore_errors=True)
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
            relative = _normalize(str(item.get("path", "") or "").strip())
            if relative:
                paths.add(relative)
    return paths


def _replace_with_retry(source: Path, target: Path) -> None:
    """模块 ZIP 替换：临时文件 + 原子替换；句柄占用时重试到超时。"""
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    deadline = time.monotonic() + LOCK_RETRY_SECONDS
    try:
        shutil.copy2(source, temporary)
        # 某些旧安装可能把同名路径留下为目录；清理这个精确目标后再替换。
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        while True:
            try:
                os.replace(temporary, target)
                return
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(3.0)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


def _copy_update_file(source: Path, target: Path, relative: str) -> None:
    if _normalize(relative) in ATOMIC_REPLACE_RELATIVES:
        _replace_with_retry(source, target)
        return
    shutil.copy2(source, target)


def _remove_obsolete_files(target_root: Path, old_paths: set[str], new_paths: set[str], *, full_payload: bool) -> None:
    obsolete = sorted(
        path
        for path in old_paths - new_paths
        if (not _is_untouchable(path) or full_payload) and not _is_preserved(path)
    )
    for relative in obsolete:
        target = target_root / relative
        if not target.exists():
            continue
        if target.is_file() or target.is_symlink():
            target.unlink(missing_ok=True)


def _copy_update_tree(source_root: Path, target_root: Path, *, full_payload: bool) -> None:
    for source in sorted(source_root.rglob("*")):
        relative = source.relative_to(source_root).as_posix()
        if _is_preserved(relative):
            continue
        if _is_untouchable(relative) and not full_payload:
            continue
        target = target_root / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_update_file(source, target, relative)


def _write_error_log(target_root: Path, message: str) -> None:
    error_dir = target_root / "runtime" / "errors"
    try:
        error_dir.mkdir(parents=True, exist_ok=True)
        (error_dir / "update_apply_error.txt").write_text(message, encoding="utf-8")
    except OSError:
        pass


def _is_full_payload(payload_root: Path) -> bool:
    """携带引擎负载（engine/ 或旧布局 runtime/app）的归档视为全量包。"""
    return (payload_root / "engine" / "app").is_dir() or (payload_root / "runtime" / "app").is_dir()


def apply_update(*, package_root: Path, archive_path: Path, pid: int) -> None:
    _wait_for_process_exit(pid, WAIT_TIMEOUT_SECONDS)
    extracted_root = _extract_archive(archive_path)
    try:
        full_payload = _is_full_payload(extracted_root)
        existing_paths = _read_manifest_paths(package_root)
        new_paths = _read_manifest_paths(extracted_root)
        _remove_obsolete_files(package_root, existing_paths, new_paths, full_payload=full_payload)
        _copy_update_tree(extracted_root, package_root, full_payload=full_payload)
    finally:
        shutil.rmtree(extracted_root, ignore_errors=True)
    archive_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply a downloaded jixiebijiaoben update package.")
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress message boxes; the caller (PowerShell orchestration) reports results",
    )
    args = parser.parse_args(argv)

    package_root = Path(args.package_root).resolve()
    archive_path = Path(args.archive).resolve()

    try:
        apply_update(package_root=package_root, archive_path=archive_path, pid=args.pid)
    except Exception as exc:
        _write_error_log(package_root, str(exc))
        if not args.quiet:
            _show_message("更新失败", f"应用更新失败：{exc}", error=True)
        return 1

    if not args.quiet:
        _show_message("更新完成", "更新已完成，请重新打开主程序。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
