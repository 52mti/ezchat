"""构建「机械臂脚本」交付包。

设计见 `打包流程.md`：
- 瘦启动器 `机械臂脚本.exe`（launcher_entry.py）与安装器 `安装环境.exe`（install_launcher_entry.py）由 PyInstaller 冻结；
- 随包带嵌入式引导 Python（`_bootstrap/python/`）与离线通用依赖（`wheelhouse/`）；
- OCR 模型与显卡相关后端由客户机首次运行 `安装环境.cmd` 联网安装；
- `config.json` 等密钥文件、内部文档、测试与缓存不打包。

构建机要求：Windows + Python 3.13 + win_amd64。

纯函数（被 tests/test_build_package*.py 约束）只用标准库，PyInstaller / pip download
等重步骤延迟导入或以子进程调用，确保 `import build` 轻量可测。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterator


PACKAGE_NAME = "机械臂脚本"
LAUNCHER_EXE = "机械臂脚本.exe"
INSTALLER_EXE = "安装环境.exe"
LAUNCHER_ALIAS_EXE = "jixiebijiaoben_launcher.exe"
INSTALLER_ALIAS_EXE = "jixiebijiaoben_installer.exe"

# pip download 默认走清华镜像，避免直连 PyPI 超时；可用 --index-url 覆盖（如阿里源）。
DEFAULT_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"

# 源码与运行所需文件（从工作树复制；未提交 git 的运行时文件也会一并带上）。
INCLUDE_PATHS = [
    "apps",
    "vision",
    "services",
    "ui",
    "workers",
    "lgb",
    "main.py",
    "launcher_entry.py",
    "install_runtime.py",
    "install_launcher_entry.py",
    "requirements.txt",
    "pyproject.toml",
    "config.json.example",
    "runtime_manifest.json",  # 可选：存在才复制（缺省走代码内置默认清单）
]

# 运行时随包带上的默认数据文件（相对工作树）。
BUNDLED_DATA_FILES = [
    "runtime/greetings/pingtai.txt",
]

# 交付包内需保留的空目录骨架。
RUNTIME_SKELETON_DIRS = [
    "runtime/logs",
    "runtime/errors",
    "runtime/memory",
    "runtime/greetings",
    "photo",
    "photo/errors",
]

# 复制源码时忽略的名称 / 后缀。
_IGNORE_NAMES = {
    ".git",
    ".gitignore",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "dist",
    "build",
    "config.json",
    "outbox.json",
    "install_state.json",
    "AGENTS.md",
}
_IGNORE_SUFFIXES = (".pyc", ".pyo", ".tmp", ".md")  # 内部 .md 文档一律不打包

# 已是压缩格式的文件，入 zip 时不再二次压缩（STORED），避免对 wheel/图片做无谓 DEFLATE。
_STORED_SUFFIXES = {".whl", ".zip", ".jpg", ".jpeg", ".png", ".gz", ".7z", ".exe"}

# lite wheelhouse 默认只随包体积小的通用 wheel（单个 ≤ 此阈值 MB），
# 大件（opencv / numpy / PyQt6-Qt6 / scipy / pywin32 等）仍由客户机联网安装。
LITE_WHEEL_MAX_MB = 5.0


def _requirements_specs(root: Path) -> list[str]:
    path = root / "requirements.txt"
    if not path.is_file():
        return []
    specs: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        specs.append(stripped)
    return specs


def _runtime_manifest_payload(root: Path) -> dict:
    """读取 runtime_manifest.json；缺省时回退到与代码一致的默认共享包清单。"""
    default = {
        "sharedPackages": [
            {"spec": "paddleocr==3.4.0"},
            {"spec": "paddlex[ocr-core]==3.4.3"},
        ],
        "backends": {},
    }
    path = root / "runtime_manifest.json"
    if not path.is_file():
        return default
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return loaded if isinstance(loaded, dict) else default


def _shared_runtime_specs(root: Path) -> set[str]:
    payload = _runtime_manifest_payload(root)
    shared = payload.get("sharedPackages", [])
    specs: set[str] = set()
    if isinstance(shared, list):
        for item in shared:
            if isinstance(item, dict):
                spec = str(item.get("spec", "") or "").strip()
                if spec:
                    specs.add(spec)
    return specs


def _bundled_dependency_specs(root: Path) -> list[str]:
    """通用离线依赖 = requirements.txt 去掉 manifest 中的共享 OCR 包。"""
    shared = _shared_runtime_specs(root)
    return [spec for spec in _requirements_specs(root) if spec not in shared]


# --------------------------------------------------------------------------- #
# 压缩档
# --------------------------------------------------------------------------- #

def _archive_excludes(relative: str, *, exclude_wheelhouse: bool) -> bool:
    """判断交付包内的相对路径是否应被压缩档排除。"""
    if relative.startswith("_bootstrap/"):
        return True
    if relative == "runtime/install_state.json":
        return True
    if relative.startswith("runtime/.paddlex/"):
        return True
    if exclude_wheelhouse and relative.startswith("wheelhouse/"):
        return True
    parts = relative.split("/")
    if "__pycache__" in parts:
        return True
    if relative.endswith((".pyc", ".pyo")):
        return True
    return False


def _iter_archive_files(package_root: Path, *, exclude_wheelhouse: bool) -> Iterator[tuple[Path, str]]:
    for path in sorted(package_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(package_root).as_posix()
        if _archive_excludes(relative, exclude_wheelhouse=exclude_wheelhouse):
            continue
        yield path, f"{package_root.name}/{relative}"


def _write_archive(package_root: Path, dist_root: Path, suffix: str, *, exclude_wheelhouse: bool) -> Path:
    dist_root.mkdir(parents=True, exist_ok=True)
    archive_path = dist_root / f"{package_root.name}{suffix}.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, arcname in _iter_archive_files(package_root, exclude_wheelhouse=exclude_wheelhouse):
            compress = zipfile.ZIP_STORED if path.suffix.lower() in _STORED_SUFFIXES else zipfile.ZIP_DEFLATED
            archive.write(path, arcname, compress_type=compress)
    return archive_path


def _write_full_package_archive(package_root: Path, dist_root: Path) -> Path:
    """全量交付包：含 wheelhouse 等离线资源，但排除引导 Python、安装状态、模型缓存。"""
    return _write_archive(package_root, dist_root, "_full", exclude_wheelhouse=False)


def _write_customer_update_archive(package_root: Path, dist_root: Path) -> Path:
    """增量更新包：仅源码/资源/exe，额外排除 wheelhouse 与引导 Python。"""
    return _write_archive(package_root, dist_root, "_update", exclude_wheelhouse=True)


# --------------------------------------------------------------------------- #
# 复制与组装
# --------------------------------------------------------------------------- #

def _ignore(_dir_path: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name in _IGNORE_NAMES:
            ignored.add(name)
        elif name.endswith(_IGNORE_SUFFIXES):
            ignored.add(name)
        elif name.endswith(".bak") or name.startswith("config.json.bak"):
            ignored.add(name)
    return ignored


def _copy_item(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, ignore=_ignore, dirs_exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _handle_remove_readonly(function, path, _excinfo) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _copy_sources(root: Path, package_root: Path) -> None:
    for relative in INCLUDE_PATHS:
        source = root / relative
        if source.exists():
            _copy_item(source, package_root / relative)
    # 调试 demo 与编译缓存不进包
    for junk in (package_root / "lgb" / "test.py",):
        junk.unlink(missing_ok=True)
    for cache in package_root.rglob("__pycache__"):
        if cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)
    for compiled in [*package_root.rglob("*.pyc"), *package_root.rglob("*.pyo")]:
        compiled.unlink(missing_ok=True)


def _copy_bundled_data(root: Path, package_root: Path) -> None:
    for relative in BUNDLED_DATA_FILES:
        source = root / relative
        if source.is_file():
            _copy_item(source, package_root / relative)
        else:
            print(f"[warn] 缺少随包数据文件：{relative}")
    for relative in RUNTIME_SKELETON_DIRS:
        (package_root / relative).mkdir(parents=True, exist_ok=True)


def _write_windows_script(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\r\n") as handle:
        handle.write("\n".join(lines) + "\n")


def _copy_executable_aliases(package_root: Path) -> None:
    for source_name, alias_name in (
        (LAUNCHER_EXE, LAUNCHER_ALIAS_EXE),
        (INSTALLER_EXE, INSTALLER_ALIAS_EXE),
    ):
        source = package_root / source_name
        if source.is_file():
            shutil.copy2(source, package_root / alias_name)


def _write_cmd_launchers(package_root: Path) -> None:
    _write_windows_script(
        package_root / "安装环境.cmd",
        [
            "@echo off",
            "setlocal",
            'cd /d "%~dp0"',
            "",
            f'set "INSTALLER_EXE=%~dp0{INSTALLER_ALIAS_EXE}"',
            'set "BOOTSTRAP_PYTHON=%~dp0_bootstrap\\python\\python.exe"',
            'set "EXIT_CODE=0"',
            "",
            "echo [JIXIEBIJIAOBEN] Installing OCR runtime...",
            "echo [JIXIEBIJIAOBEN] This may take several minutes.",
            "echo.",
            "",
            'if exist "%INSTALLER_EXE%" goto run_installer',
            'if exist "%BOOTSTRAP_PYTHON%" goto run_bootstrap',
            "echo Missing installer executable and bootstrap Python.",
            'set "EXIT_CODE=1"',
            "goto finish",
            "",
            ":run_installer",
            '\"%INSTALLER_EXE%\" %*',
            'set "EXIT_CODE=%ERRORLEVEL%"',
            "goto finish",
            "",
            ":run_bootstrap",
            'set "PYTHONNOUSERSITE=1"',
            '\"%BOOTSTRAP_PYTHON%\" -s -u \"%~dp0install_runtime.py\" %*',
            'set "EXIT_CODE=%ERRORLEVEL%"',
            "",
            ":finish",
            'if not "%EXIT_CODE%"=="0" (',
            "  echo.",
            '  echo Install failed. Exit code: %EXIT_CODE%',
            "  pause",
            "  exit /b %EXIT_CODE%",
            ")",
            "",
            "echo.",
            "echo Install completed.",
            "pause",
            "exit /b 0",
        ],
    )
    _write_windows_script(
        package_root / "启动程序.cmd",
        [
            "@echo off",
            "setlocal",
            'cd /d "%~dp0"',
            "",
            f'set "LAUNCHER_EXE=%~dp0{LAUNCHER_ALIAS_EXE}"',
            'set "RUNTIME_PYTHON=%~dp0.runtime_env\\python.exe"',
            'set "RUNTIME_PYTHON_LEGACY=%~dp0.runtime_env\\Scripts\\python.exe"',
            "",
            'if exist "%LAUNCHER_EXE%" goto start_exe',
            'if exist "%RUNTIME_PYTHON%" goto run_runtime',
            'if exist "%RUNTIME_PYTHON_LEGACY%" goto run_runtime_legacy',
            "where py >nul 2>nul",
            "if %ERRORLEVEL%==0 goto run_py",
            "where python >nul 2>nul",
            "if %ERRORLEVEL%==0 goto run_system_python",
            "echo Python was not found.",
            "pause",
            "exit /b 1",
            "",
            ":start_exe",
            'start "" "%LAUNCHER_EXE%"',
            "exit /b 0",
            "",
            ":run_runtime",
            'set "PYTHONNOUSERSITE=1"',
            '\"%RUNTIME_PYTHON%\" -s -u \"%~dp0main.py\"',
            "exit /b %ERRORLEVEL%",
            "",
            ":run_runtime_legacy",
            'set "PYTHONNOUSERSITE=1"',
            '\"%RUNTIME_PYTHON_LEGACY%\" -s -u \"%~dp0main.py\"',
            "exit /b %ERRORLEVEL%",
            "",
            ":run_py",
            'set "PYTHONNOUSERSITE=1"',
            'py -3 -s -u \"%~dp0main.py\"',
            "exit /b %ERRORLEVEL%",
            "",
            ":run_system_python",
            'set "PYTHONNOUSERSITE=1"',
            'python -s -u \"%~dp0main.py\"',
            "exit /b %ERRORLEVEL%",
        ],
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_package_manifest(package_root: Path) -> None:
    files = []
    for path in sorted(package_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(package_root).as_posix()
        if relative.startswith(("_bootstrap/", "wheelhouse/")):
            continue  # 体积大且为外部资源，不逐一登记
        files.append({"path": relative, "sha256": _sha256(path), "size": path.stat().st_size})
    manifest = {
        "schema_version": 2,
        "package_name": PACKAGE_NAME,
        "contains_private_config": False,
        "files": files,
    }
    (package_root / "package_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# 重步骤：引导 Python / wheelhouse / PyInstaller（可选，按需调用）
# --------------------------------------------------------------------------- #

def _prepare_bootstrap_python(root: Path, package_root: Path, embed_source: Path | None) -> None:
    """把嵌入式引导 Python 与 get-pip.py 放进交付包的 `_bootstrap/`。

    embed_source 指向已解压的嵌入式 Python 目录（python-3.13-embed-amd64）。
    需确保其 `pythonXX._pth` 已取消注释 `import site`，否则 venv/pip 不可用。
    缺省时回退到仓库 `_bootstrap/python/`（由 prepare_bootstrap.py 准备）。
    """
    bootstrap = package_root / "_bootstrap"
    if embed_source is None:
        repo_python = root / "_bootstrap" / "python"
        embed_source = repo_python if repo_python.is_dir() else None
    if embed_source is None:
        print("[warn] 未找到引导 Python：先运行 `python prepare_bootstrap.py` 或传 --embed-python（交付前必须补齐 _bootstrap/python/）")
        return
    if not embed_source.is_dir():
        raise FileNotFoundError(f"嵌入式 Python 源目录不存在：{embed_source}")
    target = bootstrap / "python"
    if target.exists():
        shutil.rmtree(target, onexc=_handle_remove_readonly)
    shutil.copytree(embed_source, target)
    _ensure_embeddable_site_enabled(target)
    get_pip = bootstrap / "get-pip.py"
    repo_get_pip = root / "_bootstrap" / "get-pip.py"
    if repo_get_pip.is_file():
        shutil.copy2(repo_get_pip, get_pip)
    else:
        print("[warn] 缺少 _bootstrap/get-pip.py：先运行 `python prepare_bootstrap.py`（嵌入式 Python 引导 pip 必需）")


def _ensure_embeddable_site_enabled(python_dir: Path) -> None:
    for pth in python_dir.glob("python*._pth"):
        lines = pth.read_text(encoding="utf-8").splitlines()
        changed = False
        for index, line in enumerate(lines):
            if line.strip() in {"#import site", "# import site"}:
                lines[index] = "import site"
                changed = True
        if changed:
            pth.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _download_wheels(root: Path, specs: list[str], *, index_url: str = DEFAULT_INDEX_URL) -> Path:
    """把 specs（含传递依赖）下载到持久缓存 `root/wheelhouse`（cp313 / win_amd64）。

    走 `index_url` 镜像（默认清华），并加大超时/重试，避免直连 PyPI 超时。
    缓存不随 `dist/` 删除；pip download 跳过已存在的 wheel，第二次起几乎不再下载。
    """
    cache = root / "wheelhouse"
    cache.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "pip", "download",
        "--dest", str(cache),
        "--only-binary=:all:",
        "--python-version", "3.13",
        "--platform", "win_amd64",
        "--timeout", "60",
        "--retries", "5",
    ]
    if index_url:
        command.extend(["--index-url", index_url])
    command.extend(specs)
    print("[wheelhouse] " + " ".join(command))
    subprocess.check_call(command, cwd=root)
    return cache


def _build_wheelhouse(root: Path, package_root: Path, *, lite: bool, index_url: str = DEFAULT_INDEX_URL) -> None:
    """把通用依赖 wheel 装进交付包的 `wheelhouse/`（与显卡无关的全部依赖）。

    始终下载“全通用闭包”——`requirements.txt` 全部依赖（含 paddleocr/paddlex）及其传递依赖，
    但**不含**显卡相关后端（paddlepaddle-gpu/paddlepaddle/onnxruntime，那些按硬件联网装）。

    - 默认（full）：闭包里的 wheel 全部打入包。交付包约 250~350MB；客户装时只联网下 paddle + 模型。
    - `--lite-wheelhouse`：只把单个 ≤ LITE_WHEEL_MAX_MB 的小 wheel 打入，大件仍联网装（包 ~50MB）。

    安装端 `install_runtime` 用 `--find-links wheelhouse --index-url <镜像>`：本地有就用本地、
    缺的再联网，因此即便是 lite 这种“部分 wheelhouse”也能正常工作。
    """
    specs = ["pip", "setuptools", "wheel", *_bundled_dependency_specs(root), *sorted(_shared_runtime_specs(root))]
    cache = _download_wheels(root, specs, index_url=index_url)

    dest = package_root / "wheelhouse"
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    skipped: list[str] = []
    for wheel in sorted(cache.glob("*.whl")):
        size_mb = wheel.stat().st_size / (1024 * 1024)
        if not lite or size_mb <= LITE_WHEEL_MAX_MB:
            shutil.copy2(wheel, dest / wheel.name)
            copied.append(wheel.name)
        else:
            skipped.append(f"{wheel.name}（{size_mb:.0f}MB）")
    print(f"[wheelhouse] 打入 {len(copied)} 个 wheel" + (f"；联网安装 {len(skipped)} 个大件" if lite else "（全通用依赖）"))
    if skipped:
        print("[wheelhouse] 联网安装：" + "、".join(skipped))


def _exe_cache_fresh(cached_exe: Path, root: Path, deps: list[str]) -> bool:
    """缓存的 exe 是否仍然新鲜：存在且不旧于任一源依赖。"""
    if not cached_exe.is_file():
        return False
    exe_mtime = cached_exe.stat().st_mtime
    for dep in deps:
        dep_path = root / dep
        if not dep_path.is_file() or dep_path.stat().st_mtime > exe_mtime:
            return False
    return True


def _freeze_executables(root: Path, package_root: Path, *, rebuild: bool) -> None:
    """冻结启动器与安装器；源未变（或 rebuild=False）则复用缓存 exe，跳过 PyInstaller。"""
    cache = root / "build" / "exe_cache"
    cache.mkdir(parents=True, exist_ok=True)
    work = root / "build" / "_pyinstaller"
    # deps 列出冻结结果实际依赖的源文件：安装器把 install_runtime 一起打入，故也需跟踪它。
    specs = [
        {"exe": LAUNCHER_EXE, "entry": "launcher_entry.py", "console": "--noconsole",
         "deps": ["launcher_entry.py"]},
        {"exe": INSTALLER_EXE, "entry": "install_launcher_entry.py", "console": "--console",
         "deps": ["install_launcher_entry.py", "install_runtime.py"]},
    ]
    for spec in specs:
        exe_name = spec["exe"]
        cached_exe = cache / exe_name
        if not rebuild or _exe_cache_fresh(cached_exe, root, spec["deps"]):
            if not cached_exe.is_file():
                print(f"[warn] 无缓存 exe 可复用：{exe_name}（首次构建勿用 --skip-freeze）")
                continue
            print(f"[pyinstaller] 复用缓存 exe：{exe_name}")
        else:
            try:
                import PyInstaller  # noqa: F401
            except ImportError as exc:  # pragma: no cover - 仅构建机触发
                raise RuntimeError("未安装 PyInstaller，请先 `pip install -r requirements-dev.txt`。") from exc
            name = exe_name[: -len(".exe")] if exe_name.endswith(".exe") else exe_name
            command = [
                sys.executable, "-m", "PyInstaller",
                "--onefile", spec["console"],
                "--name", name,
                "--distpath", str(cache),
                "--workpath", str(work),
                "--specpath", str(work),
                str(root / spec["entry"]),
            ]
            print("[pyinstaller] " + " ".join(command))
            subprocess.check_call(command, cwd=root)
        shutil.copy2(cached_exe, package_root / exe_name)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build 机械臂脚本 delivery package.")
    parser.add_argument("--embed-python", type=Path, default=None,
                        help="已解压的嵌入式 Python 目录（python-3.13-embed-amd64）")
    parser.add_argument("--lite-wheelhouse", action="store_true",
                        help="只打小件通用依赖（≤LITE_WHEEL_MAX_MB），大件联网装（包 ~50MB；默认打全部）")
    parser.add_argument("--no-wheelhouse", action="store_true",
                        help="完全不打 wheelhouse，通用依赖全部联网装")
    parser.add_argument("--index-url", default=DEFAULT_INDEX_URL,
                        help=f"pip download 镜像源（默认清华 {DEFAULT_INDEX_URL}；阿里源 https://mirrors.aliyun.com/pypi/simple）")
    parser.add_argument("--skip-freeze", action="store_true", help="跳过 PyInstaller，复用 exe 缓存")
    args = parser.parse_args(argv)

    root = Path.cwd()
    dist = root / "dist"
    package_root = dist / PACKAGE_NAME
    if package_root.exists():
        shutil.rmtree(package_root, onexc=_handle_remove_readonly)
    package_root.mkdir(parents=True, exist_ok=True)

    _copy_sources(root, package_root)
    _copy_bundled_data(root, package_root)

    if args.no_wheelhouse:
        print("[wheelhouse] --no-wheelhouse：通用依赖全部联网安装")
    else:
        _build_wheelhouse(root, package_root, lite=args.lite_wheelhouse, index_url=args.index_url)
    _prepare_bootstrap_python(root, package_root, args.embed_python)
    _freeze_executables(root, package_root, rebuild=not args.skip_freeze)
    _copy_executable_aliases(package_root)
    _write_cmd_launchers(package_root)

    _write_package_manifest(package_root)
    full = _write_full_package_archive(package_root, dist)
    update = _write_customer_update_archive(package_root, dist)

    print(f"交付目录：{package_root}")
    print(f"全量包：{full}")
    print(f"更新包：{update}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
