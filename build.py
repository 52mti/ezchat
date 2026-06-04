from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator


PACKAGE_NAME = "jixiebijiaoben"
LAUNCHER_EXE = "机械臂脚本.exe"
LAUNCHER_ALIAS_EXE = "jixiebijiaoben_launcher.exe"
START_CMD = "start_jixiebijiaoben.cmd"
DEFAULT_FULL_PACKAGE_NVIDIA_LANE = "cu118"

INCLUDE_PATHS = [
    "apps",
    "vision",
    "services",
    "ui",
    "workers",
    "lgb",
    "main.py",
    "launcher_entry.py",
    "ocr_subprocess_entry.py",
    "runtime_layout.py",
    "requirements.txt",
    "pyproject.toml",
    "config.json.example",
    "runtime_manifest.json",
]

BUNDLED_DATA_FILES = [
    "runtime/greetings/pingtai.txt",
]

RUNTIME_SKELETON_DIRS = [
    "runtime/app",
    "runtime/ocr_cpu",
    "runtime/ocr_nvidia",
    "runtime/ocr_directml",
    "runtime/logs",
    "runtime/errors",
    "runtime/memory",
    "runtime/reports",
    "runtime/tmp/ocr_requests",
    "runtime/greetings",
    "models/ocr/paddlex_named_models",
    "models/ocr/metadata",
    "photo",
    "photo/errors",
]

_IGNORE_NAMES = {
    ".git",
    ".gitignore",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".env",
    "dist",
    "build",
    "wheelhouse",
    "_bootstrap",
    "outbox.json",
    "install_state.json",
    "AGENTS.md",
}
_IGNORE_SUFFIXES = (".pyc", ".pyo", ".tmp", ".md")
_STORED_SUFFIXES = {".zip", ".jpg", ".jpeg", ".png", ".gz", ".7z", ".exe"}

RUNTIME_CACHE_INPUTS = (
    "runtime_builder.py",
    "runtime_manifest.json",
    "requirements.txt",
    "runtime_layout.py",
    "ocr_subprocess_entry.py",
)
PACKAGE_CACHE_EXTRA_INPUTS = (
    "build.py",
    "runtime_builder.py",
)
DEPRECATED_PACKAGE_PATHS = (
    "apps/wechat",
)


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
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return payload if isinstance(payload, dict) else default


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
    shared = _shared_runtime_specs(root)
    return [spec for spec in _requirements_specs(root) if spec not in shared]


def _manifest_default_model_spec(root: Path, *, backend_key: str, model_tier: str) -> dict[str, str]:
    payload = _runtime_manifest_payload(root)
    models = payload.get("models", {})
    if not isinstance(models, dict):
        return {}
    tiers = models.get("tiers", {})
    if not isinstance(tiers, dict):
        return {}
    tier_payload = tiers.get(model_tier, {})
    if not isinstance(tier_payload, dict):
        return {}
    backends = tier_payload.get("backends", {})
    if not isinstance(backends, dict):
        return {}
    backend_payload = backends.get(backend_key, {})
    if not isinstance(backend_payload, dict):
        return {}
    return {
        "provider": str(backend_payload.get("provider", "") or "").strip(),
        "det": str(backend_payload.get("detModelName", "") or "").strip(),
        "rec": str(backend_payload.get("recModelName", "") or "").strip(),
        "det_version": str(backend_payload.get("detVersion", "") or "").strip(),
        "rec_version": str(backend_payload.get("recVersion", "") or "").strip(),
    }


def _relative_matches_prefix(relative: str, prefix: str) -> bool:
    normalized = prefix.rstrip("/")
    return relative == normalized or relative.startswith(normalized + "/")


def _is_app_customer_config_asset(relative: str) -> bool:
    parts = relative.split("/")
    if len(parts) < 3 or parts[0] != "apps":
        return False
    if parts[2] in {"config.json", "ocr_config.py"}:
        return True
    return len(parts) >= 3 and parts[2] == "templates"


def _archive_excludes(
    relative: str,
    *,
    exclude_runtime_state: bool,
    exclude_full_runtimes: bool,
    exclude_app_customer_config_assets: bool,
) -> bool:
    if exclude_runtime_state and relative == "runtime/install_state.json":
        return True
    if exclude_app_customer_config_assets and _is_app_customer_config_asset(relative):
        return True
    if exclude_full_runtimes and (
        _relative_matches_prefix(relative, "runtime/app")
        or _relative_matches_prefix(relative, "runtime/ocr_cpu")
        or _relative_matches_prefix(relative, "runtime/ocr_nvidia")
        or _relative_matches_prefix(relative, "runtime/ocr_directml")
        or _relative_matches_prefix(relative, "models/ocr")
    ):
        return True
    if _relative_matches_prefix(relative, "runtime/.paddlex"):
        return True
    parts = relative.split("/")
    if "__pycache__" in parts:
        return True
    return relative.endswith((".pyc", ".pyo"))


def _iter_archive_dirs(
    package_root: Path,
    *,
    exclude_runtime_state: bool,
    exclude_full_runtimes: bool,
    exclude_app_customer_config_assets: bool,
) -> Iterator[str]:
    for path in sorted(package_root.rglob("*")):
        if not path.is_dir():
            continue
        relative = path.relative_to(package_root).as_posix()
        if not relative:
            continue
        if _archive_excludes(
            relative,
            exclude_runtime_state=exclude_runtime_state,
            exclude_full_runtimes=exclude_full_runtimes,
            exclude_app_customer_config_assets=exclude_app_customer_config_assets,
        ):
            continue
        yield f"{PACKAGE_NAME}/{relative}/"


def _iter_archive_files(
    package_root: Path,
    *,
    exclude_runtime_state: bool,
    exclude_full_runtimes: bool,
    exclude_app_customer_config_assets: bool,
) -> Iterator[tuple[Path, str]]:
    for path in sorted(package_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(package_root).as_posix()
        if _archive_excludes(
            relative,
            exclude_runtime_state=exclude_runtime_state,
            exclude_full_runtimes=exclude_full_runtimes,
            exclude_app_customer_config_assets=exclude_app_customer_config_assets,
        ):
            continue
        yield path, f"{PACKAGE_NAME}/{relative}"


def _write_archive(
    package_root: Path,
    dist_root: Path,
    suffix: str,
    *,
    exclude_runtime_state: bool,
    exclude_full_runtimes: bool,
    exclude_app_customer_config_assets: bool,
) -> Path:
    dist_root.mkdir(parents=True, exist_ok=True)
    archive_path = dist_root / f"{PACKAGE_NAME}{suffix}.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for arcname in _iter_archive_dirs(
            package_root,
            exclude_runtime_state=exclude_runtime_state,
            exclude_full_runtimes=exclude_full_runtimes,
            exclude_app_customer_config_assets=exclude_app_customer_config_assets,
        ):
            archive.writestr(arcname, "")
        for path, arcname in _iter_archive_files(
            package_root,
            exclude_runtime_state=exclude_runtime_state,
            exclude_full_runtimes=exclude_full_runtimes,
            exclude_app_customer_config_assets=exclude_app_customer_config_assets,
        ):
            compression = zipfile.ZIP_STORED if path.suffix.lower() in _STORED_SUFFIXES else zipfile.ZIP_DEFLATED
            archive.write(path, arcname, compress_type=compression)
    return archive_path


def _write_full_package_archive(package_root: Path, dist_root: Path) -> Path:
    return _write_archive(
        package_root,
        dist_root,
        "_full",
        exclude_runtime_state=False,
        exclude_full_runtimes=False,
        exclude_app_customer_config_assets=False,
    )


def _write_customer_update_archive(package_root: Path, dist_root: Path) -> Path:
    return _write_archive(
        package_root,
        dist_root,
        "_update",
        exclude_runtime_state=True,
        exclude_full_runtimes=True,
        exclude_app_customer_config_assets=True,
    )


def _ignore(_dir_path: str, names: list[str]) -> set[str]:
    ignored = {name for name in names if name in _IGNORE_NAMES}
    ignored.update(name for name in names if any(name.endswith(suffix) for suffix in _IGNORE_SUFFIXES))
    return ignored


def _copy_item(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, ignore=_ignore, dirs_exist_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _handle_remove_readonly(function, path, _excinfo) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _copy_sources(root: Path, package_root: Path) -> None:
    for relative in INCLUDE_PATHS:
        source = root / relative
        if not source.exists():
            continue
        _copy_item(source, package_root / relative)
    for relative in DEPRECATED_PACKAGE_PATHS:
        deprecated_path = package_root / relative
        if deprecated_path.exists():
            _remove_path(deprecated_path)
    for skeleton in RUNTIME_SKELETON_DIRS:
        (package_root / skeleton).mkdir(parents=True, exist_ok=True)
    (package_root / "runtime" / ".paddlex").mkdir(parents=True, exist_ok=True)
    root_config = package_root / "config.json"
    if root_config.exists():
        root_config.unlink()


def _copy_bundled_data(root: Path, package_root: Path) -> None:
    for relative in BUNDLED_DATA_FILES:
        source = root / relative
        if source.is_file():
            _copy_item(source, package_root / relative)


def _run_build_step(command: list[str], *, cwd: Path) -> None:
    print("[build-step] " + " ".join(str(part) for part in command))
    subprocess.check_call(command, cwd=cwd)


def _install_runtime_into_package(
    root: Path,
    package_root: Path,
    *,
    runtime_name: str,
    install_scope: str,
    backend: str | None = None,
    model_tier: str = "lite",
    cuda_lane: str | None = None,
    skip_validation: bool = False,
    bundle_models: bool = False,
    allow_missing_hardware: bool = False,
) -> None:
    command = [
        sys.executable,
        str(root / "runtime_builder.py"),
        "--package-root",
        str(package_root),
        "--runtime-name",
        runtime_name,
        "--install-scope",
        install_scope,
        "--model-tier",
        model_tier,
        "--skip-state-write",
    ]
    if backend is not None:
        command.extend(["--backend", backend])
    if cuda_lane is not None:
        command.extend(["--cuda-lane", cuda_lane])
    if skip_validation:
        command.append("--skip-validation")
    if bundle_models:
        command.append("--bundle-models")
    if allow_missing_hardware:
        command.append("--allow-missing-hardware")
    _run_build_step(command, cwd=root)


def _cleanup_packaged_ocr_cache(package_root: Path) -> None:
    cache_root = package_root / "runtime" / ".paddlex"
    if cache_root.exists():
        shutil.rmtree(cache_root, onexc=_handle_remove_readonly)


def _cleanup_packaged_transients(package_root: Path) -> None:
    _cleanup_packaged_ocr_cache(package_root)
    for cache_dir in package_root.rglob("__pycache__"):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir, onexc=_handle_remove_readonly)


def _cleanup_python_caches(package_root: Path) -> None:
    for path in sorted(package_root.rglob("__pycache__"), reverse=True):
        if path.is_dir():
            shutil.rmtree(path, onexc=_handle_remove_readonly)
    for path in package_root.rglob("*.pyc"):
        path.unlink(missing_ok=True)
    for path in package_root.rglob("*.pyo"):
        path.unlink(missing_ok=True)


def _write_full_package_runtime_state(package_root: Path, *, model_tier: str, nvidia_cuda_lane: str) -> None:
    state_path = package_root / "runtime" / "install_state.json"
    model_spec = _manifest_default_model_spec(package_root, backend_key="cpu", model_tier=model_tier)
    payload = {
        "schema_version": 5,
        "backend": "auto",
        "model_tier": model_tier,
        "execution_mode": "subprocess",
        "package_mode": "full",
        "runtime_name": "app",
        "cuda_lane": nvidia_cuda_lane,
        "cache_home": str((package_root / "runtime" / ".paddlex").resolve()),
        "status": "installed",
        "python_executable": str((package_root / "runtime" / "app" / "python.exe").resolve()),
        "runtime_path": str((package_root / "runtime" / "app").resolve()),
        "message": "Full package runtimes assembled by build.py",
        "validation": "ok",
        "validation_error": "",
        "ocr_device": "",
        "hardware": {},
        "common_packages": _bundled_dependency_specs(package_root),
        "packages": [],
        "installed_backends": ["cpu", "paddle_cuda", "onnx_directml"],
        "runtime_inventory": {
            "app": "runtime/app",
            "cpu": "runtime/ocr_cpu",
            "paddle_cuda": "runtime/ocr_nvidia",
            "onnx_directml": "runtime/ocr_directml",
        },
        "models": model_spec,
        "validation_details": {
            "build_mode": "full_package",
            "bundled_models_root": "models/ocr/paddlex_named_models",
            "nvidia_cuda_lane": nvidia_cuda_lane,
        },
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _runtime_cache_root(root: Path) -> Path:
    configured = str(os.environ.get("JXB_RUNTIME_CACHE_ROOT", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt":
        system_drive = os.environ.get("SystemDrive", "C:")
        return Path(system_drive + os.sep) / "jxb_rt_cache"
    return Path(tempfile.gettempdir()) / "jxb_rt_cache"


def _runtime_cache_key(
    root: Path,
    *,
    runtime_name: str,
    install_scope: str,
    backend: str | None,
    model_tier: str,
    cuda_lane: str | None,
    skip_validation: bool,
    bundle_models: bool,
) -> str:
    payload = {
        "runtime_name": runtime_name,
        "install_scope": install_scope,
        "backend": backend,
        "model_tier": model_tier,
        "cuda_lane": cuda_lane,
        "skip_validation": skip_validation,
        "bundle_models": bundle_models,
        "python": sys.version,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    for relative in RUNTIME_CACHE_INPUTS:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        if not path.is_file():
            digest.update(b"<missing>")
            continue
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _runtime_cache_dir(root: Path, *, runtime_name: str, cache_key: str) -> Path:
    return _runtime_cache_root(root) / runtime_name / cache_key


def _restore_runtime_from_cache(package_root: Path, cache_dir: Path, *, runtime_name: str) -> bool:
    runtime_source = cache_dir / "runtime" / runtime_name
    if not runtime_source.is_dir():
        return False
    runtime_target = package_root / "runtime" / runtime_name
    if runtime_target.exists():
        shutil.rmtree(runtime_target, onexc=_handle_remove_readonly)
    shutil.copytree(runtime_source, runtime_target)
    models_source = cache_dir / "models"
    if models_source.is_dir():
        shutil.copytree(models_source, package_root / "models", dirs_exist_ok=True)
    return True


def _store_runtime_cache(package_root: Path, cache_dir: Path, *, runtime_name: str, include_models: bool) -> None:
    if cache_dir.exists():
        shutil.rmtree(cache_dir, onexc=_handle_remove_readonly)
    (cache_dir / "runtime").mkdir(parents=True, exist_ok=True)
    shutil.copytree(package_root / "runtime" / runtime_name, cache_dir / "runtime" / runtime_name)
    if include_models:
        models_root = package_root / "models"
        if models_root.is_dir():
            shutil.copytree(models_root, cache_dir / "models")


def _ensure_runtime_available(
    root: Path,
    package_root: Path,
    *,
    runtime_name: str,
    install_scope: str,
    backend: str | None = None,
    model_tier: str = "lite",
    cuda_lane: str | None = None,
    skip_validation: bool = False,
    bundle_models: bool = False,
    allow_missing_hardware: bool = False,
    use_cache: bool = True,
) -> None:
    cache_key = _runtime_cache_key(
        root,
        runtime_name=runtime_name,
        install_scope=install_scope,
        backend=backend,
        model_tier=model_tier,
        cuda_lane=cuda_lane,
        skip_validation=skip_validation,
        bundle_models=bundle_models,
    )
    cache_dir = _runtime_cache_dir(root, runtime_name=runtime_name, cache_key=cache_key)
    if use_cache and _restore_runtime_from_cache(package_root, cache_dir, runtime_name=runtime_name):
        print(f"[runtime-cache] reuse {runtime_name} ({cache_key})")
        return
    _install_runtime_into_package(
        root,
        package_root,
        runtime_name=runtime_name,
        install_scope=install_scope,
        backend=backend,
        model_tier=model_tier,
        cuda_lane=cuda_lane,
        skip_validation=skip_validation,
        bundle_models=bundle_models,
        allow_missing_hardware=allow_missing_hardware,
    )
    if use_cache:
        _store_runtime_cache(package_root, cache_dir, runtime_name=runtime_name, include_models=bundle_models)
        print(f"[runtime-cache] store {runtime_name} ({cache_key})")


def _build_full_package_runtimes(
    root: Path,
    package_root: Path,
    *,
    model_tier: str,
    nvidia_cuda_lane: str,
    use_cache: bool,
) -> None:
    _ensure_runtime_available(root, package_root, runtime_name="app", install_scope="app", model_tier=model_tier, use_cache=use_cache)
    _ensure_runtime_available(
        root,
        package_root,
        runtime_name="ocr_cpu",
        install_scope="backend",
        backend="cpu",
        model_tier=model_tier,
        bundle_models=True,
        use_cache=use_cache,
    )
    _ensure_runtime_available(
        root,
        package_root,
        runtime_name="ocr_nvidia",
        install_scope="backend",
        backend="paddle_cuda",
        model_tier=model_tier,
        cuda_lane=nvidia_cuda_lane,
        skip_validation=True,
        allow_missing_hardware=True,
        use_cache=use_cache,
    )
    _ensure_runtime_available(
        root,
        package_root,
        runtime_name="ocr_directml",
        install_scope="backend",
        backend="onnx_directml",
        model_tier=model_tier,
        skip_validation=True,
        allow_missing_hardware=True,
        use_cache=use_cache,
    )


def _staging_root(dist_root: Path) -> Path:
    return dist_root / "_staging"


def _next_staging_session_root(dist_root: Path) -> Path:
    staging_root = _staging_root(dist_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1000):
        candidate = staging_root / f"build_{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not allocate a free staging directory under dist/_staging.")


def _prepare_staging_package_root(dist_root: Path) -> Path:
    package_root = _next_staging_session_root(dist_root) / PACKAGE_NAME
    package_root.mkdir(parents=True, exist_ok=True)
    return package_root


def _package_cache_root(root: Path) -> Path:
    configured = str(os.environ.get("JXB_PACKAGE_CACHE_ROOT", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt":
        return Path(root.anchor) / "jxb_pkg_cache"
    return Path(tempfile.gettempdir()) / "jxb_pkg_cache"


def _iter_hashable_input_files(root: Path, relatives: list[str]) -> Iterator[tuple[str, Path]]:
    for relative in relatives:
        source = root / relative
        if not source.exists():
            continue
        if source.is_file():
            yield relative.replace("\\", "/"), source
            continue
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            if path.name in _ignore(str(path.parent), [path.name]):
                continue
            yield path.relative_to(root).as_posix(), path


def _package_cache_key(
    root: Path,
    *,
    model_tier: str,
    nvidia_cuda_lane: str,
    include_full_payload: bool,
) -> str:
    digest = hashlib.sha256()
    payload = {
        "model_tier": model_tier,
        "nvidia_cuda_lane": nvidia_cuda_lane,
        "include_full_payload": include_full_payload,
        "python": sys.version,
    }
    digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    relatives = [*INCLUDE_PATHS, *BUNDLED_DATA_FILES, *PACKAGE_CACHE_EXTRA_INPUTS]
    for relative, path in _iter_hashable_input_files(root, relatives):
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _package_cache_dir(root: Path, cache_key: str) -> Path:
    return _package_cache_root(root) / cache_key


def _package_cache_complete(
    cache_dir: Path,
    *,
    require_package_dir: bool,
    require_full_zip: bool,
) -> bool:
    if require_package_dir and not (cache_dir / PACKAGE_NAME).is_dir():
        return False
    if not (cache_dir / f"{PACKAGE_NAME}_update.zip").is_file():
        return False
    if require_full_zip and not (cache_dir / f"{PACKAGE_NAME}_full.zip").is_file():
        return False
    return True


def _link_or_copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    try:
        if source.anchor.lower() == target.anchor.lower():
            os.link(source, target)
            return
    except OSError:
        pass
    shutil.copy2(source, target)


def _remove_path(target: Path) -> None:
    target_str = str(target)
    if not os.path.lexists(target_str):
        return
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "rmdir", "/S", "/Q", target_str],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        if completed.returncode == 0 or not os.path.lexists(target_str):
            return
    if target.is_symlink() or target.is_file():
        target.unlink(missing_ok=True)
        return
    shutil.rmtree(target, onexc=_handle_remove_readonly)


def _junction_or_copy_dir(source: Path, target: Path) -> None:
    if os.path.lexists(str(target)):
        _remove_path(target)
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(target), str(source)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        if completed.returncode == 0:
            return
    shutil.copytree(source, target)


def _publish_package_directory(staged_package_root: Path, dist_root: Path) -> Path:
    published_root = dist_root / PACKAGE_NAME
    try:
        _remove_path(published_root)
        shutil.copytree(staged_package_root, published_root)
        return published_root
    except PermissionError:
        locked_target = dist_root / f"{PACKAGE_NAME}_built"
        _remove_path(locked_target)
        shutil.copytree(staged_package_root, locked_target)
        print(f"[warn] publish dir busy, kept new build at: {locked_target}")
        return locked_target


def _cleanup_staging(dist_root: Path) -> None:
    staging_root = _staging_root(dist_root)
    if staging_root.exists():
        _remove_path(staging_root)


def _remove_dist_full_archive(dist_root: Path) -> None:
    full_zip = dist_root / f"{PACKAGE_NAME}_full.zip"
    if full_zip.exists():
        full_zip.unlink(missing_ok=True)


def _publish_cached_package_directory(cached_package_root: Path, dist_root: Path) -> Path:
    target = dist_root / PACKAGE_NAME
    try:
        _remove_path(target)
        shutil.copytree(cached_package_root, target)
        return target
    except PermissionError:
        fallback = dist_root / f"{PACKAGE_NAME}_built"
        _remove_path(fallback)
        shutil.copytree(cached_package_root, fallback)
        print(f"[warn] publish dir busy, kept cached build at: {fallback}")
        return fallback


def _restore_package_from_cache(root: Path, dist_root: Path, cache_dir: Path) -> tuple[Path, Path, Path]:
    dist_root.mkdir(parents=True, exist_ok=True)
    full_zip = dist_root / f"{PACKAGE_NAME}_full.zip"
    update_zip = dist_root / f"{PACKAGE_NAME}_update.zip"
    _link_or_copy_file(cache_dir / f"{PACKAGE_NAME}_full.zip", full_zip)
    _link_or_copy_file(cache_dir / f"{PACKAGE_NAME}_update.zip", update_zip)
    published_root = _publish_cached_package_directory(cache_dir / PACKAGE_NAME, dist_root)
    return published_root, full_zip, update_zip


def _restore_archives_from_cache(
    dist_root: Path,
    cache_dir: Path,
    *,
    include_full: bool,
) -> tuple[Path | None, Path]:
    dist_root.mkdir(parents=True, exist_ok=True)
    full_zip: Path | None = None
    update_zip = dist_root / f"{PACKAGE_NAME}_update.zip"
    target_full = dist_root / f"{PACKAGE_NAME}_full.zip"
    if include_full:
        cached_full = cache_dir / f"{PACKAGE_NAME}_full.zip"
        if cached_full.is_file():
            _link_or_copy_file(cached_full, target_full)
            full_zip = target_full
    elif target_full.exists():
        target_full.unlink(missing_ok=True)
    _link_or_copy_file(cache_dir / f"{PACKAGE_NAME}_update.zip", update_zip)
    return full_zip, update_zip


def _store_package_cache(
    root: Path,
    published_root: Path,
    full_zip: Path | None,
    update_zip: Path,
    *,
    cache_key: str,
) -> Path:
    cache_dir = _package_cache_dir(root, cache_key)
    if cache_dir.exists():
        shutil.rmtree(cache_dir, onexc=_handle_remove_readonly)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(published_root, cache_dir / PACKAGE_NAME)
    if full_zip is not None and full_zip.is_file():
        shutil.copy2(full_zip, cache_dir / f"{PACKAGE_NAME}_full.zip")
    shutil.copy2(update_zip, cache_dir / f"{PACKAGE_NAME}_update.zip")
    return cache_dir


def _write_windows_script(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\r\n") as handle:
        handle.write("\n".join(lines) + "\n")


def _copy_executable_aliases(package_root: Path) -> None:
    source = package_root / LAUNCHER_EXE
    if source.is_file():
        shutil.copy2(source, package_root / LAUNCHER_ALIAS_EXE)


def _write_cmd_launchers(package_root: Path) -> None:
    _write_windows_script(
        package_root / START_CMD,
        [
            "@echo off",
            "setlocal",
            'cd /d "%~dp0"',
            "",
            f'set "LAUNCHER_EXE=%~dp0{LAUNCHER_ALIAS_EXE}"',
            'set "RUNTIME_PYTHON=%~dp0runtime\\app\\python.exe"',
            'set "RUNTIME_PYTHON_ALT=%~dp0runtime\\app\\pythonw.exe"',
            "",
            'if exist "%LAUNCHER_EXE%" goto start_exe',
            'if exist "%RUNTIME_PYTHON%" goto run_runtime',
            'if exist "%RUNTIME_PYTHON_ALT%" goto run_runtime_alt',
            "echo Missing launcher executable and packaged runtime.",
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
            ":run_runtime_alt",
            'set "PYTHONNOUSERSITE=1"',
            '\"%RUNTIME_PYTHON_ALT%\" -s -u \"%~dp0main.py\"',
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
        if relative.startswith("runtime/.paddlex/"):
            continue
        files.append({"path": relative, "sha256": _sha256(path), "size": path.stat().st_size})
    payload = {
        "schema_version": 2,
        "package_name": PACKAGE_NAME,
        "contains_private_config": False,
        "files": files,
    }
    (package_root / "package_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _exe_cache_fresh(cached_exe: Path, root: Path, deps: list[str]) -> bool:
    if not cached_exe.is_file():
        return False
    exe_mtime = cached_exe.stat().st_mtime
    for dep in deps:
        dep_path = root / dep
        if not dep_path.is_file() or dep_path.stat().st_mtime > exe_mtime:
            return False
    return True


def _freeze_executables(root: Path, package_root: Path, *, rebuild: bool) -> None:
    cache = root / "build" / "exe_cache"
    cache.mkdir(parents=True, exist_ok=True)
    work = root / "build" / "_pyinstaller"
    cached_exe = cache / LAUNCHER_EXE
    deps = ["launcher_entry.py", "runtime_layout.py"]
    if rebuild and not _exe_cache_fresh(cached_exe, root, deps):
        try:
            import PyInstaller  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("PyInstaller is required. Run `pip install -r requirements-dev.txt`.") from exc
        command = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--onefile",
            "--noconsole",
            "--name",
            LAUNCHER_EXE[:-4],
            "--distpath",
            str(cache),
            "--workpath",
            str(work),
            "--specpath",
            str(work),
            str(root / "launcher_entry.py"),
        ]
        print("[pyinstaller] " + " ".join(command))
        subprocess.check_call(command, cwd=root)
    if not cached_exe.is_file():
        raise RuntimeError(f"Missing cached launcher executable: {cached_exe}")
    shutil.copy2(cached_exe, package_root / LAUNCHER_EXE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the full jixiebijiaoben package.")
    parser.add_argument("--full", action="store_true", help="also build dist/jixiebijiaoben_full.zip")
    parser.add_argument("--skip-freeze", action="store_true", help="reuse cached launcher exe")
    parser.add_argument("--no-runtime-cache", action="store_true", help="do not reuse cached runtimes")
    parser.add_argument("--publish-dir", action="store_true", help="also publish the unpacked package directory into dist/")
    parser.add_argument("--model-tier", default="lite")
    parser.add_argument("--nvidia-cuda-lane", default=DEFAULT_FULL_PACKAGE_NVIDIA_LANE)
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent
    dist = root / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    include_full_payload = bool(args.full or args.publish_dir)

    package_cache_key = _package_cache_key(
        root,
        model_tier=args.model_tier,
        nvidia_cuda_lane=args.nvidia_cuda_lane,
        include_full_payload=include_full_payload,
    )
    cache_dir = _package_cache_dir(root, package_cache_key)
    if _package_cache_complete(
        cache_dir,
        require_package_dir=include_full_payload,
        require_full_zip=False,
    ):
        print(f"[package-cache] hit {package_cache_key}")
        full_zip, update_zip = _restore_archives_from_cache(dist, cache_dir, include_full=args.full)
        if args.full and full_zip is None:
            full_zip = _write_full_package_archive(cache_dir / PACKAGE_NAME, dist)
        if args.publish_dir:
            published_root = _publish_cached_package_directory(cache_dir / PACKAGE_NAME, dist)
            print(f"[done] package={published_root}")
        if full_zip is not None:
            print(f"[done] full={full_zip}")
        print(f"[done] update={update_zip}")
        return 0

    if not args.full:
        _remove_dist_full_archive(dist)
    package_root = _prepare_staging_package_root(dist)
    _copy_sources(root, package_root)
    _copy_bundled_data(root, package_root)
    if include_full_payload:
        _build_full_package_runtimes(
            root,
            package_root,
            model_tier=args.model_tier,
            nvidia_cuda_lane=args.nvidia_cuda_lane,
            use_cache=not args.no_runtime_cache,
        )
        _cleanup_packaged_transients(package_root)
        _cleanup_python_caches(package_root)
        _write_full_package_runtime_state(package_root, model_tier=args.model_tier, nvidia_cuda_lane=args.nvidia_cuda_lane)
    _freeze_executables(root, package_root, rebuild=not args.skip_freeze)
    _copy_executable_aliases(package_root)
    _write_cmd_launchers(package_root)
    _write_package_manifest(package_root)

    full_zip: Path | None = None
    if args.full:
        full_zip = _write_full_package_archive(package_root, dist)
    update_zip = _write_customer_update_archive(package_root, dist)
    cache_dir = _store_package_cache(root, package_root, full_zip, update_zip, cache_key=package_cache_key)
    published_root: Path | None = None
    if args.publish_dir:
        published_root = _publish_package_directory(package_root, dist)
    _cleanup_staging(dist)

    print(f"[package-cache] store {package_cache_key} -> {cache_dir}")
    if published_root is not None:
        print(f"[done] package={published_root}")
    if full_zip is not None:
        print(f"[done] full={full_zip}")
    print(f"[done] update={update_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
