from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from runtime_layout import (
    bundled_paddlex_models_root,
    runtime_dir as packaged_runtime_dir,
)


DEFAULT_MANIFEST: dict[str, Any] = {
    "pip": {
        "indexUrl": "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
    },
    "sharedPackages": [
        {"spec": "paddleocr==3.4.0"},
        {"spec": "paddlex[ocr-core]==3.4.3"},
    ],
    "backends": {
        "paddle_cuda": {
            "description": "NVIDIA CUDA PaddleOCR runtime for PP-OCRv5 mobile",
            "packages": [{"spec": "paddlepaddle-gpu==3.2.2"}],
            "lanes": {
                "cu118": "https://www.paddlepaddle.org.cn/packages/stable/cu118/",
                "cu126": "https://www.paddlepaddle.org.cn/packages/stable/cu126/",
                "cu129": "https://www.paddlepaddle.org.cn/packages/stable/cu129/",
            },
            "modelProvider": "paddlex_named_models",
        },
        "onnx_directml": {
            "description": "Windows AMD/Intel DirectML OCR runtime",
            "packages": [
                {"spec": "onnxruntime-directml"},
                {"spec": "rapidocr_onnxruntime"},
            ],
            "modelProvider": "rapidocr_builtin",
        },
        "cpu": {
            "description": "CPU PaddleOCR runtime for PP-OCRv5 mobile",
            "packages": [
                {
                    "spec": "paddlepaddle==3.2.2",
                    "indexUrl": "https://www.paddlepaddle.org.cn/packages/stable/cpu/",
                }
            ],
            "modelProvider": "paddlex_named_models",
        },
    },
    "models": {
        "defaultTier": "lite",
        "tiers": {
            "lite": {
                "backends": {
                    "paddle_cuda": {
                        "provider": "paddlex_named_models",
                        "detModelName": "PP-OCRv5_mobile_det",
                        "recModelName": "PP-OCRv5_mobile_rec",
                        "detVersion": "PP-OCRv5_mobile_det",
                        "recVersion": "PP-OCRv5_mobile_rec",
                    },
                    "cpu": {
                        "provider": "paddlex_named_models",
                        "detModelName": "PP-OCRv5_mobile_det",
                        "recModelName": "PP-OCRv5_mobile_rec",
                        "detVersion": "PP-OCRv5_mobile_det",
                        "recVersion": "PP-OCRv5_mobile_rec",
                    },
                    "onnx_directml": {
                        "provider": "rapidocr_builtin",
                        "detVersion": "rapidocr_builtin_det",
                        "recVersion": "rapidocr_builtin_rec",
                    },
                }
            }
        },
    },
}

FULL_PACKAGE_RUNTIME_NAMES = ("app", "ocr_cpu", "ocr_nvidia", "ocr_directml")
_PACKAGE_ROOT_OVERRIDE: Path | None = None
_RUNTIME_NAME_OVERRIDE = "legacy"


@dataclass(slots=True)
class CommandExecutionError(RuntimeError):
    stage: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    log_path: Path

    def __str__(self) -> str:
        return (
            f"{self.stage} failed with exit code {self.returncode}. "
            f"See log: {self.log_path}"
        )


def _package_root() -> Path:
    if _PACKAGE_ROOT_OVERRIDE is not None:
        return _PACKAGE_ROOT_OVERRIDE
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _set_package_root_override(path: str | Path | None) -> None:
    global _PACKAGE_ROOT_OVERRIDE
    if path is None:
        _PACKAGE_ROOT_OVERRIDE = None
        return
    resolved = Path(path).expanduser().resolve()
    _PACKAGE_ROOT_OVERRIDE = resolved if resolved.is_dir() else resolved.parent


def _normalize_runtime_name(runtime_name: str | None) -> str:
    candidate = str(runtime_name or _RUNTIME_NAME_OVERRIDE or "legacy").strip().lower()
    if candidate in {"", "legacy", ".runtime_env"}:
        return "legacy"
    if candidate not in FULL_PACKAGE_RUNTIME_NAMES:
        raise ValueError(f"Unsupported runtime name: {runtime_name}")
    return candidate


def _set_runtime_name_override(runtime_name: str | None) -> None:
    global _RUNTIME_NAME_OVERRIDE
    _RUNTIME_NAME_OVERRIDE = _normalize_runtime_name(runtime_name)


def _runtime_dir() -> Path:
    return _package_root() / "runtime"


def _runtime_state_path() -> Path:
    return _runtime_dir() / "install_state.json"


def _install_logs_dir() -> Path:
    return _runtime_dir() / "install_logs"


def _private_runtime_dir() -> Path:
    runtime_name = _normalize_runtime_name(None)
    if runtime_name == "legacy":
        return _package_root() / ".runtime_env"
    return packaged_runtime_dir(_package_root(), runtime_name)


def _private_python() -> Path:
    if os.name == "nt":
        runtime_dir = _private_runtime_dir()
        root_python = runtime_dir / "python.exe"
        if root_python.is_file():
            return root_python
        legacy_python = runtime_dir / "Scripts" / "python.exe"
        if legacy_python.is_file():
            return legacy_python
        return root_python
    return _private_runtime_dir() / "bin" / "python"


def _private_site_packages() -> Path:
    if os.name == "nt":
        runtime_dir = _private_runtime_dir()
        root_site_packages = runtime_dir / "Lib" / "site-packages"
        if root_site_packages.is_dir():
            return root_site_packages
        bootstrap_site_packages = runtime_dir / "Scripts" / "Lib" / "site-packages"
        if bootstrap_site_packages.is_dir():
            return bootstrap_site_packages
        return root_site_packages
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return _private_runtime_dir() / "lib" / version / "site-packages"


def _bootstrap_dir() -> Path:
    return _package_root() / "_bootstrap"


def _bootstrap_python() -> Path:
    if os.name == "nt":
        return _bootstrap_dir() / "python" / "python.exe"
    return _bootstrap_dir() / "python" / "bin" / "python"


def _get_pip_script() -> Path:
    return _bootstrap_dir() / "get-pip.py"


def _wheelhouse_dir() -> Path:
    return _package_root() / "wheelhouse"


def _venv_base_command() -> list[str]:
    """Return the python command used to create `.runtime_env`.

    Prefer the bundled embeddable bootstrap Python so the customer needs no
    pre-installed Python. Fall back to the running interpreter for source
    (non-frozen) runs, and finally to a system Python launcher.
    """
    bootstrap = _bootstrap_python()
    if bootstrap.is_file():
        return [str(bootstrap)]
    if not getattr(sys, "frozen", False):
        return [sys.executable]
    for name in ("py", "python", "python3"):
        found = shutil.which(name)
        if found:
            return [found, "-3"] if name == "py" else [found]
    raise RuntimeError("Could not find a usable Python to create the runtime environment: missing _bootstrap\\python and no system Python was found.")


def _paddlex_cache_home() -> Path:
    configured = str(os.environ.get("PADDLE_PDX_CACHE_HOME", "") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if _PACKAGE_ROOT_OVERRIDE is not None or _normalize_runtime_name(None) != "legacy":
        return (_package_root() / "runtime" / ".paddlex").resolve()
    if os.name == "nt":
        public_root = Path(os.environ.get("PUBLIC", r"C:\Users\Public")).resolve()
        return public_root / "jixiebijiaoben_runtime" / ".paddlex"
    return (_package_root() / "runtime" / ".paddlex").resolve()


def _configure_paddlex_cache_home(*, force: bool = False) -> Path:
    cache_home = _paddlex_cache_home()
    cache_home.mkdir(parents=True, exist_ok=True)
    if force:
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache_home)
        os.environ["PADDLEX_HOME"] = str(cache_home)
    else:
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_home))
        os.environ.setdefault("PADDLEX_HOME", str(cache_home))
    return cache_home


def _official_models_root() -> Path:
    return _configure_paddlex_cache_home() / "official_models"


def _bundled_models_root() -> Path:
    root = bundled_paddlex_models_root(_package_root())
    root.mkdir(parents=True, exist_ok=True)
    return root


def _required_paddlex_model_files() -> tuple[str, ...]:
    return ("config.json", "inference.json", "inference.pdiparams", "inference.yml")


def _model_dir_is_complete(model_dir: Path) -> bool:
    return model_dir.is_dir() and all((model_dir / file_name).is_file() for file_name in _required_paddlex_model_files())


def _reset_incomplete_paddlex_models(model_names: list[str]) -> None:
    root = _official_models_root()
    for model_name in model_names:
        if not model_name:
            continue
        model_dir = root / model_name
        if model_dir.exists() and not _model_dir_is_complete(model_dir):
            shutil.rmtree(model_dir, ignore_errors=True)


def _bundle_paddlex_models(model_names: list[str]) -> list[str]:
    copied: list[str] = []
    source_root = _official_models_root()
    target_root = _bundled_models_root()
    for model_name in model_names:
        normalized = str(model_name or "").strip()
        if not normalized:
            continue
        source_dir = source_root / normalized
        if not _model_dir_is_complete(source_dir):
            raise RuntimeError(f"OCR model is incomplete and cannot be bundled: {normalized}")
        target_dir = target_root / normalized
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        shutil.copytree(source_dir, target_dir)
        copied.append(normalized)
    return copied


def _load_runtime_manifest() -> dict[str, Any]:
    manifest_path = _package_root() / "runtime_manifest.json"
    payload = json.loads(json.dumps(DEFAULT_MANIFEST, ensure_ascii=False))
    if not manifest_path.is_file():
        return payload
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return payload
    if not isinstance(loaded, dict):
        return payload

    if isinstance(loaded.get("pip"), dict):
        payload["pip"].update(loaded["pip"])
    if isinstance(loaded.get("sharedPackages"), list):
        payload["sharedPackages"] = loaded["sharedPackages"]
    if isinstance(loaded.get("backends"), dict):
        for backend_key, backend_payload in loaded["backends"].items():
            if not isinstance(backend_payload, dict):
                continue
            payload["backends"].setdefault(backend_key, {})
            payload["backends"][backend_key].update(backend_payload)
    if isinstance(loaded.get("models"), dict):
        payload["models"].update({k: v for k, v in loaded["models"].items() if k != "tiers"})
        tiers = loaded["models"].get("tiers")
        if isinstance(tiers, dict):
            payload["models"].setdefault("tiers", {})
            for tier_name, tier_payload in tiers.items():
                if not isinstance(tier_payload, dict):
                    continue
                payload["models"]["tiers"].setdefault(tier_name, {})
                payload["models"]["tiers"][tier_name].update({k: v for k, v in tier_payload.items() if k != "backends"})
                backends = tier_payload.get("backends")
                if isinstance(backends, dict):
                    payload["models"]["tiers"][tier_name].setdefault("backends", {})
                    for backend_name, model_payload in backends.items():
                        if not isinstance(model_payload, dict):
                            continue
                        payload["models"]["tiers"][tier_name]["backends"].setdefault(backend_name, {})
                        payload["models"]["tiers"][tier_name]["backends"][backend_name].update(model_payload)
    return payload


def _requirements_specs() -> list[str]:
    path = _package_root() / "requirements.txt"
    if not path.is_file():
        return []
    specs: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        specs.append(stripped)
    return specs


def _shared_runtime_specs(manifest: dict[str, Any]) -> set[str]:
    shared_packages = manifest.get("sharedPackages", [])
    specs: set[str] = set()
    if not isinstance(shared_packages, list):
        return specs
    for item in shared_packages:
        if not isinstance(item, dict):
            continue
        spec = str(item.get("spec", "") or "").strip()
        if spec:
            specs.add(spec)
    return specs


def _common_dependency_specs(manifest: dict[str, Any]) -> list[str]:
    shared = _shared_runtime_specs(manifest)
    return [spec for spec in _requirements_specs() if spec not in shared]


def _manifest_pip_index_url(manifest: dict[str, Any]) -> str:
    pip_block = manifest.get("pip")
    if isinstance(pip_block, dict):
        return str(pip_block.get("indexUrl", "") or "").strip()
    return ""


def _manifest_backend(manifest: dict[str, Any], backend_key: str) -> dict[str, Any]:
    backends = manifest.get("backends", {})
    if not isinstance(backends, dict):
        return {}
    backend = backends.get(backend_key, {})
    return backend if isinstance(backend, dict) else {}


def _manifest_backend_description(manifest: dict[str, Any], backend_key: str) -> str:
    return str(_manifest_backend(manifest, backend_key).get("description", "") or "").strip()


def _manifest_backend_package_plan(
    manifest: dict[str, Any],
    backend_key: str,
    *,
    cuda_lane: str | None = None,
) -> list[dict[str, str]]:
    plan: list[dict[str, str]] = []
    backend = _manifest_backend(manifest, backend_key)
    default_index_url = _manifest_pip_index_url(manifest)
    packages = backend.get("packages", [])
    if isinstance(packages, list):
        for item in packages:
            if not isinstance(item, dict):
                continue
            spec = str(item.get("spec", "") or "").strip()
            if not spec:
                continue
            index_url = str(item.get("indexUrl", "") or "").strip() or default_index_url
            if backend_key == "paddle_cuda" and spec.startswith("paddlepaddle-gpu==") and cuda_lane:
                lanes = backend.get("lanes", {})
                if isinstance(lanes, dict):
                    lane_index = str(lanes.get(cuda_lane, "") or "").strip()
                    if lane_index:
                        index_url = lane_index
            plan.append({"spec": spec, "index_url": index_url})

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in plan:
        marker = (item["spec"], item["index_url"])
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(item)
    return deduped


def _manifest_model_spec(manifest: dict[str, Any], backend_key: str, model_tier: str) -> dict[str, str]:
    models = manifest.get("models", {})
    if not isinstance(models, dict):
        return {}
    default_tier = str(models.get("defaultTier", "lite") or "lite").strip().lower()
    tier_key = default_tier if model_tier == "auto" else str(model_tier or default_tier).strip().lower()
    tiers = models.get("tiers", {})
    if not isinstance(tiers, dict):
        return {}
    tier_payload = tiers.get(tier_key, {})
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
        "model_tier": tier_key,
    }


def _resolve_executable(command_name: str) -> str | None:
    executable = shutil.which(command_name)
    if executable:
        return executable
    if not sys.platform.startswith("win"):
        return None
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    candidates: list[Path] = []
    if command_name.lower() in {"nvidia-smi", "nvidia-smi.exe"}:
        candidates.extend(
            [
                system_root / "System32" / "nvidia-smi.exe",
                program_files / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
            ]
        )
    if command_name.lower() in {"nvcc", "nvcc.exe"}:
        cuda_root = program_files / "NVIDIA GPU Computing Toolkit" / "CUDA"
        if cuda_root.is_dir():
            candidates.extend(path / "bin" / "nvcc.exe" for path in sorted(cuda_root.iterdir(), reverse=True))
        for key, value in os.environ.items():
            if key.startswith("CUDA_PATH") and value:
                candidates.append(Path(value) / "bin" / "nvcc.exe")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _run_command(command: list[str], timeout_seconds: int = 8) -> subprocess.CompletedProcess[str] | None:
    executable = _resolve_executable(command[0])
    if executable is None:
        return None
    startupinfo = None
    if sys.platform.startswith("win"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    try:
        return subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
            startupinfo=startupinfo,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _write_command_log(
    *,
    stage: str,
    command: list[str],
    returncode: int,
    stdout: str,
    stderr: str,
) -> Path:
    logs_dir = _install_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    safe_stage = re.sub(r"[^a-zA-Z0-9_.-]+", "_", stage).strip("._") or "command"
    log_path = logs_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_stage}.log"
    payload = {
        "stage": stage,
        "command": command,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
    }
    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return log_path


def _run_checked_command(
    command: list[str],
    *,
    stage: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )
    if completed.returncode == 0:
        return completed
    log_path = _write_command_log(
        stage=stage,
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    raise CommandExecutionError(
        stage=stage,
        command=list(command),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        log_path=log_path,
    )


def _normalize_major_minor(raw_version: str | None) -> tuple[int, int] | None:
    if not raw_version:
        return None
    match = re.match(r"^\s*(\d+)(?:\.(\d+))?", raw_version)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def _parse_compute_capability(raw_value: str | None) -> tuple[int, int] | None:
    if not raw_value:
        return None
    match = re.match(r"^\s*(\d+)(?:\.(\d+))?", raw_value)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def _format_compute_capability(value: tuple[int, int] | None) -> str:
    if value is None:
        return ""
    return f"{value[0]}.{value[1]}"


def _choose_cuda_lane(
    nvcc_version: str | None,
    driver_cuda_version: str | None,
    *,
    compute_capability: str | None = None,
    gpu_name: str | None = None,
) -> tuple[str | None, str]:
    capability = _parse_compute_capability(compute_capability)
    if sys.platform.startswith("win") and capability is not None:
        if capability < (6, 1):
            capability_text = _format_compute_capability(capability)
            gpu_label = gpu_name or "NVIDIA GPU"
            return (
                None,
                f"Detected {gpu_label} with compute capability {capability_text}. "
                "Full package NVIDIA support starts at compute capability 6.1, "
                "so this machine should use DirectML or CPU OCR instead.",
            )
        if capability < (8, 0):
            capability_text = _format_compute_capability(capability)
            return (
                "cu118",
                f"Detected compute capability {capability_text}; prefer Paddle cu118 for pre-Ampere NVIDIA GPUs.",
            )
    preferred = _normalize_major_minor(nvcc_version) or _normalize_major_minor(driver_cuda_version)
    if preferred is None:
        return None, "No local CUDA Toolkit version was detected, and the GPU driver did not report a usable max CUDA version."
    major, minor = preferred
    source = "local CUDA Toolkit (nvcc)" if nvcc_version else "GPU driver reported max CUDA (nvidia-smi)"
    if major > 12 or (major == 12 and minor >= 9):
        return "cu129", f"Using {source} version {major}.{minor}; selecting Paddle cu129."
    if major == 12 and minor >= 6:
        return "cu126", f"Using {source} version {major}.{minor}; selecting Paddle cu126."
    if (major == 11 and minor >= 8) or major > 11:
        return "cu118", f"Using {source} version {major}.{minor}; selecting Paddle cu118."
    return None, f"{source} version {major}.{minor} is below the minimum supported CUDA 11.8."


def _detect_gpu_environment() -> dict[str, Any]:
    query = _run_command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    gpu_names: list[str] = []
    driver_version = ""
    if query is not None and query.returncode == 0:
        for line in query.stdout.splitlines():
            parts = [part.strip() for part in line.split(",", maxsplit=1)]
            if parts and parts[0]:
                gpu_names.append(parts[0])
            if len(parts) > 1 and parts[1]:
                driver_version = parts[1]

    summary = _run_command(["nvidia-smi"])
    driver_cuda_version = ""
    if summary is not None and summary.returncode == 0:
        match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", summary.stdout)
        if match:
            driver_cuda_version = match.group(1)

    nvcc = _run_command(["nvcc", "--version"])
    nvcc_version = ""
    if nvcc is not None and nvcc.returncode == 0:
        match = re.search(r"release\s+([0-9]+(?:\.[0-9]+)?)", nvcc.stdout)
        if match:
            nvcc_version = match.group(1)
    compute_cap_query = _run_command(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    compute_capability = ""
    if compute_cap_query is not None and compute_cap_query.returncode == 0:
        for line in compute_cap_query.stdout.splitlines():
            value = line.strip()
            if value:
                compute_capability = value
                break

    return {
        "gpu_names": gpu_names,
        "driver_version": driver_version,
        "driver_cuda_version": driver_cuda_version,
        "nvcc_version": nvcc_version,
        "compute_capability": compute_capability,
    }


def _probe_windows_video_controllers() -> list[dict[str, str]]:
    if not sys.platform.startswith("win"):
        return []
    completed = _run_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterCompatibility | ConvertTo-Json -Compress",
        ]
    )
    if completed is None or completed.returncode != 0 or not completed.stdout.strip():
        return []
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []
    items = payload if isinstance(payload, list) else [payload]
    controllers: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, dict):
            controllers.append(
                {
                    "name": str(item.get("Name", "") or "").strip(),
                    "vendor": str(item.get("AdapterCompatibility", "") or "").strip(),
                }
            )
    return controllers


def _detect_backend() -> tuple[str, dict[str, Any]]:
    gpu_report = _detect_gpu_environment()
    if gpu_report["gpu_names"]:
        return "paddle_cuda", {
            "gpu_vendor": "nvidia",
            "gpu_name": gpu_report["gpu_names"][0],
            "nvidia_detected": True,
            "directml_available": False,
            "gpu_env": gpu_report,
        }

    controllers = _probe_windows_video_controllers()
    for controller in controllers:
        text = f"{controller.get('vendor', '')} {controller.get('name', '')}".lower()
        if "amd" in text or "radeon" in text or "advanced micro devices" in text:
            return "onnx_directml", {
                "gpu_vendor": "amd",
                "gpu_name": controller.get("name", ""),
                "nvidia_detected": False,
                "directml_available": True,
                "gpu_env": gpu_report,
            }
        if "intel" in text:
            return "onnx_directml", {
                "gpu_vendor": "intel",
                "gpu_name": controller.get("name", ""),
                "nvidia_detected": False,
                "directml_available": True,
                "gpu_env": gpu_report,
            }

    return "cpu", {
        "gpu_vendor": "",
        "gpu_name": "",
        "nvidia_detected": False,
        "directml_available": False,
        "gpu_env": gpu_report,
    }


def _selected_backend(explicit_backend: str, *, allow_missing_hardware: bool = False) -> tuple[str, dict[str, Any]]:
    if explicit_backend != "auto":
        backend, hardware = _detect_backend()
        if explicit_backend == "paddle_cuda" and not hardware.get("nvidia_detected") and not allow_missing_hardware:
            raise RuntimeError("OCR backend paddle_cuda requires an NVIDIA GPU.")
        if explicit_backend == "onnx_directml" and not hardware.get("directml_available") and not allow_missing_hardware:
            raise RuntimeError("OCR backend onnx_directml requires a Windows AMD/Intel GPU with DirectML support.")
        return explicit_backend, hardware
    return _detect_backend()


def _bootstrap_pip_into_venv(python_exe: Path, *, dry_run: bool) -> None:
    """Install pip into a freshly created `--without-pip` venv.

    The embeddable bootstrap Python ships no bundled ensurepip wheels, so we
    use the packaged `get-pip.py` (which carries pip internally and installs it
    offline). When `get-pip.py` is absent (source/dev runs on a normal Python),
    fall back to `ensurepip`.
    """
    env = _pip_env(python_exe)
    get_pip = _get_pip_script()
    if get_pip.is_file():
        command = [str(python_exe), str(get_pip), "--no-warn-script-location"]
    else:
        command = [str(python_exe), "-m", "ensurepip", "--upgrade"]
    if dry_run:
        print(" ".join(command))
        return
    subprocess.check_call(command, cwd=_package_root(), env=env)


def _source_python_root() -> Path:
    return Path(sys.base_prefix).resolve()


def _copy_python_tree(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    copy_dirs = ("DLLs", "Lib", "libs", "tcl")
    copy_files = ("python.exe", "pythonw.exe", "python3.dll", f"python{sys.version_info.major}{sys.version_info.minor}.dll", "vcruntime140.dll", "vcruntime140_1.dll")
    for name in copy_dirs:
        src = source / name
        if not src.exists():
            continue
        if name == "Lib":
            shutil.copytree(src, target / name, ignore=shutil.ignore_patterns("site-packages", "__pycache__", "*.pyc", "*.pyo"))
        else:
            shutil.copytree(src, target / name, dirs_exist_ok=True)
    for name in copy_files:
        src = source / name
        if src.is_file():
            shutil.copy2(src, target / name)
    (target / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
    (target / "Scripts").mkdir(parents=True, exist_ok=True)


def _create_portable_runtime(*, dry_run: bool) -> Path:
    runtime_dir = _private_runtime_dir()
    python_exe = runtime_dir / "python.exe"
    if python_exe.is_file():
        return python_exe
    source_root = _source_python_root()
    if dry_run:
        print(f"copy portable python {source_root} -> {runtime_dir}")
        return python_exe
    _copy_python_tree(source_root, runtime_dir)
    if not python_exe.is_file():
        raise RuntimeError(f"Portable runtime python.exe not found after copy: {runtime_dir}")
    _bootstrap_pip_into_venv(python_exe, dry_run=False)
    return python_exe


def _bundled_bootstrap_python_dir() -> Path | None:
    if os.name != "nt":
        return None
    bootstrap_root = _bootstrap_dir() / "python"
    return bootstrap_root if bootstrap_root.is_dir() else None


def _copy_bootstrap_python_runtime() -> Path | None:
    bootstrap_root = _bundled_bootstrap_python_dir()
    if bootstrap_root is None:
        return None
    runtime_dir = _private_runtime_dir()
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir, ignore_errors=True)
    shutil.copytree(bootstrap_root, runtime_dir)
    _ensure_runtime_package_root(runtime_dir)
    (runtime_dir / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
    return runtime_dir / "python.exe"


def _ensure_runtime_package_root(runtime_dir: Path) -> None:
    if os.name != "nt":
        return
    for pth in runtime_dir.glob("python*._pth"):
        try:
            lines = pth.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        stripped = [line.strip() for line in lines]
        if ".." in stripped:
            continue
        insert_at = next((index for index, line in enumerate(stripped) if line == "import site"), len(lines))
        lines.insert(insert_at, "..")
        pth.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _create_runtime_venv(*, dry_run: bool) -> Path:
    runtime_name = _normalize_runtime_name(None)
    python_exe = _private_python()
    if python_exe.is_file():
        return python_exe
    if runtime_name != "legacy":
        return _create_portable_runtime(dry_run=dry_run)
    if _bundled_bootstrap_python_dir() is not None:
        if dry_run:
            print(f"copy bootstrap python -> {_private_runtime_dir()}")
            return python_exe
        copied_bootstrap = _copy_bootstrap_python_runtime()
        if copied_bootstrap is None:
            raise RuntimeError("Bundled bootstrap Python disappeared while preparing the runtime environment.")
        _bootstrap_pip_into_venv(copied_bootstrap, dry_run=dry_run)
        return copied_bootstrap
    command = [*_venv_base_command(), "-m", "venv", "--without-pip", str(_private_runtime_dir())]
    if dry_run:
        print(" ".join(command))
        return python_exe
    _run_checked_command(command, stage="create_runtime_venv", cwd=_package_root())
    _bootstrap_pip_into_venv(python_exe, dry_run=dry_run)
    return python_exe


def _pip_env(python_exe: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_USER"] = "0"
    env.pop("PYTHONPATH", None)
    if python_exe is not None:
        scripts_dir = python_exe.parent if python_exe.parent.name.lower() == "scripts" else python_exe.parent / "Scripts"
        existing_path = env.get("PATH", "")
        env["PATH"] = str(scripts_dir) + (os.pathsep + existing_path if existing_path else "")
    return env


def _install_package_specs(
    python_exe: Path,
    package_specs: list[dict[str, str]],
    *,
    dry_run: bool,
) -> None:
    env = _pip_env(python_exe)
    grouped: OrderedDict[str, list[str]] = OrderedDict()
    for item in package_specs:
        index_url = str(item.get("index_url", "") or "").strip()
        grouped.setdefault(index_url, []).append(str(item["spec"]))
    for index_url, specs in grouped.items():
        if not specs:
            continue
        command = [str(python_exe), "-m", "pip", "install", "--no-cache-dir"]
        if index_url:
            command.extend(["--index-url", index_url])
        command.extend(specs)
        if dry_run:
            print(" ".join(command))
            continue
        stage = "install_runtime_packages"
        if specs:
            stage = f"{stage}_{specs[0].split('==', maxsplit=1)[0].replace('[', '_').replace(']', '_')}"
        _run_checked_command(command, stage=stage, cwd=_package_root(), env=env)


def _uninstall_conflicting_paddle_backend(python_exe: Path, backend_key: str, *, dry_run: bool) -> None:
    if backend_key == "paddle_cuda":
        specs = ["paddlepaddle"]
    elif backend_key == "cpu":
        specs = ["paddlepaddle-gpu"]
    else:
        return
    command = [str(python_exe), "-m", "pip", "uninstall", "-y", *specs]
    if dry_run:
        print(" ".join(command))
        return
    completed = subprocess.run(
        command,
        cwd=_package_root(),
        env=_pip_env(python_exe),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )
    if completed.returncode not in {0, 1}:
        log_path = _write_command_log(
            stage=f"uninstall_conflicting_{specs[0]}",
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        raise CommandExecutionError(
            f"uninstall_conflicting_{specs[0]}",
            command,
            completed.returncode,
            completed.stdout,
            completed.stderr,
            log_path,
        )


def _pip_install_specs(
    python_exe: Path,
    specs: list[str],
    default_index_url: str,
    *,
    dry_run: bool,
) -> None:
    """Install `specs`, preferring any wheels bundled in `wheelhouse/`.

    濞存嚎鍊撶划顖炲礌閸涘甯涢悹浣靛€曡ぐ褔姊捐箛鎾舵暔濞达絾鎸捐ⅶ閻忓繐绻掑▓鎴︽焻濮樿鲸鏆忓〒姘箚缁傚棝鏁嶉崸妾宼e wheelhouse闁挎稑顦埀顒€鍊界换鏍煂鐏炶姤鍊遍柡鍐ㄥ缁即宕?`--find-links
    wheelhouse` 濞?`--index-url <闂傗偓濠婂啫鍓?`闁挎稒顒猧p 濞村吋锚閸樻盯鎮介妸锔芥嫳闁革附婢橀崙锟犲嫉婢跺本鐣?wheel闁挎稑鐬煎閬嶆儍閸曨偄鏅欏ù鐘查叄閺嗗懘宕撹箛鎾磋含缂?    濞戞挸顑堝ù鍥Υ閸婂儶eelhouse 濞戞挸绉撮悺銊╁捶閵婏附顦ч悘蹇氶哺濡插摜鐥姘含缂佹儳鐏濋悾銊ф啑閸涱偀鍋?    """
    if not specs:
        return
    env = _pip_env(python_exe)
    command = [str(python_exe), "-m", "pip", "install", "--no-cache-dir"]
    command.append("--no-warn-script-location")
    if default_index_url:
        command.extend(["--index-url", default_index_url])
    wheelhouse = _wheelhouse_dir()
    if wheelhouse.is_dir():
        command.extend(["--find-links", str(wheelhouse)])
    command.extend(specs)
    if dry_run:
        print(" ".join(command))
        return
    stage = "install_common_dependencies"
    if specs:
        stage = f"{stage}_{specs[0].split('==', maxsplit=1)[0].replace('[', '_').replace(']', '_')}"
    _run_checked_command(command, stage=stage, cwd=_package_root(), env=env)


def _install_common_dependencies(
    python_exe: Path,
    manifest: dict[str, Any],
    *,
    dry_run: bool,
) -> None:
    specs = _common_dependency_specs(manifest)
    _pip_install_specs(
        python_exe,
        specs,
        _manifest_pip_index_url(manifest),
        dry_run=dry_run,
    )


def _install_shared_runtime_dependencies(
    python_exe: Path,
    manifest: dict[str, Any],
    *,
    dry_run: bool,
) -> None:
    shared_packages = manifest.get("sharedPackages", [])
    specs: list[str] = []
    if isinstance(shared_packages, list):
        for item in shared_packages:
            if not isinstance(item, dict):
                continue
            spec = str(item.get("spec", "") or "").strip()
            if spec:
                specs.append(spec)
    _pip_install_specs(
        python_exe,
        specs,
        _manifest_pip_index_url(manifest),
        dry_run=dry_run,
    )


def _ensure_pip_tooling(python_exe: Path, *, dry_run: bool) -> None:
    _pip_install_specs(
        python_exe,
        ["pip", "setuptools", "wheel"],
        _manifest_pip_index_url(_load_runtime_manifest()),
        dry_run=dry_run,
    )


def _prune_installed_docs(site_packages: Path) -> None:
    if not site_packages.is_dir():
        return

    prune_dir_names = {
        "benchmark",
        "benchmarks",
        "demo",
        "demos",
        "doc",
        "docs",
        "example",
        "examples",
        "sample",
        "samples",
        "test",
        "testing",
        "tests",
    }

    for path in site_packages.rglob("*"):
        if path.is_file():
            lower = path.name.lower()
            if lower.startswith("readme") or path.suffix.lower() in {".md", ".rst"}:
                path.unlink(missing_ok=True)

    for current_root, dir_names, _file_names in os.walk(site_packages, topdown=True):
        current_path = Path(current_root)
        kept: list[str] = []
        for dir_name in dir_names:
            if dir_name.lower() in prune_dir_names:
                target = current_path / dir_name
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                continue
            kept.append(dir_name)
        dir_names[:] = kept


def _runtime_env(cache_home: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PADDLE_PDX_CACHE_HOME"] = str(cache_home)
    env["PADDLEX_HOME"] = str(cache_home)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_USER"] = "0"
    env.pop("PYTHONPATH", None)
    return env


def _validation_script(package_root: Path, cache_home: Path, backend_key: str, model_tier: str) -> str:
    return (
        "import json, os, sys\n"
        f"os.environ['PADDLE_PDX_CACHE_HOME'] = {str(cache_home)!r}\n"
        f"os.environ['PADDLEX_HOME'] = {str(cache_home)!r}\n"
        f"sys.path.insert(0, {str(package_root)!r})\n"
        "from PIL import Image, ImageDraw, ImageFont\n"
        "from vision.ocr import ocr\n"
        "from vision.ocr_runtime import create_local_runtime_engine, load_runtime_manifest, resolve_model_spec\n"
        f"manifest = load_runtime_manifest({str(package_root)!r})\n"
        f"backend = {backend_key!r}\n"
        f"model_tier = {model_tier!r}\n"
        "model_spec = resolve_model_spec(manifest, backend, model_tier)\n"
        "engine = create_local_runtime_engine({'backend': backend, 'model_tier': model_tier, '_execution_mode': 'inprocess'})\n"
        "device = str(engine.preload() or '')\n"
        "base = Image.new('RGB', (120, 40), 'white')\n"
        "draw = ImageDraw.Draw(base)\n"
        "draw.text((8, 10), '123456', fill='black', font=ImageFont.load_default())\n"
        "sample = base.resize((480, 160), Image.Resampling.NEAREST)\n"
        "items = ocr(sample, {'engine': engine, 'use_angle_cls': False, 'rotate_to_portrait': False})\n"
        "texts = [str(item.get('text', '') or '').strip() for item in items if str(item.get('text', '') or '').strip()]\n"
        "smoke_test = {'item_count': len(items), 'texts': texts[:8]}\n"
        "if not texts:\n"
        "    raise RuntimeError('OCR smoke test returned no text')\n"
        "payload = {\n"
        "  'ocr_device': device,\n"
        "  'smoke_test': smoke_test,\n"
        "  'models': {\n"
        "    'provider': model_spec.provider,\n"
        "    'det': model_spec.det_model_name,\n"
        "    'rec': model_spec.rec_model_name,\n"
        "    'det_version': model_spec.det_version,\n"
        "    'rec_version': model_spec.rec_version,\n"
        "  }\n"
        "}\n"
        "print(json.dumps(payload, ensure_ascii=False))\n"
    )


def _validate_runtime(
    python_exe: Path,
    cache_home: Path,
    backend_key: str,
    model_tier: str,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    if dry_run:
        return {
            "ocr_device": {"paddle_cuda": "gpu:0", "onnx_directml": "dml:0", "cpu": "cpu"}[backend_key],
            "models": _manifest_model_spec(_load_runtime_manifest(), backend_key, model_tier),
        }
    completed = _run_checked_command(
        [str(python_exe), "-c", _validation_script(_package_root(), cache_home, backend_key, model_tier)],
        stage=f"validate_{backend_key}_runtime",
        cwd=_package_root(),
        env=_runtime_env(cache_home),
    )
    payload_line = ""
    for line in reversed([item.strip() for item in completed.stdout.splitlines() if item.strip()]):
        if line.startswith("{") and line.endswith("}"):
            payload_line = line
            break
    if not payload_line:
        raise RuntimeError("Runtime validation finished, but OCR payload was not returned.")
    return json.loads(payload_line)


def _write_state(payload: dict[str, Any]) -> None:
    state_path = _runtime_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _reset_private_runtime_dir() -> None:
    runtime_dir = _private_runtime_dir()
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir, ignore_errors=True)


def _install_app_runtime(
    *,
    manifest: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    python_executable = _create_runtime_venv(dry_run=dry_run)
    _ensure_pip_tooling(python_exe=python_executable, dry_run=dry_run)
    _install_common_dependencies(python_exe=python_executable, manifest=manifest, dry_run=dry_run)
    if not dry_run:
        _prune_installed_docs(_private_site_packages())
    return {
        "python_executable": python_executable,
        "packages": _common_dependency_specs(manifest),
        "runtime_path": _private_runtime_dir(),
    }


def _state_payload(
    *,
    backend_key: str,
    model_tier: str,
    cuda_lane: str | None,
    cache_home: Path,
    python_executable: Path,
    manifest: dict[str, Any],
    runtime_plan: list[dict[str, str]],
    hardware: dict[str, Any],
    model_spec: dict[str, str],
    status: str,
    validation: str,
    validation_error: str,
    ocr_device: str,
    validation_details: dict[str, Any] | None = None,
    fallback_from: str = "",
    fallback_reason: str = "",
    failure_log_path: str = "",
) -> dict[str, Any]:
    runtime_name = _normalize_runtime_name(None)
    payload = {
        "schema_version": 5,
        "backend": backend_key,
        "model_tier": model_tier,
        "execution_mode": "subprocess",
        "package_mode": "full" if runtime_name != "legacy" else "legacy",
        "runtime_name": runtime_name,
        "cuda_lane": cuda_lane,
        "cache_home": str(cache_home),
        "status": status,
        "python_executable": str(python_executable),
        "runtime_path": str(_private_runtime_dir()),
        "common_packages": _common_dependency_specs(manifest),
        "packages": runtime_plan,
        "message": _manifest_backend_description(manifest, backend_key),
        "validation": validation,
        "validation_error": validation_error,
        "ocr_device": ocr_device,
        "hardware": hardware,
        "models": {
            "provider": model_spec.get("provider", ""),
            "det": model_spec.get("det", ""),
            "rec": model_spec.get("rec", ""),
            "det_version": model_spec.get("det_version", ""),
            "rec_version": model_spec.get("rec_version", ""),
        },
    }
    if validation_details:
        payload["validation_details"] = validation_details
    if fallback_from:
        payload["fallback_from"] = fallback_from
    if fallback_reason:
        payload["fallback_reason"] = fallback_reason
    if failure_log_path:
        payload["failure_log_path"] = failure_log_path
    return payload


def _install_backend_runtime(
    *,
    manifest: dict[str, Any],
    backend_key: str,
    model_tier: str,
    cuda_lane: str | None,
    cache_home: Path,
    hardware: dict[str, Any],
    dry_run: bool,
    skip_validation: bool = False,
    bundle_models: bool = False,
) -> dict[str, Any]:
    runtime_plan = _manifest_backend_package_plan(manifest, backend_key, cuda_lane=cuda_lane)
    model_spec = _manifest_model_spec(manifest, backend_key, model_tier)
    python_executable = _private_python()

    python_executable = _create_runtime_venv(dry_run=dry_run)
    _ensure_pip_tooling(python_exe=python_executable, dry_run=dry_run)
    _install_common_dependencies(python_exe=python_executable, manifest=manifest, dry_run=dry_run)
    if backend_key in {"paddle_cuda", "cpu"}:
        _install_shared_runtime_dependencies(python_exe=python_executable, manifest=manifest, dry_run=dry_run)
        _uninstall_conflicting_paddle_backend(python_executable, backend_key, dry_run=dry_run)
    _install_package_specs(python_exe=python_executable, package_specs=runtime_plan, dry_run=dry_run)

    ocr_device = ""
    validation_payload: dict[str, Any] = {}
    if not dry_run and not skip_validation:
        if backend_key in {"paddle_cuda", "cpu"}:
            _reset_incomplete_paddlex_models(
                [
                    model_spec.get("det", ""),
                    model_spec.get("rec", ""),
                ]
            )
        validation_payload = _validate_runtime(
            python_exe=python_executable,
            cache_home=cache_home,
            backend_key=backend_key,
            model_tier=model_tier,
            dry_run=False,
        )
        ocr_device = str(validation_payload.get("ocr_device", "") or "")
        if backend_key in {"paddle_cuda", "cpu"}:
            required_model_names = [model_spec.get("det", ""), model_spec.get("rec", "")]
            missing_models = [
                model_name
                for model_name in required_model_names
                if model_name and not _model_dir_is_complete(_official_models_root() / model_name)
            ]
            if missing_models:
                raise RuntimeError("OCR model download did not complete: " + ", ".join(sorted(set(missing_models))))
        _prune_installed_docs(_private_site_packages())
    else:
        ocr_device = {"paddle_cuda": "gpu:0", "onnx_directml": "dml:0", "cpu": "cpu"}[backend_key]

    bundled_models: list[str] = []
    if bundle_models and backend_key in {"paddle_cuda", "cpu"}:
        bundled_models = _bundle_paddlex_models([model_spec.get("det", ""), model_spec.get("rec", "")])

    return {
        "backend": backend_key,
        "model_tier": model_tier,
        "cuda_lane": cuda_lane,
        "cache_home": cache_home,
        "python_executable": python_executable,
        "runtime_plan": runtime_plan,
        "hardware": hardware,
        "model_spec": model_spec,
        "ocr_device": ocr_device,
        "validation_details": validation_payload,
        "bundled_models": bundled_models,
    }


_configure_paddlex_cache_home()


def main() -> int:
    manifest = _load_runtime_manifest()
    model_tier_choices = ["auto", *sorted((manifest.get("models", {}).get("tiers", {}) or {}).keys())]

    parser = argparse.ArgumentParser(description="Install OCR runtime.")
    parser.add_argument("--package-root", type=Path, default=None)
    parser.add_argument("--runtime-name", choices=["legacy", *FULL_PACKAGE_RUNTIME_NAMES], default="legacy")
    parser.add_argument("--install-scope", choices=["backend", "app"], default="backend")
    parser.add_argument("--backend", choices=["auto", "paddle_cuda", "onnx_directml", "cpu"], default="auto")
    parser.add_argument("--model-tier", choices=model_tier_choices, default="auto")
    parser.add_argument("--cuda-lane", choices=["auto", "cu118", "cu126", "cu129"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-state-write", action="store_true")
    parser.add_argument("--bundle-models", action="store_true")
    parser.add_argument("--allow-missing-hardware", action="store_true")
    args = parser.parse_args()

    _set_package_root_override(args.package_root)
    _set_runtime_name_override(args.runtime_name)
    _configure_paddlex_cache_home(force=bool(args.package_root is not None or args.runtime_name != "legacy"))

    if args.install_scope == "app":
        result = _install_app_runtime(manifest=manifest, dry_run=args.dry_run)
        if not args.skip_state_write:
            payload = {
                "schema_version": 5,
                "package_mode": "full" if _normalize_runtime_name(None) != "legacy" else "legacy",
                "runtime_name": _normalize_runtime_name(None),
                "status": "dry_run" if args.dry_run else "installed",
                "validation": "skipped",
                "backend": "auto",
                "execution_mode": "subprocess",
                "python_executable": str(result["python_executable"]),
                "runtime_path": str(result["runtime_path"]),
                "packages": result["packages"],
                "message": "App runtime assembled.",
            }
            _write_state(payload)
        print(f"App runtime ready at {_private_runtime_dir()}")
        return 0

    backend_key, hardware = _selected_backend(args.backend, allow_missing_hardware=args.allow_missing_hardware)
    model_spec = _manifest_model_spec(manifest, backend_key, args.model_tier)
    model_tier = model_spec.get("model_tier", "lite") or "lite"

    cuda_lane = None
    cuda_lane_reason = ""
    if backend_key == "paddle_cuda":
        gpu_env = hardware.get("gpu_env", {}) if isinstance(hardware, dict) else {}
        if args.cuda_lane != "auto":
            cuda_lane = args.cuda_lane
            cuda_lane_reason = f"Using user-selected Paddle CUDA lane: {cuda_lane}"
        else:
            cuda_lane, cuda_lane_reason = _choose_cuda_lane(
                str(gpu_env.get("nvcc_version", "") or ""),
                str(gpu_env.get("driver_cuda_version", "") or ""),
                compute_capability=str(gpu_env.get("compute_capability", "") or ""),
                gpu_name=str(hardware.get("gpu_name", "") or ""),
            )
            if cuda_lane is None:
                raise RuntimeError(
                    "Could not determine a supported Paddle CUDA lane. "
                    f"driver max CUDA={gpu_env.get('driver_cuda_version') or 'unknown'}; "
                    f"nvcc(local toolkit)={gpu_env.get('nvcc_version') or 'unknown'}; reason={cuda_lane_reason}"
                )
    cache_home = _configure_paddlex_cache_home()
    runtime_dir = _runtime_dir()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    python_executable = _private_python()

    try:
        if cuda_lane_reason:
            print(cuda_lane_reason)
        result = _install_backend_runtime(
            manifest=manifest,
            backend_key=backend_key,
            model_tier=model_tier,
            cuda_lane=cuda_lane,
            cache_home=cache_home,
            hardware=hardware,
            dry_run=args.dry_run,
            skip_validation=args.skip_validation,
            bundle_models=args.bundle_models,
        )
    except Exception as exc:  # pragma: no cover - surfaced to user via CLI
        if not args.dry_run and not args.skip_validation and backend_key == "paddle_cuda":
            print(f"[warn] GPU OCR validation failed, falling back to CPU: {exc}")
            fallback_reason = str(exc)
            failure_log_path = str(exc.log_path) if isinstance(exc, CommandExecutionError) else ""
            _reset_private_runtime_dir()
            try:
                result = _install_backend_runtime(
                    manifest=manifest,
                    backend_key="cpu",
                    model_tier=model_tier,
                    cuda_lane=None,
                    cache_home=cache_home,
                    hardware=hardware,
                    dry_run=False,
                    skip_validation=False,
                    bundle_models=args.bundle_models,
                )
            except Exception as fallback_exc:
                payload = _state_payload(
                    backend_key=backend_key,
                    model_tier=model_tier,
                    cuda_lane=cuda_lane,
                    cache_home=cache_home,
                    python_executable=python_executable,
                    manifest=manifest,
                    runtime_plan=_manifest_backend_package_plan(manifest, backend_key, cuda_lane=cuda_lane),
                    hardware=hardware,
                    model_spec=model_spec,
                    status="invalid",
                    validation="failed",
                    validation_error=f"gpu_error={exc}; cpu_fallback_error={fallback_exc}",
                    ocr_device="",
                    fallback_from="paddle_cuda",
                    fallback_reason=fallback_reason,
                    failure_log_path=failure_log_path,
                )
                if not args.skip_state_write:
                    _write_state(payload)
                raise
            payload = _state_payload(
                backend_key="cpu",
                model_tier=str(result["model_tier"]),
                cuda_lane=None,
                cache_home=cache_home,
                python_executable=Path(result["python_executable"]),
                manifest=manifest,
                runtime_plan=list(result["runtime_plan"]),
                hardware=hardware,
                model_spec=dict(result["model_spec"]),
                status="installed",
                validation="ok",
                validation_error="",
                ocr_device=str(result["ocr_device"]),
                validation_details=dict(result["validation_details"]),
                fallback_from="paddle_cuda",
                fallback_reason=fallback_reason,
                failure_log_path=failure_log_path,
            )
            if not args.skip_state_write:
                _write_state(payload)
                print(f"OCR runtime state wrote {_runtime_state_path()}")
            print("[info] GPU OCR runtime fallback applied: using CPU backend.")
            return 0
        payload = _state_payload(
            backend_key=backend_key,
            model_tier=model_tier,
            cuda_lane=cuda_lane,
            cache_home=cache_home,
            python_executable=python_executable,
            manifest=manifest,
            runtime_plan=_manifest_backend_package_plan(manifest, backend_key, cuda_lane=cuda_lane),
            hardware=hardware,
            model_spec=model_spec,
            status="invalid",
            validation="failed",
            validation_error=str(exc),
            ocr_device="",
            failure_log_path=str(exc.log_path) if isinstance(exc, CommandExecutionError) else "",
        )
        if not args.skip_state_write:
            _write_state(payload)
        raise

    payload = _state_payload(
        backend_key=str(result["backend"]),
        model_tier=str(result["model_tier"]),
        cuda_lane=result["cuda_lane"] if result["backend"] == "paddle_cuda" else None,
        cache_home=cache_home,
        python_executable=Path(result["python_executable"]),
        manifest=manifest,
        runtime_plan=list(result["runtime_plan"]),
        hardware=hardware,
        model_spec=dict(result["model_spec"]),
        status="dry_run" if args.dry_run else "installed",
        validation="skipped" if args.dry_run or args.skip_validation else "ok",
        validation_error="",
        ocr_device=str(result["ocr_device"]),
        validation_details=dict(result["validation_details"]),
    )
    if result.get("bundled_models"):
        payload["bundled_models"] = list(result["bundled_models"])
    if not args.skip_state_write:
        _write_state(payload)
        print(f"OCR runtime state wrote {_runtime_state_path()}")
    print(f"OCR runtime ready at {_private_runtime_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
