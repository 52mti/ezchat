from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from runtime_layout import (
    backend_runtime_name,
    bundled_paddlex_model_dir,
    has_packaged_runtime_layout,
    model_dir_is_complete,
    package_root_from_path,
    runtime_python,
)


PADDLE_VERSION = "3.2.2"
PADDLEOCR_VERSION = "3.4.0"
PADDLEX_VERSION = "3.4.3"
DEFAULT_MODEL_TIER = "lite"
SUPPORTED_BACKENDS = ("auto", "paddle_cuda", "onnx_directml", "cpu")
SUPPORTED_EXECUTION_MODES = ("auto", "inprocess", "subprocess")
DEFAULT_OCR_GPU_ALLOCATOR_STRATEGY = "auto_growth"
DEFAULT_OCR_GPU_MEMORY_FRACTION = "0.6"
CUDA_LANE_INDEXES = {
    "cu118": "https://www.paddlepaddle.org.cn/packages/stable/cu118/",
    "cu126": "https://www.paddlepaddle.org.cn/packages/stable/cu126/",
    "cu129": "https://www.paddlepaddle.org.cn/packages/stable/cu129/",
}
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


@dataclass(slots=True)
class GpuEnvironmentReport:
    platform: str
    python_version: str
    python_bits: str
    nvidia_smi_found: bool
    gpu_names: list[str]
    driver_version: str | None
    driver_cuda_version: str | None
    nvcc_version: str | None
    compute_capability: str | None
    recommended_cuda_lane: str | None
    lane_reason: str | None
    paddle_version: str | None = None
    paddle_compiled_with_cuda: bool | None = None
    paddle_cuda_device_count: int | None = None
    paddle_current_device: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HardwareProbeResult:
    platform: str
    gpu_vendor: str
    gpu_name: str
    nvidia_detected: bool
    directml_available: bool
    recommended_backend: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RuntimePackageSpec:
    spec: str
    index_url: str = ""


@dataclass(slots=True)
class BackendManifest:
    name: str
    device: str
    packages: list[RuntimePackageSpec] = field(default_factory=list)
    lane_indexes: dict[str, str] = field(default_factory=dict)
    model_provider: str = ""
    description: str = ""


@dataclass(slots=True)
class TierBackendModelSpec:
    provider: str
    det_model_name: str = ""
    rec_model_name: str = ""
    det_version: str = ""
    rec_version: str = ""


@dataclass(slots=True)
class RuntimeManifest:
    version: str
    backends: dict[str, BackendManifest]
    model_default_tier: str
    model_tiers: dict[str, dict[str, TierBackendModelSpec]]
    shared_packages: list[RuntimePackageSpec] = field(default_factory=list)
    pip_index_url: str = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"


def _runtime_state_path(project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else _package_root()
    return root / "runtime" / "install_state.json"


def load_installed_runtime_state(project_root: str | Path | None = None) -> dict[str, Any]:
    path = _runtime_state_path(project_root)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    status = str(payload.get("status", "") or "").strip().lower()
    validation = str(payload.get("validation", "") or "").strip().lower()
    if status != "installed" or validation != "ok":
        return {}
    return payload


def _state_runtime_python(state: dict[str, Any], backend: str | None = None) -> Path | None:
    normalized_backend = str(backend or "").strip().lower()
    state_backend = str(state.get("backend", "") or "").strip().lower()
    if normalized_backend and state_backend and normalized_backend != state_backend:
        return None
    runtime_name = str(state.get("runtime_name", "") or "").strip().lower()
    package_mode = str(state.get("package_mode", "") or "").strip().lower()
    python_executable = str(state.get("python_executable", "") or "").strip()
    runtime_path = str(state.get("runtime_path", "") or "").strip()
    legacy_hint = (
        runtime_name in {"legacy", ".runtime_env"}
        or package_mode == "legacy"
        or ".runtime_env" in python_executable.replace("\\", "/")
    )
    candidates: list[Path] = []
    if python_executable:
        candidates.append(Path(python_executable))
    if legacy_hint and runtime_path:
        runtime_dir_path = Path(runtime_path)
        if os.name == "nt":
            candidates.extend(
                [
                    runtime_dir_path / "python.exe",
                    runtime_dir_path / "Scripts" / "python.exe",
                    runtime_dir_path / "pythonw.exe",
                    runtime_dir_path / "Scripts" / "pythonw.exe",
                ]
            )
        else:
            candidates.append(runtime_dir_path / "bin" / "python")
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            resolved = candidate
        if resolved.is_file():
            return resolved
    return None


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
            # pythonw 无控制台：不带此标志的子进程会走 conhost 窗口分配，与主线程
            # 消息泵并发时触发 RPC_E_CANTCALLOUTINASYNCCALL（0x8001010d）原生崩溃。
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0,
            startupinfo=startupinfo,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


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


def choose_cuda_lane(
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


def detect_gpu_environment() -> GpuEnvironmentReport:
    query = _run_command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    gpu_names: list[str] = []
    driver_version: str | None = None
    if query is not None and query.returncode == 0:
        for line in query.stdout.splitlines():
            parts = [part.strip() for part in line.split(",", maxsplit=1)]
            if parts and parts[0]:
                gpu_names.append(parts[0])
            if len(parts) > 1 and parts[1]:
                driver_version = parts[1]
    summary = _run_command(["nvidia-smi"])
    driver_cuda_version: str | None = None
    if summary is not None and summary.returncode == 0:
        match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", summary.stdout)
        if match:
            driver_cuda_version = match.group(1)
    nvcc = _run_command(["nvcc", "--version"])
    nvcc_version: str | None = None
    if nvcc is not None and nvcc.returncode == 0:
        match = re.search(r"release\s+([0-9]+(?:\.[0-9]+)?)", nvcc.stdout)
        if match:
            nvcc_version = match.group(1)
    compute_cap_query = _run_command(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    compute_capability: str | None = None
    if compute_cap_query is not None and compute_cap_query.returncode == 0:
        for line in compute_cap_query.stdout.splitlines():
            value = line.strip()
            if value:
                compute_capability = value
                break
    lane, reason = choose_cuda_lane(
        nvcc_version,
        driver_cuda_version,
        compute_capability=compute_capability,
        gpu_name=gpu_names[0] if gpu_names else None,
    )
    return GpuEnvironmentReport(
        platform=platform.platform(),
        python_version=platform.python_version(),
        python_bits=platform.architecture()[0],
        nvidia_smi_found=query is not None and query.returncode == 0,
        gpu_names=gpu_names,
        driver_version=driver_version,
        driver_cuda_version=driver_cuda_version,
        nvcc_version=nvcc_version,
        compute_capability=compute_capability,
        recommended_cuda_lane=lane,
        lane_reason=reason,
    )


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


def _detect_display_gpu() -> tuple[str, str, bool]:
    controllers = _probe_windows_video_controllers()
    for controller in controllers:
        text = f"{controller.get('vendor', '')} {controller.get('name', '')}".lower()
        if "nvidia" in text:
            return "nvidia", controller.get("name", ""), True
    for controller in controllers:
        text = f"{controller.get('vendor', '')} {controller.get('name', '')}".lower()
        if "amd" in text or "radeon" in text or "advanced micro devices" in text:
            return "amd", controller.get("name", ""), True
        if "intel" in text:
            return "intel", controller.get("name", ""), True
    if controllers:
        return "unknown", controllers[0].get("name", ""), True
    return "unknown", "", False


@lru_cache(maxsize=1)
def probe_hardware() -> HardwareProbeResult:
    gpu_report = detect_gpu_environment()
    if gpu_report.gpu_names:
        return HardwareProbeResult(
            platform=platform.platform(),
            gpu_vendor="nvidia",
            gpu_name=gpu_report.gpu_names[0],
            nvidia_detected=True,
            directml_available=False,
            recommended_backend="paddle_cuda",
            detail={"gpu_env": gpu_report.to_dict(), "video_controllers": _probe_windows_video_controllers()},
        )
    vendor, name, has_gpu = _detect_display_gpu()
    directml_available = bool(sys.platform.startswith("win") and vendor in {"amd", "intel"} and has_gpu)
    return HardwareProbeResult(
        platform=platform.platform(),
        gpu_vendor=vendor,
        gpu_name=name,
        nvidia_detected=False,
        directml_available=directml_available,
        recommended_backend="onnx_directml" if directml_available else "cpu",
        detail={"gpu_env": gpu_report.to_dict(), "video_controllers": _probe_windows_video_controllers()},
    )


def normalize_ocr_backend(raw_value: object) -> str:
    value = str(raw_value or "auto").strip().lower()
    aliases = {
        "auto": "auto",
        "gpu": "paddle_cuda",
        "cuda": "paddle_cuda",
        "nvidia": "paddle_cuda",
        "paddle": "paddle_cuda",
        "paddle_cuda": "paddle_cuda",
        "amd": "onnx_directml",
        "intel": "onnx_directml",
        "directml": "onnx_directml",
        "onnx": "onnx_directml",
        "onnx_directml": "onnx_directml",
        "cpu": "cpu",
    }
    normalized = aliases.get(value)
    if normalized not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported OCR backend: {raw_value}")
    return normalized


def select_ocr_backend(preferred_backend: object, hardware: HardwareProbeResult | None = None) -> str:
    normalized = normalize_ocr_backend(preferred_backend)
    if normalized == "auto":
        return (hardware or probe_hardware()).recommended_backend
    hardware = hardware or probe_hardware()
    if normalized == "paddle_cuda" and not hardware.nvidia_detected:
        raise RuntimeError("OCR backend paddle_cuda requires an NVIDIA GPU.")
    if normalized == "onnx_directml" and not hardware.directml_available:
        raise RuntimeError("OCR backend onnx_directml requires a Windows AMD/Intel GPU with DirectML support.")
    return normalized


def _default_runtime_manifest() -> RuntimeManifest:
    packages_common = [
        RuntimePackageSpec(f"paddleocr=={PADDLEOCR_VERSION}"),
        RuntimePackageSpec(f"paddlex[ocr-core]=={PADDLEX_VERSION}"),
    ]
    model_spec = TierBackendModelSpec(
        provider="paddlex_named_models",
        det_model_name="PP-OCRv5_mobile_det",
        rec_model_name="PP-OCRv5_mobile_rec",
        det_version="PP-OCRv5_mobile_det",
        rec_version="PP-OCRv5_mobile_rec",
    )
    return RuntimeManifest(
        version="2026.05.20",
        shared_packages=packages_common,
        backends={
            "paddle_cuda": BackendManifest(
                name="paddle_cuda",
                device="gpu:0",
                packages=[RuntimePackageSpec(f"paddlepaddle-gpu=={PADDLE_VERSION}")],
                lane_indexes=dict(CUDA_LANE_INDEXES),
                model_provider="paddlex_named_models",
                description="NVIDIA CUDA + PaddleOCR PP-OCRv5 mobile",
            ),
            "cpu": BackendManifest(
                name="cpu",
                device="cpu",
                packages=[
                    RuntimePackageSpec(
                        f"paddlepaddle=={PADDLE_VERSION}",
                        "https://www.paddlepaddle.org.cn/packages/stable/cpu/",
                    ),
                ],
                model_provider="paddlex_named_models",
                description="CPU + PaddleOCR PP-OCRv5 mobile",
            ),
            "onnx_directml": BackendManifest(
                name="onnx_directml",
                device="dml:0",
                packages=[
                    RuntimePackageSpec("onnxruntime-directml"),
                    RuntimePackageSpec("rapidocr_onnxruntime"),
                ],
                model_provider="rapidocr_builtin",
                description="Windows AMD/Intel DirectML + RapidOCR",
            ),
        },
        model_default_tier=DEFAULT_MODEL_TIER,
        model_tiers={
            "lite": {
                "paddle_cuda": model_spec,
                "cpu": model_spec,
                "onnx_directml": TierBackendModelSpec(
                    provider="rapidocr_builtin",
                    det_version="rapidocr_builtin_det",
                    rec_version="rapidocr_builtin_rec",
                ),
            }
        },
    )


def _load_package_specs(payload: object, fallback: list[RuntimePackageSpec]) -> list[RuntimePackageSpec]:
    if not isinstance(payload, list):
        return list(fallback)
    packages: list[RuntimePackageSpec] = []
    for item in payload:
        if isinstance(item, str) and item.strip():
            packages.append(RuntimePackageSpec(item.strip()))
        elif isinstance(item, dict):
            spec = str(item.get("spec", "") or "").strip()
            if spec:
                packages.append(RuntimePackageSpec(spec, str(item.get("indexUrl", "") or "").strip()))
    return packages or list(fallback)


@lru_cache(maxsize=1)
def load_runtime_manifest(project_root: str | Path | None = None) -> RuntimeManifest:
    manifest = _default_runtime_manifest()
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    path = root / "runtime_manifest.json"
    if not path.is_file():
        return manifest
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return manifest
    if not isinstance(payload, dict):
        return manifest
    pip_block = payload.get("pip")
    if isinstance(pip_block, dict):
        manifest.pip_index_url = str(pip_block.get("indexUrl", manifest.pip_index_url) or manifest.pip_index_url)
    shared_packages = payload.get("sharedPackages", payload.get("shared_packages"))
    manifest.shared_packages = _load_package_specs(shared_packages, manifest.shared_packages)
    backends = payload.get("backends")
    if isinstance(backends, dict):
        for name, backend_payload in backends.items():
            key = str(name).strip().lower()
            if key not in manifest.backends or not isinstance(backend_payload, dict):
                continue
            fallback = manifest.backends[key]
            lanes = dict(fallback.lane_indexes)
            if isinstance(backend_payload.get("lanes"), dict):
                lanes.update({str(k): str(v) for k, v in backend_payload["lanes"].items() if str(v).strip()})
            manifest.backends[key] = BackendManifest(
                name=key,
                device=str(backend_payload.get("device", fallback.device) or fallback.device),
                packages=_load_package_specs(backend_payload.get("packages"), fallback.packages),
                lane_indexes=lanes,
                model_provider=str(backend_payload.get("modelProvider", fallback.model_provider) or fallback.model_provider),
                description=str(backend_payload.get("description", fallback.description) or fallback.description),
            )
    models = payload.get("models")
    if isinstance(models, dict):
        default_tier = str(models.get("defaultTier", manifest.model_default_tier) or manifest.model_default_tier).lower()
        manifest.model_default_tier = default_tier if default_tier in manifest.model_tiers else manifest.model_default_tier
        tiers = models.get("tiers")
        if isinstance(tiers, dict):
            for tier_name, tier_payload in tiers.items():
                tier_key = str(tier_name).strip().lower()
                if tier_key not in manifest.model_tiers or not isinstance(tier_payload, dict):
                    continue
                backend_payloads = tier_payload.get("backends")
                if not isinstance(backend_payloads, dict):
                    continue
                for backend_name, spec_payload in backend_payloads.items():
                    backend_key = str(backend_name).strip().lower()
                    if backend_key not in manifest.model_tiers[tier_key] or not isinstance(spec_payload, dict):
                        continue
                    fallback = manifest.model_tiers[tier_key][backend_key]
                    manifest.model_tiers[tier_key][backend_key] = TierBackendModelSpec(
                        provider=str(spec_payload.get("provider", fallback.provider) or fallback.provider),
                        det_model_name=str(spec_payload.get("detModelName", fallback.det_model_name) or fallback.det_model_name),
                        rec_model_name=str(spec_payload.get("recModelName", fallback.rec_model_name) or fallback.rec_model_name),
                        det_version=str(spec_payload.get("detVersion", fallback.det_version) or fallback.det_version),
                        rec_version=str(spec_payload.get("recVersion", fallback.rec_version) or fallback.rec_version),
                    )
    if str(payload.get("version", "") or "").strip():
        manifest.version = str(payload["version"]).strip()
    return manifest


def build_runtime_package_specs(manifest: RuntimeManifest, backend: str) -> list[RuntimePackageSpec]:
    backend_key = str(backend or "").strip().lower()
    backend_manifest = manifest.backends.get(backend_key)
    if backend_manifest is None:
        raise ValueError(f"Unsupported OCR backend: {backend}")
    packages: list[RuntimePackageSpec] = []
    seen: set[tuple[str, str]] = set()
    for package in [*manifest.shared_packages, *backend_manifest.packages]:
        marker = (package.spec, package.index_url)
        if marker in seen:
            continue
        seen.add(marker)
        packages.append(package)
    return packages


def resolve_model_spec(manifest: RuntimeManifest, backend: str, model_tier: object) -> TierBackendModelSpec:
    tier = str(model_tier or "auto").strip().lower()
    if tier == "auto":
        tier = manifest.model_default_tier
    tier_specs = manifest.model_tiers.get(tier)
    if tier_specs is None:
        raise ValueError(f"Unsupported OCR model tier: {model_tier}")
    spec = tier_specs.get(backend)
    if spec is None:
        raise ValueError(f"Model tier '{tier}' does not support backend '{backend}'")
    return spec


def normalize_execution_mode(raw_value: object) -> str:
    value = str(raw_value or "auto").strip().lower()
    aliases = {
        "": "auto",
        "auto": "auto",
        "inprocess": "inprocess",
        "local": "inprocess",
        "subprocess": "subprocess",
        "worker": "subprocess",
    }
    normalized = aliases.get(value)
    if normalized not in SUPPORTED_EXECUTION_MODES:
        raise ValueError(f"Unsupported OCR execution mode: {raw_value}")
    return normalized


def _package_root(project_root: str | Path | None = None) -> Path:
    if project_root is not None:
        return package_root_from_path(project_root)
    configured = str(os.environ.get("JIXIEBI_PACKAGE_ROOT", "") or "").strip()
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[1]


def _load_runtime_manifest_payload(project_root: str | Path | None = None) -> dict[str, Any]:
    path = _package_root(project_root) / "runtime_manifest.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _manifest_execution_mode(project_root: str | Path | None = None) -> str:
    payload = _load_runtime_manifest_payload(project_root)
    return normalize_execution_mode(payload.get("executionMode", "auto"))


def _has_packaged_ocr_runtime_layout(project_root: str | Path | None = None) -> bool:
    root = _package_root(project_root)
    for runtime_name in ("ocr_cpu", "ocr_nvidia", "ocr_directml"):
        if runtime_python(root, runtime_name, gui=False).is_file():
            return True
    return False


def _bundled_model_mirror_root(project_root: str | Path | None = None) -> Path:
    if os.name == "nt":
        public_root = Path(os.environ.get("PUBLIC", r"C:\Users\Public")).resolve()
        return public_root / "jixiebijiaoben_runtime" / "bundled_models"
    return _package_root(project_root) / "runtime" / ".paddlex" / "bundled_models"


def _path_anchor_exists(path: Path) -> bool:
    anchor = path.anchor
    if not anchor:
        return True
    try:
        return Path(anchor).exists()
    except OSError:
        return False


def _resolve_runtime_cache_home(root: Path, configured: object = "") -> Path:
    raw = str(configured or "").strip()
    configured_is_staging = False
    if raw:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            return (root / candidate).resolve()
        normalized = str(candidate).replace("\\", "/").lower()
        configured_is_staging = "/dist/_staging/" in normalized
        if "/dist/_staging/" not in normalized and _path_anchor_exists(candidate):
            return candidate.resolve()
    env_cache = "" if configured_is_staging else str(os.environ.get("PADDLE_PDX_CACHE_HOME", "") or "").strip()
    if env_cache:
        candidate = Path(env_cache).expanduser()
        if not candidate.is_absolute():
            return (root / candidate).resolve()
        if _path_anchor_exists(candidate):
            return candidate.resolve()
    if os.name == "nt":
        public_root = Path(os.environ.get("PUBLIC", r"C:\Users\Public")).resolve()
        return public_root / "jixiebijiaoben_runtime" / ".paddlex"
    return root / "runtime" / ".paddlex"


def _model_cache_key(model_dir: Path, model_name: str) -> str:
    payload = [str(model_name or "").strip()]
    for file_name in ("config.json", "inference.json", "inference.pdiparams", "inference.yml"):
        file_path = model_dir / file_name
        try:
            stat = file_path.stat()
        except OSError:
            payload.append(f"{file_name}:missing")
            continue
        payload.append(f"{file_name}:{stat.st_size}:{stat.st_mtime_ns}")
    digest = hashlib.sha1("|".join(payload).encode("utf-8")).hexdigest()[:12]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_name or "").strip()).strip("._") or "model"
    return f"{safe_name}_{digest}"


def _copy_model_dir_atomic(source_dir: Path, target_dir: Path) -> None:
    mirror_root = target_dir.parent
    mirror_root.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="model_copy_", dir=mirror_root))
    temp_target = temp_root / target_dir.name
    try:
        shutil.copytree(source_dir, temp_target)
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        temp_target.replace(target_dir)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _materialize_ascii_safe_model_dir(model_dir: Path, model_name: str, project_root: str | Path | None = None) -> Path:
    if os.name != "nt" or str(model_dir).isascii():
        return model_dir
    cache_key = _model_cache_key(model_dir, model_name)
    target_dir = _bundled_model_mirror_root(project_root) / cache_key
    if model_dir_is_complete(target_dir):
        return target_dir
    with _ENGINE_INIT_LOCK:
        if model_dir_is_complete(target_dir):
            return target_dir
        _copy_model_dir_atomic(model_dir, target_dir)
    return target_dir


def _normalize_model_dir_for_paddle(
    model_dir_value: str | Path | None,
    model_name: str,
    project_root: str | Path | None = None,
) -> str:
    normalized = str(model_dir_value or "").strip()
    if not normalized:
        return ""
    model_dir = Path(normalized).expanduser()
    if not model_dir_is_complete(model_dir):
        return normalized
    return str(_materialize_ascii_safe_model_dir(model_dir, model_name, project_root))


def _resolve_bundled_model_dir(model_name: str, project_root: str | Path | None = None) -> str:
    normalized = str(model_name or "").strip()
    if not normalized:
        return ""
    model_dir = bundled_paddlex_model_dir(_package_root(project_root), normalized)
    if not model_dir_is_complete(model_dir):
        return ""
    return _normalize_model_dir_for_paddle(model_dir, normalized, project_root)


def _serialize_prediction_payload(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _serialize_prediction_payload(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _serialize_prediction_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_prediction_payload(item) for item in value]
    if isinstance(value, set):
        return [_serialize_prediction_payload(item) for item in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _serialize_prediction_payload(to_dict())
        except Exception:
            pass
    return str(value)


def validate_paddle_gpu_runtime() -> GpuEnvironmentReport:
    report = detect_gpu_environment()
    if not report.gpu_names:
        raise RuntimeError("No usable NVIDIA GPU was detected; paddle_cuda OCR backend cannot be used.")
    try:
        # Paddle 在 Windows 导入阶段会输出 ccache 等诊断信息；这不是 OCR
        # 初始化失败。运行时仍会把真正的异常抛给上层进行日志记录。
        with suppress_native_output():
            import paddle
    except ImportError as exc:
        raise RuntimeError("The current Python environment does not have Paddle GPU runtime installed.") from exc
    report.paddle_version = getattr(paddle, "__version__", None)
    report.paddle_compiled_with_cuda = bool(paddle.is_compiled_with_cuda())
    if not report.paddle_compiled_with_cuda:
        raise RuntimeError(f"Current Paddle {report.paddle_version or '<unknown>'} is not a GPU build.")
    report.paddle_cuda_device_count = int(paddle.device.cuda.device_count())
    if report.paddle_cuda_device_count < 1:
        raise RuntimeError("Paddle GPU is installed, but no usable CUDA device was enumerated.")
    report.paddle_current_device = str(paddle.device.get_device())
    return report


@contextlib.contextmanager
def suppress_native_output():
    with open(os.devnull, "w", encoding="utf-8", errors="ignore") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


_ENGINE_INIT_LOCK = threading.RLock()
_ENGINE_PREDICT_LOCK = threading.RLock()
_DIRECTML_ENGINE_INIT_LOCK = threading.RLock()
_RUNTIME_FALLBACK_LOCK = threading.RLock()
_PADDLE_OCR_BY_CACHE_KEY: dict[tuple[str, str, str, str], Any] = {}


def _persist_runtime_fallback(backend: str, model_tier: str, fallback_from: str, reason: Exception) -> None:
    path = _runtime_state_path()
    payload: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = {}
        if isinstance(loaded, dict):
            payload.update(loaded)
    payload["schema_version"] = max(int(payload.get("schema_version", 0) or 0), 4)
    payload["status"] = "installed"
    payload["validation"] = "ok"
    payload["validation_error"] = ""
    payload["backend"] = backend
    payload["model_tier"] = model_tier
    payload["cache_home"] = str(_resolve_runtime_cache_home(_package_root(), payload.get("cache_home", "")))
    payload["ocr_device"] = "cpu" if backend == "cpu" else payload.get("ocr_device", "")
    payload["fallback_from"] = fallback_from
    payload["fallback_reason"] = str(reason)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _persist_runtime_success(backend: str, model_tier: str, device: str) -> None:
    if backend not in {"paddle_cuda", "onnx_directml", "cpu"}:
        return
    path = _runtime_state_path()
    payload: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = {}
        if isinstance(loaded, dict):
            payload.update(loaded)
    payload["schema_version"] = max(int(payload.get("schema_version", 0) or 0), 4)
    payload["status"] = "installed"
    payload["validation"] = "ok"
    payload["validation_error"] = ""
    payload["backend"] = backend
    payload["model_tier"] = model_tier
    payload["cache_home"] = str(_resolve_runtime_cache_home(_package_root(), payload.get("cache_home", "")))
    payload["ocr_device"] = str(device or "").strip() or ("cpu" if backend == "cpu" else "")
    payload.pop("fallback_from", None)
    payload.pop("fallback_reason", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class PaddleOcrRuntimeEngine:
    def __init__(self, *, backend: str, model_tier: str, manifest: RuntimeManifest, config: dict[str, Any]) -> None:
        self.backend = backend
        self.model_tier = model_tier
        self.manifest = manifest
        self.config = dict(config)

    def _resolve_device(self) -> str:
        if self.backend == "cpu":
            return "cpu"
        report = validate_paddle_gpu_runtime()
        device = str(report.paddle_current_device or "").strip().lower()
        return device if device.startswith("gpu:") else "gpu:0"

    def _build_kwargs(self, device: str) -> dict[str, Any]:
        model_spec = resolve_model_spec(self.manifest, self.backend, self.model_tier)
        detection_model_name = self.config.get("text_detection_model_name") or model_spec.det_model_name
        recognition_model_name = self.config.get("text_recognition_model_name") or model_spec.rec_model_name
        kwargs: dict[str, Any] = {
            "lang": self.config.get("lang", "ch"),
            "use_doc_orientation_classify": bool(self.config.get("use_doc_orientation_classify", False)),
            "use_doc_unwarping": bool(self.config.get("use_doc_unwarping", False)),
            "use_textline_orientation": bool(self.config.get("use_angle_cls", False)),
            "device": device,
        }
        kwargs["text_detection_model_name"] = detection_model_name
        kwargs["text_recognition_model_name"] = recognition_model_name
        detection_dir = _normalize_model_dir_for_paddle(self.config.get("text_detection_model_dir"), detection_model_name) or _resolve_bundled_model_dir(model_spec.det_model_name)
        recognition_dir = _normalize_model_dir_for_paddle(self.config.get("text_recognition_model_dir"), recognition_model_name) or _resolve_bundled_model_dir(model_spec.rec_model_name)
        if detection_dir:
            kwargs["text_detection_model_dir"] = detection_dir
        if recognition_dir:
            kwargs["text_recognition_model_dir"] = recognition_dir
        return {key: value for key, value in kwargs.items() if value not in ("", None)}

    def _get_engine(self) -> tuple[Any, str]:
        device = self._resolve_device()
        model_spec = resolve_model_spec(self.manifest, self.backend, self.model_tier)
        cache_key = (
            self.backend,
            device,
            self.config.get("text_detection_model_name") or model_spec.det_model_name,
            self.config.get("text_recognition_model_name") or model_spec.rec_model_name,
        )
        if cache_key in _PADDLE_OCR_BY_CACHE_KEY:
            return _PADDLE_OCR_BY_CACHE_KEY[cache_key], device
        with _ENGINE_INIT_LOCK:
            if cache_key in _PADDLE_OCR_BY_CACHE_KEY:
                return _PADDLE_OCR_BY_CACHE_KEY[cache_key], device
            try:
                # PaddleOCR/PaddleX 导入时也可能直接写入 stdout/stderr（例如模型
                # 源连通性检查提示）。OCR 子进程协议依赖 stdout 为纯 JSON，主进程
                # 同样不应把这些第三方提示显示为启动异常。
                with suppress_native_output():
                    from paddleocr import PaddleOCR  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    f"Missing packaged PaddleOCR runtime for backend {self.backend}: {exc}. "
                    "Rebuild the full package."
                ) from exc
            try:
                with suppress_native_output():
                    engine = PaddleOCR(**self._build_kwargs(device))
            except TypeError:
                with suppress_native_output():
                    engine = PaddleOCR(lang=self.config.get("lang", "ch"))
            _PADDLE_OCR_BY_CACHE_KEY[cache_key] = engine
            return engine, device

    def preload(self) -> str:
        _engine, device = self._get_engine()
        return device

    def predict(self, image_array: np.ndarray, **_kwargs: Any) -> list[dict[str, Any]]:
        engine, _device = self._get_engine()
        with _ENGINE_PREDICT_LOCK:
            return engine.predict(image_array)


def _normalize_rapidocr_results(raw_result: object) -> list[dict[str, Any]]:
    if not isinstance(raw_result, list):
        return []
    rec_texts: list[str] = []
    rec_scores: list[float] = []
    rec_polys: list[list[list[float]]] = []
    for item in raw_result:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        box, text, score = item[0], item[1], item[2]
        if not isinstance(box, (list, tuple)):
            continue
        points: list[list[float]] = []
        for point in box:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                points.append([float(point[0]), float(point[1])])
        if len(points) >= 4:
            rec_polys.append(points)
            rec_texts.append(str(text or ""))
            rec_scores.append(float(score or 0.0))
    return [{"rec_texts": rec_texts, "rec_scores": rec_scores, "rec_polys": rec_polys}]


class OnnxDirectMLOcrRuntimeEngine:
    backend = "onnx_directml"

    def __init__(self, *, model_tier: str, manifest: RuntimeManifest, config: dict[str, Any]) -> None:
        self.model_tier = model_tier
        self.manifest = manifest
        self.config = dict(config)
        self._engine: Any | None = None

    @staticmethod
    def _instantiate_rapidocr(rapidocr_cls: Any, kwargs: dict[str, Any]) -> Any:
        signature = None
        try:
            signature = inspect.signature(rapidocr_cls)
        except (TypeError, ValueError):
            pass
        if signature is not None:
            accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
            if not accepts_kwargs:
                kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        try:
            return rapidocr_cls(**kwargs)
        except TypeError:
            return rapidocr_cls()

    def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        with _DIRECTML_ENGINE_INIT_LOCK:
            if self._engine is not None:
                return self._engine
            try:
                import onnxruntime as ort  # type: ignore
                from rapidocr_onnxruntime import RapidOCR  # type: ignore
            except ImportError as exc:
                raise RuntimeError("Missing packaged DirectML OCR runtime. Rebuild the full package.") from exc
            providers = list(ort.get_available_providers())
            if "DmlExecutionProvider" not in providers:
                raise RuntimeError(f"onnxruntime does not expose the DirectML provider. Providers: {providers}")
            self._engine = self._instantiate_rapidocr(RapidOCR, {"providers": ["DmlExecutionProvider", "CPUExecutionProvider"]})
            return self._engine

    def preload(self) -> str:
        self._get_engine()
        return "dml:0"

    def predict(self, image_array: np.ndarray, **_kwargs: Any) -> list[dict[str, Any]]:
        engine = self._get_engine()
        with _ENGINE_PREDICT_LOCK:
            result = engine(np.ascontiguousarray(image_array))
        if isinstance(result, tuple):
            result = result[0]
        return _normalize_rapidocr_results(result)


class ResilientRuntimeEngine:
    def __init__(
        self,
        primary_engine: Any,
        *,
        primary_backend: str,
        model_tier: str,
        fallback_factory: Any | None = None,
    ) -> None:
        self._primary_engine = primary_engine
        self._fallback_engine: Any | None = None
        self._fallback_factory = fallback_factory
        self._primary_backend = primary_backend
        self._model_tier = model_tier

    def _activate_fallback(self, exc: Exception) -> Any:
        if self._fallback_factory is None:
            raise exc
        with _RUNTIME_FALLBACK_LOCK:
            if self._fallback_engine is not None:
                return self._fallback_engine
            self._fallback_engine = self._fallback_factory()
            _persist_runtime_fallback("cpu", self._model_tier, self._primary_backend, exc)
            return self._fallback_engine

    def preload(self) -> str:
        try:
            device = str(self._primary_engine.preload() or "")
            _persist_runtime_success(self._primary_backend, self._model_tier, device)
            return device
        except Exception as exc:
            fallback = self._activate_fallback(exc)
            return str(fallback.preload() or "")

    def predict(self, image_array: np.ndarray, **kwargs: Any) -> list[dict[str, Any]]:
        engine = self._fallback_engine or self._primary_engine
        try:
            return engine.predict(image_array, **kwargs)
        except Exception as exc:
            fallback = self._activate_fallback(exc)
            return fallback.predict(image_array, **kwargs)


class SubprocessOcrRuntimeEngine:
    def __init__(self, *, backend: str, model_tier: str, manifest: RuntimeManifest, config: dict[str, Any]) -> None:
        self.backend = backend
        self.model_tier = model_tier
        self.manifest = manifest
        self.config = dict(config)
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.RLock()

    def _runtime_python_path(self) -> Path:
        root = _package_root()
        candidate = runtime_python(root, backend_runtime_name(self.backend), gui=False)
        if candidate.is_file():
            return candidate
        state_python = _state_runtime_python(load_installed_runtime_state(root), self.backend)
        if state_python is not None:
            return state_python
        raise RuntimeError(f"Missing OCR runtime python for backend: {self.backend}")

    def _subprocess_config(self) -> dict[str, Any]:
        allowed_keys = {
            "lang",
            "use_angle_cls",
            "use_doc_orientation_classify",
            "use_doc_unwarping",
            "ocr_gpu_allocator_strategy",
            "ocr_gpu_memory_fraction",
            "text_detection_model_name",
            "text_recognition_model_name",
            "text_detection_model_dir",
            "text_recognition_model_dir",
            "det_db_thresh",
            "rec_thresh",
        }
        payload = {key: self.config[key] for key in allowed_keys if key in self.config}
        payload["backend"] = self.backend
        payload["model_tier"] = self.model_tier
        payload["_execution_mode"] = "inprocess"
        return payload

    def _start_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        root = _package_root()
        python_exe = self._runtime_python_path()
        command = [
            str(python_exe),
            "-s",
            "-u",
            str(root / "ocr_subprocess_entry.pyc"),
            "--root",
            str(root),
            "--backend",
            self.backend,
            "--model-tier",
            self.model_tier,
            "--config-json",
            json.dumps(self._subprocess_config(), ensure_ascii=False),
        ]
        startupinfo = None
        creationflags = 0
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONNOUSERSITE"] = "1"
        env["PIP_USER"] = "0"
        env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        env["FLAGS_allocator_strategy"] = str(
            self.config.get("ocr_gpu_allocator_strategy", DEFAULT_OCR_GPU_ALLOCATOR_STRATEGY) or DEFAULT_OCR_GPU_ALLOCATOR_STRATEGY
        )
        env["FLAGS_fraction_of_gpu_memory_to_use"] = str(
            self.config.get("ocr_gpu_memory_fraction", DEFAULT_OCR_GPU_MEMORY_FRACTION) or DEFAULT_OCR_GPU_MEMORY_FRACTION
        )
        cache_home = _resolve_runtime_cache_home(root, load_installed_runtime_state(root).get("cache_home", ""))
        env["PADDLE_PDX_CACHE_HOME"] = str(cache_home)
        env["PADDLEX_HOME"] = str(cache_home)
        runtime_dir_path = python_exe.parent
        if runtime_dir_path.name.lower() == "scripts":
            runtime_dir_path = runtime_dir_path.parent
        scripts_dir = runtime_dir_path / "Scripts"
        path_parts = [str(runtime_dir_path)]
        if scripts_dir.is_dir():
            path_parts.append(str(scripts_dir))
        existing_path = env.get("PATH", "")
        env["PATH"] = os.pathsep.join([*path_parts, existing_path] if existing_path else path_parts)
        env.pop("PYTHONPATH", None)
        if sys.platform.startswith("win"):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(
            command,
            cwd=root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            env=env,
            startupinfo=startupinfo,
            creationflags=creationflags,
        )
        return self._process

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            process = self._start_process()
            if process.stdin is None or process.stdout is None:
                raise RuntimeError("OCR subprocess pipes are unavailable.")
            process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            if not line:
                stderr_output = ""
                if process.stderr is not None:
                    stderr_output = process.stderr.read().strip()
                self.close()
                raise RuntimeError(f"OCR subprocess exited unexpectedly. {stderr_output}".strip())
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                self.close()
                raise
            if not isinstance(response, dict):
                self.close()
                raise RuntimeError("OCR subprocess returned an invalid response payload.")
            if not bool(response.get("ok", False)):
                self.close()
                raise RuntimeError(str(response.get("error", "OCR subprocess request failed.")))
            return response

    def preload(self) -> str:
        response = self._request({"action": "preload"})
        return str(response.get("device", "") or "")

    def predict(self, image_array: np.ndarray, **_kwargs: Any) -> list[dict[str, Any]]:
        root = _package_root()
        temp_dir = root / "runtime" / "tmp" / "ocr_requests"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix="ocr_", suffix=".npy", dir=temp_dir, delete=False) as handle:
                temp_path = handle.name
            np.save(temp_path, np.ascontiguousarray(image_array))
            response = self._request({"action": "predict", "image_path": temp_path})
            result = response.get("result", [])
            return result if isinstance(result, list) else []
        finally:
            if temp_path:
                with contextlib.suppress(OSError):
                    Path(temp_path).unlink()

    def close(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
            if process is None:
                return
            try:
                if process.poll() is None and process.stdin is not None and process.stdout is not None:
                    process.stdin.write(json.dumps({"action": "shutdown"}) + "\n")
                    process.stdin.flush()
            except OSError:
                pass
            with contextlib.suppress(Exception):
                process.terminate()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


def _effective_backend_config(config: dict[str, Any]) -> tuple[str, str]:
    preferred = config.get("backend", config.get("ocr_backend", "auto"))
    installed_state = load_installed_runtime_state()
    installed_backend = str(installed_state.get("backend", "") or "").strip().lower()
    preferred_backend = normalize_ocr_backend(preferred)
    if preferred_backend == "auto" and installed_backend in {"paddle_cuda", "onnx_directml", "cpu"}:
        fallback_from = str(installed_state.get("fallback_from", "") or "").strip().lower()
        if installed_backend == "cpu" and fallback_from == "paddle_cuda":
            detected_backend = select_ocr_backend("auto")
            backend = detected_backend if detected_backend == "paddle_cuda" else installed_backend
        else:
            backend = installed_backend
    else:
        backend = select_ocr_backend(preferred_backend)
    model_tier = str(config.get("model_tier", config.get("modelTier", "auto")) or "auto").strip().lower()
    if model_tier == "auto":
        installed_tier = str(installed_state.get("model_tier", "") or "").strip().lower()
        model_tier = installed_tier or load_runtime_manifest().model_default_tier
    return backend, model_tier


def _effective_execution_mode(config: dict[str, Any]) -> str:
    explicit = normalize_execution_mode(config.get("_execution_mode", config.get("ocr_execution_mode", config.get("execution_mode", "auto"))))
    if explicit != "auto":
        return explicit
    installed_state = load_installed_runtime_state()
    state_mode = str(installed_state.get("execution_mode", "") or "").strip().lower()
    if state_mode in {"inprocess", "subprocess"}:
        return state_mode
    root = _package_root()
    has_ocr_runtime = _has_packaged_ocr_runtime_layout(root)
    if getattr(sys, "frozen", False) and (has_packaged_runtime_layout(root) or has_ocr_runtime):
        return "subprocess"
    manifest_mode = _manifest_execution_mode(root)
    if manifest_mode != "auto" and has_ocr_runtime:
        return manifest_mode
    return "inprocess"


def _create_engine_for_backend(
    backend: str,
    model_tier: str,
    manifest: RuntimeManifest,
    config: dict[str, Any],
) -> Any:
    if backend == "onnx_directml":
        return OnnxDirectMLOcrRuntimeEngine(model_tier=model_tier, manifest=manifest, config=config)
    if backend in {"paddle_cuda", "cpu"}:
        return PaddleOcrRuntimeEngine(backend=backend, model_tier=model_tier, manifest=manifest, config=config)
    raise RuntimeError(f"Unsupported OCR backend: {backend}")


def create_local_runtime_engine(config: dict[str, Any], *, allow_fallback: bool = True) -> Any:
    manifest = load_runtime_manifest()
    backend, model_tier = _effective_backend_config(config)
    engine = _create_engine_for_backend(backend, model_tier, manifest, config)
    if allow_fallback and backend == "paddle_cuda":
        fallback_config = dict(config)
        fallback_config["backend"] = "cpu"
        fallback_config["ocr_backend"] = "cpu"
        return ResilientRuntimeEngine(
            engine,
            primary_backend=backend,
            model_tier=model_tier,
            fallback_factory=lambda: _create_engine_for_backend("cpu", model_tier, manifest, fallback_config),
        )
    return engine


def create_ocr_runtime_engine(config: dict[str, Any]) -> Any:
    execution_mode = _effective_execution_mode(config)
    if execution_mode == "inprocess":
        return create_local_runtime_engine(config)

    manifest = load_runtime_manifest()
    backend, model_tier = _effective_backend_config(config)
    engine = SubprocessOcrRuntimeEngine(backend=backend, model_tier=model_tier, manifest=manifest, config=config)
    if backend == "paddle_cuda":
        fallback_config = dict(config)
        fallback_config["backend"] = "cpu"
        fallback_config["ocr_backend"] = "cpu"
        fallback_config["_execution_mode"] = "subprocess"
        return ResilientRuntimeEngine(
            engine,
            primary_backend=backend,
            model_tier=model_tier,
            fallback_factory=lambda: SubprocessOcrRuntimeEngine(
                backend="cpu",
                model_tier=model_tier,
                manifest=manifest,
                config=fallback_config,
            ),
        )
    return engine


@lru_cache(maxsize=8)
def _cached_runtime_engine(cache_key: str) -> Any:
    return create_ocr_runtime_engine(json.loads(cache_key))


def get_ocr_runtime_engine(config: dict[str, Any]) -> Any:
    relevant = {
        key: value
        for key, value in config.items()
        if key
        in {
            "backend",
            "ocr_backend",
            "execution_mode",
            "ocr_execution_mode",
            "_execution_mode",
            "model_tier",
            "modelTier",
            "lang",
            "use_angle_cls",
            "use_doc_orientation_classify",
            "use_doc_unwarping",
            "ocr_gpu_allocator_strategy",
            "ocr_gpu_memory_fraction",
            "text_detection_model_name",
            "text_recognition_model_name",
            "text_detection_model_dir",
            "text_recognition_model_dir",
        }
    }
    return _cached_runtime_engine(json.dumps(relevant, sort_keys=True, ensure_ascii=False))
