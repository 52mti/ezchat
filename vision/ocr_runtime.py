from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import inspect
import json
import logging
import os
import platform
import queue
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
from collections import deque
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
# GPU 显存兜底值：paddle_cuda 未探测到显存信息时的保守上限。常规路径走
# _adaptive_gpu_memory_fraction()（按卡容量自适应，mobile 模型 ~1GB 即足），
# 不再默认 0.6——0.6 在被桌面程序占用的 8GB 卡上曾引发连环 CUDA OOM，
# 进而静默降级 Paddle-CPU（全屏单次 5s+，20260917 OOM 事故的根因）。
DEFAULT_OCR_GPU_MEMORY_FRACTION = "0.25"
# 每车道显存预算（mobile det+rec 稳态 ~1GB，含 cuDNN workspace 与预热峰值）
# 与自适应 fraction 的上限：budget = min(_GPU_MEM_BUDGET_MB, 总显存×上限)。
_GPU_MEM_BUDGET_MB = 1280
_GPU_MEM_FRACTION_CAP = 0.25
_GPU_MEM_FRACTION_FLOOR = 0.05
# OCR 子进程生命周期留痕：经主程序的 jixiebijiaoben logger 落
# runtime/logs/app_YYYYMMDD.log。客户机"一轮拉起几次/是否每调用一拉"
# 由此可直接数行得出（此前 stderr 只进内存环形缓冲，重启无任何痕迹）。
_OCR_RUNTIME_LOG = logging.getLogger("jixiebijiaoben").getChild("ocr_runtime")
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


def _probe_cache_path() -> Path | None:
    """探测结果磁盘缓存只在打包环境启用（JIXIEBI_PACKAGE_ROOT 已设），
    开发机与测试保持每次真实探测，行为不变。"""
    configured = str(os.environ.get("JIXIEBI_PACKAGE_ROOT", "") or "").strip()
    if not configured:
        return None
    return Path(configured) / "runtime" / "ocr_probe_cache.json"


def _load_cached_probe() -> HardwareProbeResult | None:
    path = _probe_cache_path()
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return None
    if payload.get("platform") != platform.platform():
        return None
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    try:
        return HardwareProbeResult(
            platform=str(result["platform"]),
            gpu_vendor=str(result["gpu_vendor"]),
            gpu_name=str(result["gpu_name"]),
            nvidia_detected=bool(result["nvidia_detected"]),
            directml_available=bool(result["directml_available"]),
            recommended_backend=str(result["recommended_backend"]),
            detail=dict(result.get("detail") or {}),
        )
    except (KeyError, TypeError):
        return None


def _save_cached_probe(result: HardwareProbeResult) -> None:
    path = _probe_cache_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "platform": platform.platform(),
            "result": result.to_dict(),
        }
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        # 缓存写入失败只影响下次启动多探测一次，不影响本次结果。
        pass


@lru_cache(maxsize=1)
def probe_hardware() -> HardwareProbeResult:
    """硬件探测结果磁盘缓存优先：命中后启动不再创建任何探测子进程。

    客户机 pythonw 环境下，GPU/显卡探测子进程（nvidia-smi/wmic）曾被卷入
    0x8001010d 原生崩溃现场；能少起的子进程一律少起。
    """
    cached = _load_cached_probe()
    if cached is not None:
        return cached
    result = _probe_hardware_uncached()
    _save_cached_probe(result)
    return result


def _probe_hardware_uncached() -> HardwareProbeResult:
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
    device_seen: str = ""

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
        # det 输入分辨率限长：检测耗时随输入面积近似平方增长，全屏竖屏图
        # 限到 960/max 实测省 ~20%（3060 Ti 246→195ms），识别条目数不变。
        # 配置 text_det_limit_side_len=0 可关闭。
        limit_side = int(self.config.get("text_det_limit_side_len", 960) or 0)
        if limit_side > 0:
            kwargs["text_det_limit_side_len"] = limit_side
            kwargs["text_det_limit_type"] = str(self.config.get("text_det_limit_type", "max") or "max")
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
        # 记录实际落到的设备，供运行观测日志读取（backend 标注的金标准）。
        self.device_seen = device
        return device

    def predict(self, image_array: np.ndarray, **_kwargs: Any) -> list[dict[str, Any]]:
        engine, device = self._get_engine()
        self.device_seen = device
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


def _engine_runtime_names(backend: str) -> tuple[str, ...]:
    """后端对应的引擎目录候选（先序优先）。

    cpu 自 20260917 起切 ORT（CPU EP），复用 engine/ocr_directml 的
    onnxruntime-directml 运行时；engine/ocr_cpu（paddle CPU，1.1GB）已
    从全量包移除，仅作为旧安装目录的兜底候选保留。
    """
    normalized = str(backend or "").strip().lower()
    if normalized == "cpu":
        return ("ocr_directml", "ocr_cpu")
    if normalized == "onnx_directml":
        return ("ocr_directml",)
    if normalized == "paddle_cuda":
        return ("ocr_nvidia",)
    return (backend_runtime_name(normalized),)


def _resolve_bundled_rapid_onnx_models() -> dict[str, str] | None:
    """随包分发的 v5 onnx 模型（models/ocr/rapid_onnx_models/，与 GPU 路径
    同源 PP-OCRv5 mobile，识别结果一致）。模型在更新器保护区（models/ 前缀），
    只随全量包走、不进 update.zip；缺失时回退 rapidocr 内置模型（v3 时代）。"""
    root = _package_root() / "models" / "ocr" / "rapid_onnx_models"
    det = root / "PP-OCRv5_mobile_det.onnx"
    rec = root / "PP-OCRv5_mobile_rec.onnx"
    dict_path = root / "PP-OCRv5_rec_dict.txt"
    if det.is_file() and rec.is_file():
        payload = {"det": str(det), "rec": str(rec)}
        if dict_path.is_file():
            payload["dict"] = str(dict_path)
        return payload
    return None


class OnnxDirectMLOcrRuntimeEngine:
    """ORT/RapidOCR 引擎：onnx_directml（DML EP）与 cpu（CPU EP）共用。

    20260917 两项修正：
    1. rapidocr_onnxruntime 1.2.3 的 OrtInferSession 只有 CPU/CUDA 分支，
       构造参数里的 providers 会被静默吞掉——"DirectML 后端"此前从未真正
       用过 GPU。现改为构造期临时替换其会话工厂，强制注入 EP 列表。
    2. backend=cpu 不再走 paddlepaddle（Windows CPU 推理实测比 ORT 慢
       ~6 倍：全屏 5.4s vs 0.9s），改用本引擎 CPU EP。
    优先加载包内 v5 onnx 模型（与 GPU 路径同源，识别一致）；缺失时回退
    rapidocr 内置模型（v3 时代，角标数字偶有差异）。
    """

    device_seen: str = ""

    def __init__(self, *, backend: str = "onnx_directml", model_tier: str, manifest: RuntimeManifest, config: dict[str, Any]) -> None:
        self.backend = "onnx_directml" if backend == "onnx_directml" else "cpu"
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

    def _execution_providers(self) -> list[str]:
        if self.backend == "cpu":
            return ["CPUExecutionProvider"]
        return ["DmlExecutionProvider", "CPUExecutionProvider"]

    def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        with _DIRECTML_ENGINE_INIT_LOCK:
            if self._engine is not None:
                return self._engine
            try:
                import onnxruntime as ort  # type: ignore
                import rapidocr_onnxruntime.utils as rapid_utils  # type: ignore
                from rapidocr_onnxruntime import RapidOCR  # type: ignore
            except ImportError as exc:
                raise RuntimeError("Missing packaged ORT OCR runtime (engine/ocr_directml). Rebuild the full package.") from exc
            providers = self._execution_providers()
            if "DmlExecutionProvider" in providers and "DmlExecutionProvider" not in list(ort.get_available_providers()):
                raise RuntimeError(f"onnxruntime does not expose the DirectML provider. Providers: {ort.get_available_providers()}")
            kwargs: dict[str, Any] = dict(self.config.get("rapidocr_kwargs") or {})
            # rapid 默认 det 限长 min/736 会把竖屏图"放大"（564→736 短边，
            # det 输入像素 ×2.6）；对齐 paddle 路径的 max/960 策略，CPU EP
            # 全屏实测 1320→853ms（-35%）且文本零差异。配置
            # det_limit_side_len=0 可关闭。
            limit_side = int(self.config.get("det_limit_side_len", 960) or 0)
            if limit_side > 0:
                kwargs.setdefault("det_limit_type", str(self.config.get("det_limit_type", "max") or "max"))
                kwargs.setdefault("det_limit_side_len", limit_side)
            onnx = _resolve_bundled_rapid_onnx_models()
            if onnx is not None:
                kwargs.setdefault("det_model_path", onnx["det"])
                kwargs.setdefault("rec_model_path", onnx["rec"])
                if onnx.get("dict"):
                    kwargs.setdefault("rec_dict_path", onnx["dict"])
            # 构造期补丁：OrtInferSession 不读 providers 配置，临时替换其
            # 会话工厂强制注入 EP（构造完立即还原，只在 init 锁内发生）。
            real_session = rapid_utils.InferenceSession
            requested_providers = providers

            def _session_with_providers(path_or_bytes, sess_options=None, providers=None, **session_kwargs):
                return real_session(path_or_bytes, sess_options=sess_options, providers=requested_providers, **session_kwargs)

            rapid_utils.InferenceSession = _session_with_providers
            try:
                self._engine = self._instantiate_rapidocr(RapidOCR, kwargs)
            finally:
                rapid_utils.InferenceSession = real_session
            return self._engine

    def preload(self) -> str:
        self._get_engine()
        self.device_seen = "cpu" if self.backend == "cpu" else "dml:0"
        return self.device_seen

    def predict(self, image_array: np.ndarray, **_kwargs: Any) -> list[dict[str, Any]]:
        engine = self._get_engine()
        with _ENGINE_PREDICT_LOCK:
            result = engine(np.ascontiguousarray(image_array))
        if isinstance(result, tuple):
            result = result[0]
        return _normalize_rapidocr_results(result)


class ResilientRuntimeEngine:
    @property
    def device_seen(self) -> str:
        # 透传实际生效引擎（主或 fallback）观测到的设备，供运行日志读取。
        engine = self._fallback_engine or self._primary_engine
        return str(getattr(engine, "device_seen", "") or "")

    @property
    def last_timings(self) -> dict[str, float]:
        # 透传实际生效引擎的分段耗时（解码/纯推理 ms）。缺这个属性时
        # getattr(engine, "last_timings") 在包装层上恒为 None，统计行的
        # "子进程纯推理"字段永远出不来（曾长期掩盖真实推理水位）。
        engine = self._fallback_engine or self._primary_engine
        timings = getattr(engine, "last_timings", None)
        return dict(timings) if isinstance(timings, dict) else {}

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


def _frame_write(stream, payload: bytes, blob: bytes = b"") -> None:
    """二进制帧协议：[8B 头: json_len, bin_len][json][blob]。

    图像以原始字节直走管道，不再 base64 塞进 JSON 行——3.6MB 字符串的
    json.dumps/loads 与分块行传输实测占稳态单次耗时的一半以上。
    """
    stream.write(struct.pack("<II", len(payload), len(blob)))
    if payload:
        stream.write(payload)
    if blob:
        stream.write(blob)
    stream.flush()


def _read_exact(stream, count: int) -> bytes:
    """从管道读满 count 字节（os.read 直读文件描述符）。

    BufferedReader 的缓冲层在读端逐 8KB 填充且小读触发整块底层读，
    响应方向实测引入数百毫秒级延迟；os.read 直读避免缓冲层二次调度。
    """
    chunks: list[bytes] = []
    remaining = count
    fd = stream.fileno()
    while remaining > 0:
        chunk = os.read(fd, remaining)
        if not chunk:
            raise EOFError("pipe closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class SubprocessOcrRuntimeEngine:
    # 子进程单次请求的等待上限：CPU 全图 OCR 可达数十秒，取宽裕值；
    # 超时杀进程重启（模型重载），远好于无限挂死拖住全局引擎锁。
    RESPONSE_TIMEOUT_SECONDS = 120.0
    STDERR_TAIL_LINES = 30
    # 共享内存传图：跨进程管道逐块搬运实测 ~250ms/MB（同进程仅 3.5ms/2.7MB，
    # 慢在进程边界的内核调度），SHM 一次内存拷贝替代。块大小取对话
    # scale=2 全幅上限再留余量；超限或 SHM 创建失败回退 blob 直传。
    SHM_CAPACITY_BYTES = 16 * 1024 * 1024

    def __init__(self, *, backend: str, model_tier: str, manifest: RuntimeManifest, config: dict[str, Any]) -> None:
        self.backend = backend
        self.model_tier = model_tier
        self.manifest = manifest
        self.config = dict(config)
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.RLock()
        self.device_seen: str = ""
        self.last_timings: dict[str, float] = {}
        self._shm: Any = None
        self._shm_failed = False
        self._response_queue: "queue.Queue[bytes | None]" = queue.Queue()
        self._stderr_tail: "deque[str]" = deque(maxlen=self.STDERR_TAIL_LINES)
        self._spawn_count = 0
        # OOM 阶梯降级：CUDA OOM 时逐次减半 fraction 重拉子进程（至多
        # _OOM_FRACTION_RETRIES 次），仍失败才交给 Resilient 落 CPU(ORT)——
        # 不再一步跳崖到慢引擎。
        self._gpu_memory_fraction_override: str | None = None
        self._oom_fraction_retries = 0
        self._OOM_FRACTION_RETRIES = 2

    def _runtime_python_path(self) -> Path:
        root = _package_root()
        # cpu 后端 20260917 起复用 ocr_directml 运行时跑 ORT CPU EP；
        # 旧安装残留的 engine/ocr_cpu 仅作兜底候选（新全量包已不再收录）。
        runtime_names = _engine_runtime_names(self.backend)
        for runtime_name in runtime_names:
            candidate = runtime_python(root, runtime_name, gui=False)
            if candidate.is_file():
                return candidate
        state_python = _state_runtime_python(load_installed_runtime_state(root), self.backend)
        if state_python is not None:
            return state_python
        if not getattr(sys, "frozen", False):
            # 开发机源码运行：engine/ 运行时不随仓库，主进程解释器本身
            # 装有 paddle（inprocess 模式同依赖），直接复用，保证子进程
            # 路径在开发机上可跑（否则与客户机行为脱节，优化无法本地复现）。
            return Path(sys.executable)
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
            "text_det_limit_side_len",
            "text_det_limit_type",
            "det_limit_side_len",
            "det_limit_type",
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
        # 入口优先 .py（源码/全量布局）：根目录若残留旧 .pyc，直接执行会
        # 跑过期代码；打包引擎桥（仅 .pyc）布局再回退 .pyc。
        entry_path = root / "ocr_subprocess_entry.py"
        if not entry_path.is_file():
            entry_path = root / "ocr_subprocess_entry.pyc"
        command = [
            str(python_exe),
            "-s",
            "-u",
            str(entry_path),
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
        env["FLAGS_fraction_of_gpu_memory_to_use"] = self._current_gpu_memory_fraction()
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
            env=env,
            startupinfo=startupinfo,
            creationflags=creationflags,
        )
        self._spawn_count += 1
        _OCR_RUNTIME_LOG.info(
            "[ocr-subprocess] 拉起 #%d backend=%s tier=%s pid=%s",
            self._spawn_count,
            self.backend,
            self.model_tier,
            getattr(self._process, "pid", "?"),
        )
        # 每代进程配全新响应队列，避免旧进程残留响应污染新会话。
        self._response_queue = queue.Queue()
        self._start_pipe_workers(self._process)
        return self._process

    def _current_gpu_memory_fraction(self) -> str:
        # 优先级：OOM 阶梯覆盖值 > 显式配置 > 按卡自适应 > 兜底常量。
        if self._gpu_memory_fraction_override:
            return self._gpu_memory_fraction_override
        configured = str(self.config.get("ocr_gpu_memory_fraction", "") or "").strip()
        if configured:
            return configured
        if self.backend == "paddle_cuda":
            return _adaptive_gpu_memory_fraction()
        return DEFAULT_OCR_GPU_MEMORY_FRACTION

    def _start_pipe_workers(self, process: subprocess.Popen[bytes]) -> None:
        """stdout reader + stderr 排空两个常驻 daemon 线程。

        Paddle/PaddleX 运行时会向 stderr 打日志；Windows 管道缓冲仅
        4-8KB，无人消费时子进程会阻塞在 stderr 写上、不再产出 stdout
        响应，主进程 readline 挂死且持有全局引擎锁——所有设备 OCR
        一起排队停摆。排空线程持续消费 stderr（保留尾部若干行供报错），
        reader 线程让 _request 可以带超时等待。
        """

        def read_stdout() -> None:
            try:
                while True:
                    if process.stdout is None:
                        break
                    header = _read_exact(process.stdout, 8)
                    json_len, blob_len = struct.unpack("<II", header)
                    json_bytes = _read_exact(process.stdout, json_len)
                    if blob_len:
                        _read_exact(process.stdout, blob_len)  # 响应不带 blob，防御性丢弃
                    self._response_queue.put(json_bytes)
            except Exception:
                pass
            finally:
                self._response_queue.put(None)

        def drain_stderr() -> None:
            try:
                while True:
                    raw = process.stderr.readline() if process.stderr else b""
                    if not raw:
                        break
                    self._stderr_tail.append(raw.decode("utf-8", "replace"))
            except Exception:
                pass

        threading.Thread(target=read_stdout, name="ocr-sub-stdout", daemon=True).start()
        threading.Thread(target=drain_stderr, name="ocr-sub-stderr", daemon=True).start()

    def _request(self, payload: dict[str, Any], blob: bytes = b"") -> dict[str, Any]:
        # OOM 阶梯：CUDA OOM 时减半 fraction 重拉子进程重试；仍失败才抛给
        # 上层（Resilient 落 CPU——现已是 ORT，不再是慢速 Paddle-CPU）。
        try:
            return self._request_once(payload, blob=blob)
        except Exception as exc:
            if not self._prepare_oom_retry(str(exc)):
                raise
            return self._request_once(payload, blob=blob)

    def _prepare_oom_retry(self, error_text: str) -> bool:
        lowered = str(error_text or "").lower()
        if self.backend != "paddle_cuda" or "out of memory" not in lowered and "cuda error" not in lowered:
            return False
        if self._oom_fraction_retries >= self._OOM_FRACTION_RETRIES:
            return False
        try:
            current = float(self._current_gpu_memory_fraction())
        except (TypeError, ValueError):
            return False
        halved = max(_GPU_MEM_FRACTION_FLOOR, current / 2.0)
        self._gpu_memory_fraction_override = f"{halved:.3f}"
        self._oom_fraction_retries += 1
        self.close()
        _OCR_RUNTIME_LOG.warning(
            "[ocr-subprocess] CUDA OOM，显存 fraction %.3f→%.3f 重拉重试（第 %d 次）",
            current,
            halved,
            self._oom_fraction_retries,
        )
        return True

    def _request_once(self, payload: dict[str, Any], blob: bytes = b"") -> dict[str, Any]:
        with self._lock:
            process = self._start_process()
            if process.stdin is None:
                raise RuntimeError("OCR subprocess pipes are unavailable.")
            _frame_write(process.stdin, json.dumps(payload, ensure_ascii=False).encode("utf-8"), blob)
            try:
                line = self._response_queue.get(timeout=self.RESPONSE_TIMEOUT_SECONDS)
            except queue.Empty:
                # 不再无限挂死：杀进程（下次调用自动重启重载模型），
                # 全局引擎锁随之释放，其他设备可继续。
                tail = "".join(list(self._stderr_tail)[-5:]).strip()
                _OCR_RUNTIME_LOG.error(
                    "[ocr-subprocess] 响应超时 %.0fs（第 %d 代），关闭待重拉；stderr 尾部: %s",
                    self.RESPONSE_TIMEOUT_SECONDS,
                    self._spawn_count,
                    tail[:300],
                )
                self.close()
                raise TimeoutError(
                    f"OCR subprocess did not respond within {self.RESPONSE_TIMEOUT_SECONDS:.0f}s. {tail}"
                ) from None
            if line is None:
                stderr_output = "".join(list(self._stderr_tail)[-10:]).strip()
                _OCR_RUNTIME_LOG.error(
                    "[ocr-subprocess] 子进程意外退出（第 %d 代），关闭待重拉；stderr 尾部: %s",
                    self._spawn_count,
                    stderr_output[:300],
                )
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
                error_text = str(response.get("error", "OCR subprocess request failed."))
                _OCR_RUNTIME_LOG.error(
                    "[ocr-subprocess] 请求报错（第 %d 代），关闭待重拉: %s",
                    self._spawn_count,
                    error_text[:300],
                )
                self.close()
                raise RuntimeError(error_text)
            # predict 响应顺带实际设备与子进程内部分段耗时，记录供观测。
            device_seen = str(response.get("device", "") or "").strip()
            if device_seen:
                self.device_seen = device_seen
            timings = response.get("timings")
            if isinstance(timings, dict):
                self.last_timings = dict(timings)
            return response

    def preload(self) -> str:
        response = self._request({"action": "preload"})
        device = str(response.get("device", "") or "")
        # 记录子进程实际生效的设备，供运行观测日志读取。
        self.device_seen = device
        return device

    def predict(self, image_array: np.ndarray, **_kwargs: Any) -> list[dict[str, Any]]:
        # 图像走共享内存：主进程写入复用块（一次内存拷贝），帧里只传块名
        # 与字节数——跨进程管道逐块搬运的 ~250ms/MB 调度税整体绕开。
        # 超容量或 SHM 不可用时回退 blob 直传（帧协议不变，两端兼容）。
        contiguous = np.ascontiguousarray(image_array)
        payload = {
            "action": "predict",
            "shape": [int(dim) for dim in contiguous.shape],
        }
        blob = contiguous.tobytes()
        shm = self._usable_shm(len(blob))
        if shm is not None:
            shm.buf[: len(blob)] = blob
            payload["shm_name"] = shm.name
            payload["size"] = len(blob)
            blob = b""
        response = self._request(payload, blob=blob)
        result = response.get("result", [])
        return result if isinstance(result, list) else []

    def _usable_shm(self, size: int) -> Any:
        if self._shm_failed or size > self.SHM_CAPACITY_BYTES:
            return None
        if self._shm is not None:
            return self._shm
        try:
            from multiprocessing import shared_memory

            # 名字带主进程 pid：主进程退出后 Windows 句柄机制自动回收，
            # 不依赖 unlink；子进程重启后按名重新 attach 即可。
            self._shm = shared_memory.SharedMemory(
                create=True,
                size=self.SHM_CAPACITY_BYTES,
                name=f"jixiebi_ocr_{os.getpid()}",
            )
            return self._shm
        except Exception:
            self._shm_failed = True
            return None

    def close(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
            shm = self._shm
            self._shm = None
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
            if process is None:
                return
            try:
                if process.poll() is None and process.stdin is not None and process.stdout is not None:
                    _frame_write(process.stdin, json.dumps({"action": "shutdown"}).encode("utf-8"))
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
    if manifest_mode != "auto" and (has_ocr_runtime or not getattr(sys, "frozen", False)):
        # 源码运行（开发机）没有 engine/ 目录也要跟随 manifest：否则开发机
        # 退回 inprocess，与客户机的子进程路径不一致，性能与观测字段都
        # 无法在开发机上复现。
        return manifest_mode
    return "inprocess"


def _create_engine_for_backend(
    backend: str,
    model_tier: str,
    manifest: RuntimeManifest,
    config: dict[str, Any],
) -> Any:
    if backend in {"onnx_directml", "cpu"}:
        return OnnxDirectMLOcrRuntimeEngine(backend=backend, model_tier=model_tier, manifest=manifest, config=config)
    if backend == "paddle_cuda":
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


_RUNTIME_ENGINE_RELEVANT_KEYS = {
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
    "text_det_limit_side_len",
    "text_det_limit_type",
    "det_limit_side_len",
    "det_limit_type",
}

# 引擎实例池（车道模型）：GPU 每实例 ~1GB（CUDA 上下文是大头），池大小
# 与设备数解耦；超过池大小的调用方在并发槽排队（车道模型）。CPU/DirectML
# 多实例只会互相抢核，恒为 1。
#
# 车道数受两重钳制：空闲显存自适应上限（adaptive_engine_pool_cap，防
# 连环 CUDA OOM）与并发槽硬顶 16。20260917 曾短暂取消显存钳制，当天
# 即在 8GB 卡上连环 OOM→静默降级 Paddle-CPU（单次 4s+），随即恢复。
# ocr_engine_pool_ignore_vram_cap=true 可绕过显存钳制（仅调试）。
_DEFAULT_GPU_ENGINE_POOL_SIZE = 2
_FALLBACK_GPU_ENGINE_POOL_CAP = 4
_ABSOLUTE_MAX_ENGINE_POOL_SIZE = 16  # 配置硬顶：并发槽实现的自然天花板
_ADAPTIVE_HINT_MAX_ENGINE_POOL = 8  # 自适应上限的提示封顶：再多只添噪（子进程开销）
_ENGINE_INSTANCE_MB = 1280
_ENGINE_POOL_VRAM_RESERVE_MB = 1024
_POOLED_ENGINES: dict[str, list[Any]] = {}
_POOLED_ENGINES_LOCK = threading.Lock()
_ADAPTIVE_POOL_CAP_CACHE: int | None = None


def _probe_gpu_free_mb() -> int | None:
    """nvidia-smi 读空闲显存（MB）；失败/无 N 卡返回 None。"""
    result = _run_command(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        timeout_seconds=5,
    )
    if result is None or not result.stdout:
        return None
    try:
        return int(str(result.stdout).strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def _probe_gpu_total_mb() -> int | None:
    """nvidia-smi 读显卡总显存（MB）；失败/无 N 卡返回 None。"""
    result = _run_command(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        timeout_seconds=5,
    )
    if result is None or not result.stdout:
        return None
    try:
        return int(str(result.stdout).strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


@lru_cache(maxsize=1)
def _adaptive_gpu_memory_fraction() -> str:
    """按显卡总显存自适应的每进程显存上限（FLAGS_fraction_of_gpu_memory_to_use）。

    mobile det+rec 实测稳态 ~1GB/车道，给到 _GPU_MEM_BUDGET_MB 再封顶 25%：
    8GB 卡 ≈0.16、4GB 卡 ≈0.25（1GB）。旧默认 0.6 允许单进程吃 4.8GB，
    在被桌面程序占用的卡上连环触发 CUDA OOM（20260917 事故），废弃。
    探测失败回退 DEFAULT_OCR_GPU_MEMORY_FRACTION。配置键
    ocr_gpu_memory_fraction 仍可显式覆盖。
    """
    total_mb = _probe_gpu_total_mb()
    if not total_mb or total_mb <= 0:
        return DEFAULT_OCR_GPU_MEMORY_FRACTION
    budget_mb = min(_GPU_MEM_BUDGET_MB, total_mb * _GPU_MEM_FRACTION_CAP)
    fraction = max(_GPU_MEM_FRACTION_FLOOR, min(_GPU_MEM_FRACTION_CAP, budget_mb / total_mb))
    return f"{fraction:.3f}"


def adaptive_engine_pool_cap() -> int:
    """按空闲显存估算的车道数上限（钳制配置；缓存一次）。

    20260917 起恢复钳制（曾短暂取消，当天即连环 OOM）：空闲显存扣除
    安全余量后按每车道预算折算，避免配置车道数超出显存放不下。
    ocr_engine_pool_ignore_vram_cap=true 可显式绕过（调试用）。
    """
    global _ADAPTIVE_POOL_CAP_CACHE
    if _ADAPTIVE_POOL_CAP_CACHE is None:
        free_mb = _probe_gpu_free_mb()
        if free_mb is None or free_mb <= 0:
            _ADAPTIVE_POOL_CAP_CACHE = _FALLBACK_GPU_ENGINE_POOL_CAP
        else:
            budget = free_mb - _ENGINE_POOL_VRAM_RESERVE_MB
            cap = budget // _ENGINE_INSTANCE_MB if budget > 0 else 0
            _ADAPTIVE_POOL_CAP_CACHE = min(
                _ADAPTIVE_HINT_MAX_ENGINE_POOL, max(1, int(cap))
            )
    return _ADAPTIVE_POOL_CAP_CACHE


def _engine_pool_size(config: dict[str, Any], backend: str) -> int:
    if backend != "paddle_cuda":
        return 1
    requested = _DEFAULT_GPU_ENGINE_POOL_SIZE
    for key in ("ocr_engine_pool_size", "engine_pool_size"):
        if key in config:
            try:
                value = int(config.get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            if value >= 1:
                requested = value
                break
    size = max(1, min(requested, _ABSOLUTE_MAX_ENGINE_POOL_SIZE))
    if bool(config.get("ocr_engine_pool_ignore_vram_cap", False)):
        return size
    cap = adaptive_engine_pool_cap()
    if size > cap:
        _OCR_RUNTIME_LOG.warning(
            "[ocr-pool] 配置车道数 %d 超出空闲显存自适应上限 %d，已钳制"
            "（ocr_engine_pool_ignore_vram_cap=true 可绕过）",
            size,
            cap,
        )
        size = max(1, cap)
    return size


def engine_pool_size_for(config: dict[str, Any]) -> int:
    """按配置与实际后端解析引擎池大小（并发槽的默认车道数同源于此）。"""
    backend, _model_tier = _effective_backend_config(config)
    return _engine_pool_size(config, backend)


def get_pooled_runtime_engine(config: dict[str, Any], *, lane: int = 0) -> Any:
    """按车道取引擎实例：车道=并发槽位，同一车道恒复用同一实例（子进程
    常驻），不同车道各自独立实例并行推理。

    lane>0 实例创建失败（显存不足等）自动降级共享 0 号实例——子进程桥
    内部自带锁，共享只损失并行度、不损失正确性，绝不让池问题阻断 OCR。
    """
    relevant = {key: value for key, value in config.items() if key in _RUNTIME_ENGINE_RELEVANT_KEYS}
    backend, _model_tier = _effective_backend_config(config)
    size = _engine_pool_size(config, backend)
    cache_key = json.dumps(relevant, sort_keys=True, ensure_ascii=False) + f"|pool{size}"
    with _POOLED_ENGINES_LOCK:
        instances = _POOLED_ENGINES.setdefault(cache_key, [])
        index = max(0, min(int(lane), size - 1))
        while len(instances) <= index:
            instances.append(None)
        if instances[index] is None:
            engine_config = json.loads(cache_key.split("|pool")[0])
            try:
                instances[index] = create_ocr_runtime_engine(engine_config)
            except Exception as exc:
                if index == 0:
                    raise
                _OCR_RUNTIME_LOG.error(
                    "[ocr-pool] 实例 #%d 创建失败，降级共享 #0: %s", index, exc
                )
                if instances[0] is None:
                    instances[0] = create_ocr_runtime_engine(engine_config)
                instances[index] = instances[0]
        return instances[index]


def get_ocr_runtime_engine(config: dict[str, Any]) -> Any:
    relevant = {
        key: value
        for key, value in config.items()
        if key in _RUNTIME_ENGINE_RELEVANT_KEYS
    }
    return get_pooled_runtime_engine(config)
