from __future__ import annotations

import contextlib
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


PADDLE_VERSION = "3.2.2"
PADDLEOCR_VERSION = "3.4.0"
PADDLEX_VERSION = "3.4.3"
DEFAULT_MODEL_TIER = "lite"
SUPPORTED_BACKENDS = ("auto", "paddle_cuda", "onnx_directml", "cpu")
CUDA_LANE_INDEXES = {
    "cu118": "https://www.paddlepaddle.org.cn/packages/stable/cu118/",
    "cu126": "https://www.paddlepaddle.org.cn/packages/stable/cu126/",
    "cu129": "https://www.paddlepaddle.org.cn/packages/stable/cu129/",
}


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
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
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


def _normalize_major_minor(raw_version: str | None) -> tuple[int, int] | None:
    if not raw_version:
        return None
    match = re.match(r"^\s*(\d+)(?:\.(\d+))?", raw_version)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def choose_cuda_lane(nvcc_version: str | None, driver_cuda_version: str | None) -> tuple[str | None, str]:
    preferred = _normalize_major_minor(nvcc_version) or _normalize_major_minor(driver_cuda_version)
    if preferred is None:
        return None, "未检测到 CUDA Toolkit 或显卡驱动 CUDA 版本。"
    major, minor = preferred
    source = "nvcc" if nvcc_version else "nvidia-smi"
    if major > 12 or (major == 12 and minor >= 9):
        return "cu129", f"从 {source} 检测到 CUDA {major}.{minor}，选择 Paddle cu129。"
    if major == 12 and minor >= 6:
        return "cu126", f"从 {source} 检测到 CUDA {major}.{minor}，选择 Paddle cu126。"
    if (major == 11 and minor >= 8) or major > 11:
        return "cu118", f"从 {source} 检测到 CUDA {major}.{minor}，选择 Paddle cu118。"
    return None, f"检测到 CUDA {major}.{minor}，低于 Paddle 当前支持的 CUDA 11.8。"


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
    lane, reason = choose_cuda_lane(nvcc_version, driver_cuda_version)
    return GpuEnvironmentReport(
        platform=platform.platform(),
        python_version=platform.python_version(),
        python_bits=platform.architecture()[0],
        nvidia_smi_found=query is not None and query.returncode == 0,
        gpu_names=gpu_names,
        driver_version=driver_version,
        driver_cuda_version=driver_cuda_version,
        nvcc_version=nvcc_version,
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
        raise RuntimeError("OCR 后端 paddle_cuda 需要 NVIDIA 显卡。")
    if normalized == "onnx_directml" and not hardware.directml_available:
        raise RuntimeError("OCR 后端 onnx_directml 需要 Windows AMD/Intel 显卡和 DirectML。")
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


def validate_paddle_gpu_runtime() -> GpuEnvironmentReport:
    report = detect_gpu_environment()
    if not report.gpu_names:
        raise RuntimeError("未检测到可用 NVIDIA 显卡，无法使用 paddle_cuda OCR 后端。")
    try:
        import paddle
    except ImportError as exc:
        raise RuntimeError("当前 Python 环境没有安装 Paddle GPU 运行时。") from exc
    report.paddle_version = getattr(paddle, "__version__", None)
    report.paddle_compiled_with_cuda = bool(paddle.is_compiled_with_cuda())
    if not report.paddle_compiled_with_cuda:
        raise RuntimeError(f"当前 Paddle {report.paddle_version or '<unknown>'} 不是 GPU 版本。")
    report.paddle_cuda_device_count = int(paddle.device.cuda.device_count())
    if report.paddle_cuda_device_count < 1:
        raise RuntimeError("Paddle GPU 已安装，但没有枚举到可用 CUDA 设备。")
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
_PADDLE_OCR_BY_CACHE_KEY: dict[tuple[str, str, str, str], Any] = {}


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
        kwargs: dict[str, Any] = {
            "lang": self.config.get("lang", "ch"),
            "use_doc_orientation_classify": bool(self.config.get("use_doc_orientation_classify", False)),
            "use_doc_unwarping": bool(self.config.get("use_doc_unwarping", False)),
            "use_textline_orientation": bool(self.config.get("use_angle_cls", False)),
            "device": device,
        }
        kwargs["text_detection_model_name"] = self.config.get("text_detection_model_name") or model_spec.det_model_name
        kwargs["text_recognition_model_name"] = self.config.get("text_recognition_model_name") or model_spec.rec_model_name
        for source_key, target_key in (
            ("text_detection_model_dir", "text_detection_model_dir"),
            ("text_recognition_model_dir", "text_recognition_model_dir"),
        ):
            if self.config.get(source_key):
                kwargs[target_key] = self.config[source_key]
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
                from paddleocr import PaddleOCR  # type: ignore
            except ImportError as exc:
                raise RuntimeError("缺少 paddleocr，请先运行 `python install_runtime.py --backend cpu|paddle_cuda`。") from exc
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
                raise RuntimeError("缺少 DirectML OCR 依赖，请先运行 `python install_runtime.py --backend onnx_directml`。") from exc
            providers = list(ort.get_available_providers())
            if "DmlExecutionProvider" not in providers:
                raise RuntimeError(f"当前 onnxruntime 没有 DirectML Provider，可用 Provider: {providers}")
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


def _effective_backend_config(config: dict[str, Any]) -> tuple[str, str]:
    preferred = config.get("backend", config.get("ocr_backend", "auto"))
    installed_state = load_installed_runtime_state()
    installed_backend = str(installed_state.get("backend", "") or "").strip().lower()
    if normalize_ocr_backend(preferred) == "auto" and installed_backend in {"paddle_cuda", "onnx_directml", "cpu"}:
        backend = installed_backend
    else:
        backend = select_ocr_backend(preferred)
    model_tier = str(config.get("model_tier", config.get("modelTier", "auto")) or "auto").strip().lower()
    if model_tier == "auto":
        installed_tier = str(installed_state.get("model_tier", "") or "").strip().lower()
        model_tier = installed_tier or load_runtime_manifest().model_default_tier
    return backend, model_tier


def create_ocr_runtime_engine(config: dict[str, Any]) -> Any:
    manifest = load_runtime_manifest()
    backend, model_tier = _effective_backend_config(config)
    if backend == "onnx_directml":
        return OnnxDirectMLOcrRuntimeEngine(model_tier=model_tier, manifest=manifest, config=config)
    if backend in {"paddle_cuda", "cpu"}:
        return PaddleOcrRuntimeEngine(backend=backend, model_tier=model_tier, manifest=manifest, config=config)
    raise RuntimeError(f"Unsupported OCR backend: {backend}")


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
            "model_tier",
            "modelTier",
            "lang",
            "use_angle_cls",
            "use_doc_orientation_classify",
            "use_doc_unwarping",
            "text_detection_model_name",
            "text_recognition_model_name",
            "text_detection_model_dir",
            "text_recognition_model_dir",
        }
    }
    return _cached_runtime_engine(json.dumps(relevant, sort_keys=True, ensure_ascii=False))
