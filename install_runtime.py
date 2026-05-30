from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from vision.ocr_runtime import detect_gpu_environment, load_runtime_manifest, normalize_ocr_backend, select_ocr_backend


def detect_backend() -> str:
    report = detect_gpu_environment()
    if report.gpu_names:
        return "paddle_cuda"
    hardware = None
    try:
        from vision.ocr_runtime import probe_hardware

        hardware = probe_hardware()
    except Exception:
        hardware = None
    if hardware and hardware.directml_available:
        return "onnx_directml"
    return "cpu"


def validate_backend(backend_key: str) -> None:
    if backend_key in {"paddle_cuda", "cpu"}:
        import paddle  # noqa: F401
        import paddleocr  # noqa: F401
        return
    if backend_key == "onnx_directml":
        import onnxruntime  # noqa: F401
        import rapidocr_onnxruntime  # noqa: F401
        return
    raise ValueError(f"Unsupported backend: {backend_key}")


def _backend_packages(manifest: dict[str, object], backend_key: str) -> list[str]:
    backend = manifest.get("backends", {})
    if not isinstance(backend, dict):
        return []
    entry = backend.get(backend_key, {})
    if not isinstance(entry, dict):
        return []
    packages = entry.get("packages", [])
    if not isinstance(packages, list):
        return []
    normalized: list[str] = []
    for item in packages:
        if isinstance(item, str) and item.strip():
            normalized.append(item.strip())
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description="Install OCR runtime.")
    parser.add_argument("--backend", choices=["auto", "paddle_cuda", "onnx_directml", "cpu"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path("runtime_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    backend_key = detect_backend() if args.backend == "auto" else args.backend
    backend_key = select_ocr_backend(normalize_ocr_backend(backend_key), None) if backend_key != "auto" else backend_key
    packages = _backend_packages(manifest, backend_key)

    runtime_dir = Path("runtime")
    runtime_dir.mkdir(exist_ok=True)
    state_path = runtime_dir / "install_state.json"
    status = "dry_run" if args.dry_run else "installed"
    validation = "skipped"
    validation_error = ""
    if packages and not args.dry_run:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])
        try:
            validate_backend(backend_key)
        except Exception as exc:  # pragma: no cover - surfaced to user via CLI
            status = "invalid"
            validation = "failed"
            validation_error = str(exc)
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "backend": backend_key,
                        "status": status,
                        "packages": packages,
                        "message": manifest["backends"][backend_key]["description"],
                        "validation": validation,
                        "validation_error": validation_error,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            raise
        validation = "ok"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": backend_key,
                "status": status,
                "packages": packages,
                "message": manifest["backends"][backend_key]["description"],
                "validation": validation,
                "validation_error": validation_error,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"OCR runtime state wrote {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
