from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def detect_backend() -> str:
    if shutil.which("nvidia-smi"):
        return "paddle_cuda"
    return "cpu"


def validate_backend(backend_key: str) -> None:
    if backend_key in {"paddle_cuda", "cpu"}:
        importlib.import_module("paddleocr")
        importlib.import_module("paddle")
        return
    if backend_key == "onnx_directml":
        importlib.import_module("onnxruntime")
        importlib.import_module("rapidocr_onnxruntime")
        return
    raise ValueError(f"Unsupported backend: {backend_key}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install OCR runtime skeleton.")
    parser.add_argument("--backend", choices=["auto", "paddle_cuda", "onnx_directml", "cpu"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path("runtime_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    backend_key = detect_backend() if args.backend == "auto" else args.backend
    backend = manifest["backends"][backend_key]
    packages = list(backend.get("packages", []))

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
                        "message": backend["description"],
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
                "message": backend["description"],
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
