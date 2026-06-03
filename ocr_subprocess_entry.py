from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _write_response(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OCR subprocess bridge entry.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model-tier", required=True)
    parser.add_argument("--config-json", default="{}")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from vision.ocr_runtime import create_local_runtime_engine, _serialize_prediction_payload

    try:
        config = json.loads(args.config_json)
    except json.JSONDecodeError as exc:
        _write_response({"ok": False, "error": f"Invalid config JSON: {exc}"})
        return 1
    if not isinstance(config, dict):
        _write_response({"ok": False, "error": "Config payload must be a JSON object."})
        return 1
    config = dict(config)
    config["backend"] = args.backend
    config["model_tier"] = args.model_tier
    config["_execution_mode"] = "inprocess"

    try:
        engine = create_local_runtime_engine(config)
    except Exception as exc:
        _write_response({"ok": False, "error": str(exc)})
        return 1

    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            _write_response({"ok": False, "error": f"Invalid request JSON: {exc}"})
            continue
        if not isinstance(payload, dict):
            _write_response({"ok": False, "error": "Request payload must be a JSON object."})
            continue
        action = str(payload.get("action", "") or "").strip().lower()
        try:
            if action == "shutdown":
                _write_response({"ok": True, "device": ""})
                return 0
            if action == "preload":
                _write_response({"ok": True, "device": str(engine.preload() or "")})
                continue
            if action != "predict":
                raise RuntimeError(f"Unsupported OCR subprocess action: {action}")
            image_path = str(payload.get("image_path", "") or "").strip()
            if not image_path:
                raise RuntimeError("Missing image_path for OCR subprocess predict request.")
            image_array = np.load(image_path, allow_pickle=False)
            result = engine.predict(np.ascontiguousarray(image_array))
            _write_response({"ok": True, "result": _serialize_prediction_payload(result)})
        except Exception as exc:
            _write_response({"ok": False, "error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
