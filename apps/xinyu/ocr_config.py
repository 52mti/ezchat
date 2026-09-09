from __future__ import annotations

import copy
import json
import os
from pathlib import Path


def _load_mode_config(filename: str) -> dict:
    root = str(os.environ.get("JIXIEBI_PACKAGE_ROOT", "") or "").strip()
    path = Path(root) / "apps" / "xinyu" / filename if root else Path(__file__).with_name(filename)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return copy.deepcopy(payload) if isinstance(payload, dict) else {}


DIALOGUE_OCR_MODE = _load_mode_config("dialogue_ocr_config.json")
LIST_OCR_MODE = _load_mode_config("list_ocr_config.json")
LLM_DIALOGUE_CONFIG = _load_mode_config("dialogue_llm_config.json")


POINTS_OCR_MODE = {
    "preprocess": {
        "mode": "original",
        "scale": 2,
        "rotate_to_portrait": True,
    },
    "rules": {
        "min_confidence": 0.30,
        "min_box_height_ratio": 0.01,
        "line_gap": 24,
        "keywords": ["我的积分", "积分"],
        "neighbor_y_ratio": 0.06,
        "neighbor_x_ratio": 0.45,
    },
}


OCR_MODES = {
    "dialogue": DIALOGUE_OCR_MODE,
    "list": LIST_OCR_MODE,
    "points": POINTS_OCR_MODE,
}
