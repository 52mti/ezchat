from __future__ import annotations

import copy
import json
import os
from pathlib import Path


def _load_mode_config(filename: str) -> dict:
    root = str(os.environ.get("JIXIEBI_PACKAGE_ROOT", "") or "").strip()
    path = Path(root) / "apps" / "huanyou" / filename if root else Path(__file__).with_name(filename)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return copy.deepcopy(payload) if isinstance(payload, dict) else {}


DIALOGUE_OCR_MODE = _load_mode_config("dialogue_ocr_config.json")
LIST_OCR_MODE = _load_mode_config("list_ocr_config.json")
LLM_DIALOGUE_CONFIG = _load_mode_config("dialogue_llm_config.json")


# 聚合入口：保留旧导入路径不变。
# dialogue 和 list 的具体规则已经拆到独立文件，避免互相影响。

POINTS_KEYWORDS = ["我的收益", "收益", "金币"]

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
        "keywords": POINTS_KEYWORDS,
        "neighbor_y_ratio": 0.08,
        "neighbor_x_ratio": 0.65,
    },
}

OCR_MODES = {
    "dialogue": DIALOGUE_OCR_MODE,
    "list": LIST_OCR_MODE,
    "points": POINTS_OCR_MODE,
}
