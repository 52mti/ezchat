from __future__ import annotations

import copy
import json
import os
from pathlib import Path


def _load_mode_config(filename: str) -> dict:
    """三层回退：客户层(runtime/configs) -> 包内 json(开发机/迁移前) -> 代码内置默认。

    App 级配置 json 不随包分发（见 build.py 归档排除），全新装机没有
    json 文件，由 ocr_defaults.py 的代码层默认兜底。
    """
    root = str(os.environ.get("JIXIEBI_PACKAGE_ROOT", "") or "").strip()
    base = Path(root) if root else Path(__file__).resolve().parents[2]
    candidates = (
        base / "runtime" / "configs" / "huanyou" / filename,
        base / "apps" / "huanyou" / filename,
        Path(__file__).with_name(filename),
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload:
            return copy.deepcopy(payload)
    from .ocr_defaults import DIALOGUE_OCR_DEFAULT, LIST_OCR_DEFAULT

    return {"dialogue_ocr_config.json": DIALOGUE_OCR_DEFAULT, "list_ocr_config.json": LIST_OCR_DEFAULT}.get(filename, {})


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
