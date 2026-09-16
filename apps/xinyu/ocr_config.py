from __future__ import annotations

import copy
import json
import os
from pathlib import Path


def _read_json_dict(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) and payload else None


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _builtin_default(filename: str) -> dict:
    from .ocr_defaults import DIALOGUE_OCR_DEFAULT, LIST_OCR_DEFAULT

    return {
        "dialogue_ocr_config.json": DIALOGUE_OCR_DEFAULT,
        "list_ocr_config.json": LIST_OCR_DEFAULT,
    }.get(filename, {})


def _load_mode_config(filename: str) -> dict:
    """开发域基线 + 客户层覆盖：包内 json(apps) -> 客户层(runtime/configs) 按键覆盖。

    preprocess（mode/scale/rotate_to_portrait）属开发域性能与布局旋钮，
    客户层副本里的 preprocess 一律剥离，不参与覆盖——旧布局迁移曾把整份
    默认值冻进 runtime/configs，导致开发端后续调 scale 对迁移装机永久
    失效（scale=2 的旧副本一直压着新默认）。客户可调的 rules 等键仍以
    客户层为准；两级都没有 json 时由 ocr_defaults.py 代码层默认兜底。
    """
    root = str(os.environ.get("JIXIEBI_PACKAGE_ROOT", "") or "").strip()
    base = Path(root) if root else Path(__file__).resolve().parents[2]

    dev_payload: dict | None = None
    for path in (
        base / "apps" / "xinyu" / filename,
        Path(__file__).with_name(filename),
    ):
        dev_payload = _read_json_dict(path)
        if dev_payload is not None:
            break
    customer_payload = _read_json_dict(base / "runtime" / "configs" / "xinyu" / filename)

    if dev_payload is None and customer_payload is None:
        return copy.deepcopy(_builtin_default(filename))

    merged = copy.deepcopy(dev_payload) if dev_payload is not None else {}
    if customer_payload is not None:
        override = copy.deepcopy(customer_payload)
        override.pop("preprocess", None)
        _deep_merge(merged, override)
    if "preprocess" not in merged:
        default = _builtin_default(filename)
        if "preprocess" in default:
            merged["preprocess"] = copy.deepcopy(default["preprocess"])
    return merged


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
