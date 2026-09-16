"""OCR 配置分层回归：客户层副本不得冻结开发域 preprocess 调优。

背景（2026-09-16 事故）：旧布局迁移把整份默认值（含 preprocess.scale=2）
复制进 runtime/configs/<app>/，而 _load_mode_config 曾是"客户层整文件优先"，
导致开发端把 L1 json 改成 scale=1 后，迁移装机（含客户机）永远跑 scale=2，
OCR 提速对所有存量机器不生效。同时 ResilientRuntimeEngine 漏透传
last_timings，"子进程纯推理"字段从未出现在统计行。
"""

import json
from pathlib import Path

from apps.xinyu import ocr_config
from services.config_manager import ConfigManager
from vision.ocr_runtime import ResilientRuntimeEngine


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_list_mode(package_root: Path) -> dict:
    import os

    old = os.environ.get("JIXIEBI_PACKAGE_ROOT")
    os.environ["JIXIEBI_PACKAGE_ROOT"] = str(package_root)
    try:
        return ocr_config._load_mode_config("list_ocr_config.json")
    finally:
        if old is None:
            os.environ.pop("JIXIEBI_PACKAGE_ROOT", None)
        else:
            os.environ["JIXIEBI_PACKAGE_ROOT"] = old


def test_customer_layer_preprocess_cannot_pin_scale(tmp_path):
    """L2 旧副本 preprocess.scale=2 不再遮蔽 L1 的 scale=1，rules 仍以 L2 为准。"""
    _write_json(
        tmp_path / "apps" / "xinyu" / "list_ocr_config.json",
        {"preprocess": {"mode": "original", "scale": 1, "rotate_to_portrait": True}, "rules": {"min_confidence": 0.3}},
    )
    _write_json(
        tmp_path / "runtime" / "configs" / "xinyu" / "list_ocr_config.json",
        {"preprocess": {"mode": "original", "scale": 2, "rotate_to_portrait": True}, "rules": {"min_confidence": 0.55}},
    )

    mode = _load_list_mode(tmp_path)

    assert mode["preprocess"]["scale"] == 1
    assert mode["rules"]["min_confidence"] == 0.55


def test_customer_layer_only_uses_builtin_preprocess(tmp_path):
    """打包装机没有 L1 json 时，preprocess 由代码内置默认兜底（scale=1）。"""
    _write_json(
        tmp_path / "runtime" / "configs" / "xinyu" / "list_ocr_config.json",
        {"preprocess": {"mode": "original", "scale": 2, "rotate_to_portrait": True}, "rules": {"min_confidence": 0.6}},
    )

    mode = _load_list_mode(tmp_path)

    assert mode["preprocess"]["scale"] == 1
    assert mode["rules"]["min_confidence"] == 0.6


def test_empty_customer_layer_keeps_dev_payload(tmp_path):
    _write_json(
        tmp_path / "apps" / "xinyu" / "list_ocr_config.json",
        {"preprocess": {"mode": "original", "scale": 1, "rotate_to_portrait": True}, "rules": {"min_confidence": 0.31}},
    )

    mode = _load_list_mode(tmp_path)

    assert mode["preprocess"]["scale"] == 1
    assert mode["rules"]["min_confidence"] == 0.31


def test_update_app_config_strips_preprocess_from_customer_mode_files(tmp_path):
    """UI 保存带 preprocess 的 ocr_modes：客户层 mode 文件不落 preprocess（写端剥离）。"""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"devices": []}, ensure_ascii=False), encoding="utf-8")
    manager = ConfigManager(config_path)

    manager.update_app_config(
        "xinyu",
        {
            "ocr_modes": {
                "list": {
                    "preprocess": {"mode": "original", "scale": 2, "rotate_to_portrait": True},
                    "rules": {"min_confidence": 0.42},
                }
            }
        },
    )

    mode_file = tmp_path / "runtime" / "configs" / "xinyu" / "list_ocr_config.json"
    assert mode_file.is_file()
    payload = json.loads(mode_file.read_text(encoding="utf-8"))
    assert "preprocess" not in payload
    assert payload["rules"]["min_confidence"] == 0.42


def test_get_app_overrides_customer_preprocess_not_pinned(tmp_path):
    """读端剥离：L2 mode 文件带旧 preprocess，读取后 preprocess 由随包默认提供。"""
    _write_json(
        tmp_path / "apps" / "xinyu" / "list_ocr_config.json",
        {"preprocess": {"mode": "original", "scale": 1, "rotate_to_portrait": True}, "rules": {"min_confidence": 0.3}},
    )
    _write_json(
        tmp_path / "runtime" / "configs" / "xinyu" / "list_ocr_config.json",
        {"preprocess": {"mode": "original", "scale": 2, "rotate_to_portrait": True}, "rules": {"min_confidence": 0.58}},
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"devices": []}, ensure_ascii=False), encoding="utf-8")
    manager = ConfigManager(config_path)

    overrides = manager.get_app_overrides("xinyu")
    list_mode = overrides["ocr_modes"]["list"]

    assert list_mode["preprocess"]["scale"] == 1
    assert list_mode["rules"]["min_confidence"] == 0.58


def test_resilient_engine_passes_through_last_timings():
    """包装层必须透传 last_timings，否则统计行的"子进程纯推理"永远为空。"""

    class SubprocessLike:
        device_seen = "gpu:0"
        last_timings = {"decode_ms": 11.0, "predict_ms": 1016.0}

        def preload(self) -> str:
            return "gpu:0"

        def predict(self, image_array, **kwargs):
            return []

    engine = ResilientRuntimeEngine(SubprocessLike(), primary_backend="paddle_cuda", model_tier="lite")

    assert engine.last_timings == {"decode_ms": 11.0, "predict_ms": 1016.0}
    # 透传的是副本：外部改返回值不得污染引擎状态。
    engine.last_timings["predict_ms"] = 0.0
    assert engine._primary_engine.last_timings["predict_ms"] == 1016.0

    class WithoutTimings:
        device_seen = "cpu"

    bare = ResilientRuntimeEngine(WithoutTimings(), primary_backend="cpu", model_tier="lite")
    assert bare.last_timings == {}
