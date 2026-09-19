"""列表固定条带（compact 与顶部条带共用）测试。

背景（20260919 客户口径统一）：自适应 compact 从上一轮全扫行界推导并缓存，
横幅把「未回复」Tab 带压低后缓存边界不再覆盖首行，且只在条带完全无行时才
重算——客户机出现"第一条永远不回复"。改为固定条带 [0, 0.05, 1, 0.30]，
Tab/首行被压出条带时自然回退完整 OCR，首行不漏。

历史背景（20260915 形状稳定）：边界按 bucket（0.05）档位对齐，同设备行数
小变不换引擎输入形状，避免 GPU 引擎每种新形状首次调用 ~10 秒调机。
"""

from __future__ import annotations

import pytest

from apps.xinyu.config import DEFAULT_CONFIG
from apps.xinyu import steps


def _config(**overrides) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg.setdefault("_runtime_cache", {})
    cfg.update(overrides)
    return cfg


def test_fixed_band_default_is_005_to_030():
    """默认固定条带 [0, 0.05, 1, 0.30]，compact 与顶部条带完全一致。"""
    cfg = _config()
    band = pytest.approx([0.0, 0.05, 1.0, 0.30])
    assert steps._fixed_list_band_region(cfg) == band
    assert steps._list_compact_region(cfg) == band
    assert steps._list_top_strip_region(cfg) == band


def test_fixed_band_bucket_aligns_custom_bottom_outward():
    """自定义底界向外吸附到 0.05 档：0.37 → 0.40（高度 0.35），保持形状稳定。"""
    cfg = _config(list_top_strip_bottom=0.37)
    assert steps._fixed_list_band_region(cfg) == pytest.approx([0.0, 0.05, 1.0, 0.35])


def test_fixed_band_bucket_zero_keeps_exact_bottom():
    """bucket=0 关闭对齐，底界保持原值（高度 = 底界 - 0.05）。"""
    cfg = _config(list_top_strip_bottom=0.32, list_compact_region_bucket=0)
    assert steps._fixed_list_band_region(cfg) == [0.0, 0.05, 1.0, 0.27]


def test_fixed_band_flags_gate_independently():
    """两个开关各自独立：关 compact 不影响条带，反之亦然。"""
    assert steps._list_compact_region(_config(enable_compact_list_ocr_region=False)) is None
    assert steps._list_top_strip_region(_config(enable_list_top_strip_scan=False)) is None
    assert steps._list_compact_region(_config(enable_list_top_strip_scan=False)) == pytest.approx([0.0, 0.05, 1.0, 0.30])


def test_stale_cached_compact_region_is_ignored():
    """旧版本写入的自适应缓存不再生效：区域恒为固定条带。"""
    cfg = _config()
    cfg["_runtime_cache"]["ocr_regions"] = {"list_compact": [0.0, 0.06, 1.0, 0.82]}
    assert steps._list_compact_region(cfg) == pytest.approx([0.0, 0.05, 1.0, 0.30])


def test_unread_tab_band_no_longer_grows_band_region():
    """Tab 带缓存只记录自身位置，不再回写列表条带（区域已固定）。"""
    cfg = _config()
    steps._cache_unread_tab_band(cfg, [0.0, 0.10, 1.0, 0.15])
    assert cfg["_runtime_cache"]["ocr_regions"]["unread_tab_band"][1] <= 0.10
    assert "list_compact" not in cfg["_runtime_cache"]["ocr_regions"]
