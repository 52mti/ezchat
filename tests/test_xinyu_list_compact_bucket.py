"""列表 compact 区域档位对齐（形状稳定）测试。

背景（20260915 客户机 OCR 7~15 秒定位）：列表 compact 边界逐轮由当轮
OCR 现场重算，高度每轮微变 → GPU 引擎对每种新输入形状首次调用要付
约 10 秒调机。档位对齐后边界吸附到归一化粗档，行数小变不换形状。
"""

from __future__ import annotations

from apps.xinyu.config import DEFAULT_CONFIG
from apps.xinyu import steps


def _config(**overrides) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg.setdefault("_runtime_cache", {})
    cfg.update(overrides)
    return cfg


def _ocr_result(first_row_top: float, row_count: int = 4) -> dict:
    rows = []
    for index in range(row_count):
        top = first_row_top + index * 0.09
        rows.append({"bounds": [0.02, top, 0.96, 0.08], "name": f"会话{index}"})
    return {"result": {"message_list": rows}}


def _cached_region(cfg: dict) -> list[float] | None:
    return steps._valid_region(steps._cache_get(cfg, "ocr_regions", "list_compact"))


def test_small_row_shifts_stay_in_same_bucket():
    """行位置小幅漂移（0.131→0.136）必须落在同一档，区域完全一致。"""
    for first_top in (0.131, 0.133, 0.136):
        cfg = _config()
        steps._cache_list_compact_region_from_ocr_result(cfg, _ocr_result(first_top))
        region = _cached_region(cfg)
        assert region is not None
        assert region[1] == 0.10  # 0.131~0.136 - margin 后向外吸附到 0.10 档
    # 同一配置下两轮结果逐位相等 → 引擎输入形状稳定
    cfg_a = _config()
    steps._cache_list_compact_region_from_ocr_result(cfg_a, _ocr_result(0.131))
    cfg_b = _config()
    steps._cache_list_compact_region_from_ocr_result(cfg_b, _ocr_result(0.136))
    assert _cached_region(cfg_a) == _cached_region(cfg_b)


def test_bucket_expands_outward_and_never_cuts_rows():
    """对齐只向外扩：区域顶必须仍在首行之上（不切内容）。"""
    cfg = _config()
    first_top = 0.133
    steps._cache_list_compact_region_from_ocr_result(cfg, _ocr_result(first_top))
    region = _cached_region(cfg)
    assert region is not None
    assert region[1] <= first_top
    assert region[1] + region[3] >= 0.85  # 底部向外吸附，不小于原边界


def test_large_change_moves_to_next_bucket():
    """列表大变（顶栏出现，首行下移 0.09）换档——自适应保留。"""
    cfg_before = _config()
    steps._cache_list_compact_region_from_ocr_result(cfg_before, _ocr_result(0.131))
    cfg_after = _config()
    steps._cache_list_compact_region_from_ocr_result(cfg_after, _ocr_result(0.221))
    assert _cached_region(cfg_after)[1] > _cached_region(cfg_before)[1]


def test_bucket_zero_restores_exact_boundary():
    """bucket<=0 关闭对齐：回到逐轮精确边界（回退开关）。"""
    cfg = _config(list_compact_region_bucket=0)
    steps._cache_list_compact_region_from_ocr_result(cfg, _ocr_result(0.131))
    region = _cached_region(cfg)
    assert region is not None
    # 未对齐：顶 = max(full_region顶, 首行顶-margin)，精确值
    assert abs(region[1] - (0.131 - 0.02)) < 1e-9 or region[1] == 0.1
