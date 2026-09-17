"""引擎实例池（车道模型）：GPU 多实例并行、池大小与设备数解耦、降级共享。"""

from __future__ import annotations

import threading

import vision.ocr as ocr_module
import vision.ocr_runtime as runtime_module


def _use_backend(monkeypatch, backend: str):
    monkeypatch.setattr(runtime_module, "_effective_backend_config", lambda config: (backend, "lite"))
    monkeypatch.setattr(ocr_module.runtime_module if hasattr(ocr_module, "runtime_module") else runtime_module, "_effective_backend_config", lambda config: (backend, "lite"))


def _reset_pools(monkeypatch):
    monkeypatch.setattr(runtime_module, "_POOLED_ENGINES", {})


class _FakeEngine:
    def __init__(self, name: str) -> None:
        self.name = name


def test_engine_pool_size_resolution(monkeypatch):
    _use_backend(monkeypatch, "paddle_cuda")
    monkeypatch.setattr(runtime_module, "adaptive_engine_pool_cap", lambda: 4)
    assert runtime_module.engine_pool_size_for({"backend": "paddle_cuda"}) == 2
    assert runtime_module.engine_pool_size_for({"backend": "paddle_cuda", "ocr_engine_pool_size": 4}) == 4
    # 自适应上限：请求值恒被空闲显存上限钳制（30 台设备显存也不越界）
    assert runtime_module.engine_pool_size_for({"backend": "paddle_cuda", "ocr_engine_pool_size": 30}) == 4
    # 小显存卡（cap 2）：请求 4 只给 2；cap 1 时默认 2 也压到 1
    monkeypatch.setattr(runtime_module, "adaptive_engine_pool_cap", lambda: 2)
    assert runtime_module.engine_pool_size_for({"backend": "paddle_cuda", "ocr_engine_pool_size": 4}) == 2
    monkeypatch.setattr(runtime_module, "adaptive_engine_pool_cap", lambda: 1)
    assert runtime_module.engine_pool_size_for({"backend": "paddle_cuda"}) == 1
    _use_backend(monkeypatch, "cpu")
    assert runtime_module.engine_pool_size_for({"backend": "cpu", "ocr_engine_pool_size": 4}) == 1


def test_adaptive_engine_pool_cap_from_free_vram(monkeypatch):
    """自适应上限 = (空闲MB - 1GB 保留) / 1GB 每实例，绝对上限 8，
    探测失败回退保守 4；结果进程内缓存（探测一次）。"""
    monkeypatch.setattr(runtime_module, "_ADAPTIVE_POOL_CAP_CACHE", None)

    monkeypatch.setattr(runtime_module, "_probe_gpu_free_mb", lambda: 5980)  # 8GB 卡
    assert runtime_module.adaptive_engine_pool_cap() == 4

    monkeypatch.setattr(runtime_module, "_ADAPTIVE_POOL_CAP_CACHE", None)
    monkeypatch.setattr(runtime_module, "_probe_gpu_free_mb", lambda: 14336)  # 16GB 卡
    assert runtime_module.adaptive_engine_pool_cap() == 8  # 绝对上限

    monkeypatch.setattr(runtime_module, "_ADAPTIVE_POOL_CAP_CACHE", None)
    monkeypatch.setattr(runtime_module, "_probe_gpu_free_mb", lambda: 1500)  # 小显存
    assert runtime_module.adaptive_engine_pool_cap() == 1

    monkeypatch.setattr(runtime_module, "_ADAPTIVE_POOL_CAP_CACHE", None)
    monkeypatch.setattr(runtime_module, "_probe_gpu_free_mb", lambda: None)  # 探测失败
    assert runtime_module.adaptive_engine_pool_cap() == 4


def test_pooled_engine_assigns_instance_per_lane(monkeypatch):
    _use_backend(monkeypatch, "paddle_cuda")
    _reset_pools(monkeypatch)
    created: list[_FakeEngine] = []
    monkeypatch.setattr(
        runtime_module,
        "create_ocr_runtime_engine",
        lambda config: created.append(_FakeEngine(f"engine-{len(created)}")) or created[-1],
    )
    config = {"backend": "paddle_cuda"}

    first = runtime_module.get_pooled_runtime_engine(config, lane=0)
    again = runtime_module.get_pooled_runtime_engine(config, lane=0)
    lane1 = runtime_module.get_pooled_runtime_engine(config, lane=1)

    assert first is again  # 同车道恒复用（子进程常驻）
    assert lane1 is not first  # 不同车道独立实例
    assert len(created) == 2


def test_pooled_engine_creation_failure_degrades_to_shared(monkeypatch):
    _use_backend(monkeypatch, "paddle_cuda")
    _reset_pools(monkeypatch)
    created: list[_FakeEngine] = []

    def fake_create(config):
        if created:  # 第二个实例创建失败（显存不足等）
            raise RuntimeError("CUDA out of memory")
        created.append(_FakeEngine("engine-0"))
        return created[0]

    monkeypatch.setattr(runtime_module, "create_ocr_runtime_engine", fake_create)

    engine0 = runtime_module.get_pooled_runtime_engine({"backend": "paddle_cuda"}, lane=0)
    engine1 = runtime_module.get_pooled_runtime_engine({"backend": "paddle_cuda"}, lane=1)

    assert engine1 is engine0  # 降级共享 0 号，不阻断 OCR


def test_native_slot_assigns_distinct_lanes_and_keeps_reentrant_lane():
    lanes: dict[int, int] = {}
    barrier = threading.Barrier(2)

    def worker(tag: int) -> None:
        with ocr_module._ocr_native_call_slot(2):
            with ocr_module._ocr_native_call_slot(2):  # 可重入：沿用外层车道
                lanes[tag] = ocr_module._current_ocr_lane()
                barrier.wait(timeout=5)

    threads = [threading.Thread(target=worker, args=(tag,)) for tag in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert set(lanes.values()) == {0, 1}  # 并发持有者车道互异
    assert ocr_module._current_ocr_lane() == 0  # 退出槽后回到默认


def test_ocr_max_concurrency_follows_pool_for_gpu(monkeypatch):
    monkeypatch.setattr(runtime_module, "engine_pool_size_for", lambda config: 3)
    assert ocr_module._ocr_max_concurrency({"backend": "paddle_cuda"}) == 3
    assert ocr_module._ocr_max_concurrency({"backend": "paddle_cuda", "max_concurrency": 1}) == 1
    # 无运行时引擎键的传统路径保持串行
    assert ocr_module._ocr_max_concurrency({}) == 1


def test_worker_preload_warms_all_pool_lanes(tmp_path, monkeypatch):
    """worker 启动预热应暖齐池内全部车道（弱卡首加载 77~100s 从第一轮
    挪到启动 runway），且与生产调用同池同实例。"""
    from services.config_manager import ConfigManager
    from workers.device_worker import DeviceWorker

    warmed: list[int] = []

    class _FakeEngine:
        def preload(self) -> str:
            return "gpu:0"

    def fake_pool(config, lane=0):
        warmed.append(lane)
        return _FakeEngine()

    import vision.ocr_runtime as runtime_mod
    monkeypatch.setattr(runtime_mod, "get_pooled_runtime_engine", fake_pool)
    monkeypatch.setattr(runtime_mod, "engine_pool_size_for", lambda config: 2)

    worker = DeviceWorker(
        {"ip": "192.168.1.108", "selected_app": "xinyu"},
        app_config={"poll_interval": 1, "ocr_modes": {"list": {"ocr_backend": "paddle_cuda"}}},
        events=__import__("queue").Queue(),
        config_manager=ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json"),
        robot=object(),
    )

    worker._preload_ocr_engines()

    assert sorted(warmed) == [0, 1]


def test_ocr_max_concurrency_reads_backend_from_ocr_subtree(monkeypatch):
    """回归（20260917 开发机实测排队未消）：生产调用链里 backend/model_tier
    在 config["ocr"] 子树，模式级并发槽拿的是整个 app config——顶层漏看
    子树会让槽恒为 1，引擎池永远接不上。"""
    import vision.ocr_runtime as runtime_mod
    monkeypatch.setattr(runtime_mod, "engine_pool_size_for", lambda config: 2)

    assert ocr_module._ocr_max_concurrency({"ocr": {"backend": "paddle_cuda"}}) == 2
    assert ocr_module._ocr_max_concurrency({"ocr": {"model_tier": "lite"}}) == 2
    # 顶层键仍然生效
    assert ocr_module._ocr_max_concurrency({"backend": "paddle_cuda"}) == 2
    # 两处都没有引擎键：传统路径保持串行
    assert ocr_module._ocr_max_concurrency({"ocr": {"det_db_thresh": 0.3}}) == 1
    assert ocr_module._ocr_max_concurrency({}) == 1
