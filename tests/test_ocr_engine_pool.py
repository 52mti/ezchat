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
    """车道数 = 配置值，双重钳制：空闲显存自适应上限（防连环 CUDA OOM，
    20260917 恢复）+ 并发槽硬顶 16；ocr_engine_pool_ignore_vram_cap=true
    可绕过显存钳制（仅调试）。"""
    _use_backend(monkeypatch, "paddle_cuda")
    monkeypatch.setattr(runtime_module, "_ADAPTIVE_POOL_CAP_CACHE", 6)
    assert runtime_module.engine_pool_size_for({"backend": "paddle_cuda"}) == 2
    assert runtime_module.engine_pool_size_for({"backend": "paddle_cuda", "ocr_engine_pool_size": 4}) == 4
    # 超出显存自适应上限被钳制
    assert runtime_module.engine_pool_size_for({"backend": "paddle_cuda", "ocr_engine_pool_size": 10}) == 6
    assert runtime_module.engine_pool_size_for({"backend": "paddle_cuda", "ocr_engine_pool_size": 30}) == 6
    # 显式绕过显存钳制：配置值即车道数，唯一硬顶 16
    bypass = {"backend": "paddle_cuda", "ocr_engine_pool_ignore_vram_cap": True}
    assert runtime_module.engine_pool_size_for({**bypass, "ocr_engine_pool_size": 10}) == 10
    assert runtime_module.engine_pool_size_for({**bypass, "ocr_engine_pool_size": 30}) == 16
    _use_backend(monkeypatch, "cpu")
    assert runtime_module.engine_pool_size_for({"backend": "cpu", "ocr_engine_pool_size": 4}) == 1


def test_adaptive_engine_pool_cap_from_free_vram(monkeypatch):
    """自适应上限 = (空闲MB - 1GB 保留) / 1280MB 每车道（mobile 模型实测
    ~1GB+峰值），提示封顶 8，探测失败回退保守 4；结果进程内缓存（探测一次）。"""
    monkeypatch.setattr(runtime_module, "_ADAPTIVE_POOL_CAP_CACHE", None)

    monkeypatch.setattr(runtime_module, "_probe_gpu_free_mb", lambda: 5980)  # 8GB 卡
    assert runtime_module.adaptive_engine_pool_cap() == 3

    monkeypatch.setattr(runtime_module, "_ADAPTIVE_POOL_CAP_CACHE", None)
    monkeypatch.setattr(runtime_module, "_probe_gpu_free_mb", lambda: 14336)  # 16GB 卡
    assert runtime_module.adaptive_engine_pool_cap() == 8  # 提示封顶

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


def test_worker_preload_stops_after_degraded_lane(tmp_path, monkeypatch):
    """车道创建失败降级共享后，预热不再往后空试（设 10 台时避免启动期
    连续 9 次注定失败的创建尝试）。"""
    from services.config_manager import ConfigManager
    from workers.device_worker import DeviceWorker

    class _SharedEngine:
        def preload(self) -> str:
            return "gpu:0"

    shared = _SharedEngine()
    warmed: list[int] = []

    def fake_pool(config, lane=0):
        warmed.append(lane)
        return shared

    import vision.ocr_runtime as runtime_mod
    monkeypatch.setattr(runtime_mod, "get_pooled_runtime_engine", fake_pool)
    monkeypatch.setattr(runtime_mod, "engine_pool_size_for", lambda config: 3)

    worker = DeviceWorker(
        {"ip": "192.168.1.108", "selected_app": "xinyu"},
        app_config={"poll_interval": 1, "ocr_modes": {"list": {"ocr_backend": "paddle_cuda"}}},
        events=__import__("queue").Queue(),
        config_manager=ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json"),
        robot=object(),
    )

    worker._preload_ocr_engines()

    # lane1 拿到的是 0 号共享实例（降级形态）→ 立即止损，不再尝试 lane2
    assert warmed == [0, 1]


def test_mode_file_backend_reaches_production_engine_config(monkeypatch):
    """回归（20260917 PDX 崩溃链）：UI「OCR 引擎」切换写进模式文件的
    ocr_backend 必须并入生产调用配置——此前生产只看 ocr 子树的 auto，
    切换不生效，且与预热按模式配置建的引擎形成同进程双后端，进程内
    PaddleX 拒绝二次初始化（每次 OCR 必炸）。"""
    from PIL import Image

    import copy

    import apps.xinyu.ocr_rules as rules_module
    from apps.xinyu.ocr_config import DIALOGUE_OCR_MODE

    captured: list[dict] = []

    def fake_ocr(image, config):
        captured.append({k: config[k] for k in ("backend", "ocr_backend", "model_tier") if k in config})
        return []

    monkeypatch.setattr(rules_module, "ocr", fake_ocr)
    image = Image.new("RGB", (100, 200), "white")
    config = {
        "ocr": {"backend": "auto", "model_tier": "lite", "det_db_thresh": 0.3},
        "ocr_modes": {
            "dialogue": {
                **copy.deepcopy(DIALOGUE_OCR_MODE),
                "ocr_backend": "cpu",
            }
        },
    }

    rules_module.run_ocr_mode(image, config, "dialogue")

    assert captured and captured[0]["ocr_backend"] == "cpu"  # 切换生效
    assert captured[0]["model_tier"] == "lite"  # ocr 子树键保留


def test_worker_preload_merges_ocr_subtree_and_mode_keys(tmp_path, monkeypatch):
    """预热与生产必须命中同一池：引擎配置 = ocr 子树 + 模式文件引擎键。"""
    from services.config_manager import ConfigManager
    from workers.device_worker import DeviceWorker

    captured: list[dict] = []

    class _Engine:
        def preload(self) -> str:
            return "cpu"

    def fake_pool(config, lane=0):
        captured.append(dict(config))
        return _Engine()

    import vision.ocr_runtime as runtime_mod
    monkeypatch.setattr(runtime_mod, "get_pooled_runtime_engine", fake_pool)
    monkeypatch.setattr(runtime_mod, "engine_pool_size_for", lambda config: 1)

    worker = DeviceWorker(
        {"ip": "192.168.1.108", "selected_app": "xinyu"},
        app_config={
            "poll_interval": 1,
            "ocr": {"backend": "auto", "model_tier": "lite", "det_db_thresh": 0.3},
            "ocr_modes": {"list": {"ocr_backend": "cpu", "rules": {"min_confidence": 0.3}}},
        },
        events=__import__("queue").Queue(),
        config_manager=ConfigManager(tmp_path / "config.json", tmp_path / "outbox.json"),
        robot=object(),
    )

    worker._preload_ocr_engines()

    assert captured and captured[0]["ocr_backend"] == "cpu"
    assert captured[0]["backend"] == "auto"  # ocr 子树基底保留
    assert captured[0]["model_tier"] == "lite"
