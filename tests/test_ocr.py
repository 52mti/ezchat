from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
import importlib

import numpy as np
import pytest
from PIL import Image

ocr_module = importlib.import_module("vision.ocr")
from vision.ocr import ocr
from vision.screen import annotate_image_bytes, load_image


class FakeEngine:
    def ocr(self, image, cls=True):
        return [
            [
                [
                    [[10, 20], [50, 20], [50, 40], [10, 40]],
                    ("hello", 0.92),
                ]
            ]
        ]


def test_ocr_normalizes_engine_boxes():
    image = Image.new("RGB", (100, 100), "white")

    result = ocr(image, {"engine": FakeEngine()})

    assert result == [
        {
            "text": "hello",
            "x": 0.1,
            "y": 0.2,
            "w": 0.4,
            "h": 0.2,
            "confidence": 0.92,
        }
    ]


def test_ocr_returns_empty_without_backend():
    image = Image.new("RGB", (100, 100), "white")

    result = ocr(image, {})

    assert isinstance(result, list)


def test_load_image_error_includes_file_path(tmp_path):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"\xff\xd8broken")

    with pytest.raises(RuntimeError) as exc_info:
        load_image(broken)

    message = str(exc_info.value)
    assert "图片读取失败" in message
    assert str(broken.resolve()) in message


def test_load_image_error_includes_capture_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image = annotate_image_bytes(b"\xff\xd8broken", "机械臂截图 device=192.168.5.4 app=huanyou label=after_bottom_nav_message")

    with pytest.raises(RuntimeError) as exc_info:
        load_image(image)

    message = str(exc_info.value)
    assert "device=192.168.5.4" in message
    assert "app=huanyou" in message
    assert "label=after_bottom_nav_message" in message
    assert "问题图片已保存=" in message
    saved = list((tmp_path / "photo" / "errors").glob("image_load_failed_*.jpg"))
    assert len(saved) == 1


class FakePredictEngine:
    def __init__(self) -> None:
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append((image, kwargs))
        return [
            {
                "dt_polys": [
                    np.array([[10, 20], [30, 20], [30, 60], [10, 60]], dtype=np.float32),
                ],
                "rec_texts": ["runtime"],
                "rec_scores": [0.88],
            }
        ]


def test_ocr_runtime_predict_result_is_normalized(monkeypatch):
    image = Image.new("RGB", (100, 50), "white")
    engine = FakePredictEngine()

    monkeypatch.setattr(ocr_module, "_get_paddle_engine", lambda config: engine)

    result = ocr(
        image,
        {
            "runtime_enabled": True,
            "lang": "en",
            "use_angle_cls": False,
            "det_db_thresh": 0.25,
            "rec_thresh": 0.6,
        },
    )

    assert len(engine.calls) == 1
    called_image, called_kwargs = engine.calls[0]
    assert isinstance(called_image, np.ndarray)
    assert called_image.shape == (100, 50, 3)
    assert called_kwargs == {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "text_det_thresh": 0.25,
        "text_rec_score_thresh": 0.6,
    }
    assert result == [
        {
            "text": "runtime",
            "x": 0.2,
            "y": 0.2,
            "w": 0.4,
            "h": 0.4,
            "confidence": 0.88,
        }
    ]


def test_ocr_runtime_predict_result_uses_portrait_coordinates_for_landscape_input(monkeypatch):
    image = Image.new("RGB", (100, 50), "white")
    engine = FakePredictEngine()

    monkeypatch.setattr(ocr_module, "_get_paddle_engine", lambda config: engine)

    result = ocr(
        image,
        {
            "runtime_enabled": True,
            "rotate_to_portrait": True,
        },
    )

    called_image, _called_kwargs = engine.calls[0]
    assert called_image.shape == (100, 50, 3)
    assert result[0]["x"] == 0.2
    assert result[0]["y"] == 0.2
    assert result[0]["w"] == 0.4
    assert result[0]["h"] == 0.4


def test_ocr_runtime_uses_mobile_model_names_and_gpu(monkeypatch):
    image = Image.new("RGB", (100, 50), "white")
    captured = {}

    def fake_cached_engine(cache_key: str):
        captured["cache_key"] = cache_key
        return FakePredictEngine()

    monkeypatch.setattr(ocr_module, "_cached_paddle_engine", fake_cached_engine)

    ocr(
        image,
        {
            "runtime_enabled": True,
            "lang": "ch",
            "use_gpu": True,
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "PP-OCRv5_mobile_rec",
        },
    )

    assert captured["cache_key"] is not None
    assert "\"device\": \"gpu:0\"" in captured["cache_key"]
    assert "\"text_detection_model_name\": \"PP-OCRv5_mobile_det\"" in captured["cache_key"]
    assert "\"text_recognition_model_name\": \"PP-OCRv5_mobile_rec\"" in captured["cache_key"]


def test_ocr_runtime_prefers_auto_backend_entry(monkeypatch):
    image = Image.new("RGB", (100, 50), "white")
    captured = {}

    class DummyRuntimeEngine(FakePredictEngine):
        pass

    import vision.ocr_runtime as runtime_module

    def fake_runtime_engine(config):
        captured["config"] = dict(config)
        return DummyRuntimeEngine()

    monkeypatch.setattr(
        runtime_module,
        "get_ocr_runtime_engine",
        fake_runtime_engine,
    )

    result = ocr(
        image,
        {
            "runtime_enabled": True,
            "backend": "auto",
            "model_tier": "lite",
            "use_gpu": False,
        },
    )

    assert captured["config"]["backend"] == "auto"
    assert result[0]["text"] == "runtime"


def test_ocr_predict_calls_are_serialized_across_threads():
    image = Image.new("RGB", (100, 50), "white")

    class SlowPredictEngine:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def predict(self, image, **kwargs):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.03)
                return [
                    {
                        "dt_polys": [
                            np.array([[10, 20], [30, 20], [30, 40], [10, 40]], dtype=np.float32),
                        ],
                        "rec_texts": ["threadsafe"],
                        "rec_scores": [0.99],
                    }
                ]
            finally:
                with self.lock:
                    self.active -= 1

    engine = SlowPredictEngine()

    def run_once():
        return ocr(image, {"engine": engine})

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: run_once(), range(2)))

    assert engine.max_active == 1
    assert [result[0]["text"] for result in results] == ["threadsafe", "threadsafe"]


def test_ocr_predict_concurrency_can_be_configured():
    image = Image.new("RGB", (100, 50), "white")
    entered = threading.Barrier(2)

    class SlowPredictEngine:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def predict(self, image, **kwargs):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                entered.wait(timeout=1)
                time.sleep(0.03)
                return [
                    {
                        "dt_polys": [
                            np.array([[10, 20], [30, 20], [30, 40], [10, 40]], dtype=np.float32),
                        ],
                        "rec_texts": ["parallel"],
                        "rec_scores": [0.99],
                    }
                ]
            finally:
                with self.lock:
                    self.active -= 1

    engine = SlowPredictEngine()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: ocr(image, {"engine": engine, "max_concurrency": 2}), range(2)))

    assert engine.max_active == 2
    assert [result[0]["text"] for result in results] == ["parallel", "parallel"]


def test_ocr_engine_initialization_is_serialized_across_threads(monkeypatch):
    image = Image.new("RGB", (100, 50), "white")
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_cached_engine(cache_key: str):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.03)
            return FakePredictEngine()
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(ocr_module, "_cached_paddle_engine", fake_cached_engine)

    def run_once():
        return ocr(image, {"runtime_enabled": True})

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: run_once(), range(2)))

    assert max_active == 1
    assert [result[0]["text"] for result in results] == ["runtime", "runtime"]
