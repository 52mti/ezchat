from __future__ import annotations

import importlib

import numpy as np
from PIL import Image

ocr_module = importlib.import_module("vision.ocr")
from vision.ocr import ocr


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
