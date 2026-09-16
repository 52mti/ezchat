from __future__ import annotations

from io import BytesIO
from pathlib import Path

from services import photo_capture as photo_capture_module
from apps.base import BaseApp
from services.photo_capture import capture_transform_image
from services.photo_capture import _sync_coordinate_cache, _sync_photo_result_cache
from services import coordinate_cache
from PIL import Image


class CaptureRobot:
    host = "192.168.1.108"

    def __init__(self):
        self.pre_moves = []
        self.status_calls = 0

    def _image_bytes(self) -> bytes:
        buffer = BytesIO()
        Image.new("RGB", (8, 8), "white").save(buffer, format="JPEG")
        return buffer.getvalue()

    def get_status(self):
        self.status_calls += 1
        return {
            "data": {
                "transform_width": 8,
                "transform_height": 8,
                "points": [
                    {"x": 0, "y": 0},
                    {"x": 7, "y": 0},
                    {"x": 7, "y": 7},
                    {"x": 0, "y": 7},
                ],
            }
        }

    def get_capture_image(self, pre_move=True):
        self.pre_moves.append(pre_move)
        return self._image_bytes()


class FlakyCaptureRobot(CaptureRobot):
    def __init__(self):
        super().__init__()
        self.capture_calls = 0

    def get_capture_image(self, pre_move=True):
        self.capture_calls += 1
        if self.capture_calls == 1:
            raise RuntimeError("Connection broken: IncompleteRead")
        return super().get_capture_image(pre_move=pre_move)


class DemoApp(BaseApp):
    def run_once(self):
        return None


class FailingRecoverApp(BaseApp):
    def run_once(self):
        return None

    def recover(self):
        raise RuntimeError("recover boom")


class SolidCaptureRobot(CaptureRobot):
    def __init__(self, color: tuple[int, int, int]):
        super().__init__()
        self.color = color

    def _image_bytes(self) -> bytes:
        buffer = BytesIO()
        Image.new("RGB", (80, 120), self.color).save(buffer, format="JPEG")
        return buffer.getvalue()


def test_base_app_opens_circuit_after_threshold():
    app = DemoApp(robot=object(), config={"circuit_breaker_threshold": 2})

    app.on_error(RuntimeError("one"))
    app.on_error(RuntimeError("two"))

    status = app.get_status()
    assert status["error_count"] == 2
    assert status["is_circuit_open"] is True
    assert status["last_error"] == "two"


def test_capture_reports_phone_screen_health_after_two_frames():
    robot = SolidCaptureRobot((20, 20, 20))
    config = {"_runtime": {"device_ip": "192.168.1.108", "app_name": "demo"}}

    capture_transform_image(robot, config)
    assert config["_runtime"]["phone_screen_health"] == "unknown"
    capture_transform_image(robot, config)

    assert config["_runtime"]["phone_screen_health"] == "black_screen"
    assert config["_runtime"]["phone_screen_health_metrics"]["dark_ratio"] >= 0.7


def test_capture_reports_overexposed_and_black_screen_health():
    for color, expected in [((255, 255, 255), "overexposed"), ((0, 0, 0), "black_screen")]:
        robot = SolidCaptureRobot(color)
        config = {"_runtime": {"device_ip": "192.168.1.108", "app_name": "demo"}}
        capture_transform_image(robot, config)
        capture_transform_image(robot, config)
        assert config["_runtime"]["phone_screen_health"] == expected


def test_capture_reports_color_cast_overexposure_after_confirmation():
    # 两个颜色通道大面积饱和，模拟摄像头曝光/白平衡异常造成的偏粉、偏蓝画面。
    robot = SolidCaptureRobot((255, 203, 255))
    config = {"_runtime": {"device_ip": "192.168.1.108", "app_name": "demo"}}

    capture_transform_image(robot, config)
    assert config["_runtime"]["phone_screen_health"] == "unknown"
    capture_transform_image(robot, config)

    assert config["_runtime"]["phone_screen_health"] == "overexposed"
    assert config["_runtime"]["phone_screen_health_metrics"]["clipped_channel_count"] >= 2


def test_phone_screen_health_keeps_previous_state_until_recovery_is_confirmed():
    config = {"_runtime": {"device_ip": "192.168.1.108", "app_name": "demo"}}
    dark_robot = SolidCaptureRobot((0, 0, 0))
    capture_transform_image(dark_robot, config)
    capture_transform_image(dark_robot, config)
    assert config["_runtime"]["phone_screen_health"] == "black_screen"

    normal_robot = SolidCaptureRobot((120, 120, 120))
    capture_transform_image(normal_robot, config)
    assert config["_runtime"]["phone_screen_health"] == "black_screen"
    capture_transform_image(normal_robot, config)
    assert config["_runtime"]["phone_screen_health"] == "normal"


def test_base_app_opens_circuit_after_recover_limit():
    app = FailingRecoverApp(
        robot=object(),
        config={"circuit_breaker_threshold": 99, "recover_failure_limit": 2},
    )

    app.on_error(RuntimeError("one"))
    app.on_error(RuntimeError("two"))

    assert app.get_status()["is_circuit_open"] is True
    assert "设备复位失败" in app.get_status()["last_error"]


def test_base_app_recover_logs_include_current_step():
    logs = []
    app = FailingRecoverApp(
        robot=object(),
        config={"circuit_breaker_threshold": 99, "recover_failure_limit": 1},
    )
    app.set_log_callback(lambda level, message, payload=None: logs.append((level, message, payload)))
    app.set_runtime_step("input_box_click", "点击输入框")

    app.on_error(RuntimeError("input failed"))

    assert ("info", "准备执行设备复位") in [(level, message) for level, message, _payload in logs]
    failure = [entry for entry in logs if entry[1] == "设备复位失败"][-1]
    assert failure[0] == "warning"
    assert failure[2]["step"] == "input_box_click"
    assert failure[2]["label"] == "点击输入框"
    assert failure[2]["is_circuit_open"] is True


def test_base_app_error_capture_is_saved_under_photo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    robot = CaptureRobot()
    app = DemoApp(robot=robot, config={})

    app.on_error(RuntimeError("boom"))

    saved = list((tmp_path / "photo" / "errors").glob("192.168.1.108_*.jpg"))
    assert len(saved) == 1
    assert app.config["_runtime"]["last_error_capture_path"] == str(saved[0].resolve())
    assert app.config["_runtime"]["error_screenshot"] == str(saved[0].resolve())
    assert robot.pre_moves == [True]
    assert robot.status_calls == 1
    assert not (tmp_path / "runtime" / "errors").exists()


def test_runtime_robot_capture_records_saved_capture_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    photo_capture_module._STATUS_CACHE.clear()
    photo_capture_module._PROCESSED_IMAGE_CACHE.clear()
    robot = CaptureRobot()
    config = {
        "save_captures": True,
        "_runtime": {"device_ip": "192.168.1.108", "app_name": "wechat"},
    }

    image = capture_transform_image(robot, config, label="screen")

    assert image.startswith(b"\xff\xd8")
    capture_path = config["_runtime"]["current_capture_path"]
    assert capture_path.endswith("_screen.jpg")
    assert (tmp_path / Path(capture_path).relative_to(tmp_path)).exists()


def test_runtime_robot_capture_is_not_saved_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    photo_capture_module._STATUS_CACHE.clear()
    photo_capture_module._PROCESSED_IMAGE_CACHE.clear()
    robot = CaptureRobot()

    image = capture_transform_image(
        robot,
        {"_runtime": {"device_ip": "192.168.1.108", "app_name": "wechat"}},
        label="screen",
    )

    assert image.startswith(b"\xff\xd8")
    assert "device=192.168.1.108" in image.image_source
    assert "app=wechat" in image.image_source
    assert "label=screen" in image.image_source
    assert robot.pre_moves == [True]
    assert robot.status_calls == 1
    assert not (tmp_path / "photo" / "captures").exists()


def test_capture_reads_current_perspective_status_for_each_screenshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    photo_capture_module._STATUS_CACHE.clear()
    photo_capture_module._PROCESSED_IMAGE_CACHE.clear()
    robot = CaptureRobot()

    capture_transform_image(robot, {"_runtime": {"device_ip": "192.168.1.108", "app_name": "wechat"}})
    capture_transform_image(robot, {"_runtime": {"device_ip": "192.168.1.108", "app_name": "wechat"}})

    assert robot.status_calls == 2
    assert robot.pre_moves == [True, True]


def test_capture_reuses_processed_image_when_raw_image_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    photo_capture_module._STATUS_CACHE.clear()
    photo_capture_module._PROCESSED_IMAGE_CACHE.clear()
    robot = CaptureRobot()
    build_calls = 0
    original_build = photo_capture_module.build_local_perspective_image

    def counted_build(raw_bytes, status):
        nonlocal build_calls
        build_calls += 1
        return original_build(raw_bytes, status)

    monkeypatch.setattr(photo_capture_module, "build_local_perspective_image", counted_build)

    first = capture_transform_image(robot, {"_runtime": {"device_ip": "192.168.1.108", "app_name": "wechat"}})
    second = capture_transform_image(robot, {"_runtime": {"device_ip": "192.168.1.108", "app_name": "wechat"}})

    assert first == second
    assert build_calls == 1
    assert robot.status_calls == 2
    assert robot.pre_moves == [True, True]


def test_capture_time_is_excluded_from_step_timeout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    photo_capture_module._STATUS_CACHE.clear()
    photo_capture_module._PROCESSED_IMAGE_CACHE.clear()
    robot = CaptureRobot()
    config = {"_runtime": {"device_ip": "192.168.1.108", "app_name": "wechat"}}

    capture_transform_image(robot, config)

    assert float(config["_runtime"].get("_step_timeout_excluded_seconds", 0.0)) > 0.0


def test_coordinate_cache_survives_same_perspective_and_resets_on_change():
    cache = {
        "template_points": {"back": (0.2, 0.3)},
        "screen_anchors": {"bottom_my": (0.5, 0.9)},
        "ocr_regions": {"dialogue_message": [0.0, 0.2, 1.0, 0.6]},
        "detection_result_reuse": {"key": {"result": True}},
        "ocr_result_reuse": {"key": {"result": True}},
        "metrics": {"capture_count": 8},
        "contact_send_counts": {"user": 1},
    }
    config = {"_runtime": {"perspective_signature": "same"}, "_runtime_cache": cache}

    _sync_coordinate_cache(config, "same")
    assert cache["template_points"]["back"] == (0.2, 0.3)
    assert cache["metrics"]["capture_count"] == 8

    _sync_coordinate_cache(config, "changed")
    assert "template_points" not in cache
    assert "screen_anchors" not in cache
    assert "ocr_regions" not in cache
    # 透视变化阶段只负责坐标缓存；照片识别缓存由照片签名单独管理。
    assert "detection_result_reuse" in cache
    assert "ocr_result_reuse" in cache
    assert cache["metrics"]["capture_count"] == 8
    assert cache["contact_send_counts"]["user"] == 1


def test_coordinate_cache_persists_by_device_app_and_perspective(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinate_cache, "DEFAULT_ROOT", tmp_path / "coordinate_cache")
    config = {
        "_runtime": {
            "device_ip": "192.168.1.10",
            "app_name": "xinyu",
            "perspective_signature": "same",
        },
        "_runtime_cache": {"template_points": {"back": [0.2, 0.3]}},
    }
    coordinate_cache.persist_from_runtime(config, immediate=True)
    restored = {
        "_runtime": {
            "device_ip": "192.168.1.10",
            "app_name": "xinyu",
            "perspective_signature": "same",
        },
        "_runtime_cache": {},
    }
    assert coordinate_cache.load_into_runtime(restored)
    assert restored["_runtime_cache"]["template_points"]["back"] == [0.2, 0.3]


def test_photo_change_clears_recognition_cache_but_keeps_coordinates():
    cache = {
        "template_points": {"back": (0.2, 0.3)},
        "screen_anchors": {"bottom_my": (0.5, 0.9)},
        "detection_result_reuse": {"key": {"result": True}},
        "ocr_result_reuse": {"key": {"result": True}},
        "metrics": {"capture_count": 8},
    }
    config = {"_runtime": {}, "_runtime_cache": cache}

    _sync_photo_result_cache(config, b"photo-a")
    _sync_photo_result_cache(config, b"photo-a")
    assert cache["template_points"]["back"] == (0.2, 0.3)
    assert "detection_result_reuse" in cache
    assert "ocr_result_reuse" in cache

    _sync_photo_result_cache(config, b"photo-b")
    assert cache["template_points"]["back"] == (0.2, 0.3)
    assert cache["screen_anchors"]["bottom_my"] == (0.5, 0.9)
    assert "detection_result_reuse" not in cache
    assert "ocr_result_reuse" not in cache
    assert config["_runtime"]["photo_cache_reset"] is True


def test_capture_retries_transient_connection_break(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    photo_capture_module._STATUS_CACHE.clear()
    photo_capture_module._PROCESSED_IMAGE_CACHE.clear()
    robot = FlakyCaptureRobot()
    config = {
        "_runtime": {"device_ip": "192.168.1.108", "app_name": "huanyou"},
        "capture": {"retry_attempts": 1, "retry_delay_seconds": 0},
    }

    image = capture_transform_image(robot, config, label="send_button")

    assert image.startswith(b"\xff\xd8")
    assert robot.capture_calls == 2
    assert config["_runtime"]["capture_retry_count"] == 1
    assert "IncompleteRead" in config["_runtime"]["last_capture_error"]


def test_base_app_runtime_step_updates_status_and_callback():
    app = DemoApp(robot=object(), config={})
    payloads = []

    app.set_status_callback(payloads.append)
    app.set_runtime_step("list_ocr_scan", "List OCR Scan")

    assert app.get_status()["current_step"] == "list_ocr_scan"
    assert app.get_status()["current_step_label"] == "List OCR Scan"
    assert payloads[-1]["current_step"] == "list_ocr_scan"
    assert payloads[-1]["current_step_label"] == "List OCR Scan"


def test_base_app_tracks_average_reply_seconds():
    app = DemoApp(robot=object(), config={})

    app.record_reply_round(8.0)
    app.record_reply_round(12.0)

    status = app.get_status()
    assert status["reply_round_count"] == 2
    assert status["average_reply_seconds"] == 10.0


def test_coordinate_cache_invalidates_across_app_versions():
    import contextlib
    import json

    import services.coordinate_cache as coordinate_cache

    runtime = {
        "device_ip": "192.168.1.10",
        "app_name": "xinyu",
        "perspective_signature": "version-probe",
    }
    saved_cache = coordinate_cache._APP_VERSION_CACHE[:]
    path_v1 = path_v2 = None
    try:
        coordinate_cache._APP_VERSION_CACHE[:] = ["test-1.0"]
        coordinate_cache.persist_from_runtime(
            {"_runtime": runtime, "_runtime_cache": {"template_points": {"back": [0.2, 0.3]}}},
            immediate=True,
        )
        path_v1 = coordinate_cache._path({"_runtime": runtime})

        coordinate_cache._APP_VERSION_CACHE[:] = ["test-2.0"]
        path_v2 = coordinate_cache._path({"_runtime": runtime})
        assert path_v1 != path_v2 and path_v1.parent != path_v2.parent

        restored = {"_runtime": runtime, "_runtime_cache": {}}
        assert not coordinate_cache.load_into_runtime(restored)
        assert restored["_runtime_cache"] == {}

        payload = json.loads(path_v1.read_text(encoding="utf-8"))
        assert payload["app_version"] == "test-1.0"
    finally:
        coordinate_cache._APP_VERSION_CACHE[:] = saved_cache
        for path in (path_v1, path_v2):
            if path is not None:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
