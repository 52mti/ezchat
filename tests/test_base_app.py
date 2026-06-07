from __future__ import annotations

from io import BytesIO

from apps import photo_capture as photo_capture_module
from apps.base import BaseApp
from apps.photo_capture import capture_transform_image
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


class DemoApp(BaseApp):
    def run_once(self):
        return None


class FailingRecoverApp(BaseApp):
    def run_once(self):
        return None

    def recover(self):
        raise RuntimeError("recover boom")


def test_base_app_opens_circuit_after_threshold():
    app = DemoApp(robot=object(), config={"circuit_breaker_threshold": 2})

    app.on_error(RuntimeError("one"))
    app.on_error(RuntimeError("two"))

    status = app.get_status()
    assert status["error_count"] == 2
    assert status["is_circuit_open"] is True
    assert status["last_error"] == "two"


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
    assert robot.pre_moves == [True]
    assert robot.status_calls == 1
    assert not (tmp_path / "runtime" / "errors").exists()


def test_runtime_robot_capture_is_not_saved_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    robot = CaptureRobot()

    image = capture_transform_image(
        robot,
        {"_runtime": {"device_ip": "192.168.1.108", "app_name": "wechat"}},
        label="screen",
    )

    assert image.startswith(b"\xff\xd8")
    assert robot.pre_moves == [True]
    assert robot.status_calls == 1
    assert not (tmp_path / "photo" / "captures").exists()


def test_capture_reuses_perspective_status_for_process_lifetime(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    photo_capture_module._STATUS_CACHE.clear()
    photo_capture_module._PROCESSED_IMAGE_CACHE.clear()
    robot = CaptureRobot()

    capture_transform_image(robot, {"_runtime": {"device_ip": "192.168.1.108", "app_name": "wechat"}})
    capture_transform_image(robot, {"_runtime": {"device_ip": "192.168.1.108", "app_name": "wechat"}})

    assert robot.status_calls == 1
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
    assert robot.status_calls == 1
    assert robot.pre_moves == [True, True]


def test_capture_time_is_excluded_from_step_timeout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    photo_capture_module._STATUS_CACHE.clear()
    photo_capture_module._PROCESSED_IMAGE_CACHE.clear()
    robot = CaptureRobot()
    config = {"_runtime": {"device_ip": "192.168.1.108", "app_name": "wechat"}}

    capture_transform_image(robot, config)

    assert float(config["_runtime"].get("_step_timeout_excluded_seconds", 0.0)) > 0.0


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
