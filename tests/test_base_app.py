from __future__ import annotations

from io import BytesIO

from apps.base import BaseApp
from apps.photo_capture import capture_transform_image
from PIL import Image


class CaptureRobot:
    host = "192.168.1.108"

    def __init__(self):
        self.save_paths = []
        self.pre_moves = []

    def _image_bytes(self) -> bytes:
        buffer = BytesIO()
        Image.new("RGB", (8, 8), "white").save(buffer, format="JPEG")
        return buffer.getvalue()

    def get_transform_image(self, pre_move=True, save_path=None):
        self.pre_moves.append(pre_move)
        self.save_paths.append(save_path)
        image_bytes = self._image_bytes()
        if save_path:
            save_path.write_bytes(image_bytes)
        return image_bytes


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


def test_base_app_error_capture_is_saved_under_photo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    robot = CaptureRobot()
    app = DemoApp(robot=robot, config={})

    app.on_error(RuntimeError("boom"))

    saved = list((tmp_path / "photo" / "errors").glob("192.168.1.108_*.jpg"))
    assert len(saved) == 1
    assert robot.save_paths == [saved[0]]
    assert robot.pre_moves == [True]
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
    assert robot.save_paths == [None]
    assert robot.pre_moves == [True]
    assert not (tmp_path / "photo" / "captures").exists()


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
