from __future__ import annotations

from services.robot_monitor import MonitoredRobot, RobotArmMonitor


class FakeRobot:
    host = "192.168.1.108"

    def __init__(self) -> None:
        self.status_index = 0
        self.resets = 0
        self.captures: list[bool] = []

    def get_status(self) -> dict[str, object]:
        self.status_index += 1
        return {"step": self.status_index}

    def reset_device(self, wait: bool = True) -> str:
        self.resets += 1
        return "ok"

    def get_transform_image(self, pre_move: bool = True, save_path=None) -> bytes:
        self.captures.append(pre_move)
        return b"image"


def test_monitor_records_recent_statuses_before_reset() -> None:
    robot = FakeRobot()
    events: list[tuple[str, str, dict | None]] = []
    monitor = RobotArmMonitor(robot, device_ip=robot.host, emit=lambda level, message, payload=None: events.append((level, message, payload)))
    wrapped = MonitoredRobot(robot, monitor)

    monitor.sample("first")
    monitor.sample("second")
    result = wrapped.reset_device(wait=True)

    assert result == "ok"
    before = [event for event in events if event[1] == "robot_monitor_before_operation"][-1]
    after = [event for event in events if event[1] == "robot_monitor_after_operation"][-1]
    assert before[0] == "info"
    assert before[2]["operation"] == "reset_device"
    assert len(before[2]["recent_statuses"]) >= 3
    assert after[2]["ok"] is True
    assert after[2]["operation"] == "reset_device"


def test_monitor_records_capture_pre_move_only_when_enabled() -> None:
    robot = FakeRobot()
    events: list[tuple[str, str, dict | None]] = []
    monitor = RobotArmMonitor(robot, device_ip=robot.host, emit=lambda level, message, payload=None: events.append((level, message, payload)))
    wrapped = MonitoredRobot(robot, monitor)

    wrapped.get_transform_image(pre_move=False)
    assert [event for event in events if event[1].startswith("robot_monitor_")] == []

    wrapped.get_transform_image(pre_move=True)
    monitor_events = [event for event in events if event[1].startswith("robot_monitor_")]
    assert [event[1] for event in monitor_events] == [
        "robot_monitor_before_operation",
        "robot_monitor_after_operation",
    ]
    assert monitor_events[0][2]["operation"] == "capture_pre_move"
