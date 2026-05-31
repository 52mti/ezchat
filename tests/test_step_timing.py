from __future__ import annotations

from apps.step_timing import exclude_ocr_wait_from_step_timeout, step_deadline


def test_ocr_wait_is_excluded_from_step_deadline() -> None:
    current_time = {"value": 0.0}

    def now() -> float:
        return current_time["value"]

    config: dict[str, object] = {"exclude_ocr_wait_from_step_timeout": True}
    deadline = step_deadline(config, 1.0, now=now)

    current_time["value"] = 0.8
    assert deadline.expired() is False

    with exclude_ocr_wait_from_step_timeout(config, now=now):
        current_time["value"] = 2.8

    assert deadline.expired() is False

    current_time["value"] = 3.1
    assert deadline.expired() is True


def test_ocr_wait_counts_when_exclusion_is_disabled() -> None:
    current_time = {"value": 0.0}

    def now() -> float:
        return current_time["value"]

    config: dict[str, object] = {"exclude_ocr_wait_from_step_timeout": False}
    deadline = step_deadline(config, 1.0, now=now)

    with exclude_ocr_wait_from_step_timeout(config, now=now):
        current_time["value"] = 1.1

    assert deadline.expired() is True
