from __future__ import annotations

import threading
import time

from vision.ocr import _ocr_native_call_slot, get_ocr_queue_status


def test_ocr_queue_reports_waiting_calls() -> None:
    entered = threading.Event()
    release = threading.Event()

    def occupy() -> None:
        with _ocr_native_call_slot(1):
            entered.set()
            release.wait(2)

    first = threading.Thread(target=occupy)
    second = threading.Thread(target=occupy)
    first.start()
    assert entered.wait(1)
    second.start()
    for _ in range(100):
        if get_ocr_queue_status()["waiting"] == 1:
            break
        time.sleep(0.01)
    assert get_ocr_queue_status()["waiting"] == 1
    assert get_ocr_queue_status()["active"] == 1

    release.set()
    first.join(2)
    second.join(2)
    assert get_ocr_queue_status() == {"waiting": 0, "active": 0}
