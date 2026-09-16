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


def test_ocr_slot_reentrant_same_thread_no_deadlock():
    """同线程嵌套进槽不自等（run_ocr_mode 全流程进槽后内部 ocr() 再进槽）。"""
    import threading
    import time

    from vision.ocr import _ocr_native_call_slot, get_ocr_queue_status

    entered_inner = threading.Event()
    release_inner = threading.Event()

    with _ocr_native_call_slot(1):
        with _ocr_native_call_slot(1):
            entered_inner.set()
            # 若嵌套会自等，这里 0.3s 后仍进不来；可重入则立即通过。
            time.sleep(0.05)
    assert entered_inner.is_set()
    status = get_ocr_queue_status()
    assert status["active"] == 0 and status["waiting"] == 0


def test_ocr_slot_still_serializes_across_threads():
    import threading

    from vision.ocr import _ocr_native_call_slot, get_ocr_queue_status

    seen_concurrent = []
    lock = threading.Lock()

    def worker():
        with _ocr_native_call_slot(1):
            with lock:
                active = get_ocr_queue_status()["active"]
                seen_concurrent.append(active)
            time.sleep(0.05)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert not any(t.is_alive() for t in threads)
    assert all(value == 1 for value in seen_concurrent)
