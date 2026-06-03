from __future__ import annotations

import pytest

from apps.xinyu import steps as xinyu_steps


_XINYU_LIST_SELECTION_TESTS = {
    "test_xinyu_list_falls_back_to_content_when_unread_is_missing",
    "test_xinyu_list_ignores_empty_placeholder_when_falling_back_to_content",
    "test_xinyu_list_still_prefers_unread_target_over_content_fallback",
}


@pytest.fixture(autouse=True)
def _stabilize_xinyu_list_selection_tests(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    if request.node.name not in _XINYU_LIST_SELECTION_TESTS:
        return
    monkeypatch.setattr(xinyu_steps, "ensure_list_screen", lambda robot, config, reporter=None: "list")
    monkeypatch.setattr(
        xinyu_steps,
        "check_unread_button",
        lambda robot, config, image, reporter=None, ocr_result=None: (image, ocr_result),
    )
