from __future__ import annotations

import inspect
import json

from PIL import Image

import services.standard_api as standard_api


def _square_scene() -> tuple[Image.Image, Image.Image]:
    """100x100 灰底图，右下角 (60,40) 贴一个 20x20 深色方块。"""
    base = Image.new("RGB", (100, 100), (240, 240, 240))
    square = Image.new("RGB", (20, 20), (10, 10, 10))
    base.paste(square, (60, 40))
    return base, square


class FakeRobot:
    def __init__(self) -> None:
        self.clicked: list[tuple[float, float]] = []
        self.swipes: list[tuple[float, float, float, float]] = []

    def move_click(self, x: float, y: float) -> dict:
        self.clicked.append((x, y))
        return {"ok": True}

    def swipe(self, start_x: float, start_y: float, end_x: float, end_y: float, wait: bool = True) -> dict:
        self.swipes.append((start_x, start_y, end_x, end_y))
        return {"ok": True}


def test_manifest_covers_all_registered_apis() -> None:
    manifest = standard_api.api_manifest()
    names = {entry["name"] for entry in manifest}
    expected = set(standard_api.__all__) - {"api_manifest", "api_tools_schema", "api_markdown"}
    assert names == expected

    for entry in manifest:
        assert entry["summary"] and entry["returns"]
        func = getattr(standard_api, entry["name"])
        signature = inspect.signature(func)
        assert [param["name"] for param in entry["params"]] == list(signature.parameters)
        for param, (_, real) in zip(entry["params"], signature.parameters.items()):
            assert param["required"] == (real.default is inspect.Parameter.empty)


def test_manifest_and_tools_are_json_serializable() -> None:
    json.dumps(standard_api.api_manifest(), ensure_ascii=False)
    tools = standard_api.api_tools_schema()
    json.dumps(tools, ensure_ascii=False)
    manifest_names = {entry["name"] for entry in standard_api.api_manifest()}
    assert {tool["function"]["name"] for tool in tools} == manifest_names
    allowed_types = {"number", "integer", "string", "boolean", "array", "object"}
    for tool in tools:
        for property_schema in tool["function"]["parameters"]["properties"].values():
            assert property_schema["type"] in allowed_types


def test_find_template_full_image_and_region_offset() -> None:
    base, square = _square_scene()

    full = standard_api.find_template(base, square, threshold=0.8)
    assert full is not None and full["score"] >= 0.8
    assert abs(full["x"] - 0.6) < 0.03
    assert abs(full["y"] - 0.4) < 0.03
    assert abs(full["w"] - 0.2) < 0.03
    assert abs(full["h"] - 0.2) < 0.03

    # region 只覆盖右下象限时，返回坐标必须换算回整图坐标系。
    part = standard_api.find_template(base, square, threshold=0.8, region=[0.5, 0.25, 0.5, 0.5])
    assert part is not None
    for key in ("x", "y", "w", "h"):
        assert abs(part[key] - full[key]) < 0.02, key


def test_template_exists_and_negative() -> None:
    base, square = _square_scene()
    assert standard_api.template_exists(base, square, threshold=0.8) is True

    # 模板比图大时永远匹配不上。
    big = Image.new("RGB", (120, 120), (10, 10, 10))
    assert standard_api.template_exists(base, big, threshold=0.5) is False
    assert standard_api.find_template(base, big, threshold=0.5) is None


def test_click_and_swipe_delegate_to_robot() -> None:
    robot = FakeRobot()
    standard_api.click(robot, 0.25, 0.75)
    assert robot.clicked == [(0.25, 0.75)]

    match = {"x": 0.6, "y": 0.4, "w": 0.2, "h": 0.2}
    standard_api.click_match(robot, match)
    assert robot.clicked[-1] == (0.7, 0.5)

    result = standard_api.swipe(robot, 0.5, 0.8, 0.5, 0.2)
    assert result == {"ok": True}
    assert robot.swipes == [(0.5, 0.8, 0.5, 0.2)]


def test_read_text_without_engine_returns_empty() -> None:
    base, _ = _square_scene()
    assert standard_api.read_text(base) == []
    assert standard_api.read_text(base, region=[0.1, 0.1, 0.5, 0.5]) == []


def test_speak_delegates_to_tts_input(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        standard_api,
        "tts_input",
        lambda robot, text, max_chars=40, pause_seconds=0.8: calls.append((text, max_chars)),
    )
    standard_api.speak(FakeRobot(), "你好", max_chars=20)
    assert calls == [("你好", 20)]


def test_understand_image_merges_prompt_path(monkeypatch) -> None:
    captured: dict = {}

    class FakeClient:
        def __init__(self, *, workspace_root=None) -> None:
            captured["workspace_root"] = workspace_root

        def parse_dialogue(self, image, config):
            captured["config"] = config
            return "reply"

    monkeypatch.setattr(standard_api, "VisionDialogueClient", FakeClient)
    reply = standard_api.understand_image("img-bytes", {"other": 1}, prompt_path="apps/xinyu/prompt.txt")
    assert reply == "reply"
    assert captured["config"]["other"] == 1
    assert captured["config"]["dialogue_llm"]["prompt_path"] == "apps/xinyu/prompt.txt"

    standard_api.understand_image("img-bytes", {"dialogue_llm": {"prompt_path": "keep.txt"}})
    assert captured["config"]["dialogue_llm"]["prompt_path"] == "keep.txt"


def test_api_markdown_renders_every_api() -> None:
    markdown = standard_api.api_markdown()
    for name in set(standard_api.__all__) - {"api_manifest", "api_tools_schema", "api_markdown"}:
        assert f"`{name}(" in markdown
