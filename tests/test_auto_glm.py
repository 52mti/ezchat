from __future__ import annotations

import numpy as np
import pytest

from auto_glm.adapter import RobotActionAdapter
from auto_glm.gpt_planner import GPTPlannerClient, GPTPlannerConfig
from auto_glm.model import AutoGLMConfig, AutoGLMProtocolError
from auto_glm.planner import ExecutionPlan
from auto_glm.protocol import parse_action
from auto_glm.verification import compare_images, image_signature, verify_transition


class FakeRobot:
    def __init__(self) -> None:
        self.clicks: list[tuple[float, float]] = []
        self.swipes: list[tuple[float, float, float, float]] = []

    def move_click(self, x: float, y: float) -> None:
        self.clicks.append((x, y))

    def swipe(self, start_x: float, start_y: float, end_x: float, end_y: float, wait: bool = True) -> None:
        self.swipes.append((start_x, start_y, end_x, end_y))


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {"choices": [{"message": {"content": '{"summary":"测试","nodes":[{"id":"n1","goal":"定位","executor_task":"点击按钮","verification":"页面变化","on_failure":"记录并停止"}]}'}}]}


class FakeSession:
    def post(self, *args, **kwargs):
        return FakeResponse()


def test_parse_autoglm_tap_from_answer_markup():
    action = parse_action('<think>点击相册</think><answer>do(action="Tap", element=[86,875])</answer>')

    assert action.name == "Tap"
    assert action.arguments["element"] == [86, 875]


def test_parse_finish_keeps_quotes_in_official_terminal_message():
    action = parse_action('finish(message="已进入\"照片\"选择界面")')

    assert action.finished
    assert action.name == "finish"
    assert action.message == '已进入"照片"选择界面'


def test_parse_official_launch_action():
    action = parse_action('do(action="Launch", app="相册")')

    assert action.name == "Launch"
    assert action.arguments["app"] == "相册"


def test_parse_rejects_out_of_range_coordinates():
    with pytest.raises(ValueError, match="0 到 999"):
        parse_action('do(action="Tap", element=[1000, 2])')


def test_adapter_converts_thousandth_coordinates_to_robot_coordinates():
    robot = FakeRobot()
    result = RobotActionAdapter(robot, point_resolver=lambda point, action: point).execute(
        parse_action('do(action="Tap", element=[250,750])')
    )

    assert result.success
    assert robot.clicks == [(0.25, 0.75)]


def test_adapter_rejects_unverified_model_click():
    result = RobotActionAdapter(FakeRobot()).execute(parse_action('do(action="Tap", element=[250,750])'))

    assert not result.success
    assert "局部复核器" in result.message


def test_adapter_keeps_back_app_specific():
    result = RobotActionAdapter(FakeRobot()).execute(parse_action('do(action="Back")'))

    assert not result.success
    assert "返回执行器" in result.message


def test_plan_requires_verification_and_failure_policy():
    plan = ExecutionPlan.from_json(
        {
            "summary": "打开相册",
            "nodes": [{"id": "open_photo", "goal": "打开相册", "executor_task": "点击照片按钮", "verification": "出现相册", "on_failure": "保存证据并停止"}],
        }
    )

    assert plan.nodes[0].executor_type == "autoglm"


def test_gpt_planner_returns_strict_execution_plan():
    client = GPTPlannerClient(
        GPTPlannerConfig(api_key="test", model="gpt", endpoint="https://example.test"), session=FakeSession()
    )

    plan = client.plan("发照片", current_state="聊天界面", capabilities={"camera": True})

    assert plan.summary == "测试"
    assert plan.nodes[0].verification == "页面变化"


def test_model_configs_load_keys_from_workspace_env(tmp_path):
    env_dir = tmp_path / ".env"
    env_dir.mkdir()
    (env_dir / "gpt_planner.txt").write_text("OPENAI_API_KEY=gpt-test\n", encoding="utf-8")
    (env_dir / "autoglm.txt").write_text("AUTOGLM_API_KEY=glm-test\n", encoding="utf-8")

    planner = GPTPlannerConfig.from_workspace(tmp_path, model="gpt", endpoint="https://example.test")
    executor = AutoGLMConfig.from_workspace(tmp_path)

    assert planner.api_key == "gpt-test"
    assert executor.api_key == "glm-test"


def test_protocol_error_preserves_raw_model_reply():
    error = AutoGLMProtocolError("未找到动作", "模型只返回了说明文字")

    assert str(error) == "未找到动作"
    assert error.raw_content == "模型只返回了说明文字"


def test_visual_transition_requires_meaningful_change():
    before = np.zeros((100, 60, 3), dtype=np.uint8)
    same = before.copy()
    changed = np.full((100, 60, 3), 255, dtype=np.uint8)

    assert image_signature(before) == image_signature(same)
    assert not compare_images(before, same).changed
    assert compare_images(before, changed).changed
    assert verify_transition(before, lambda: changed, stable_samples=2).changed
