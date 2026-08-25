"""Interactive GPT-planned / AutoGLM-executed task diagnostic runner.

Run ``python auto_glm/run_task_test.py`` and enter any task at the prompt.
The default mode records the complete plan and model actions without touching
the device. Add ``--execute-model-actions`` only for a deliberate live test.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import tomllib
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from apps.coordinates import click_image_point  # noqa: E402
from apps.xinyu import steps  # noqa: E402
from apps.xinyu.config import DEFAULT_CONFIG  # noqa: E402
from auto_glm.gpt_planner import GPTPlannerClient, GPTPlannerConfig  # noqa: E402
from auto_glm.model import AutoGLMClient, AutoGLMConfig  # noqa: E402
from auto_glm.planner import ExecutionPlan, PlanNode  # noqa: E402
from auto_glm.verification import compare_images  # noqa: E402
from lgb.lgb import RobotInterface  # noqa: E402


class RollingLog:
    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.path = output_dir / "运行日志.txt"

    def log(self, level: str, message: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] [{level}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def _is_photo_send_task(task: str) -> bool:
    normalized = "".join(str(task).split())
    return ("相册" in normalized or "照片" in normalized or "图片" in normalized) and any(
        word in normalized for word in ("发", "发送", "给对面", "给他", "给她")
    )


def _photo_send_plan() -> ExecutionPlan:
    return ExecutionPlan(
        summary="心遇已知发图流程：打开相册、随机选图、完成发送、回到聊天界面验证。",
        nodes=(
            PlanNode("photo_open", "打开相册", "点击聊天编辑区左下角的照片按钮，打开系统相册。", "截图不再是聊天界面。", "保存模板与截图诊断并停止。", "template"),
            PlanNode("photo_random_select", "随机选择一张照片", "在图片列表的已配置随机范围中随机点击一张照片。", "截图进入照片预览或出现完成按钮。", "保存截图并停止。", "program"),
            PlanNode("photo_complete", "完成并发送照片", "点击照片预览页的完成按钮，发送照片。", "完成按钮消失且返回聊天界面。", "保存模板候选和截图并停止。", "template"),
            PlanNode("photo_verify", "验证发送结果", "确认已回到聊天界面。", "聊天界面模板命中。", "保存发送后截图并停止。", "program"),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="交互式 GPT + AutoGLM 任务诊断")
    parser.add_argument("--device", default="192.168.1.210", help="机械臂设备 IP")
    parser.add_argument("--task", default="", help="任务文本；不传时启动后输入")
    parser.add_argument("--max-steps", type=int, default=8, help="最多执行的规划节点数")
    parser.add_argument(
        "--execute-model-actions",
        action="store_true",
        help="允许将模型 Tap/Swipe 坐标直接交给机械臂，仅用于明确授权的测试",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task = str(args.task).strip()
    if not task:
        task = input("请输入要测试的任务：").strip()
    if not task:
        print("未输入任务，已退出。", flush=True)
        return 2
    if not args.execute_model_actions:
        choice = input("是否将模型 Tap/Swipe 坐标直接执行到机械臂？输入 YES 执行，其它任意输入仅诊断：").strip()
        args.execute_model_actions = choice.upper() == "YES"

    output_dir = ROOT / "runtime" / "autoglm_task_tests" / datetime.now().strftime("%Y%m%d_%H%M%S")
    log = RollingLog(output_dir)
    log.log("信息", f"任务={task}")
    log.log("信息", f"设备={args.device}，输出目录={output_dir}")
    if args.execute_model_actions:
        log.log("警告", "已启用模型坐标直接执行：仅适合明确授权的测试设备")
    else:
        log.log("信息", "当前是诊断模式：只记录模型动作，不点击设备。追加 --execute-model-actions 才会实际执行")

    try:
        plan = _build_plan(task, output_dir, log)
        robot = RobotInterface(args.device)
        model = AutoGLMClient(AutoGLMConfig.from_workspace(ROOT))
        config = _xinyu_config()
        history: list[str] = []
        for index, node in enumerate(plan.nodes[: max(1, args.max_steps)], start=1):
            if not _run_node(robot, model, node, index, config, output_dir, log, history, args.execute_model_actions):
                log.log("警告", f"节点 {node.id} 未验证，停止后续节点。请检查本次输出目录。")
                return 3
        log.log("信息", "任务测试结束。请依次检查运行日志、截图、GPT规划和 AutoGLM 动作文件。")
        return 0
    except Exception as error:
        log.log("错误", f"任务测试异常：{error}")
        return 1


def _build_plan(task: str, output_dir: Path, log: RollingLog) -> ExecutionPlan:
    if _is_photo_send_task(task):
        plan = _photo_send_plan()
        log.log("信息", "识别到“相册随机发图”任务，使用已知的直接执行节点，不生成观察节点")
        return _save_plan(plan, output_dir, log)
    try:
        plan = GPTPlannerClient(_planner_config()).plan(
            task,
            current_state="启动时必须先截图确认；状态未知时不能猜测点击。",
            capabilities={
                "感知": ["相机截图", "模板匹配", "OCR", "颜色检测", "AutoGLM 截图理解"],
                "执行": ["机械臂点击", "机械臂滑动"],
                "限制": ["每次仅一个动作", "每个动作后截图验证", "不确定时停止并记录"],
            },
            project_skills="当前项目使用真实摄像头和机械臂，不使用 ADB 或无障碍节点。",
        )
    except Exception as error:
        log.log("警告", f"GPT 规划失败，回退为单节点模型探索：{error}")
        plan = ExecutionPlan(
            summary="GPT 规划不可用，使用单个 AutoGLM 探索节点。",
            nodes=(
                PlanNode(
                    id="model_explore",
                    goal=task,
                    executor_task=task,
                    verification="动作后截图应出现与任务目标一致的可观察变化。",
                    on_failure="保存截图和动作结果后停止。",
                ),
            ),
        )
    return _save_plan(plan, output_dir, log)


def _save_plan(plan: ExecutionPlan, output_dir: Path, log: RollingLog) -> ExecutionPlan:
    payload = {"summary": plan.summary, "nodes": [node.__dict__ for node in plan.nodes]}
    (output_dir / "GPT规划.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.log("信息", f"GPT 规划：{plan.summary}")
    for node in plan.nodes:
        log.log("信息", f"节点 {node.id}：{node.goal}；验证={node.verification}")
    return plan


def _run_node(
    robot: RobotInterface,
    model: AutoGLMClient,
    node: PlanNode,
    index: int,
    config: dict[str, Any],
    output_dir: Path,
    log: RollingLog,
    history: list[str],
    execute: bool,
) -> bool:
    before = _capture(robot, output_dir, f"{index:02d}_动作前", log)
    state = _screen_state(before, config)
    if node.id.startswith("photo_"):
        return _run_known_photo_node(robot, model, node, index, before, state, config, output_dir, log, execute)
    log.log("信息", f"节点 {node.id} 当前界面={state}；请求 AutoGLM：{node.executor_task}")
    decision = model.decide(node.executor_task, before, history=history)
    action = decision.action
    action_path = output_dir / f"{index:02d}_AutoGLM动作.json"
    action_path.write_text(
        json.dumps(
            {
                "node": node.__dict__,
                "thinking": decision.thinking,
                "raw_content": decision.raw_content,
                "action": {"name": action.name, "arguments": action.arguments, "finished": action.finished},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log.log("信息", f"AutoGLM 动作={action.name}，参数={action.arguments}，文件={action_path.name}")
    if action.finished:
        log.log("信息", f"模型声明完成：{action.message}")
        return True
    if not execute:
        log.log("信息", "诊断模式未执行动作；已保存候选动作供查看")
        return False
    if action.name == "Tap":
        point = _model_point(action.arguments.get("element"))
        if point is None:
            log.log("错误", "模型 Tap 动作没有 0~999 坐标")
            return False
        log.log("警告", f"执行模型 Tap：({point[0]:.3f}, {point[1]:.3f})")
        click_image_point(robot, *point)
    elif action.name == "Swipe":
        start, end = _model_point(action.arguments.get("start")), _model_point(action.arguments.get("end"))
        if start is None or end is None:
            log.log("错误", "模型 Swipe 动作没有有效起止坐标")
            return False
        robot.swipe(*start, *end, wait=True)
        log.log("警告", f"执行模型滑动：{start} -> {end}")
    elif action.name == "Wait":
        seconds = min(10.0, _wait_seconds(action.arguments.get("duration")))
        log.log("信息", f"执行等待：{seconds:.1f} 秒")
        time.sleep(seconds)
    else:
        log.log("错误", f"通用测试器不执行 {action.name}；需由具体 App 提供输入/返回适配器")
        return False

    time.sleep(0.8)
    after = _capture(robot, output_dir, f"{index:02d}_动作后", log)
    transition = compare_images(before, after)
    history.append(f"节点 {node.id} 动作={action.name}；画面变化={transition.changed}；差异={transition.difference_ratio:.4f}")
    log.log("信息", f"节点 {node.id} 画面验证：{transition.message}，差异={transition.difference_ratio:.4f}")
    return transition.changed


def _run_known_photo_node(
    robot: RobotInterface,
    model: AutoGLMClient,
    node: PlanNode,
    index: int,
    before: Image.Image,
    state: str,
    config: dict[str, Any],
    output_dir: Path,
    log: RollingLog,
    execute: bool,
) -> bool:
    """Run the known Xinyu photo flow directly; no observation is delegated to the model."""
    if node.id == "photo_verify":
        log.log("信息", f"发送后验证：当前界面={state}")
        return state == steps.CHAT_WINDOW_SCREEN
    if node.id == "photo_open":
        if state != steps.CHAT_WINDOW_SCREEN:
            log.log("错误", f"当前不是聊天界面，不能打开相册：{state}")
            return False
        point = steps._template_center(before, config, steps.PHOTO_BUTTON, group="buttons")
        if point is None:
            return _photo_model_fallback(robot, model, node, index, before, config, steps.PHOTO_BUTTON, output_dir, log, execute)
        log.log("信息", f"本地照片模板命中：{point}，直接点击")
    elif node.id == "photo_random_select":
        if state == steps.CHAT_WINDOW_SCREEN:
            log.log("错误", "仍在聊天界面，不能随机选图")
            return False
        left, top, right, bottom = map(float, config["photo_select_random_bounds"])
        point = (random.uniform(left, right), random.uniform(top, bottom))
        log.log("信息", f"随机选图：范围={config['photo_select_random_bounds']}，点击={point}")
    elif node.id == "photo_complete":
        point = steps._template_center(before, config, steps.COMPLETE_BUTTON, group="buttons")
        if point is None:
            return _photo_model_fallback(robot, model, node, index, before, config, steps.COMPLETE_BUTTON, output_dir, log, execute)
        log.log("信息", f"本地完成按钮模板命中：{point}，直接点击发送")
    else:
        log.log("错误", f"未知内置发图节点：{node.id}")
        return False
    if not execute:
        log.log("信息", "诊断模式未执行机械臂点击")
        return False
    click_image_point(robot, *point)
    time.sleep(0.8)
    after = _capture(robot, output_dir, f"{index:02d}_动作后", log)
    transition = compare_images(before, after)
    log.log("信息", f"节点 {node.id} 画面验证：{transition.message}，差异={transition.difference_ratio:.4f}")
    return transition.changed


def _photo_model_fallback(
    robot: RobotInterface,
    model: AutoGLMClient,
    node: PlanNode,
    index: int,
    image: Image.Image,
    config: dict[str, Any],
    template_key: str,
    output_dir: Path,
    log: RollingLog,
    execute: bool,
) -> bool:
    """Use model position only after it stays in the configured template range."""
    log.log("警告", f"本地模板未命中：{template_key}；请求 AutoGLM 候选定位")
    decision = model.decide(node.executor_task, image)
    (output_dir / f"{index:02d}_AutoGLM模板候选.json").write_text(
        json.dumps({"raw_content": decision.raw_content, "action": decision.action.arguments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    point = _model_point(decision.action.arguments.get("element")) if decision.action.name == "Tap" else None
    if point is None:
        log.log("错误", f"AutoGLM 未给出有效点击候选：{decision.action.name}")
        return False
    region = config["template_match_regions"].get(template_key)
    if not isinstance(region, list) or not _in_region(point, region):
        log.log("错误", f"AutoGLM 候选超出 {template_key} 配置范围：候选={point}，范围={region}")
        _save_candidate_crop(image, point, template_key, output_dir)
        return False
    _save_candidate_crop(image, point, template_key, output_dir)
    log.log("信息", f"AutoGLM 候选已落在范围内：{point}；候选模板截图已保存")
    if not execute:
        return False
    click_image_point(robot, *point)
    time.sleep(0.8)
    after = _capture(robot, output_dir, f"{index:02d}_模型候选动作后", log)
    transition = compare_images(image, after)
    log.log("信息", f"模型候选动作验证：{transition.message}，差异={transition.difference_ratio:.4f}")
    return transition.changed


def _capture(robot: RobotInterface, output_dir: Path, name: str, log: RollingLog) -> Image.Image:
    log.log("信息", f"截图：{name}")
    image = Image.open(BytesIO(robot.get_transform_image(pre_move=True))).convert("RGB")
    image.save(output_dir / f"{name}.jpg", quality=95)
    return image


def _planner_config() -> GPTPlannerConfig:
    path = Path.home() / ".codex" / "config.toml"
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    provider_id = str(payload.get("model_provider", ""))
    provider = payload.get("model_providers", {}).get(provider_id, {})
    base_url = str(provider.get("base_url", "")).rstrip("/")
    model = str(payload.get("model", ""))
    if not base_url or not model:
        raise RuntimeError("未找到 GPT 模型地址或模型名")
    return GPTPlannerConfig.from_workspace(ROOT, model=model, endpoint=f"{base_url}/chat/completions")


def _xinyu_config() -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config["templates"] = dict(DEFAULT_CONFIG["templates"])
    config["template_match_regions"] = dict(DEFAULT_CONFIG["template_match_regions"])
    config["template_match_thresholds"] = dict(DEFAULT_CONFIG["template_match_thresholds"])
    return config


def _model_point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not 0 <= x <= 999 or not 0 <= y <= 999:
        return None
    return x / 1000, y / 1000


def _in_region(point: tuple[float, float], region: list[float]) -> bool:
    left, top, width, height = map(float, region)
    return left <= point[0] <= left + width and top <= point[1] <= top + height


def _save_candidate_crop(image: Image.Image, point: tuple[float, float], key: str, output_dir: Path) -> None:
    width, height = image.size
    center_x, center_y = int(point[0] * width), int(point[1] * height)
    half_width, half_height = max(32, width // 16), max(24, height // 22)
    crop = image.crop(
        (
            max(0, center_x - half_width),
            max(0, center_y - half_height),
            min(width, center_x + half_width),
            min(height, center_y + half_height),
        )
    )
    crop.save(output_dir / f"候选模板_{key.replace('.', '_')}.jpg", quality=95)


def _wait_seconds(value: Any) -> float:
    try:
        return max(0.0, float(str(value).split()[0]))
    except (TypeError, ValueError, IndexError):
        return 1.0


def _screen_state(image: Image.Image, config: dict[str, Any]) -> str:
    """Diagnostic runner must not invoke OCR merely to label the page."""
    if steps._matches_screen(image, config, steps.CHAT_WINDOW_SCREEN):
        return steps.CHAT_WINDOW_SCREEN
    if steps._matches_screen(image, config, steps.CHAT_LIST_SCREEN):
        return steps.CHAT_LIST_SCREEN
    if steps._matches_screen(image, config, steps.CALL_SCREEN):
        return steps.CALL_SCREEN
    return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
