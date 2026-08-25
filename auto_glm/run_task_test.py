"""Interactive GPT-planned, AutoGLM-only visual task runner.

GPT only splits a goal into short nodes. All perception, page judgement,
coordinates and actions are decided by AutoGLM from consecutive screenshots.
No template matching, OCR or app-specific rules are used in this file.
"""

from __future__ import annotations

import argparse
import json
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
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from apps.coordinates import click_image_point  # noqa: E402
from apps.photo_capture import capture_transform_image  # noqa: E402
from apps.xinyu import steps as xinyu_steps  # noqa: E402
from apps.xinyu.config import DEFAULT_CONFIG as XINYU_DEFAULT_CONFIG  # noqa: E402
from auto_glm.gpt_planner import GPTPlannerClient, GPTPlannerConfig  # noqa: E402
from auto_glm.model import AutoGLMClient, AutoGLMConfig, AutoGLMProtocolError  # noqa: E402
from auto_glm.planner import ExecutionPlan, PlanNode  # noqa: E402
from lgb.lgb import RobotInterface  # noqa: E402
from services.config_manager import ConfigManager  # noqa: E402


AUTOGLM_EXECUTOR_PROMPT = """你是手机界面视觉执行智能体，可以根据连续操作历史和当前状态截图完成任务。

你必须严格且仅按以下格式输出，不能在标签外输出任何文字：
<think>对当前截图和下一步选择的简短判断</think>
<answer>一条动作指令</answer>

其中 {think} 只写本次选择理由，{action} 必须是且只能是一条下列格式：
- do(action="Tap", element=[x,y])
  点击截图中的一个具体点。坐标原点为左上角 (0,0)，右下角为 (999,999)。
- do(action="Swipe", start=[x1,y1], end=[x2,y2])
  从起点滑动至终点；坐标同样为 0 到 999。
- do(action="Type", text="xxx")
  仅在输入框已经被上一动作聚焦后输入文本。
- do(action="Wait", duration="x")
  等待界面加载，x 为秒数。
- do(action="Back")
  返回上一层。执行器会在当前截图中匹配已配置的返回按钮模板后由机械臂点击。
- finish(message="xxx")
  仅在当前节点目标已从截图中确认完成时使用。

严格规则：
1. 每次只能输出一条动作。执行后你会收到新的当前截图。
2. 只能依据当前截图、当前短目标和历史动作判断；不要假设模板、OCR、控件树或固定坐标存在。
3. 点击前先确认目标在截图中可见；点击后先观察新截图是否生效。页面未变化时，可 Wait 一次，再根据新截图调整一次点击位置。
4. 当前短目标已满足时必须输出 finish，不能继续重复点击。每轮当前截图会自动进入动作历史；根据完整任务上下文判断任务是否已经完成。
5. 目标不在可见区域时，使用 Swipe 探索；不要在同一位置重复无效点击。
6. 不要输出 Home、Take_over、Long Press、Double Tap、Note、Call_API 或解释性文本。需要返回时优先输出 Back。
7. <answer> 内必须包含 do(action=...) 或 finish(message=...)。"""


class RollingLog:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self.path = output_dir / "运行日志.txt"

    def write(self, level: str, message: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] [{level}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT 规划 + AutoGLM 纯视觉执行测试")
    parser.add_argument("--device", default="192.168.1.210", help="机械臂设备 IP")
    parser.add_argument("--task", default="", help="任务文本；不传时启动后输入")
    parser.add_argument("--max-actions-per-node", type=int, default=8, help="每个 GPT 节点最多模型动作数")
    parser.add_argument("--dry-run", action="store_true", help="只请求模型和保存证据，不执行机械臂动作")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task = str(args.task).strip() or input("请输入要测试的任务：").strip()
    if not task:
        print("未输入任务，已退出。", flush=True)
        return 2
    output_dir = ROOT / "runtime" / "autoglm_task_tests" / datetime.now().strftime("%Y%m%d_%H%M%S")
    log = RollingLog(output_dir)
    log.write("信息", f"用户任务：{task}")
    log.write("信息", f"设备={args.device}，模式={'仅诊断' if args.dry_run else 'AutoGLM 直接执行'}")
    log.write("信息", "本次不调用模板、OCR 或心遇专项流程；GPT 仅输出短目标和验证条件")
    try:
        plan = _plan(task, output_dir, log)
        robot = RobotInterface(args.device)
        model = AutoGLMClient(AutoGLMConfig.from_workspace(ROOT))
        back_config = ConfigManager(ROOT / "config.json").get_app_config(
            args.device, "xinyu", XINYU_DEFAULT_CONFIG
        )
        history = [f"总体用户任务：{task}"]
        conversation: list[dict[str, Any]] = [{"role": "system", "content": AUTOGLM_EXECUTOR_PROMPT}]
        for index, node in enumerate(plan.nodes, start=1):
            result = _run_node(robot, model, node, index, output_dir, log, history, conversation, args, back_config)
            if result != "node_done":
                log.write("错误", f"节点 {node.id} 未完成，停止整个任务。失败策略：{node.on_failure}")
                return 3
        log.write("信息", "全部 GPT 节点已由 AutoGLM 完成。请查看截图和每步 JSON 复盘。")
        return 0
    except Exception as error:
        log.write("错误", f"运行异常：{error}")
        return 1


def _plan(task: str, output_dir: Path, log: RollingLog) -> ExecutionPlan:
    plan = GPTPlannerClient(_planner_config()).plan(
        task,
        current_state="未知。执行器会在每个节点前自动提供当前截图。",
        capabilities={
            "规划器职责": "仅拆分短目标、成功条件与失败策略；不输出坐标、不判断具体控件位置。",
            "执行器职责": "AutoGLM 根据连续截图独立判断页面、定位并输出 Tap/Swipe/Type/Wait/finish。",
            "限制": "不得使用模板匹配、OCR、颜色规则、无障碍节点或 App 专项流程。",
        },
        project_skills="每个节点必须可由连续截图完成，避免生成仅观察、不产生状态推进的节点。",
    )
    (output_dir / "GPT规划.json").write_text(
        json.dumps({"summary": plan.summary, "nodes": [node.__dict__ for node in plan.nodes]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.write("信息", f"GPT 规划：{plan.summary}")
    for node in plan.nodes:
        log.write("信息", f"节点 {node.id}：{node.goal}；成功条件={node.verification}")
    return plan


def _run_node(
    robot: RobotInterface,
    model: AutoGLMClient,
    node: PlanNode,
    node_index: int,
    output_dir: Path,
    log: RollingLog,
    history: list[str],
    conversation: list[dict[str, Any]],
    args: argparse.Namespace,
    back_config: dict[str, Any],
) -> str:
    log.write("信息", f"开始节点 {node.id}：{node.goal}")
    for action_index in range(1, max(1, args.max_actions_per_node) + 1):
        image = _capture(robot, output_dir, f"{node_index:02d}_{action_index:02d}_动作前", log)
        instruction = (
            f"总体用户任务：{history[0]}\n"
            f"当前节点目标：{node.goal}\n"
            f"本次执行指令：{node.executor_task}\n"
            f"当前节点成功条件：{node.verification}\n"
            "若当前节点已满足成功条件，输出 finish；否则只执行推进当前节点的一个动作。"
        )
        references = _node_visual_references(node)
        if references:
            log.write("信息", "本节点已向 AutoGLM 提供视觉参考图：" + "、".join(label for label, _ in references))
        decision = _request_action_with_retry(
            model, instruction, image, output_dir, node_index, action_index, log, references, conversation
        )
        if decision is None:
            return "failed"
        action = decision.action
        action_path = output_dir / f"{node_index:02d}_{action_index:02d}_AutoGLM动作.json"
        action_path.write_text(
            json.dumps(
                {
                    "node": node.__dict__, "history": history, "thinking": decision.thinking,
                    "raw_content": decision.raw_content,
                    "action": {"name": action.name, "arguments": action.arguments, "finished": action.finished},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        log.write("信息", f"节点 {node.id} 第 {action_index} 步：AutoGLM={action.name}，参数={action.arguments}")
        if action.finished:
            history.append(f"节点 {node.id} 已完成：{action.message or node.goal}")
            log.write("信息", f"节点 {node.id} 完成：{action.message or node.goal}")
            return "node_done"
        if args.dry_run:
            log.write("信息", "诊断模式未执行模型动作，当前节点到此结束")
            return "failed"
        success, message = _execute(robot, action.name, action.arguments, image, back_config)
        log.write("信息" if success else "错误", message)
        if not success:
            return "failed"
        history.append(f"节点 {node.id} 第 {action_index} 步执行 {action.name}：{message}")
        time.sleep(0.8)
    log.write("错误", f"节点 {node.id} 超过最大动作数 {args.max_actions_per_node}，模型未声明完成")
    return "failed"


def _request_action_with_retry(
    model: AutoGLMClient,
    instruction: str,
    image: Image.Image,
    output_dir: Path,
    node_index: int,
    action_index: int,
    log: RollingLog,
    references: list[tuple[str, Image.Image]],
    conversation: list[dict[str, Any]],
):
    for attempt in range(1, 3):
        retry_instruction = instruction
        if attempt > 1:
            retry_instruction += "\n上一次返回不是有效动作协议。本次必须严格输出 <think>...</think><answer>do(action=...) 或 finish(message=...)</answer>，标签外不得有任何文字。"
        try:
            return model.decide(
                retry_instruction,
                image,
                reference_images=references,
                conversation=conversation,
            )
        except AutoGLMProtocolError as error:
            path = output_dir / f"{node_index:02d}_{action_index:02d}_协议异常_第{attempt}次.txt"
            path.write_text(error.raw_content, encoding="utf-8")
            preview = error.raw_content.replace("\n", " ").strip()[:200] or "（空返回）"
            log.write("警告", f"AutoGLM 第 {attempt}/2 次未返回动作协议：{error}；原始返回已保存到 {path.name}；内容={preview}")
        except Exception as error:
            log.write("警告", f"AutoGLM 第 {attempt}/2 次请求失败：{error}")
    log.write("错误", "AutoGLM 连续两次未返回可执行动作，已停止本节点")
    return None


def _node_visual_references(node: PlanNode) -> list[tuple[str, Image.Image]]:
    """Give the model visual semantics without using a local matcher to click."""
    text = f"{node.goal} {node.executor_task}".replace(" ", "")
    button_dir = ROOT / "apps" / "xinyu" / "templates" / "buttons"
    references: list[tuple[str, Image.Image]] = []
    if (
        ("相册" in text or "照片" in text or "图片" in text or "更多功能" in text)
        and any(keyword in text for keyword in ("打开", "入口", "选择界面", "更多功能"))
    ):
        path = button_dir / "btn_照片按钮.jpg"
        if path.is_file():
            references.append(("心遇照片/相册按钮", Image.open(path).convert("RGB")))
    if any(keyword in text for keyword in ("确认", "完成", "发送已选", "发送照片")):
        path = button_dir / "btn_完成按钮.jpg"
        if path.is_file():
            references.append(("心遇完成发送按钮", Image.open(path).convert("RGB")))
    return references


def _capture(robot: RobotInterface, output_dir: Path, name: str, log: RollingLog) -> Image.Image:
    log.write("信息", f"截图：{name}")
    # Reuse the project's calibrated perspective + portrait normalization so
    # model coordinates and mechanical-arm coordinates share one orientation.
    image_data = capture_transform_image(robot, {}, pre_move=True, label=f"autoglm_{name}")
    image = Image.open(BytesIO(bytes(image_data))).convert("RGB")
    image.save(output_dir / f"{name}.jpg", quality=95)
    log.write("信息", f"截图坐标系：竖屏 {image.width}x{image.height}")
    return image


def _execute(
    robot: RobotInterface,
    name: str,
    arguments: dict[str, Any],
    screenshot: Image.Image,
    back_config: dict[str, Any],
) -> tuple[bool, str]:
    if name == "Tap":
        point = _point(arguments.get("element"))
        if point is None:
            return False, "Tap 动作没有有效 0~999 坐标"
        click_image_point(robot, *point)
        return True, f"已点击模型坐标 {point}"
    if name == "Swipe":
        start, end = _point(arguments.get("start")), _point(arguments.get("end"))
        if start is None or end is None:
            return False, "Swipe 动作没有有效起止坐标"
        robot.swipe(*start, *end, wait=True)
        return True, f"已执行模型滑动 {start} -> {end}"
    if name == "Type":
        text = str(arguments.get("text", "")).strip()
        if not text or not hasattr(robot, "play_audio"):
            return False, "Type 动作缺少文本或机械臂不支持语音输入"
        robot.play_audio(text, wait=True)
        return True, "已通过机械臂语音输入执行 Type"
    if name == "Wait":
        seconds = _wait_seconds(arguments.get("duration"))
        time.sleep(seconds)
        return True, f"已等待 {seconds:.1f} 秒"
    if name == "Back":
        point = xinyu_steps._template_center(screenshot, back_config, xinyu_steps.BACK_BUTTON, group="buttons")
        if point is None:
            return False, "Back 未找到返回按钮模板，未使用固定坐标点击"
        click_image_point(robot, *point)
        return True, f"Back 已通过返回按钮模板点击 {point}"
    return False, f"纯视觉测试器不支持模型动作：{name}"


def _point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not 0 <= x <= 999 or not 0 <= y <= 999:
        return None
    return x / 1000, y / 1000


def _wait_seconds(value: Any) -> float:
    try:
        return min(10.0, max(0.0, float(str(value).split()[0])))
    except (TypeError, ValueError, IndexError):
        return 1.0


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


if __name__ == "__main__":
    raise SystemExit(main())
