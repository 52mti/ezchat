"""Run a traceable GPT + AutoGLM photo-send test on one Xinyu device.

Run from the project root:
    python auto_glm/run_photo_send_test.py
    python auto_glm/run_photo_send_test.py --device 192.168.1.210

The script never uses a model click point directly.  It records the model
candidate, then requires a local template/OCR rule to provide the actual
mechanical-arm click point.
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
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from apps.coordinates import click_image_point  # noqa: E402
from apps.xinyu import steps  # noqa: E402
from apps.xinyu.config import DEFAULT_CONFIG  # noqa: E402
from apps.visual_assets import format_template_match_diagnostic  # noqa: E402
from auto_glm.gpt_planner import GPTPlannerClient, GPTPlannerConfig  # noqa: E402
from auto_glm.model import AutoGLMClient, AutoGLMConfig  # noqa: E402
from auto_glm.verification import compare_images  # noqa: E402
from lgb.lgb import RobotInterface  # noqa: E402


class TestLogger:
    """Write line-by-line Chinese logs to both the console and a run folder."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "运行日志.txt"

    def write(self, level: str, message: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] [{level}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def info(self, message: str) -> None:
        self.write("信息", message)

    def warning(self, message: str) -> None:
        self.write("警告", message)

    def error(self, message: str) -> None:
        self.write("错误", message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="心遇 GPT + AutoGLM 发图链路诊断")
    parser.add_argument("--device", default="192.168.1.210", help="机械臂设备 IP")
    parser.add_argument("--skip-planner", action="store_true", help="不调用 GPT 规划器")
    parser.add_argument("--skip-model", action="store_true", help="不调用 AutoGLM 候选定位")
    parser.add_argument("--dry-run", action="store_true", help="只识别和记录，不执行机械臂点击")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = ROOT / "runtime" / "autoglm_photo_tests" / datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TestLogger(output_dir)
    logger.info(f"开始心遇发图诊断，设备={args.device}，输出目录={output_dir}")
    logger.info("所有日志会实时滚动输出；每一步的截图和模型结果均保存到输出目录")

    config = _xinyu_config()
    robot = RobotInterface(args.device)
    try:
        _write_plan(logger, output_dir, args)
        before = _capture(robot, output_dir, "01_开始截图", logger)
        screen = steps.current_screen(before, config)
        logger.info(f"当前界面判定={screen}")

        if screen == steps.CHAT_WINDOW_SCREEN:
            if not _open_album(robot, before, config, output_dir, logger, args):
                return 2
            selection = _capture(robot, output_dir, "03_相册页面", logger)
            if not _select_random_photo(robot, selection, config, output_dir, logger, args):
                return 3
        else:
            logger.warning("当前不在已确认聊天页，跳过打开相册和随机选图，直接检查是否已在选图预览页")

        preview = _capture(robot, output_dir, "05_发送前预览", logger)
        if not _complete_send(robot, preview, config, output_dir, logger, args):
            return 4
        after = _capture(robot, output_dir, "07_发送后页面", logger)
        _verify_after_send(after, config, logger)
        logger.info("测试完成：已执行到发送后验证。请查看输出目录中的截图和运行日志。")
        return 0
    except Exception as error:
        logger.error(f"测试停止：{error}")
        logger.error("未满足定位或验证条件时不会继续猜测点击。请查看最后一张截图和模板诊断。")
        return 1


def _xinyu_config() -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config["templates"] = dict(DEFAULT_CONFIG["templates"])
    config["template_match_regions"] = dict(DEFAULT_CONFIG["template_match_regions"])
    config["template_match_thresholds"] = dict(DEFAULT_CONFIG["template_match_thresholds"])
    return config


def _capture(robot: RobotInterface, output_dir: Path, label: str, logger: TestLogger) -> Image.Image:
    target = output_dir / f"{label}.jpg"
    logger.info(f"截图：{label}")
    image = Image.open(BytesIO(robot.get_transform_image(pre_move=True))).convert("RGB")
    image.save(target, quality=95)
    logger.info(f"已保存截图：{target.name}，尺寸={image.width}x{image.height}")
    return image


def _write_plan(logger: TestLogger, output_dir: Path, args: argparse.Namespace) -> None:
    if args.skip_planner:
        logger.info("已跳过 GPT 规划器")
        return
    try:
        planner = GPTPlannerClient(_planner_config())
        plan = planner.plan(
            "在当前心遇聊天界面，打开相册，随机选择一张照片并发送给对面。",
            current_state="需以启动截图确认；本地模板、OCR、颜色规则优先。",
            capabilities={
                "感知": ["截图", "模板匹配", "OCR", "颜色检测"],
                "执行": ["机械臂点击", "机械臂滑动"],
                "限制": ["模型坐标仅是候选，必须局部复核", "每步后截图验证"],
            },
            project_skills="照片按钮=button.photo；随机选图范围=[0.3,0.2,0.9,0.5]；完成按钮=button.complete。",
        )
        payload = {
            "summary": plan.summary,
            "nodes": [node.__dict__ for node in plan.nodes],
        }
        (output_dir / "GPT规划.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"GPT 规划完成：{plan.summary}")
        for node in plan.nodes:
            logger.info(f"规划节点 {node.id}：{node.goal}；执行器={node.executor_type}；验证={node.verification}")
    except Exception as error:
        logger.warning(f"GPT 规划器调用失败，继续执行本地诊断：{error}")


def _planner_config() -> GPTPlannerConfig:
    codex_config = Path.home() / ".codex" / "config.toml"
    if not codex_config.is_file():
        raise RuntimeError("未找到 GPT 规划器配置：~/.codex/config.toml")
    payload = tomllib.loads(codex_config.read_text(encoding="utf-8"))
    provider_id = str(payload.get("model_provider", ""))
    provider = payload.get("model_providers", {}).get(provider_id, {})
    base_url = str(provider.get("base_url", "")).rstrip("/")
    model = str(payload.get("model", ""))
    if not base_url or not model:
        raise RuntimeError("GPT 规划器缺少模型名或服务地址")
    return GPTPlannerConfig.from_workspace(ROOT, model=model, endpoint=f"{base_url}/chat/completions")


def _open_album(
    robot: RobotInterface,
    image: Image.Image,
    config: dict[str, Any],
    output_dir: Path,
    logger: TestLogger,
    args: argparse.Namespace,
) -> bool:
    candidate = _model_candidate(image, "点击底部左侧相册/照片按钮，打开照片选择界面。只执行本步骤。", output_dir, logger, args)
    region = config["template_match_regions"][steps.PHOTO_BUTTON]
    if candidate and not _in_region(candidate, region):
        logger.warning(f"AutoGLM 照片候选超出配置范围：{_point(candidate)}，范围={region}")
    elif candidate:
        logger.info(f"AutoGLM 照片候选在配置范围内：{_point(candidate)}")

    point = steps._template_center(image, config, steps.PHOTO_BUTTON, group="buttons")
    if point is None:
        logger.error(f"未找到照片按钮模板：{format_template_match_diagnostic(config, steps.PHOTO_BUTTON)}")
        return False
    logger.info(f"本地照片模板命中：{_point(point)}，将使用该点而非模型候选")
    if args.dry_run:
        logger.warning("试运行模式：未点击照片按钮")
        return False
    click_image_point(robot, *point)
    logger.info("已点击照片按钮，等待界面切换")
    time.sleep(0.8)
    return True


def _select_random_photo(
    robot: RobotInterface,
    image: Image.Image,
    config: dict[str, Any],
    output_dir: Path,
    logger: TestLogger,
    args: argparse.Namespace,
) -> bool:
    if steps.current_screen(image, config) == steps.CHAT_WINDOW_SCREEN:
        logger.error("点击照片按钮后仍在聊天界面，未进入选图页")
        return False
    left, top, right, bottom = map(float, config["photo_select_random_bounds"])
    point = random.uniform(left, right), random.uniform(top, bottom)
    logger.info(f"随机选图范围={config['photo_select_random_bounds']}，本次点击={_point(point)}")
    if args.dry_run:
        logger.warning("试运行模式：未点击随机照片")
        return False
    click_image_point(robot, *point)
    logger.info("已点击随机照片，等待预览页")
    time.sleep(0.8)
    return True


def _complete_send(
    robot: RobotInterface,
    image: Image.Image,
    config: dict[str, Any],
    output_dir: Path,
    logger: TestLogger,
    args: argparse.Namespace,
) -> bool:
    candidate = _model_candidate(image, "当前已选择一张照片。点击用于确认发送该照片的完成按钮。只执行本步骤。", output_dir, logger, args)
    if candidate:
        logger.info(f"AutoGLM 完成按钮候选：{_point(candidate)}")
    point = steps._template_center(image, config, steps.COMPLETE_BUTTON, group="buttons")
    if point is None:
        logger.error(f"未找到完成按钮模板：{format_template_match_diagnostic(config, steps.COMPLETE_BUTTON)}")
        logger.error("本次已停止发送；请检查 05_发送前预览.jpg 后补充当前界面模板或范围")
        return False
    logger.info(f"本地完成按钮模板命中：{_point(point)}")
    if args.dry_run:
        logger.warning("试运行模式：未点击完成按钮")
        return False
    click_image_point(robot, *point)
    logger.info("已点击完成按钮，等待发送完成")
    time.sleep(1.0)
    return True


def _model_candidate(
    image: Image.Image,
    task: str,
    output_dir: Path,
    logger: TestLogger,
    args: argparse.Namespace,
) -> tuple[float, float] | None:
    if args.skip_model:
        logger.info("已跳过 AutoGLM 候选定位")
        return None
    try:
        decision = AutoGLMClient(AutoGLMConfig.from_workspace(ROOT)).decide(task, image)
        payload = {
            "task": task,
            "thinking": decision.thinking,
            "raw_content": decision.raw_content,
            "action": {"name": decision.action.name, "arguments": decision.action.arguments},
        }
        name = f"AutoGLM_{len(list(output_dir.glob('AutoGLM_*.json'))) + 1}.json"
        (output_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if decision.action.name != "Tap":
            logger.warning(f"AutoGLM 未返回点击动作：{decision.action.name}")
            return None
        point = decision.action.arguments.get("element")
        if not isinstance(point, list) or len(point) != 2:
            logger.warning("AutoGLM 点击动作没有有效坐标")
            return None
        return float(point[0]) / 1000, float(point[1]) / 1000
    except Exception as error:
        logger.warning(f"AutoGLM 候选定位失败：{error}")
        return None


def _verify_after_send(image: Image.Image, config: dict[str, Any], logger: TestLogger) -> None:
    screen = steps.current_screen(image, config)
    complete = steps._template_center(image, config, steps.COMPLETE_BUTTON, group="buttons")
    logger.info(f"发送后界面判定={screen}，完成按钮={'仍存在' if complete else '未检测到'}")
    if screen != steps.CHAT_WINDOW_SCREEN:
        logger.warning("发送后未能通过模板确认聊天界面，请查看最后截图人工确认")


def _in_region(point: tuple[float, float], region: list[float]) -> bool:
    left, top, width, height = map(float, region)
    return left <= point[0] <= left + width and top <= point[1] <= top + height


def _point(point: tuple[float, float]) -> str:
    return f"({point[0]:.3f}, {point[1]:.3f})"


if __name__ == "__main__":
    raise SystemExit(main())
