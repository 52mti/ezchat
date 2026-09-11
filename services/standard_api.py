"""基础服务标准调用接口（AI 优先，同时供业务代码使用）。

这是全部基础能力的唯一推荐入口：业务流程、AutoGLM 适配层、AI 代码
生成 / 运行时决策都只 import 本模块，不直接碰 apps/ vision/ lgb/ 里
散落的实现。给人看的讲解在 docs/10-基础服务调用手册.md；给 AI 看的
接口清单就是本模块本身：

- 每个接口经 ``@api(summary=..., returns=...)`` 登记，函数签名即参数
  schema（类型注解 + 默认值 + docstring）；
- ``api_manifest()`` 输出机器可读 JSON（名称/用途/参数/返回说明）；
- ``api_tools_schema()`` 直接生成 OpenAI function-calling 风格工具清单；
- ``python -m services.standard_api --format markdown`` 导出函数表。

共享约定（适用于本模块所有接口）：
- 坐标一律 [0.0, 1.0] 归一化（透视图比例 = 机械臂运动平面比例），
  框统一为 [x, y, w, h]；
- image 接受 bytes / str / Path / PIL.Image / np.ndarray；
- robot 为 ``connect_arm()`` 返回的 ``lgb.lgb.RobotInterface`` 实例；
- 动作类接口内嵌取消检查，用户停止时抛 ``OperationCancelled``；
- 本模块只做薄委托与坐标语义换算，不含业务逻辑；实现仍各归各模块。
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

from apps.coordinates import click_image_point
from apps.utils.text_input import tts_input
from apps.visual_steps import current_screen as _current_screen
from lgb.lgb import RobotInterface, scan_devices
from services.cancellation import raise_if_cancelled
from services.photo_capture import capture_transform_image
from services.vision_dialogue_client import VisionDialogueClient, VisionDialogueReply
from vision.matcher import match_enhanced
from vision.ocr import ocr as _ocr
from vision.screen import ImageInput, crop_region, screen_matches


Region = Sequence[float]

_API_REGISTRY: list[dict[str, Any]] = []


def api(summary: str, *, returns: str, notes: str = "") -> Callable:
    """登记一个标准接口：summary 一句话用途；returns 返回结构说明。"""

    def decorate(func: Callable) -> Callable:
        entry = {"name": func.__name__, "summary": summary, "returns": returns, "notes": notes}
        _API_REGISTRY.append(entry)
        func.api_spec = entry  # type: ignore[attr-defined]
        return func

    return decorate


def _package_root() -> Path:
    configured = str(os.environ.get("JIXIEBI_PACKAGE_ROOT", "") or "").strip()
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 机械臂控制
# ---------------------------------------------------------------------------


@api(summary="扫描局域网内的机械臂设备", returns="设备 IP 字符串列表，如 ['192.168.3.200']", notes="ip_prefix 传 '192.168.3'，None 时自动用本机网段；扫描 8888 端口")
def scan_arms(ip_prefix: str | None = None, port: int = 8888, timeout: float = 0.5) -> list[str]:
    return scan_devices(ip_prefix, port=port, timeout=timeout)


@api(summary="连接一台机械臂设备", returns="RobotInterface 实例（后续动作/截图接口的第一个参数）", notes="rotate 是透视图旋转角 0/90/180/270，安装方向不对时调整")
def connect_arm(ip: str, rotate: int = 0) -> RobotInterface:
    return RobotInterface(ip, rotate=rotate)


@api(summary="读取设备状态", returns="状态 dict；透视标定参数在 data.transform_width / transform_height / points（四点）")
def arm_status(robot: RobotInterface) -> dict:
    return robot.get_status()


@api(summary="点击归一化坐标（标准点击入口）", returns="None", notes="内嵌取消检查与拟人抖动（robot._codex_click_jitter_enabled=True 时幅度 0.1%~0.3%）")
def click(robot: RobotInterface, x: float, y: float) -> None:
    click_image_point(robot, x, y)


@api(summary="点击 find_template 命中框的中心", returns="None")
def click_match(robot: RobotInterface, match: dict) -> None:
    click_image_point(robot, float(match["x"]) + float(match["w"]) / 2, float(match["y"]) + float(match["h"]) / 2)


@api(summary="滑屏", returns="设备执行结果 dict", notes="内嵌取消检查")
def swipe(robot: RobotInterface, start_x: float, start_y: float, end_x: float, end_y: float) -> dict:
    raise_if_cancelled(robot)
    return robot.swipe(start_x, start_y, end_x, end_y, wait=True)


# ---------------------------------------------------------------------------
# 截图（透视取图）
# ---------------------------------------------------------------------------


@api(summary="截图：相机原图 → 透视矫正 → 竖屏 JPEG（视觉流水线入口）", returns="JPEG bytes（ImageBytes 子类，带 .image_source 溯源标签）", notes="内部完成：读设备四点标定并 cv2.warpPerspective、标定变化自动失效坐标缓存、同图复用、亮度健康上报、失败重试；pre_move=True 先抬臂到安全位")
def capture_screen(robot: RobotInterface, config: dict | None = None, *, pre_move: bool = True, label: str = "capture") -> bytes:
    return capture_transform_image(robot, config, pre_move=pre_move, label=label)


# ---------------------------------------------------------------------------
# 模板匹配
# ---------------------------------------------------------------------------


@api(summary="模板匹配：在图中找模板图的位置", returns="命中返回 {x, y, w, h, score, channel, scale}（整图归一化坐标），未命中返回 None", notes="region 限定搜索范围时返回坐标已换算回整图坐标系；threshold 默认 0.85")
def find_template(image: ImageInput, template: ImageInput, threshold: float = 0.85, region: Region | None = None) -> dict | None:
    if region is None:
        return match_enhanced(image, template, threshold=threshold)
    region_x, region_y, region_w, region_h = (float(value) for value in region)
    found = match_enhanced(crop_region(image, region), template, threshold=threshold)
    if found is None:
        return None
    return {
        "x": region_x + float(found["x"]) * region_w,
        "y": region_y + float(found["y"]) * region_h,
        "w": float(found["w"]) * region_w,
        "h": float(found["h"]) * region_h,
        "score": float(found["score"]),
        "channel": str(found.get("channel", "")),
        "scale": float(found.get("scale", 1.0)),
    }


@api(summary="判断模板是否出现在图中", returns="bool", notes="只需存在性判断时用它，比 find_template 快")
def template_exists(image: ImageInput, template: ImageInput, threshold: float = 0.85, region: Region | None = None) -> bool:
    return screen_matches(image, template, threshold=threshold, region=region)


@api(summary="识别当前手机处于哪个业务界面", returns="界面 key 字符串（对应 apps/<app>/config 的 screens 配置）")
def current_screen(image: ImageInput, config: dict) -> str:
    return _current_screen(image, config)


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


@api(summary="OCR 文字识别", returns="list[dict]，每项 {text, x, y, w, h, confidence}，框为归一化", notes="需要 config['runtime_enabled']=True 且引擎就绪（vision/ocr_runtime 自动选 CUDA/DirectML/CPU 后端）；region 限定识别范围可显著提速")
def read_text(image: ImageInput, region: Region | None = None, config: dict | None = None) -> list[dict]:
    merged = dict(config or {})
    if region is not None:
        merged["region"] = list(region)
    return _ocr(image, merged)


# ---------------------------------------------------------------------------
# TTS 语音
# ---------------------------------------------------------------------------


@api(summary="语音播报（本地 piper TTS，模拟语音输入通道）", returns="None", notes="长文本自动按 max_chars 分段播放；设备需已绑定本机 TTS 地址（主程序启动时 bind_device_tts 完成）")
def speak(robot: RobotInterface, text: str, max_chars: int = 40) -> None:
    tts_input(robot, text, max_chars=max_chars)


# ---------------------------------------------------------------------------
# 云端视觉理解
# ---------------------------------------------------------------------------


@api(summary="云端视觉理解：把截图交给大模型解析", returns="VisionDialogueReply（payload 解析结果 / provider / model / raw_text 原文）", notes="本地 OCR 兜不住时的增强通道；prompt_path 不传则用 config['dialogue_llm']['prompt_path']")
def understand_image(image: ImageInput, config: dict, prompt_path: str | None = None) -> VisionDialogueReply:
    merged = dict(config or {})
    dialogue = dict(merged.get("dialogue_llm") or {})
    if prompt_path:
        dialogue["prompt_path"] = prompt_path
    merged["dialogue_llm"] = dialogue
    client = VisionDialogueClient(workspace_root=_package_root())
    return client.parse_dialogue(image, merged)


# ---------------------------------------------------------------------------
# 机器可读清单
# ---------------------------------------------------------------------------

_PY_TYPE_NAMES = {
    "float": "number",
    "int": "integer",
    "str": "string",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "tuple": "array",
}


def _annotation_name(annotation: Any) -> str:
    """把类型注解转成 AI 可读的类型描述（保留 Python 语义，如 float | None）。"""
    if annotation is inspect.Parameter.empty or annotation is None:
        return "any"
    if isinstance(annotation, type):
        return _PY_TYPE_NAMES.get(annotation.__name__, annotation.__name__)
    return str(annotation).replace("typing.", "") or "any"


def _param_default(default: Any) -> Any:
    if isinstance(default, (bool, int, float, str)) or default is None:
        return default
    return repr(default)


def api_manifest() -> list[dict[str, Any]]:
    """全部标准接口的机器可读清单（JSON 可序列化）。"""
    module = globals()
    entries: list[dict[str, Any]] = []
    for entry in _API_REGISTRY:
        func = module[entry["name"]]
        try:
            signature = inspect.signature(func, eval_str=True)
        except (TypeError, NameError):
            signature = inspect.signature(func)
        params = []
        for name, param in signature.parameters.items():
            params.append(
                {
                    "name": name,
                    "type": _annotation_name(param.annotation),
                    "required": param.default is inspect.Parameter.empty,
                    **({"default": _param_default(param.default)} if param.default is not inspect.Parameter.empty else {}),
                }
            )
        entries.append({**entry, "params": params})
    return entries


def _json_param_type(type_text: str) -> str:
    base = type_text.split("|")[0].strip().strip("[]")
    for token, json_type in _PY_TYPE_NAMES.items():
        if base.startswith(token):
            return json_type
    if base.startswith("Path"):
        return "string"
    return "object"


def api_tools_schema() -> list[dict[str, Any]]:
    """OpenAI function-calling 风格的工具定义，可直接交给 LLM。"""
    tools = []
    for entry in api_manifest():
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param in entry["params"]:
            properties[param["name"]] = {"type": _json_param_type(param["type"])}
            if param["required"]:
                required.append(param["name"])
        description = "；".join(part for part in (entry["summary"], entry["notes"]) if part)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": entry["name"],
                    "description": description,
                    "parameters": {"type": "object", "properties": properties, "required": required},
                },
            }
        )
    return tools


def api_markdown() -> str:
    """从登记信息导出 Markdown 函数表（与代码同源，不会漂移）。"""
    lines = [
        "| 接口 | 用途 | 返回 |",
        "| --- | --- | --- |",
    ]
    for entry in api_manifest():
        params = ", ".join(
            f"{param['name']}={param.get('default')!r}" if "default" in param else param["name"]
            for param in entry["params"]
        )
        lines.append(f"| `{entry['name']}({params})` | {entry['summary']} | {entry['returns']} |")
    return "\n".join(lines)


__all__ = [
    "scan_arms",
    "connect_arm",
    "arm_status",
    "capture_screen",
    "click",
    "click_match",
    "swipe",
    "find_template",
    "template_exists",
    "current_screen",
    "read_text",
    "speak",
    "understand_image",
    "api_manifest",
    "api_tools_schema",
    "api_markdown",
]


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="导出基础服务标准接口清单")
    parser.add_argument("--format", choices=("json", "markdown"), default="json", help="json=机器可读清单，markdown=函数表")
    args = parser.parse_args(argv)
    if args.format == "json":
        print(json.dumps(api_manifest(), ensure_ascii=False, indent=2))
    else:
        print(api_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
