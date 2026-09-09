"""Parser for the public AutoGLM single-action protocol."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


_SUPPORTED_ACTIONS = {
    "Launch",
    "Tap",
    "Double Tap",
    "Long Press",
    "Swipe",
    "Type",
    "Type_Name",
    "Back",
    "Home",
    "Wait",
    "Take_over",
    "Interact",
    "Note",
    "Call_API",
}


@dataclass(frozen=True)
class AutoGLMAction:
    """One model action. Coordinates retain AutoGLM's 0..999 convention."""

    name: str
    arguments: dict[str, Any]
    finished: bool = False

    @property
    def message(self) -> str:
        return str(self.arguments.get("message", ""))


def parse_action(content: str) -> AutoGLMAction:
    """Extract exactly one ``do(...)`` or ``finish(...)`` action safely."""
    action_text = _extract_action_text(content)
    # Follow Open-AutoGLM: finish is a terminal marker, so its human-readable
    # message is extracted without asking Python to parse its inner prose. This
    # prevents ordinary quotes in a description such as "照片" from making an
    # otherwise valid completion fail. Only the terminal finish action uses this
    # permissive path; executable do(...) actions remain AST-validated.
    if action_text.startswith("finish("):
        return _parse_finish_action(action_text)
    try:
        expression = ast.parse(action_text, mode="eval").body
    except SyntaxError as error:
        raise ValueError(f"AutoGLM 动作格式错误：{error.msg}") from error
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        raise ValueError("AutoGLM 动作必须是 do(...) 或 finish(...)")
    if expression.args:
        raise ValueError("AutoGLM 动作只允许关键字参数")
    arguments = {keyword.arg: ast.literal_eval(keyword.value) for keyword in expression.keywords if keyword.arg}
    if expression.func.id == "finish":
        return AutoGLMAction("finish", arguments, finished=True)
    if expression.func.id != "do":
        raise ValueError("AutoGLM 动作必须是 do(...) 或 finish(...)")
    name = str(arguments.pop("action", "")).strip()
    if name not in _SUPPORTED_ACTIONS:
        raise ValueError(f"不支持的 AutoGLM 动作：{name or '空动作'}")
    _validate_coordinates(name, arguments)
    return AutoGLMAction(name, arguments)


def _parse_finish_action(action_text: str) -> AutoGLMAction:
    prefix = "finish(message="
    if not action_text.startswith(prefix) or not action_text.endswith(")"):
        raise ValueError("AutoGLM finish 动作格式必须是 finish(message=\"...\")")
    raw_message = action_text[len(prefix) : -1].strip()
    if len(raw_message) >= 2 and raw_message[0] in {"'", '"'} and raw_message[-1] == raw_message[0]:
        raw_message = raw_message[1:-1]
    return AutoGLMAction("finish", {"message": raw_message}, finished=True)


def _extract_action_text(content: str) -> str:
    text = str(content).strip()
    if "<answer>" in text:
        text = text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    starts = [index for index in (text.find("do("), text.find("finish(")) if index >= 0]
    if not starts:
        raise ValueError("AutoGLM 返回中未找到动作")
    text = text[min(starts) :].strip()
    # Model replies may append an explanation after the action. Keep only the
    # balanced call expression before passing it to the Python AST.
    depth = 0
    quote = ""
    escaped = False
    for index, character in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = ""
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return text[: index + 1]
    raise ValueError("AutoGLM 动作括号未闭合")


def _validate_coordinates(name: str, arguments: dict[str, Any]) -> None:
    fields = ("element",) if name in {"Tap", "Double Tap", "Long Press"} else ("start", "end") if name == "Swipe" else ()
    for field in fields:
        value = arguments.get(field)
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"AutoGLM 动作 {name} 缺少坐标参数 {field}")
        try:
            x, y = float(value[0]), float(value[1])
        except (TypeError, ValueError) as error:
            raise ValueError(f"AutoGLM 坐标 {field} 必须是数字") from error
        if not 0 <= x <= 999 or not 0 <= y <= 999:
            raise ValueError(f"AutoGLM 坐标 {field} 必须在 0 到 999 之间")
