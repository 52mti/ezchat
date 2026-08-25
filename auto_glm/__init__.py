"""AutoGLM planning and robot-execution primitives.

This package contains only the protocol shared with the previous AutoGLM
workbench.  Android/ADB-specific code intentionally does not live here.
"""

from auto_glm.adapter import ActionResult, RobotActionAdapter
from auto_glm.gpt_planner import GPTPlannerClient, GPTPlannerConfig
from auto_glm.model import AutoGLMClient, AutoGLMConfig, AutoGLMProtocolError, ModelDecision
from auto_glm.planner import ExecutionPlan, PlanNode
from auto_glm.protocol import AutoGLMAction, parse_action
from auto_glm.verification import VisualTransition, image_signature, verify_transition

__all__ = [
    "ActionResult",
    "AutoGLMAction",
    "AutoGLMClient",
    "AutoGLMConfig",
    "AutoGLMProtocolError",
    "ExecutionPlan",
    "GPTPlannerClient",
    "GPTPlannerConfig",
    "ModelDecision",
    "PlanNode",
    "RobotActionAdapter",
    "VisualTransition",
    "image_signature",
    "parse_action",
    "verify_transition",
]
