from __future__ import annotations

from pathlib import Path
from typing import Any

from apps.base import ReplyAction
from apps.coordinates import image_to_robot_point
from apps.utils.text_input import play_prepared_tts_input, prepare_tts_input
from services.cancellation import raise_if_cancelled

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
BUTTON_DIR = PACKAGE_ROOT / "apps" / "xinyu" / "templates" / "buttons"
TEMPLATE_CANDIDATES = {
    "unread_tab": ("btn_未回复.jpg", "btn_未读消息按钮.jpg", "btn_未读消息-2.jpg"),
    "next_unread": ("btn_下一条按钮.jpg", "btn_未读消息按钮.jpg", "btn_next_msg.jpg"),
    "back": ("btn_返回按钮.jpg", "btn_返回按钮_2.jpg", "btn_返回按钮_3.jpg", "btn_返回按钮_4.jpg", "btn_返回按钮_5.jpg", "btn_back.jpg"),
    "microphone": ("btn_麦克风按钮.jpg", "btn_麦克风按钮_2.jpg", "btn_麦克风按钮_3.jpg", "btn_voice_input.jpg"),
}


class ArmPorts:
    """机械臂、OCR、模板和现有 TTS 的真实调用。"""

    def __init__(self, robot: Any, config: dict[str, Any], reply_provider: Any, sent_callback: Any) -> None:
        self.robot = robot
        self.config = config
        self.reply_provider = reply_provider
        self.sent_callback = sent_callback
        self._pending_action: ReplyAction | None = None

    def capture(self) -> bytes:
        raise_if_cancelled(self.robot)
        return self.robot.get_transform_image(pre_move=True)

    def ocr(self, image: Any) -> list[dict[str, Any]]:
        from vision.ocr import ocr

        ocr_config = self.config.get("ocr") if isinstance(self.config.get("ocr"), dict) else {}
        payload = dict(ocr_config)
        payload.setdefault("runtime_enabled", True)
        return ocr(image, payload)

    def match(self, image: Any, key: str) -> tuple[float, float] | None:
        from vision.matcher import match, match_enhanced

        threshold = float(self.config.get("match_threshold", 0.7) or 0.7)
        for name in TEMPLATE_CANDIDATES.get(key, ()):
            path = BUTTON_DIR / name
            if not path.is_file():
                continue
            found = match(image, path, threshold=threshold)
            if found is None:
                found = match_enhanced(image, path, threshold=threshold)
            if found is not None:
                return found["x"] + found["w"] / 2, found["y"] + found["h"] / 2
        return None

    def click(self, x: float, y: float) -> None:
        raise_if_cancelled(self.robot)
        robot_x, robot_y = image_to_robot_point(x, y)
        self.robot.move_click(robot_x, robot_y, wait=True)

    def reset(self) -> None:
        raise_if_cancelled(self.robot)
        self.robot.reset_device(wait=True)

    def swipe_down(self) -> None:
        raise_if_cancelled(self.robot)
        self.robot.swipe(0.5, 0.32, 0.5, 0.78, wait=True)

    def reply(self, contact: str, text: str) -> tuple[str, str]:
        message = {"contact": contact, "user": contact, "content": text, "message_time": "", "unread_count": ""}
        if self.reply_provider is None:
            self._pending_action = ReplyAction(text=text, reason="ai_reply")
            return text, ""
        result = self.reply_provider(message)
        if isinstance(result, ReplyAction):
            self._pending_action = result
            return result.text, result.message_fingerprint
        self._pending_action = ReplyAction(text=str(result or ""), reason="ai_reply")
        return self._pending_action.text, ""

    def prepare_tts(self, text: str) -> list[str]:
        return prepare_tts_input(self.robot, text, max_chars=20)

    def play(self, chunks: list[str]) -> None:
        play_prepared_tts_input(self.robot, chunks)

    def mark_sent(self, contact: str, incoming: str, outgoing: str, fingerprint: str) -> None:
        if self.sent_callback is None:
            return
        action = self._pending_action or ReplyAction(text=outgoing, message_fingerprint=fingerprint, reason="ai_reply")
        self.sent_callback(action, {"contact": contact, "user": contact, "content": incoming})
