from __future__ import annotations

import re
from typing import Any

from apps.cancellation import cancellable_sleep, raise_if_cancelled


def split_tts_chunks(text: str, max_chars: int = 20) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = [part for part in re.split(r"([,.;!?，。！？；、\n])", text) if part]
    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(current) + len(part) <= max_chars:
            current += part
            continue
        if current:
            chunks.append(current)
            current = ""
        while len(part) > max_chars:
            chunks.append(part[:max_chars])
            part = part[max_chars:]
        current = part
    if current:
        chunks.append(current)
    return chunks


def tts_input(robot: Any, text: str, max_chars: int = 20, pause_seconds: float = 0.8) -> None:
    for chunk in split_tts_chunks(text, max_chars=max_chars):
        raise_if_cancelled(robot)
        robot.play_audio(chunk, wait=True)
        cancellable_sleep(pause_seconds, robot)
