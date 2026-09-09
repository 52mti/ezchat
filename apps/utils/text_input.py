from __future__ import annotations

import re
import threading
from typing import Any

import requests

from apps.cancellation import cancellable_sleep, raise_if_cancelled
from services.local_tts import ensure_local_tts_server, tts_url_for_device


# 多台设备共用一个本地 TTS 服务。先固定为单并发，避免请求堆积导致
# Edge TTS、端口连接和本地音频文件同时竞争，优先保证服务持续可用。
_TTS_REQUEST_GATE = threading.BoundedSemaphore(1)


def split_tts_chunks(text: str, max_chars: int = 40) -> list[str]:
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


def tts_input(robot: Any, text: str, max_chars: int = 40, pause_seconds: float = 0.8) -> None:
    play_prepared_tts_input(robot, split_tts_chunks(text, max_chars=max_chars), pause_seconds=pause_seconds)


def prepare_tts_input(
    robot: Any,
    text: str,
    max_chars: int = 40,
    *,
    timeout_seconds: float = 45.0,
) -> list[str]:
    """在点击麦克风前生成音频，使设备后续 /api/audio 请求命中本地缓存。"""
    chunks = split_tts_chunks(text, max_chars=max_chars)
    if not chunks:
        return chunks
    host = str(getattr(robot, "host", "") or "").strip()
    if not host:
        raise RuntimeError("无法确定设备 IP，不能预合成 TTS 音频")
    endpoint = tts_url_for_device(host)
    recovered = False
    for chunk in chunks:
        raise_if_cancelled(robot)
        with _TTS_REQUEST_GATE:
            try:
                response = requests.post(endpoint, json={"text": chunk}, timeout=timeout_seconds)
            except requests.RequestException:
                if recovered:
                    raise
                recovered = True
                ensure_local_tts_server(force_restart=True)
                response = requests.post(endpoint, json={"text": chunk}, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not str(payload.get("url", "")).strip():
            raise RuntimeError("本地 TTS 服务未返回音频地址")
    return chunks


def play_prepared_tts_input(robot: Any, chunks: list[str], pause_seconds: float = 0.8) -> None:
    """播放已由 prepare_tts_input 预合成的语音分段。"""
    for chunk in chunks:
        raise_if_cancelled(robot)
        robot.play_audio(chunk, wait=True)
        cancellable_sleep(pause_seconds, robot)
