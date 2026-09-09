from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


DEFAULT_SYNTHESIS_TIMEOUT_SECONDS = 25.0
DEFAULT_TTS_ENGINE = "windows_sapi"
FALLBACK_TTS_ENGINE = "piper"
DEFAULT_SPEAKER_RATE = 1.0
PIPER_MODEL_FILENAME = "zh_CN-huayan-medium.onnx"
PIPER_ENGINE_VERSION = "piper-tts-1.7.0"
LOGGER = logging.getLogger("local_tts")
if not LOGGER.handlers:
    _log_handler = logging.StreamHandler()
    _log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(_log_handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False
_PIPER_VOICE_LOCK = threading.RLock()
_PIPER_VOICE: object | None = None
_PIPER_MODEL_PATH: Path | None = None


@dataclass(frozen=True)
class SynthesisResult:
    engine: str
    fallback_reason: str = ""


def _cleanup_old_audio(directory: Path, *, max_age_seconds: int = 3600) -> None:
    cutoff = time.time() - max_age_seconds
    for path in directory.glob("*.wav"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                path.with_suffix(".json").unlink(missing_ok=True)
        except OSError:
            continue


def _audio_filename(
    text: str,
    voice: str,
    *,
    engine: str = DEFAULT_TTS_ENGINE,
    speaker_rate: float = DEFAULT_SPEAKER_RATE,
) -> str:
    """缓存键包含文字、引擎版本和语速，避免错误复用不同合成参数的音频。"""
    cache_key = f"{engine}\0{PIPER_ENGINE_VERSION}\0{speaker_rate:.3f}\0{voice}\0{text}"
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    return f"{digest}.wav"


def _read_cache_result(filepath: Path) -> SynthesisResult:
    try:
        payload = json.loads(filepath.with_suffix(".json").read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return SynthesisResult(
                engine=str(payload.get("engine", DEFAULT_TTS_ENGINE) or DEFAULT_TTS_ENGINE),
                fallback_reason=str(payload.get("fallback_reason", "") or ""),
            )
    except (OSError, ValueError, TypeError):
        pass
    return SynthesisResult(engine=DEFAULT_TTS_ENGINE)


def _write_cache_result(filepath: Path, result: SynthesisResult) -> None:
    filepath.with_suffix(".json").write_text(
        json.dumps({"engine": result.engine, "fallback_reason": result.fallback_reason}, ensure_ascii=False),
        encoding="utf-8",
    )


def _project_roots() -> list[Path]:
    roots: list[Path] = []
    configured = str(os.environ.get("JIXIEBI_PACKAGE_ROOT", "") or "").strip()
    if configured:
        roots.append(Path(configured))
    roots.extend((Path.cwd(), Path(__file__).resolve().parents[1]))
    unique_roots: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in unique_roots:
            unique_roots.append(resolved)
    return unique_roots


def _find_piper_model() -> Path:
    for root in _project_roots():
        for directory in (root / "runtime" / "tts" / "piper", root / "tts" / "piper"):
            model_path = directory / PIPER_MODEL_FILENAME
            config_path = model_path.with_suffix(".onnx.json")
            if model_path.is_file() and config_path.is_file():
                return model_path
    raise RuntimeError("缺少 Piper 中文离线语音模型，请使用包含 runtime/tts/piper 的完整包安装")


def _is_chinese_sapi_voice(voice: object) -> bool:
    try:
        name = str(voice.GetDescription() or "").lower()
        language = str(voice.GetAttribute("Language") or "").lower()
    except Exception:
        return False
    return any(token in language for token in ("804", "2052")) or any(
        token in name for token in ("chinese", "中文", "xiaoxiao", "yunxi")
    )


def _synthesize_windows_sapi(text: str, output_path: Path, speaker_rate: float = DEFAULT_SPEAKER_RATE) -> None:
    """使用客户 Windows 已安装的语音包离线生成 WAV，不访问任何网络服务。"""
    if os.name != "nt":
        raise RuntimeError("离线 TTS 仅支持 Windows SAPI")
    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise RuntimeError("离线 TTS 缺少 pywin32 运行时") from exc

    pythoncom.CoInitialize()
    stream = None
    try:
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        voices = list(speaker.GetVoices())
        chinese_voice = next((voice for voice in voices if _is_chinese_sapi_voice(voice)), None)
        if chinese_voice is None:
            raise RuntimeError("客户 Windows 未安装中文语音包，无法进行离线中文 TTS")
        speaker.Voice = chinese_voice
        speaker.Rate = max(-10, min(10, round((float(speaker_rate) - 1.0) * 10)))
        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        stream.Open(str(output_path), 3, False)
        speaker.AudioOutputStream = stream
        speaker.Speak(text)
    finally:
        if stream is not None:
            stream.Close()
        pythoncom.CoUninitialize()


def _synthesize_piper(text: str, output_path: Path, speaker_rate: float = DEFAULT_SPEAKER_RATE) -> None:
    """通过打包的 Piper 中文模型离线生成 22050Hz WAV。"""
    try:
        from piper import PiperVoice
        from piper.config import SynthesisConfig
    except ImportError as exc:
        raise RuntimeError("缺少 Piper 离线 TTS 推理运行时，请使用完整包安装") from exc

    global _PIPER_MODEL_PATH, _PIPER_VOICE
    model_path = _find_piper_model()
    with _PIPER_VOICE_LOCK:
        if _PIPER_VOICE is None or _PIPER_MODEL_PATH != model_path:
            _PIPER_VOICE = PiperVoice.load(model_path)
            _PIPER_MODEL_PATH = model_path
        voice = _PIPER_VOICE
    config = SynthesisConfig(length_scale=1.0 / max(0.5, min(2.0, float(speaker_rate))))
    with wave.open(str(output_path), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=config)  # type: ignore[union-attr]


def _synthesize_offline(text: str, output_path: Path, speaker_rate: float) -> SynthesisResult:
    try:
        _synthesize_windows_sapi(text, output_path, speaker_rate)
        return SynthesisResult(engine=DEFAULT_TTS_ENGINE)
    except Exception as sapi_error:
        try:
            _synthesize_piper(text, output_path, speaker_rate)
            return SynthesisResult(engine=FALLBACK_TTS_ENGINE, fallback_reason=str(sapi_error))
        except Exception as piper_error:
            raise RuntimeError(f"Windows SAPI5 合成失败：{sapi_error}；Piper 回退失败：{piper_error}") from piper_error


def create_app(
    static_dir: str | Path,
    *,
    synthesis_timeout_seconds: float = DEFAULT_SYNTHESIS_TIMEOUT_SECONDS,
    speaker_rate: float = DEFAULT_SPEAKER_RATE,
) -> FastAPI:
    """创建仅供局域网机械臂调用的离线 TTS 服务。"""
    audio_dir = Path(static_dir).resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_old_audio(audio_dir)
    synthesis_lock = asyncio.Lock()

    app = FastAPI()
    app.mount("/static", StaticFiles(directory=audio_dir), name="static")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "engine": DEFAULT_TTS_ENGINE, "fallback": FALLBACK_TTS_ENGINE}

    @app.post("/tts")
    async def tts(request: Request):
        try:
            data = await request.json()
        except ValueError:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
        if not isinstance(data, dict):
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
        text = str(data.get("text", "") or "").strip()
        voice = str(data.get("voice", "zh-CN-XiaoxiaoNeural") or "zh-CN-XiaoxiaoNeural")
        if not text:
            return JSONResponse({"error": "Missing text"}, status_code=400)

        filename = _audio_filename(text, voice, speaker_rate=speaker_rate)
        filepath = audio_dir / filename
        base_url = str(request.base_url).rstrip("/")
        if filepath.is_file() and filepath.stat().st_size > 0:
            cached_result = _read_cache_result(filepath)
            LOGGER.info("TTS 引擎=%s 缓存命中=是 排队等待=0ms 合成耗时=0ms", cached_result.engine)
            return {
                "url": f"{base_url}/static/{filename}",
                "duration": 0,
                "cached": True,
                "engine": cached_result.engine,
                "queue_wait_ms": 0,
                "synthesis_ms": 0,
            }

        queue_started_at = time.monotonic()
        async with synthesis_lock:
            queue_wait_ms = int((time.monotonic() - queue_started_at) * 1000)
            if filepath.is_file() and filepath.stat().st_size > 0:
                cached_result = _read_cache_result(filepath)
                LOGGER.info(
                    "TTS 引擎=%s 缓存命中=是 排队等待=%dms 合成耗时=0ms",
                    cached_result.engine,
                    queue_wait_ms,
                )
                return {
                    "url": f"{base_url}/static/{filename}",
                    "duration": queue_wait_ms,
                    "cached": True,
                    "engine": cached_result.engine,
                    "queue_wait_ms": queue_wait_ms,
                    "synthesis_ms": 0,
                }

            temporary_path = filepath.with_suffix(".part")
            synthesis_started_at = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(_synthesize_offline, text, temporary_path, speaker_rate),
                    timeout=max(1.0, float(synthesis_timeout_seconds)),
                )
                if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
                    raise RuntimeError("离线 TTS 未生成有效 WAV 文件")
                temporary_path.replace(filepath)
                _write_cache_result(filepath, result)
            except TimeoutError:
                temporary_path.unlink(missing_ok=True)
                return JSONResponse(
                    {"error": "TTS generation timed out", "detail": "Piper/SAPI5 did not finish in time"},
                    status_code=504,
                )
            except Exception as exc:
                temporary_path.unlink(missing_ok=True)
                return JSONResponse({"error": "TTS generation failed", "detail": str(exc)}, status_code=500)

        synthesis_ms = int((time.monotonic() - synthesis_started_at) * 1000)
        fallback_detail = f"，Piper 回退原因={result.fallback_reason}" if result.fallback_reason else ""
        LOGGER.info(
            "TTS 引擎=%s 缓存命中=否 排队等待=%dms 合成耗时=%dms%s",
            result.engine,
            queue_wait_ms,
            synthesis_ms,
            fallback_detail,
        )
        return {
            "url": f"{base_url}/static/{filename}",
            "duration": queue_wait_ms + synthesis_ms,
            "cached": False,
            "engine": result.engine,
            "queue_wait_ms": queue_wait_ms,
            "synthesis_ms": synthesis_ms,
            "fallback_reason": result.fallback_reason,
        }

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="机械臂本地 TTS 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--data-dir", default="runtime/tts")
    parser.add_argument("--synthesis-timeout", type=float, default=DEFAULT_SYNTHESIS_TIMEOUT_SECONDS)
    parser.add_argument("--speaker-rate", type=float, default=DEFAULT_SPEAKER_RATE)
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(
        create_app(args.data_dir, synthesis_timeout_seconds=args.synthesis_timeout, speaker_rate=args.speaker_rate),
        host=args.host,
        port=args.port,
        log_level="warning",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
