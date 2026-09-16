from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# 二进制帧协议：请求/响应均为 [8B 头: json_len, bin_len][json utf-8][bin]。
# stdout 专用于协议（第三方库的 print 一律重定向到 stderr，任何杂散字节
# 进入 stdout 都会破坏帧同步）；图像以原始字节走 bin 段，不做 base64。
_PROTO_OUT = sys.stdout.buffer
_SHM_CACHE: dict[str, Any] = {}


def _attach_shm(name: str) -> Any:
    """按名挂载主进程创建的共享内存块并缓存；进程退出统一释放。"""
    shm = _SHM_CACHE.get(name)
    if shm is not None:
        return shm
    from multiprocessing import shared_memory

    shm = shared_memory.SharedMemory(name=name)
    _SHM_CACHE[name] = shm
    return shm


def _release_shm_cache() -> None:
    for shm in _SHM_CACHE.values():
        try:
            shm.close()  # 只解除挂载；块生命周期由主进程管理
        except Exception:
            pass
    _SHM_CACHE.clear()


def _write_frame(payload: dict[str, Any], blob: bytes = b"") -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    _PROTO_OUT.write(struct.pack("<II", len(data), len(blob)))
    _PROTO_OUT.write(data)
    if blob:
        _PROTO_OUT.write(blob)
    _PROTO_OUT.flush()


def _write_response(payload: dict[str, Any]) -> None:
    _write_frame(payload)


def _read_exact(stream, count: int) -> bytes:
    """从管道读满 count 字节。

    用 os.read 直读文件描述符：BufferedReader 的 8KB 缓冲层读 2.7MB
    图像要数百次填充调用（实测 >1 秒），os.read 单次最多返回管道当前
    可用量（通常 64KB），同样次数量级但无逐块缓冲拷贝，实测降到几十
    毫秒级。
    """
    import os as _os

    chunks: list[bytes] = []
    remaining = count
    fd = stream.fileno()
    while remaining > 0:
        chunk = _os.read(fd, remaining)
        if not chunk:
            raise EOFError("pipe closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OCR subprocess bridge entry.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model-tier", required=True)
    parser.add_argument("--config-json", default="{}")
    args = parser.parse_args(argv)

    # 协议输出独占 stdout.buffer 之后，把文本层 stdout 指向 stderr：
    # Paddle/PaddleX 等第三方库的任何 print 都进 stderr 排空线程，
    # 不会再污染帧流。
    sys.stdout = sys.stderr

    root = Path(args.root).resolve()
    bridge_modules = root / "ocr_bridge_modules.zip"
    if bridge_modules.is_file() and str(bridge_modules) not in sys.path:
        sys.path.insert(0, str(bridge_modules))
    os.environ["JIXIEBI_PACKAGE_ROOT"] = str(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from vision.ocr_runtime import create_local_runtime_engine, _serialize_prediction_payload

    def _serialize_prediction_fast(result: Any) -> Any:
        """PaddleOCR 结果的快速序列化：三键直取 + 一次性 tolist。

        通用递归版对 rec_polys 每个坐标点做多级 isinstance 链，实测
        193~617ms/次（主进程侧 json 解析同步变贵）；此版本 <10ms，结构
        与通用版一致，非 PaddleOCR 形状时回退通用版。
        """
        try:
            serialized = []
            for item in result or []:
                if not isinstance(item, dict) or "rec_texts" not in item:
                    return _serialize_prediction_payload(result)
                polys = []
                for poly in item.get("rec_polys") or []:
                    polys.append(poly.tolist() if hasattr(poly, "tolist") else [[float(v) for v in point] for point in poly])
                serialized.append(
                    {
                        "rec_texts": [str(text) for text in item["rec_texts"]],
                        "rec_scores": [float(score) for score in (item.get("rec_scores") or [])],
                        "rec_polys": polys,
                    }
                )
            return serialized
        except Exception:
            return _serialize_prediction_payload(result)

    try:
        config = json.loads(args.config_json)
    except json.JSONDecodeError as exc:
        _write_response({"ok": False, "error": f"Invalid config JSON: {exc}"})
        return 1
    if not isinstance(config, dict):
        _write_response({"ok": False, "error": "Config payload must be a JSON object."})
        return 1
    config = dict(config)
    config["backend"] = args.backend
    config["model_tier"] = args.model_tier
    config["_execution_mode"] = "inprocess"

    try:
        engine = create_local_runtime_engine(config, allow_fallback=False)
    except Exception as exc:
        _write_response({"ok": False, "error": str(exc)})
        return 1

    # 启动预热：进程第一个 GPU 前向要付 cuDNN 初始化 + 首次 kernel 编译
    # （实测 ~13 秒），不预热的话这笔钱由业务第一个请求承担——表现为
    # "单次 OCR 十几秒"。这里用业务主尺寸的空图提前付掉，失败不阻塞，
    # 但把栈打到 stderr：预热失败往往预示业务首请求同样失败，静默吞掉
    # 会让定位变成猜谜。
    try:
        engine.predict(np.full((1280, 720, 3), 128, dtype=np.uint8))
    except Exception:
        import traceback

        print("[ocr-subprocess] warmup failed:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    stdin_buffer = sys.stdin.buffer
    while True:
        _frame_started = time.perf_counter()
        try:
            header = _read_exact(stdin_buffer, 8)
        except (EOFError, OSError):
            _release_shm_cache()
            return 0
        json_len, blob_len = struct.unpack("<II", header)
        frame_read_ms = (time.perf_counter() - _frame_started) * 1000.0
        try:
            json_bytes = _read_exact(stdin_buffer, json_len)
            blob = _read_exact(stdin_buffer, blob_len) if blob_len else b""
            payload = json.loads(json_bytes)
        except (EOFError, OSError, json.JSONDecodeError) as exc:
            _write_response({"ok": False, "error": f"Invalid request frame: {exc}"})
            continue
        if not isinstance(payload, dict):
            _write_response({"ok": False, "error": "Request payload must be a JSON object."})
            continue
        action = str(payload.get("action", "") or "").strip().lower()
        try:
            if action == "shutdown":
                _write_response({"ok": True, "device": ""})
                _release_shm_cache()
                return 0
            if action == "preload":
                _write_response({"ok": True, "device": str(engine.preload() or "")})
                continue
            if action != "predict":
                raise RuntimeError(f"Unsupported OCR subprocess action: {action}")
            decode_ms = 0.0
            shm_name = str(payload.get("shm_name", "") or "")
            if shm_name:
                # 图像在共享内存块中（主进程已写好，串行帧协议保证先写后
                # 读）：一次内存读取替代跨进程管道逐块搬运。copy() 使数组
                # 可写——PaddleOCR 预处理会原地写输入，只读视图导致识别异常。
                started = time.perf_counter()
                shm = _attach_shm(shm_name)
                size = int(payload.get("size", 0) or 0)
                shape = tuple(int(v) for v in (payload.get("shape") or []) if isinstance(v, int))
                image_array = np.frombuffer(shm.buf[:size], dtype=np.uint8)
                if len(shape) == 3:
                    image_array = image_array.reshape(shape).copy()
                decode_ms = (time.perf_counter() - started) * 1000.0
            elif blob:
                # blob 直传（SHM 不可用/超容量的回退通道）。copy() 同上：
                # 只读视图会让预处理原地写失败、识别结果异常。
                started = time.perf_counter()
                shape = tuple(int(v) for v in (payload.get("shape") or []) if isinstance(v, int))
                image_array = np.frombuffer(blob, dtype=np.uint8)
                if len(shape) == 3:
                    image_array = np.ascontiguousarray(image_array.reshape(shape)).copy()
                decode_ms = (time.perf_counter() - started) * 1000.0
            else:
                image_path = str(payload.get("image_path", "") or "").strip()
                if not image_path:
                    raise RuntimeError("Missing image payload for OCR subprocess predict request.")
                started = time.perf_counter()
                image_array = np.load(image_path, allow_pickle=False)
                decode_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            result = engine.predict(np.ascontiguousarray(image_array))
            predict_ms = (time.perf_counter() - started) * 1000.0
            _ser_started = time.perf_counter()
            _serialized = _serialize_prediction_fast(result)
            serialize_ms = (time.perf_counter() - _ser_started) * 1000.0
            # 响应顺带实际生效的设备与内部分段耗时：主进程的运行观测依赖它
            # （懒加载场景不会单独发 preload，没有它 device 永远显示为空）。
            _resp_started = time.perf_counter()
            _write_response(
                {
                    "ok": True,
                    "result": _serialized,
                    "device": str(getattr(engine, "device_seen", "") or ""),
                    "timings": {
                        "read_ms": round(frame_read_ms, 1),
                        "decode_ms": round(decode_ms, 1),
                        "predict_ms": round(predict_ms, 1),
                        "serialize_ms": round(serialize_ms, 1),
                        "respwrite_ms": round((time.perf_counter() - _resp_started) * 1000.0, 1),
                    },
                }
            )
        except Exception as exc:
            _write_response({"ok": False, "error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
