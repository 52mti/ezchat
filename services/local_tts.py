from __future__ import annotations

import ipaddress
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable

import requests


DEFAULT_TTS_PORT = 9999
DEFAULT_DEVICE_BACKEND_PASSWORD = "admin"
DEFAULT_DEVICE_BACKEND_TIMEOUT_SECONDS = 30.0
_RECOVERY_LOCK = threading.RLock()
_RECOVERY_SERVER: "LocalTtsServer | None" = None


def _project_root() -> Path:
    configured = str(os.environ.get("JIXIEBI_PACKAGE_ROOT", "") or "").strip()
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[1]


def local_ip_for_device(device_ip: str) -> str:
    """取访问该设备时 Windows 会选用的本机 IPv4，避免多网卡绑定错误。"""
    target = str(ipaddress.ip_address(device_ip))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect((target, 9))
        return str(sock.getsockname()[0])


def tts_url_for_device(device_ip: str, *, port: int = DEFAULT_TTS_PORT) -> str:
    return f"http://{local_ip_for_device(device_ip)}:{int(port)}/tts"


def bind_device_tts(
    device_ip: str,
    *,
    port: int = DEFAULT_TTS_PORT,
    timeout: float = DEFAULT_DEVICE_BACKEND_TIMEOUT_SECONDS,
) -> str:
    """登录设备后台并保存本机 TTS 地址，返回实际写入的地址。"""
    normalized_ip = str(ipaddress.ip_address(device_ip))
    base_url = f"http://{normalized_ip}:8888"
    tts_url = tts_url_for_device(normalized_ip, port=port)
    session = requests.Session()
    login = session.post(f"{base_url}/login", data={"password": DEFAULT_DEVICE_BACKEND_PASSWORD}, timeout=timeout)
    login.raise_for_status()
    response = session.post(f"{base_url}/save_sys_config", data={"tts_url": tts_url}, timeout=timeout)
    response.raise_for_status()
    return tts_url


class LocalTtsServer:
    """主 UI 生命周期内管理的本地 TTS 子进程。"""

    def __init__(self, *, port: int = DEFAULT_TTS_PORT, root: str | Path | None = None) -> None:
        global _RECOVERY_SERVER
        self.port = int(port)
        self.root = Path(root) if root is not None else _project_root()
        self._process: subprocess.Popen[object] | None = None
        self._lock = threading.RLock()
        with _RECOVERY_LOCK:
            _RECOVERY_SERVER = self

    @property
    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/health"

    def is_ready(self) -> bool:
        try:
            response = requests.get(self.health_url, timeout=0.5)
            return response.ok and response.json().get("status") == "ok"
        except (OSError, ValueError, requests.RequestException):
            return False

    def start(self, *, timeout: float = 8.0) -> None:
        with self._lock:
            if self.is_ready():
                return
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("本地 TTS 服务启动超时")
            log_path = self.root / "runtime" / "logs" / "local_tts_server.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("a", encoding="utf-8")
            packaged_server = self.root / "tts_server.pyc"
            command = [
                sys.executable,
                "-s",
                "-u",
                str(packaged_server),
                "--host",
                "0.0.0.0",
                "--port",
                str(self.port),
                "--data-dir",
                str(self.root / "runtime" / "tts"),
            ]
            if not packaged_server.is_file():
                command = [sys.executable, "-m", "tts.server", *command[4:]]
            try:
                self._process = subprocess.Popen(
                    command,
                    cwd=self.root,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
                )
            except OSError:
                log_file.close()
                raise
            finally:
                log_file.close()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.is_ready():
                    return
                if self._process.poll() is not None:
                    raise RuntimeError("本地 TTS 服务启动失败，请查看 runtime/logs/local_tts_server.log")
                time.sleep(0.1)
            raise RuntimeError("本地 TTS 服务启动超时")

    def bind_devices(self, device_ips: Iterable[str]) -> list[tuple[str, str, str]]:
        self.start()
        results: list[tuple[str, str, str]] = []
        for device_ip in device_ips:
            normalized_ip = str(device_ip or "").strip()
            if not normalized_ip:
                continue
            try:
                results.append((normalized_ip, bind_device_tts(normalized_ip, port=self.port), ""))
            except Exception as exc:
                results.append((normalized_ip, "", str(exc)))
        return results

    def stop(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
            if process is not None and process.poll() is None:
                process.terminate()

    def restart(self) -> None:
        """停止并重新拉起同一个受管理的 TTS 子进程。"""
        with self._lock:
            process = self._process
            self._process = None
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3.0)
        self.start()


def ensure_local_tts_server(*, force_restart: bool = False) -> None:
    """在 TTS 子进程退出后，按单例方式恢复服务，避免多设备重复拉起进程。"""
    global _RECOVERY_SERVER
    with _RECOVERY_LOCK:
        if _RECOVERY_SERVER is None:
            _RECOVERY_SERVER = LocalTtsServer()
        if force_restart:
            _RECOVERY_SERVER.restart()
        else:
            _RECOVERY_SERVER.start()
