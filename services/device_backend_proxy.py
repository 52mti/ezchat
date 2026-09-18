from __future__ import annotations

import http.server
import html as html_module
import ipaddress
import re
import threading
from typing import Any

import requests


DEVICE_BACKEND_PORT = 8888
DEFAULT_BACKEND_PASSWORD = "admin"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 15.0

# 逐跳头与本机代理无关，转发时一律剥掉；Content-Length 按新响应体重算。
_STRIPPED_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "content-encoding",
    "set-cookie",
    "host",
    "accept-encoding",
}
_LOGIN_PAGE_MARKER = "loginForm"

# 新后台设置页的 TTS 初始化脚本会把"自定义地址"输入框清空展示（固件显示
# 缺陷）：已存地址不匹配任何预设时下拉框切到 __custom__，随后 onTtsPreset
# 执行 inp.value=''。代理在 input 上补 data-orig 并注入恢复脚本，让用户
# 看到服务端真实存储的地址。
_TTS_INPUT_RE = re.compile(
    r"(<input\b[^>]*\bid=[\"']tts_url[\"'][^>]*?\bvalue=[\"']([^\"']*)[\"'][^>]*?)(/?>)",
    re.IGNORECASE,
)
_TTS_RESTORE_SCRIPT = (
    "<script>(function(){var i=document.getElementById('tts_url');"
    "if(i&&i.dataset.orig&&!i.value){i.value=i.dataset.orig;}})();</script>"
)


def _patch_tts_display(payload: bytes) -> bytes:
    """把固件页里被清空展示的自定义 TTS 地址恢复显示真实存储值。"""
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload
    match = _TTS_INPUT_RE.search(text)
    if match is None or not match.group(2):
        return payload
    escaped = html_module.escape(match.group(2), quote=True)
    patched_input = f"{match.group(1)} data-orig=\"{escaped}\"{match.group(3)}"
    text = text[: match.start()] + patched_input + text[match.end() :]
    if "</body>" in text:
        text = text.replace("</body>", f"{_TTS_RESTORE_SCRIPT}</body>", 1)
    else:
        text += _TTS_RESTORE_SCRIPT
    return text.encode("utf-8")


class DeviceBackendProxy:
    """把设备后台反向代理到本机回环地址，供浏览器直接打开。

    新版设备后台的会话 Cookie 带 SameSite=Strict，浏览器里任何由页面发起的
    跨站跳转都带不上 Cookie；旧方案要先在 %TEMP% 落一个跳板页、延迟后再以
    OS 级方式打开后台首页，跳板页会一直留在浏览器和临时目录里。改为代理后
    浏览器只与 127.0.0.1 同源交互，登录 Cookie 由代理内部的 requests 会话
    持有，跳板页不再存在；老版后台（Cookie 无 SameSite 属性）走同一条路
    同样成立。
    """

    def __init__(
        self,
        device_ip: str,
        *,
        port: int = DEVICE_BACKEND_PORT,
        password: str = DEFAULT_BACKEND_PASSWORD,
        timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        normalized_ip = str(ipaddress.ip_address(device_ip))
        self.device_ip = normalized_ip
        self.device_base = f"http://{normalized_ip}:{int(port)}"
        self.password = password
        self.timeout = timeout
        self._session = requests.Session()
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        if self._server is None:
            return ""
        return f"http://127.0.0.1:{self._server.server_address[1]}/"

    def start(self) -> str:
        """登录设备并启动本机回环代理，返回可在浏览器打开的地址。"""
        with self._lock:
            if self._server is not None:
                return self.url
            self._login()
            proxy = self

            class _Handler(http.server.BaseHTTPRequestHandler):
                protocol_version = "HTTP/1.1"

                def _relay(self) -> None:
                    proxy._relay_request(self)

                do_GET = _relay
                do_POST = _relay
                do_HEAD = _relay

                def log_message(self, *_args: Any) -> None:
                    # 后台页面会周期性轮询 /api/status，不刷日志。
                    return

            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
            server.daemon_threads = True
            thread = threading.Thread(
                target=server.serve_forever,
                name=f"device-backend-proxy-{self.device_ip}",
                daemon=True,
            )
            thread.start()
            self._server = server
            self._thread = thread
            return self.url

    def stop(self) -> None:
        with self._lock:
            server = self._server
            thread = self._thread
            self._server = None
            self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2.0)

    def _login(self) -> None:
        self._session.cookies.clear()
        response = self._session.post(
            f"{self.device_base}/login",
            data={"password": self.password},
            timeout=self.timeout,
        )
        response.raise_for_status()

    def _proxy_base(self) -> str:
        return self.url.rstrip("/")

    def _relay_request(self, handler: http.server.BaseHTTPRequestHandler) -> None:
        try:
            response = self._forward(handler)
            if self._session_expired(response) and handler.path != "/login":
                # 设备重启或会话过期后内部 Cookie 失效：重新登录再试一次。
                self._login()
                response = self._forward(handler)
        except (requests.RequestException, OSError):
            handler.send_error(502, "设备后台不可达")
            return
        self._send(handler, response)

    def _forward(self, handler: http.server.BaseHTTPRequestHandler) -> requests.Response:
        length = int(handler.headers.get("Content-Length") or 0)
        body = handler.rfile.read(length) if length > 0 else None
        headers = {
            key: value
            for key, value in handler.headers.items()
            if key.lower() not in _STRIPPED_HEADERS
        }
        return self._session.request(
            handler.command,
            self.device_base + handler.path,
            data=body,
            headers=headers,
            timeout=self.timeout,
            allow_redirects=False,
        )

    def _session_expired(self, response: requests.Response) -> bool:
        location = str(response.headers.get("Location") or "")
        if "login" in location.lower():
            return True
        content_type = str(response.headers.get("Content-Type") or "")
        if "text/html" not in content_type.lower():
            return False
        # 未登录时后台首页本身也会渲染登录页，据此识别会话失效。
        return _LOGIN_PAGE_MARKER in response.text

    def _send(self, handler: http.server.BaseHTTPRequestHandler, response: requests.Response) -> None:
        payload = response.content
        content_type = str(response.headers.get("Content-Type") or "")
        if "text/html" in content_type.lower():
            payload = _patch_tts_display(payload)
        handler.send_response(response.status_code)
        for key, value in response.headers.items():
            if key.lower() in _STRIPPED_HEADERS:
                continue
            if key.lower() == "location":
                value = value.replace(self.device_base, self._proxy_base())
            handler.send_header(key, value)
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        if handler.command != "HEAD":
            handler.wfile.write(payload)


class DeviceBackendProxyManager:
    """每台设备一个常驻代理：重复打开复用同一个地址，退出时统一关闭。"""

    def __init__(self, *, device_port: int = DEVICE_BACKEND_PORT) -> None:
        self._device_port = int(device_port)
        self._proxies: dict[str, DeviceBackendProxy] = {}
        self._lock = threading.Lock()

    def open_url(self, device_ip: str) -> str:
        normalized_ip = str(ipaddress.ip_address(device_ip))
        with self._lock:
            proxy = self._proxies.get(normalized_ip)
            if proxy is None:
                proxy = DeviceBackendProxy(normalized_ip, port=self._device_port)
                self._proxies[normalized_ip] = proxy
        return proxy.start()

    def close_all(self) -> None:
        with self._lock:
            proxies = list(self._proxies.values())
            self._proxies.clear()
        for proxy in proxies:
            proxy.stop()
