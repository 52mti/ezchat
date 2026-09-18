from __future__ import annotations

import http.server
import threading

import pytest
import requests

from services.device_backend_proxy import DeviceBackendProxy, DeviceBackendProxyManager


class _FakeDevice:
    """模拟设备后台：/login 通过密码换发会话 Cookie，其余路径凭证访问。

    行为对齐真实固件——新旧后台共同点是"未登录访问首页返回登录页"，新后台
    额外给 Cookie 带 SameSite=Strict（对本测试无影响，代理在服务端持证）。
    """

    def __init__(self) -> None:
        self.logins = 0
        self.valid_tokens: set[str] = set()
        device = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _respond(self, status: int, body: bytes, content_type: str, headers: dict[str, str] | None = None) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _authed(self) -> bool:
                for part in (self.headers.get("Cookie") or "").split(";"):
                    key, _, value = part.strip().partition("=")
                    if key == "esp_token" and value in device.valid_tokens:
                        return True
                return False

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length).decode("utf-8")
                if self.path == "/login":
                    device.logins += 1
                    if "password=admin" in body:
                        token = f"token-{device.logins}"
                        device.valid_tokens.add(token)
                        self._respond(
                            302,
                            b"",
                            "text/html",
                            {
                                "Set-Cookie": f"esp_token={token}; Path=/; HttpOnly; SameSite=Strict",
                                "Location": "/",
                            },
                        )
                    else:
                        self._respond(401, b"unauthorized", "text/html")
                    return
                if not self._authed():
                    self._respond(200, b'<form id="loginForm"></form>', "text/html")
                    return
                self._respond(200, body.encode("utf-8"), "text/plain")

            def do_GET(self) -> None:
                if not self._authed():
                    self._respond(200, b'<form id="loginForm"></form>', "text/html")
                    return
                if self.path == "/":
                    # 对齐新版固件首页：自定义 TTS 地址以 value 属性服务端渲染，
                    # 页面脚本会在加载后把它清空展示
                    body = (
                        "<html><body>SETTINGS-MARKER"
                        "<input type='text' name='tts_url' id='tts_url' "
                        "value='http://192.168.1.96:9999/tts' style='display:none'>"
                        "</body></html>"
                    )
                    self._respond(200, body.encode(), "text/html")
                    return
                self._respond(200, f'{{"path": "{self.path}"}}'.encode(), "application/json")

            def log_message(self, *_args: object) -> None:
                return

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def start(self) -> "_FakeDevice":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture()
def fake_device():
    device = _FakeDevice().start()
    yield device
    device.stop()


def _proxy_for(fake_device: _FakeDevice) -> DeviceBackendProxy:
    return DeviceBackendProxy("127.0.0.1", port=fake_device.port)


def test_proxy_serves_logged_in_settings_page(fake_device):
    proxy = _proxy_for(fake_device)
    url = proxy.start()
    try:
        assert url.startswith("http://127.0.0.1:")
        response = requests.get(url, timeout=5)
        assert response.status_code == 200
        assert "SETTINGS-MARKER" in response.text
        assert fake_device.logins == 1
    finally:
        proxy.stop()


def test_proxy_relays_query_and_post(fake_device):
    proxy = _proxy_for(fake_device)
    url = proxy.start().rstrip("/")
    try:
        echoed = requests.get(f"{url}/api/echo?x=1", timeout=5).json()
        assert echoed == {"path": "/api/echo?x=1"}
        posted = requests.post(f"{url}/api/submit", data={"k": "v"}, timeout=5)
        assert posted.text == "k=v"
    finally:
        proxy.stop()


def test_proxy_restores_wiped_tts_display_value(fake_device):
    proxy = _proxy_for(fake_device)
    url = proxy.start()
    try:
        page = requests.get(url, timeout=5).text
        # 固件脚本清空输入框后，注入的 data-orig + 恢复脚本把真实存储值带回界面
        assert 'data-orig="http://192.168.1.96:9999/tts"' in page
        assert "i.dataset.orig" in page
        # 原始 value 属性与恢复脚本注入位置
        assert "value='http://192.168.1.96:9999/tts'" in page
        assert page.rstrip().endswith("</script></body></html>")
    finally:
        proxy.stop()


def test_proxy_relogins_after_session_expiry(fake_device):
    proxy = _proxy_for(fake_device)
    url = proxy.start()
    try:
        assert "SETTINGS-MARKER" in requests.get(url, timeout=5).text
        # 模拟设备重启：旧 token 全部失效，代理应自动重登
        fake_device.valid_tokens.clear()
        assert "SETTINGS-MARKER" in requests.get(url, timeout=5).text
        assert fake_device.logins == 2
    finally:
        proxy.stop()


def test_proxy_start_fails_when_device_unreachable():
    proxy = DeviceBackendProxy("127.0.0.1", port=1, timeout=1.0)
    with pytest.raises(requests.RequestException):
        proxy.start()


def test_manager_reuses_proxy_and_close_all_stops_it(fake_device):
    manager = DeviceBackendProxyManager(device_port=fake_device.port)
    first = manager.open_url("127.0.0.1")
    second = manager.open_url("127.0.0.1")
    assert first == second
    assert "SETTINGS-MARKER" in requests.get(first, timeout=5).text

    manager.close_all()
    with pytest.raises(requests.RequestException):
        requests.get(first, timeout=2)
    # 关闭后可重新创建
    reopened = manager.open_url("127.0.0.1")
    assert "SETTINGS-MARKER" in requests.get(reopened, timeout=5).text
    manager.close_all()
