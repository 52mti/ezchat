"""本机控制台 HTTP 服务。只转发到 ConsoleApp，不写业务流程。"""

from __future__ import annotations

import asyncio
import base64
import queue
import threading
import webbrowser
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from services.console_app import ConsoleApp, InstanceLock


HOST = "127.0.0.1"
PORT = 8765
WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


class DeviceBody(BaseModel):
    ip: str


class LicenseBody(BaseModel):
    license_key: str = ""


class AppBody(BaseModel):
    app_name: str


class SettingsBody(BaseModel):
    device_ip: str
    app_name: str = ""
    values: dict[str, Any] = Field(default_factory=dict)


class PoolBody(BaseModel):
    size: int = 2


class VisionBody(BaseModel):
    device_ip: str
    app_name: str = ""


class TemplateBody(BaseModel):
    app_name: str
    group: str
    name: str
    image_base64: str
    region: list[float] | None = None


def create_app(console: ConsoleApp) -> FastAPI:
    app = FastAPI(title="机械臂控制台")
    app.state.console = console

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return console.status_snapshot()

    @app.get("/api/devices")
    def devices() -> dict[str, Any]:
        return {"devices": console.list_devices(), "status": console.status_snapshot()}

    @app.post("/api/devices")
    def add_device(body: DeviceBody) -> dict[str, Any]:
        try:
            created = console.add_device(body.ip.strip())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not created:
            raise HTTPException(status_code=409, detail="该设备已存在")
        return {"ok": True, "devices": console.list_devices()}

    @app.delete("/api/devices/{device_ip}")
    def delete_device(device_ip: str) -> dict[str, Any]:
        if not console.delete_device(device_ip):
            raise HTTPException(status_code=404, detail="设备不存在")
        return {"ok": True, "devices": console.list_devices()}

    @app.post("/api/devices/scan")
    def scan() -> dict[str, Any]:
        return console.scan_devices()

    @app.post("/api/devices/{device_ip}/start")
    def start_device(device_ip: str) -> dict[str, Any]:
        result = console.start_device(device_ip)
        if not result.get("ok"):
            raise HTTPException(status_code=409, detail=str(result.get("error") or "无法启动"))
        return result

    @app.post("/api/devices/{device_ip}/stop")
    def stop_device(device_ip: str) -> dict[str, Any]:
        return console.stop_device(device_ip)

    @app.post("/api/devices/{device_ip}/app")
    def set_app(device_ip: str, body: AppBody) -> dict[str, Any]:
        if console.config_manager.find_device(device_ip) is None:
            raise HTTPException(status_code=404, detail="设备不存在")
        console.set_selected_app(device_ip, body.app_name)
        return {"ok": True, "devices": console.list_devices()}

    @app.post("/api/devices/{device_ip}/license")
    def bind_license(device_ip: str, body: LicenseBody) -> dict[str, Any]:
        result = console.bind_license(device_ip, body.license_key)
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=str(result.get("error") or "卡密无效"))
        return result

    @app.get("/api/devices/{device_ip}/logs")
    def logs(device_ip: str) -> dict[str, Any]:
        return {"lines": console.logs_for(device_ip)}

    @app.get("/api/logs")
    def all_logs(device_ip: str = "") -> dict[str, Any]:
        return {"lines": console.logs_for(device_ip)}

    @app.get("/api/settings")
    def get_settings(device_ip: str, app_name: str = "") -> dict[str, Any]:
        return console.get_settings(device_ip, app_name or None)

    @app.put("/api/settings")
    def save_settings(body: SettingsBody) -> dict[str, Any]:
        return console.save_settings(body.device_ip, body.app_name, body.values)

    @app.put("/api/ocr-pool")
    def set_pool(body: PoolBody) -> dict[str, Any]:
        return {"size": console.set_pool_size(body.size)}

    @app.post("/api/vision/reset")
    def reset_vision(body: VisionBody) -> dict[str, Any]:
        console.reset_vision(body.device_ip, body.app_name)
        return {"ok": True}

    @app.post("/api/devices/{device_ip}/backend")
    def open_backend(device_ip: str) -> dict[str, Any]:
        try:
            url = console.open_backend(device_ip)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"url": url}

    @app.get("/api/devices/{device_ip}/preview.jpg")
    def preview(device_ip: str) -> Response:
        data = console.screenshots.get(device_ip)
        if not data:
            raise HTTPException(status_code=404, detail="还没有截图")
        return Response(content=data, media_type="image/jpeg")

    @app.post("/api/devices/{device_ip}/screenshot")
    def screenshot(device_ip: str) -> Response:
        try:
            data = console.capture_screenshot(device_ip)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"无法获取设备截图：{exc}") from exc
        return Response(content=data, media_type="image/jpeg")

    @app.post("/api/calibration")
    def start_calibration(body: VisionBody) -> dict[str, Any]:
        result = console.start_calibration(body.device_ip, body.app_name or None)
        if not result.get("ok"):
            raise HTTPException(status_code=409, detail=str(result.get("error") or "无法标定"))
        return result

    @app.get("/api/calibration/{device_ip}")
    def calibration(device_ip: str) -> dict[str, Any]:
        return console.calibration_state.get(device_ip, {"phase": "idle"})

    @app.get("/api/templates")
    def templates(app_name: str = "xinyu") -> dict[str, Any]:
        return {"templates": console.list_templates(app_name)}

    @app.post("/api/templates")
    def save_template(body: TemplateBody) -> dict[str, Any]:
        try:
            image = base64.b64decode(body.image_base64)
            saved = console.save_template(body.app_name, body.group, body.name, image, body.region)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, **saved}

    @app.get("/api/chat/{device_ip}/contacts")
    def contacts(device_ip: str, app_name: str = "") -> dict[str, Any]:
        return {"contacts": console.chat_contacts(device_ip, app_name or None)}

    @app.get("/api/chat/{device_ip}/messages")
    def messages(device_ip: str, contact: str = "", app_name: str = "") -> dict[str, Any]:
        return {"messages": console.chat_messages(device_ip, app_name or None, contact)}

    @app.get("/api/monitor")
    def monitor() -> dict[str, Any]:
        return {"frames": console.monitor_meta()}

    @app.get("/api/monitor/{device_ip}.jpg")
    def monitor_image(device_ip: str) -> Response:
        data = console.monitor_image(device_ip)
        if not data:
            raise HTTPException(status_code=404, detail="还没有截图")
        media = "image/jpeg" if data[:2] == b"\xff\xd8" else "application/octet-stream"
        return Response(content=data, media_type=media)

    @app.post("/api/updates/check")
    def check_update() -> dict[str, Any]:
        return console.check_update()

    @app.get("/api/updates")
    def update_state() -> dict[str, Any]:
        return console.update_state or {"phase": "idle"}

    @app.post("/api/updates/download")
    def download_update() -> dict[str, Any]:
        result = console.download_update()
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=str(result.get("error") or "无法下载"))
        return result

    @app.post("/api/updates/cancel")
    def cancel_update() -> dict[str, Any]:
        console.cancel_download()
        return {"ok": True}

    @app.post("/api/updates/apply")
    def apply_update() -> dict[str, Any]:
        result = console.apply_update()
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=str(result.get("error") or "无法应用更新"))
        return result

    @app.websocket("/api/events")
    async def events(websocket: WebSocket) -> None:
        await websocket.accept()
        listener = console.subscribe()
        try:
            await websocket.send_json({"type": "devices", "devices": console.list_devices(), "status": console.status_snapshot()})
            while True:
                try:
                    payload = await asyncio.to_thread(listener.get, True, 0.5)
                except queue.Empty:
                    continue
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            pass
        finally:
            console.unsubscribe(listener)

    if WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    else:

        @app.get("/")
        def missing_web() -> dict[str, str]:
            return {"message": "页面还没有构建。在 web 目录执行 npm install 和 npm run build。"}

    return app


def serve(host: str = HOST, port: int = PORT, *, open_browser: bool = True) -> int:
    import uvicorn

    from services import ConfigManager

    lock = InstanceLock()
    if not lock.acquire():
        _tell_already_running()
        return 1
    console = ConsoleApp(ConfigManager())
    console.start()
    app = create_app(console)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)

    def request_exit() -> None:
        server.should_exit = True

    console.on_exit = request_exit
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    try:
        server.run()
    finally:
        console.shutdown()
        lock.release()
    return 0


def _tell_already_running() -> None:
    message = "控制台已经在运行。请使用已经打开的页面，不要再启动第二个进程。"
    if __import__("os").name == "nt":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, "机械臂脚本", 0x40)
            return
        except Exception:
            pass
    print(message)
