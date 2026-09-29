import { useEffect, useRef, useState } from "react";
import { Button, Card, Input, List, Select, Space, message } from "antd";
import { api } from "../api";

export default function CanvasPage({ devices }) {
  const [deviceIp, setDeviceIp] = useState(devices[0]?.ip || "");
  const [group, setGroup] = useState("buttons");
  const [name, setName] = useState("popup_custom");
  const [templates, setTemplates] = useState([]);
  const [previewUrl, setPreviewUrl] = useState("");
  const [selection, setSelection] = useState(null);
  const [drag, setDrag] = useState(null);
  const [calibration, setCalibration] = useState(null);
  const imageRef = useRef(null);

  useEffect(() => {
    if (!deviceIp && devices[0]) setDeviceIp(devices[0].ip);
  }, [devices, deviceIp]);

  useEffect(() => {
    const appName = devices.find((item) => item.ip === deviceIp)?.selected_app || "xinyu";
    api.templates(appName).then((data) => setTemplates(data.templates || [])).catch(() => {});
  }, [deviceIp, devices]);

  function imagePoint(event) {
    const image = imageRef.current;
    if (!image) return null;
    const box = image.getBoundingClientRect();
    const x = (event.clientX - box.left) / box.width;
    const y = (event.clientY - box.top) / box.height;
    return {
      x: Math.min(1, Math.max(0, x)),
      y: Math.min(1, Math.max(0, y)),
    };
  }

  function onMouseDown(event) {
    const point = imagePoint(event);
    if (!point) return;
    setDrag(point);
    setSelection({ x: point.x, y: point.y, w: 0, h: 0 });
  }

  function onMouseMove(event) {
    if (!drag) return;
    const point = imagePoint(event);
    if (!point) return;
    setSelection({
      x: Math.min(drag.x, point.x),
      y: Math.min(drag.y, point.y),
      w: Math.abs(point.x - drag.x),
      h: Math.abs(point.y - drag.y),
    });
  }

  function onMouseUp() {
    setDrag(null);
  }

  async function capture() {
    if (!deviceIp) return;
    try {
      const response = await fetch(`/api/devices/${encodeURIComponent(deviceIp)}/screenshot`, { method: "POST" });
      if (!response.ok) throw new Error(await response.text());
      const blob = await response.blob();
      setPreviewUrl(URL.createObjectURL(blob));
      setSelection(null);
    } catch (error) {
      message.error(error.message || "截图失败");
    }
  }

  async function saveCrop() {
    const image = imageRef.current;
    if (!image || !selection || selection.w < 0.01 || selection.h < 0.01) {
      message.warning("请先在截图上拖出一个区域");
      return;
    }
    const canvas = document.createElement("canvas");
    const width = Math.max(1, Math.round(selection.w * image.naturalWidth));
    const height = Math.max(1, Math.round(selection.h * image.naturalHeight));
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    context.drawImage(
      image,
      selection.x * image.naturalWidth,
      selection.y * image.naturalHeight,
      width,
      height,
      0,
      0,
      width,
      height,
    );
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
    const bytes = new Uint8Array(await blob.arrayBuffer());
    let binary = "";
    bytes.forEach((value) => {
      binary += String.fromCharCode(value);
    });
    const appName = devices.find((item) => item.ip === deviceIp)?.selected_app || "xinyu";
    try {
      await api.saveTemplate({
        app_name: appName,
        group,
        name,
        image_base64: btoa(binary),
        region: [selection.x, selection.y, selection.w, selection.h],
      });
      message.success("模板已保存");
      const data = await api.templates(appName);
      setTemplates(data.templates || []);
    } catch (error) {
      message.error(error.message);
    }
  }

  async function calibrate() {
    if (!deviceIp) return;
    try {
      const appName = devices.find((item) => item.ip === deviceIp)?.selected_app || "xinyu";
      await api.calibrate(deviceIp, appName);
      message.info("正在标定，请让手机停留在聊天窗口");
      for (let attempt = 0; attempt < 40; attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 1500));
        const state = await api.calibration(deviceIp);
        setCalibration(state);
        if (state.phase === "done" || state.phase === "error") {
          setPreviewUrl(`/api/devices/${encodeURIComponent(deviceIp)}/preview.jpg?t=${Date.now()}`);
          if (state.phase === "error") message.error(state.error || "标定失败");
          return;
        }
      }
    } catch (error) {
      message.error(error.message);
    }
  }

  const region = calibration?.result?.chat_area_dynamic;
  const box = selection || (Array.isArray(region) && region.length === 4
    ? { x: region[0], y: region[1], w: region[2], h: region[3] }
    : null);

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
      <Card title="在截图上拖拽框选模板，或执行视觉标定">
        <Space wrap className="mb-4">
          <Select
            className="w-56"
            value={deviceIp || undefined}
            placeholder="选择设备"
            options={devices.map((item) => ({ value: item.ip, label: item.ip }))}
            onChange={setDeviceIp}
          />
          <Button type="primary" onClick={capture}>刷新截图</Button>
          <Button onClick={calibrate}>视觉标定</Button>
          <Select
            value={group}
            onChange={setGroup}
            options={[
              { value: "buttons", label: "按钮模板" },
              { value: "screens", label: "界面模板" },
            ]}
          />
          <Input className="w-48" value={name} onChange={(event) => setName(event.target.value)} placeholder="模板名称" />
          <Button onClick={saveCrop}>保存框选</Button>
        </Space>
        <div
          className="relative inline-block max-w-full"
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseUp}
        >
          {previewUrl ? (
            <img ref={imageRef} alt="设备截图" src={previewUrl} className="max-h-[78vh] max-w-full select-none" draggable={false} />
          ) : (
            <div className="py-24 text-slate-500">先刷新截图，再在画面上拖出模板区域。</div>
          )}
          {previewUrl && box && (
            <div
              className="pointer-events-none absolute border-2 border-red-500 bg-red-500/20"
              style={{
                left: `${box.x * 100}%`,
                top: `${box.y * 100}%`,
                width: `${box.w * 100}%`,
                height: `${box.h * 100}%`,
              }}
            />
          )}
        </div>
        {calibration?.result?.reason ? <div className="mt-3">{calibration.result.reason}</div> : null}
        {calibration?.phase === "done" && calibration.result?.ok ? <div className="mt-3">聊天区域已写入坐标缓存。</div> : null}
      </Card>
      <Card title="已有模板">
        <List
          dataSource={templates}
          locale={{ emptyText: "还没有模板" }}
          renderItem={(item) => <List.Item>{item.group} / {item.name}</List.Item>}
        />
      </Card>
    </div>
  );
}
