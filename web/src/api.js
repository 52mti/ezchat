async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { detail: text };
    }
  }
  if (!response.ok) {
    const detail = data && (data.detail || data.message);
    throw new Error(typeof detail === "string" ? detail : "请求失败");
  }
  return data;
}

export const api = {
  devices: () => request("/api/devices"),
  addDevice: (ip) => request("/api/devices", { method: "POST", body: JSON.stringify({ ip }) }),
  deleteDevice: (ip) => request(`/api/devices/${encodeURIComponent(ip)}`, { method: "DELETE" }),
  scan: () => request("/api/devices/scan", { method: "POST" }),
  start: (ip) => request(`/api/devices/${encodeURIComponent(ip)}/start`, { method: "POST" }),
  stop: (ip) => request(`/api/devices/${encodeURIComponent(ip)}/stop`, { method: "POST" }),
  setApp: (ip, appName) =>
    request(`/api/devices/${encodeURIComponent(ip)}/app`, {
      method: "POST",
      body: JSON.stringify({ app_name: appName }),
    }),
  license: (ip, licenseKey) =>
    request(`/api/devices/${encodeURIComponent(ip)}/license`, {
      method: "POST",
      body: JSON.stringify({ license_key: licenseKey }),
    }),
  logs: (ip) => request(`/api/logs?device_ip=${encodeURIComponent(ip || "")}`),
  settings: (ip, appName) =>
    request(`/api/settings?device_ip=${encodeURIComponent(ip)}&app_name=${encodeURIComponent(appName || "")}`),
  saveSettings: (deviceIp, appName, values) =>
    request("/api/settings", {
      method: "PUT",
      body: JSON.stringify({ device_ip: deviceIp, app_name: appName, values }),
    }),
  setPool: (size) => request("/api/ocr-pool", { method: "PUT", body: JSON.stringify({ size }) }),
  resetVision: (deviceIp, appName) =>
    request("/api/vision/reset", { method: "POST", body: JSON.stringify({ device_ip: deviceIp, app_name: appName }) }),
  openBackend: (ip) => request(`/api/devices/${encodeURIComponent(ip)}/backend`, { method: "POST" }),
  contacts: (ip, appName) =>
    request(`/api/chat/${encodeURIComponent(ip)}/contacts?app_name=${encodeURIComponent(appName || "")}`),
  messages: (ip, contact, appName) =>
    request(
      `/api/chat/${encodeURIComponent(ip)}/messages?contact=${encodeURIComponent(contact)}&app_name=${encodeURIComponent(appName || "")}`,
    ),
  monitor: () => request("/api/monitor"),
  templates: (appName) => request(`/api/templates?app_name=${encodeURIComponent(appName || "xinyu")}`),
  saveTemplate: (body) => request("/api/templates", { method: "POST", body: JSON.stringify(body) }),
  calibrate: (deviceIp, appName) =>
    request("/api/calibration", { method: "POST", body: JSON.stringify({ device_ip: deviceIp, app_name: appName }) }),
  calibration: (ip) => request(`/api/calibration/${encodeURIComponent(ip)}`),
  checkUpdate: () => request("/api/updates/check", { method: "POST" }),
  updateState: () => request("/api/updates"),
  downloadUpdate: () => request("/api/updates/download", { method: "POST" }),
  cancelUpdate: () => request("/api/updates/cancel", { method: "POST" }),
  applyUpdate: () => request("/api/updates/apply", { method: "POST" }),
};

export function connectEvents(onMessage) {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${window.location.host}/api/events`);
  socket.onmessage = (event) => {
    try {
      onMessage(JSON.parse(event.data));
    } catch {
      /* 忽略无法解析的推送 */
    }
  };
  return socket;
}

export const STATUS_LABELS = {
  stopped: "已停止",
  running: "运行中",
  stopping: "正在停止",
  paused: "已暂停",
  circuit_open: "熔断",
  abnormal: "画面异常",
  license_error: "卡密异常",
  normal: "正常",
};
