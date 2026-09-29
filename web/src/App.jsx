import { useEffect, useState } from "react";
import { Layout, Menu, Typography } from "antd";
import DevicesPage from "./pages/Devices";
import SettingsPage from "./pages/Settings";
import ChatPage from "./pages/Chat";
import MonitorPage from "./pages/Monitor";
import CanvasPage from "./pages/CanvasTools";
import { api, connectEvents } from "./api";

const { Header, Content } = Layout;

const PAGES = [
  { key: "devices", label: "设备值守" },
  { key: "settings", label: "设置" },
  { key: "chat", label: "聊天记录" },
  { key: "monitor", label: "监控墙" },
  { key: "canvas", label: "标定与模板" },
];

export default function App() {
  const [page, setPage] = useState("devices");
  const [snapshot, setSnapshot] = useState({ devices: [], status: {} });
  const [update, setUpdate] = useState(null);

  async function reload() {
    const data = await api.devices();
    setSnapshot(data);
  }

  useEffect(() => {
    reload().catch(() => {});
    const socket = connectEvents((message) => {
      if (message.type === "devices" || message.type === "scan") {
        if (message.devices) {
          setSnapshot({ devices: message.devices, status: message.status || {} });
        } else {
          reload().catch(() => {});
        }
      }
      if (message.type === "update") {
        setUpdate(message.update);
      }
    });
    const timer = window.setInterval(() => {
      reload().catch(() => {});
    }, 3000);
    return () => {
      socket.close();
      window.clearInterval(timer);
    };
  }, []);

  const devices = snapshot.devices || [];
  const status = snapshot.status || {};

  return (
    <Layout className="min-h-screen">
      <Header className="flex items-center gap-6 bg-slate-900 px-6">
        <Typography.Title level={4} className="!mb-0 !text-white">
          机械臂控制台
        </Typography.Title>
        <Menu
          theme="dark"
          mode="horizontal"
          selectedKeys={[page]}
          items={PAGES}
          onClick={(item) => setPage(item.key)}
          className="min-w-0 flex-1 bg-transparent"
        />
        <span className="text-slate-200">{status.ocr_mode || ""}</span>
      </Header>
      <Content className="page p-6">
        {page === "devices" && (
          <DevicesPage devices={devices} status={status} update={update} onChanged={reload} />
        )}
        {page === "settings" && <SettingsPage devices={devices} status={status} />}
        {page === "chat" && <ChatPage devices={devices} />}
        {page === "monitor" && <MonitorPage devices={devices} />}
        {page === "canvas" && <CanvasPage devices={devices} />}
      </Content>
    </Layout>
  );
}
