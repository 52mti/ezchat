import { useEffect, useState } from "react";
import { Card, List, Select, Typography, message } from "antd";
import { api } from "../api";

export default function ChatPage({ devices }) {
  const [deviceIp, setDeviceIp] = useState(devices[0]?.ip || "");
  const [contacts, setContacts] = useState([]);
  const [contact, setContact] = useState("");
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    if (!deviceIp && devices[0]) setDeviceIp(devices[0].ip);
  }, [devices, deviceIp]);

  useEffect(() => {
    if (!deviceIp) return;
    api.contacts(deviceIp)
      .then((data) => {
        const rows = data.contacts || [];
        setContacts(rows);
        setContact(rows[0]?.contact_name || rows[0]?.name || "");
      })
      .catch((error) => message.error(error.message));
  }, [deviceIp]);

  useEffect(() => {
    if (!deviceIp || !contact) {
      setMessages([]);
      return;
    }
    api.messages(deviceIp, contact)
      .then((data) => setMessages(data.messages || []))
      .catch((error) => message.error(error.message));
  }, [deviceIp, contact]);

  return (
    <div className="grid gap-4 md:grid-cols-[280px_1fr]">
      <Card title="联系人">
        <Select
          className="mb-3 w-full"
          value={deviceIp || undefined}
          placeholder="选择设备"
          options={devices.map((item) => ({ value: item.ip, label: item.ip }))}
          onChange={setDeviceIp}
        />
        <List
          dataSource={contacts}
          locale={{ emptyText: "还没有聊天记录" }}
          renderItem={(item) => {
            const name = item.contact_name || item.name || "未命名";
            return (
              <List.Item className={name === contact ? "bg-slate-100" : ""} onClick={() => setContact(name)}>
                {name}
              </List.Item>
            );
          }}
        />
      </Card>
      <Card title={contact || "消息"}>
        <List
          dataSource={messages}
          locale={{ emptyText: "选择一个联系人" }}
          renderItem={(item) => (
            <List.Item>
              <div>
                <Typography.Text type="secondary">{item.created_at} · {item.direction}</Typography.Text>
                <div>{item.content}</div>
              </div>
            </List.Item>
          )}
        />
      </Card>
    </div>
  );
}
