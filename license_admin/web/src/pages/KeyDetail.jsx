import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { App, Button, Descriptions, Space, Table, Tag } from "antd";
import { api } from "../api";
import { BIND_LABELS, STATUS_COLORS, STATUS_LABELS, formatTime } from "../format";

export default function KeyDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { message } = App.useApp();
  const [record, setRecord] = useState(null);

  async function load() {
    setRecord(await api(`/api/admin/keys/${id}`));
  }

  useEffect(() => {
    load().catch((error) => message.error(error.message));
  }, [id]);

  async function act(path, success) {
    try {
      await api(path, { method: "POST", body: JSON.stringify({}) });
      message.success(success);
      await load();
    } catch (error) {
      message.error(error.message);
    }
  }

  if (!record) {
    return null;
  }

  return (
    <div className="page">
      <div className="page-heading">
        <h2 style={{ margin: 0 }}>{record.code}</h2>
        <Space>
          <Button onClick={() => navigate("/keys")}>返回</Button>
          <Button onClick={() => navigator.clipboard.writeText(record.code).then(() => message.success("已复制"))}>
            复制卡密
          </Button>
          <Button onClick={() => act(`/api/admin/keys/${id}/renew`, "已续期 30 天")} disabled={record.status === "revoked"}>
            续期 30 天
          </Button>
          <Button onClick={() => act(`/api/admin/keys/${id}/unbind`, "已解绑")} disabled={!record.bind_mode}>
            解绑
          </Button>
          <Button danger onClick={() => act(`/api/admin/keys/${id}/revoke`, "已作废")} disabled={record.status === "revoked"}>
            作废
          </Button>
        </Space>
      </div>
      <Descriptions bordered column={2}>
        <Descriptions.Item label="客户">{record.customer_name}</Descriptions.Item>
        <Descriptions.Item label="状态">
          <Tag color={STATUS_COLORS[record.status]}>{STATUS_LABELS[record.status]}</Tag>
        </Descriptions.Item>
        <Descriptions.Item label="绑定方式">{BIND_LABELS[record.bind_mode] || "-"}</Descriptions.Item>
        <Descriptions.Item label="机器标识">{record.machine_id || "-"}</Descriptions.Item>
        <Descriptions.Item label="绑定 IP">{record.bound_device_ip || "-"}</Descriptions.Item>
        <Descriptions.Item label="最后 IP">{record.last_device_ip || "-"}</Descriptions.Item>
        <Descriptions.Item label="签发">{formatTime(record.issued_at)}</Descriptions.Item>
        <Descriptions.Item label="激活">{formatTime(record.activated_at)}</Descriptions.Item>
        <Descriptions.Item label="到期">{formatTime(record.expires_at)}</Descriptions.Item>
        <Descriptions.Item label="最后校验">{formatTime(record.last_verified_at)}</Descriptions.Item>
      </Descriptions>
      <h3 style={{ margin: 0 }}>续期记录</h3>
      <Table
        rowKey="id"
        dataSource={record.renewals}
        pagination={false}
        columns={[
          { title: "时间", dataIndex: "created_at", render: formatTime },
          { title: "操作者", dataIndex: "operator" },
          { title: "原到期", dataIndex: "old_expires_at", render: formatTime },
          { title: "新到期", dataIndex: "new_expires_at", render: formatTime },
          { title: "备注", dataIndex: "note", render: (value) => value || "-" },
        ]}
      />
    </div>
  );
}
