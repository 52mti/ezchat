import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { App, Button, Form, Input, Modal, Space, Table } from "antd";
import { api } from "../api";
import { formatTime } from "../format";

const EMPTY = { name: "", contact: "", note: "" };

export default function Customers() {
  const navigate = useNavigate();
  const { message } = App.useApp();
  const [query, setQuery] = useState("");
  const [rows, setRows] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [editing, setEditing] = useState(null);
  const [saving, setSaving] = useState(false);

  async function load(nextPage = page, nextQuery = query) {
    const params = new URLSearchParams({ page: String(nextPage), page_size: "20" });
    if (nextQuery.trim()) {
      params.set("q", nextQuery.trim());
    }
    const data = await api(`/api/admin/customers?${params.toString()}`);
    setRows(data.items);
    setTotal(data.total);
  }

  useEffect(() => {
    load(1, "").catch((error) => message.error(error.message));
  }, []);

  return (
    <div className="page">
      <div className="page-heading">
        <h2 style={{ margin: 0 }}>客户</h2>
        <Space>
          <Input.Search
            allowClear
            placeholder="名称或联系方式"
            onSearch={(value) => {
              setQuery(value);
              setPage(1);
              load(1, value).catch((error) => message.error(error.message));
            }}
            style={{ width: 240 }}
          />
          <Button type="primary" onClick={() => setEditing({ ...EMPTY })}>
            新建客户
          </Button>
        </Space>
      </div>
      <Table
        rowKey="id"
        dataSource={rows}
        pagination={{ current: page, pageSize: 20, total, onChange: (next) => { setPage(next); load(next).catch((error) => message.error(error.message)); } }}
        columns={[
          { title: "名称", dataIndex: "name" },
          { title: "联系方式", dataIndex: "contact", render: (value) => value || "-" },
          { title: "备注", dataIndex: "note", render: (value) => value || "-" },
          { title: "更新时间", dataIndex: "updated_at", render: formatTime },
          {
            title: "操作",
            render: (_, row) => (
              <Space>
                <Button size="small" onClick={() => setEditing(row)}>编辑</Button>
                <Button size="small" type="primary" onClick={() => navigate(`/keys?customer=${row.id}`)}>发卡</Button>
              </Space>
            ),
          },
        ]}
      />
      <Modal
        title={editing?.id ? "编辑客户" : "新建客户"}
        open={Boolean(editing)}
        confirmLoading={saving}
        onCancel={() => setEditing(null)}
        onOk={async () => {
          if (!editing?.name?.trim()) {
            message.error("请填写客户名称");
            return;
          }
          setSaving(true);
          try {
            const payload = { name: editing.name, contact: editing.contact || "", note: editing.note || "" };
            if (editing.id) {
              await api(`/api/admin/customers/${editing.id}`, { method: "PUT", body: JSON.stringify(payload) });
            } else {
              await api("/api/admin/customers", { method: "POST", body: JSON.stringify(payload) });
            }
            setEditing(null);
            await load();
          } catch (error) {
            message.error(error.message);
          } finally {
            setSaving(false);
          }
        }}
      >
        {editing ? (
          <Form layout="vertical">
            <Form.Item label="名称" required>
              <Input value={editing.name} onChange={(event) => setEditing((current) => ({ ...current, name: event.target.value }))} />
            </Form.Item>
            <Form.Item label="联系方式">
              <Input value={editing.contact} onChange={(event) => setEditing((current) => ({ ...current, contact: event.target.value }))} />
            </Form.Item>
            <Form.Item label="备注">
              <Input.TextArea value={editing.note} onChange={(event) => setEditing((current) => ({ ...current, note: event.target.value }))} />
            </Form.Item>
          </Form>
        ) : null}
      </Modal>
    </div>
  );
}
