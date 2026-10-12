import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { App, Button, Select, Space, Table, Tag } from "antd";
import { api } from "../api";
import { BIND_LABELS, STATUS_COLORS, STATUS_LABELS, formatTime } from "../format";

export default function Keys() {
  const navigate = useNavigate();
  const { message } = App.useApp();
  const [searchParams, setSearchParams] = useSearchParams();
  const [customers, setCustomers] = useState([]);
  const [rows, setRows] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [status, setStatus] = useState("");
  const [customerId, setCustomerId] = useState(searchParams.get("customer") || "");
  const [expiring, setExpiring] = useState(false);
  const [selected, setSelected] = useState([]);
  const [issuing, setIssuing] = useState(false);
  const [issuedCode, setIssuedCode] = useState("");

  async function loadCustomers() {
    const data = await api("/api/admin/customers?page=1&page_size=100");
    setCustomers(data.items);
  }

  async function load(nextPage = page) {
    const params = new URLSearchParams({ page: String(nextPage), page_size: "20" });
    if (status) {
      params.set("status", status);
    }
    if (customerId) {
      params.set("customer_id", customerId);
    }
    if (expiring) {
      params.set("expiring_within_days", "7");
    }
    const data = await api(`/api/admin/keys?${params.toString()}`);
    setRows(data.items);
    setTotal(data.total);
  }

  useEffect(() => {
    loadCustomers().catch((error) => message.error(error.message));
  }, []);

  useEffect(() => {
    load(1).catch((error) => message.error(error.message));
    setPage(1);
  }, [status, customerId, expiring]);

  async function issue() {
    if (!customerId) {
      message.error("请先选择客户");
      return;
    }
    setIssuing(true);
    try {
      const created = await api("/api/admin/keys", {
        method: "POST",
        body: JSON.stringify({ customer_id: Number(customerId) }),
      });
      setIssuedCode(created.code);
      setSearchParams({});
      await load(1);
    } catch (error) {
      message.error(error.message);
    } finally {
      setIssuing(false);
    }
  }

  async function renewSelected() {
    if (!selected.length) {
      return;
    }
    try {
      const data = await api("/api/admin/keys/renew", {
        method: "POST",
        body: JSON.stringify({ ids: selected }),
      });
      const failed = data.results.filter((item) => !item.ok);
      if (failed.length) {
        message.warning(`${failed.length} 张未能续期`);
      } else {
        message.success("已续期 30 天");
      }
      setSelected([]);
      await load();
    } catch (error) {
      message.error(error.message);
    }
  }

  return (
    <div className="page">
      <div className="page-heading">
        <h2 style={{ margin: 0 }}>卡密</h2>
        <Space wrap>
          <Select
            allowClear
            placeholder="客户"
            style={{ width: 180 }}
            value={customerId || undefined}
            options={customers.map((item) => ({ value: String(item.id), label: item.name }))}
            onChange={(value) => setCustomerId(value || "")}
          />
          <Select
            allowClear
            placeholder="状态"
            style={{ width: 140 }}
            value={status || undefined}
            options={Object.entries(STATUS_LABELS).map(([value, label]) => ({ value, label }))}
            onChange={(value) => setStatus(value || "")}
          />
          <Button type={expiring ? "primary" : "default"} onClick={() => setExpiring((value) => !value)}>
            7 天内到期
          </Button>
          <Button onClick={renewSelected} disabled={!selected.length}>
            批量续期 30 天
          </Button>
          <Button type="primary" loading={issuing} onClick={issue}>
            发放卡密
          </Button>
        </Space>
      </div>
      {issuedCode ? (
        <div style={{ background: "#fff", padding: 16, borderRadius: 8 }}>
          新卡密：<strong>{issuedCode}</strong>
          <Button
            type="link"
            onClick={() => navigator.clipboard.writeText(issuedCode).then(() => message.success("已复制"))}
          >
            复制
          </Button>
        </div>
      ) : null}
      <Table
        rowKey="id"
        dataSource={rows}
        rowSelection={{ selectedRowKeys: selected, onChange: setSelected }}
        pagination={{
          current: page,
          pageSize: 20,
          total,
          onChange: (next) => {
            setPage(next);
            load(next).catch((error) => message.error(error.message));
          },
        }}
        columns={[
          { title: "卡密", dataIndex: "code" },
          { title: "客户", dataIndex: "customer_name" },
          {
            title: "状态",
            dataIndex: "status",
            render: (value) => <Tag color={STATUS_COLORS[value]}>{STATUS_LABELS[value] || value}</Tag>,
          },
          { title: "绑定", dataIndex: "bind_mode", render: (value) => BIND_LABELS[value] || "-" },
          { title: "到期", dataIndex: "expires_at", render: formatTime },
          { title: "最后校验", dataIndex: "last_verified_at", render: formatTime },
          {
            title: "操作",
            render: (_, row) => (
              <Button size="small" onClick={() => navigate(`/keys/${row.id}`)}>
                详情
              </Button>
            ),
          },
        ]}
      />
    </div>
  );
}
