import { useEffect, useState } from "react";
import { App, Table } from "antd";
import { api } from "../api";
import { ACTION_LABELS, formatTime } from "../format";

export default function Audit() {
  const { message } = App.useApp();
  const [rows, setRows] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);

  async function load(nextPage = page) {
    const data = await api(`/api/admin/audit?page=${nextPage}&page_size=20`);
    setRows(data.items);
    setTotal(data.total);
  }

  useEffect(() => {
    load(1).catch((error) => message.error(error.message));
  }, []);

  return (
    <div className="page">
      <h2 style={{ margin: 0 }}>审计</h2>
      <Table
        rowKey="id"
        dataSource={rows}
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
          { title: "时间", dataIndex: "created_at", render: formatTime },
          { title: "操作者", dataIndex: "operator" },
          { title: "动作", dataIndex: "action", render: (value) => ACTION_LABELS[value] || value },
          { title: "摘要", dataIndex: "summary" },
        ]}
      />
    </div>
  );
}
