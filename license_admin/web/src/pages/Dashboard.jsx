import { useEffect, useState } from "react";
import { Card, Col, Row, Statistic } from "antd";
import { api } from "../api";

const CARDS = [
  ["active", "使用中"],
  ["expiring_soon", "7 天内到期"],
  ["expired", "已过期"],
  ["unbound", "未绑定"],
  ["customers", "客户"],
];

export default function Dashboard() {
  const [stats, setStats] = useState(null);

  useEffect(() => {
    api("/api/admin/stats").then(setStats);
  }, []);

  return (
    <div className="page">
      <h2 style={{ margin: 0 }}>总览</h2>
      <Row gutter={[16, 16]}>
        {CARDS.map(([key, title]) => (
          <Col key={key} xs={24} sm={12} lg={8} xl={4}>
            <Card>
              <Statistic title={title} value={stats ? stats[key] : "-"} />
            </Card>
          </Col>
        ))}
      </Row>
    </div>
  );
}
