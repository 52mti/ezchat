import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { App, Button, Card, Form, Input } from "antd";
import { login } from "../api";

export default function Login() {
  const navigate = useNavigate();
  const { message } = App.useApp();
  const [submitting, setSubmitting] = useState(false);

  return (
    <div style={{ minHeight: "100vh", display: "grid", placeItems: "center" }}>
      <Card title="卡密管理" style={{ width: 380 }}>
        <Form
          layout="vertical"
          onFinish={async (values) => {
            setSubmitting(true);
            try {
              await login(values.username, values.password);
              navigate("/", { replace: true });
            } catch (error) {
              message.error(error.message);
            } finally {
              setSubmitting(false);
            }
          }}
        >
          <Form.Item label="用户名" name="username" rules={[{ required: true, message: "请输入用户名" }]}>
            <Input autoComplete="username" />
          </Form.Item>
          <Form.Item label="口令" name="password" rules={[{ required: true, message: "请输入口令" }]}>
            <Input.Password autoComplete="current-password" />
          </Form.Item>
          <Button type="primary" htmlType="submit" block loading={submitting}>
            登录
          </Button>
        </Form>
      </Card>
    </div>
  );
}
