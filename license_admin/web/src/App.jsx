import { useEffect, useState } from "react";
import { Navigate, Outlet, Route, BrowserRouter, Routes, useLocation, useNavigate } from "react-router-dom";
import { Button, Layout, Menu, Spin } from "antd";
import { ensureSession, logout } from "./api";
import Audit from "./pages/Audit";
import Customers from "./pages/Customers";
import Dashboard from "./pages/Dashboard";
import KeyDetail from "./pages/KeyDetail";
import Keys from "./pages/Keys";
import Login from "./pages/Login";

const { Header, Sider, Content } = Layout;

function RequireAuth() {
  const [state, setState] = useState("checking");

  useEffect(() => {
    let active = true;
    ensureSession().then((ok) => {
      if (active) {
        setState(ok ? "ready" : "denied");
      }
    });
    return () => {
      active = false;
    };
  }, []);

  if (state === "checking") {
    return (
      <div style={{ minHeight: "100vh", display: "grid", placeItems: "center" }}>
        <Spin />
      </div>
    );
  }
  if (state === "denied") {
    return <Navigate to="/login" replace />;
  }
  return <Outlet />;
}

function AdminLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  const selected = location.pathname.startsWith("/keys") ? "/keys" : location.pathname;

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Sider theme="light" width={200}>
        <div style={{ padding: "20px 16px", fontWeight: 650, fontSize: 16 }}>卡密管理</div>
        <Menu
          mode="inline"
          selectedKeys={[selected === "/" ? "/" : selected]}
          items={[
            { key: "/", label: "总览" },
            { key: "/customers", label: "客户" },
            { key: "/keys", label: "卡密" },
            { key: "/audit", label: "审计" },
          ]}
          onClick={({ key }) => navigate(key)}
        />
      </Sider>
      <Layout>
        <Header style={{ background: "#fff", display: "flex", justifyContent: "flex-end", alignItems: "center" }}>
          <Button
            onClick={() => {
              logout().finally(() => navigate("/login", { replace: true }));
            }}
          >
            退出
          </Button>
        </Header>
        <Content style={{ margin: 24 }}>
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route element={<RequireAuth />}>
          <Route element={<AdminLayout />}>
            <Route index element={<Dashboard />} />
            <Route path="customers" element={<Customers />} />
            <Route path="keys" element={<Keys />} />
            <Route path="keys/:id" element={<KeyDetail />} />
            <Route path="audit" element={<Audit />} />
          </Route>
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
