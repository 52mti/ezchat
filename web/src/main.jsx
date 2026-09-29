import React from "react";
import ReactDOM from "react-dom/client";
import { ConfigProvider } from "antd";
import zhCN from "antd/locale/zh_CN";
import App from "./App";
import "antd/dist/reset.css";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ConfigProvider locale={zhCN} theme={{ token: { fontSize: 15, borderRadius: 8 } }}>
      <App />
    </ConfigProvider>
  </React.StrictMode>,
);
