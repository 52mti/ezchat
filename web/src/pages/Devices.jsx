import { useEffect, useState } from "react";
import { Button, Card, Input, Modal, Space, Statistic, Table, Typography, message } from "antd";
import { STATUS_LABELS, api } from "../api";

export default function DevicesPage({ devices, status, update, onChanged }) {
  const [ip, setIp] = useState("");
  const [logs, setLogs] = useState([]);
  const [selectedIp, setSelectedIp] = useState("");
  const [licenseIp, setLicenseIp] = useState("");
  const [licenseKey, setLicenseKey] = useState("");
  const [updateInfo, setUpdateInfo] = useState(update);

  useEffect(() => {
    setUpdateInfo(update);
  }, [update]);

  useEffect(() => {
    api.logs(selectedIp).then((data) => setLogs(data.lines || [])).catch(() => {});
  }, [selectedIp, devices]);

  async function run(action) {
    try {
      await action();
      if (onChanged) await onChanged();
    } catch (error) {
      message.error(error.message);
    }
  }

  const columns = [
    { title: "设备 IP", dataIndex: "ip" },
    { title: "App", dataIndex: "selected_app" },
    {
      title: "状态",
      dataIndex: "status",
      render: (value, row) => `${STATUS_LABELS[value] || value}${row.detail ? ` · ${row.detail}` : ""}`,
    },
    { title: "画面", dataIndex: "health" },
    {
      title: "平均用时",
      render: (_, row) => (row.average_reply_seconds == null ? "--" : `${Number(row.average_reply_seconds).toFixed(1)} 秒`),
    },
    { title: "无干预时长", dataIndex: "unattended_label" },
    { title: "余额", dataIndex: "balance" },
    { title: "卡密", dataIndex: "license_masked" },
    { title: "到期", dataIndex: "license_expires", render: (value) => value || "-" },
    {
      title: "操作",
      render: (_, row) => (
        <Space wrap>
          <Button type="primary" onClick={() => run(() => api.start(row.ip))}>
            启动
          </Button>
          <Button onClick={() => run(() => api.stop(row.ip))}>停止</Button>
          <Button onClick={() => setLicenseIp(row.ip)}>卡密</Button>
          <Button
            onClick={() =>
              run(async () => {
                const result = await api.openBackend(row.ip);
                window.open(result.url, "_blank");
              })
            }
          >
            设备后台
          </Button>
          <Button danger onClick={() => confirmDelete(row.ip)}>
            删除
          </Button>
        </Space>
      ),
    },
  ];

  function confirmDelete(deviceIp) {
    Modal.confirm({
      title: `确认删除设备 ${deviceIp} 吗？`,
      okText: "删除",
      okButtonProps: { danger: true },
      cancelText: "取消",
      onOk: () => run(() => api.deleteDevice(deviceIp)),
    });
  }

  async function refreshUpdate() {
    const data = await api.updateState();
    setUpdateInfo(data);
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <Card><Statistic title="设备" value={status.device_count || devices.length} /></Card>
        <Card><Statistic title="运行中" value={status.running_count || 0} /></Card>
        <Card><Statistic title="异常" value={status.alert_count || 0} /></Card>
        <Card><Statistic title="OCR 等待" value={status.ocr_waiting || 0} /></Card>
      </div>
      <Card>
        <Space wrap>
          <Button type="primary" loading={status.scan_running} onClick={() => run(() => api.scan())}>
            扫描设备
          </Button>
          <Input
            className="w-56"
            placeholder="输入控制盒 IP"
            value={ip}
            onChange={(event) => setIp(event.target.value)}
          />
          <Button
            onClick={() =>
              run(async () => {
                await api.addDevice(ip.trim());
                setIp("");
              })
            }
          >
            添加
          </Button>
          <Button onClick={() => run(() => api.checkUpdate())}>检查更新</Button>
          {updateInfo && updateInfo.phase && updateInfo.phase !== "idle" && (
            <Typography.Text>{updateInfo.message || updateInfo.phase}</Typography.Text>
          )}
          {updateInfo && updateInfo.phase === "ready" && (
            <Button onClick={() => run(async () => { await api.downloadUpdate(); await refreshUpdate(); })}>下载更新</Button>
          )}
          {updateInfo && updateInfo.phase === "downloading" && (
            <Button onClick={() => run(() => api.cancelUpdate())}>取消下载</Button>
          )}
          {updateInfo && updateInfo.phase === "downloaded" && (
            <Button type="primary" onClick={() => run(() => api.applyUpdate())}>
              应用更新并退出
            </Button>
          )}
        </Space>
        {status.scan_message ? <div className="mt-3 text-slate-600">{status.scan_message}</div> : null}
      </Card>
      <Table
        rowKey="ip"
        columns={columns}
        dataSource={devices}
        pagination={false}
        onRow={(row) => ({ onClick: () => setSelectedIp(row.ip) })}
      />
      <Card title={selectedIp ? `${selectedIp} 的日志` : "日志"}>
        <pre className="max-h-72 overflow-auto whitespace-pre-wrap text-sm">{logs.join("\n") || "暂无日志"}</pre>
      </Card>
      <Modal
        title={`为 ${licenseIp} 绑定卡密`}
        open={Boolean(licenseIp)}
        onCancel={() => setLicenseIp("")}
        onOk={() =>
          run(async () => {
            await api.license(licenseIp, licenseKey);
            message.success("正在校验卡密");
            setLicenseIp("");
            setLicenseKey("");
          })
        }
        okText="校验"
        cancelText="取消"
      >
        <Input.Password value={licenseKey} onChange={(event) => setLicenseKey(event.target.value)} placeholder="输入卡密" />
      </Modal>
    </div>
  );
}
