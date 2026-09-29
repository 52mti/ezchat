import { useEffect, useState } from "react";
import { Button, Card, Form, Input, InputNumber, Select, Space, Switch, message } from "antd";
import { api } from "../api";

const PERSONA_FIELDS = [
  ["基础设定", "稳定前缀，改动即时生效"],
  ["人物信息", "和基础设定放在同一段"],
  ["边界条件", "建议不超过 200 字"],
  ["审核标准", "独立审核，建议写清楚"],
];

export default function SettingsPage({ devices, status }) {
  const [deviceIp, setDeviceIp] = useState(devices[0]?.ip || "");
  const [form] = Form.useForm();

  useEffect(() => {
    if (!deviceIp && devices[0]) setDeviceIp(devices[0].ip);
  }, [devices, deviceIp]);

  useEffect(() => {
    if (!deviceIp) return;
    api.settings(deviceIp).then((data) => {
      form.setFieldsValue({
        app_name: data.app_name,
        persona_chat_model: data.persona_chat_model,
        ...data.runtime,
        ...data.device_app,
        persona_texts: data.persona_texts,
        ocr_backend: data.app?.ocr_modes?.dialogue?.ocr_backend || "auto",
      });
    }).catch((error) => message.error(error.message));
  }, [deviceIp, form]);

  async function save() {
    const values = form.getFieldsValue();
    const backend = values.ocr_backend || "auto";
    const ocrModes = {};
    for (const mode of ["dialogue", "list"]) {
      ocrModes[mode] = backend === "auto" ? {} : { ocr_backend: backend };
    }
    try {
      await api.saveSettings(deviceIp, values.app_name, {
        poll_interval: values.poll_interval,
        step_timeout: values.step_timeout,
        keep_awake_interval_seconds: values.keep_awake_interval_seconds,
        exclude_ocr_wait_from_step_timeout: values.exclude_ocr_wait_from_step_timeout,
        photo_sent_cooldown_seconds: values.photo_sent_cooldown_seconds,
        online_photo_send_count: values.online_photo_send_count,
        greetings_txt: values.greetings_txt,
        judge_enabled: values.judge_enabled,
        greeting_ai_enabled: values.greeting_ai_enabled,
        history_turns: values.history_turns,
        summary_interval: values.summary_interval,
        persona_profile_json: values.persona_profile_json,
        persona_texts: values.persona_texts,
        persona_chat_model: values.persona_chat_model,
        deepseek_api_key: values.deepseek_api_key,
        ocr_modes: ocrModes,
      });
      if (values.pool_size) await api.setPool(values.pool_size);
      message.success("配置已保存");
    } catch (error) {
      message.error(error.message);
    }
  }

  return (
    <Card title="运行、OCR 和人设">
      <Form form={form} layout="vertical" initialValues={{ pool_size: status.pool_size || 2 }}>
        <Space wrap className="mb-4">
          <Select
            className="w-64"
            placeholder="选择设备"
            value={deviceIp}
            options={devices.map((item) => ({ value: item.ip, label: item.ip }))}
            onChange={setDeviceIp}
          />
          <Form.Item name="app_name" label="App" className="mb-0">
            <Input className="w-40" />
          </Form.Item>
          <Form.Item name="pool_size" label="OCR 引擎并发" className="mb-0">
            <InputNumber min={1} max={16} />
          </Form.Item>
        </Space>
        <div className="grid gap-4 md:grid-cols-3">
          <Form.Item name="poll_interval" label="轮询间隔（秒）"><InputNumber className="w-full" min={1} /></Form.Item>
          <Form.Item name="step_timeout" label="步骤超时（秒）"><InputNumber className="w-full" min={1} /></Form.Item>
          <Form.Item name="keep_awake_interval_seconds" label="防息屏间隔（秒）"><InputNumber className="w-full" min={1} /></Form.Item>
          <Form.Item name="photo_sent_cooldown_seconds" label="发图冷却（秒）"><InputNumber className="w-full" min={0} /></Form.Item>
          <Form.Item name="online_photo_send_count" label="在线发图次数"><InputNumber className="w-full" min={0} /></Form.Item>
          <Form.Item name="history_turns" label="记忆轮数"><InputNumber className="w-full" min={0} /></Form.Item>
          <Form.Item name="summary_interval" label="摘要间隔"><InputNumber className="w-full" min={0} /></Form.Item>
        </div>
        <Space className="mb-4">
          <Form.Item name="exclude_ocr_wait_from_step_timeout" valuePropName="checked" label="OCR 等待不计入超时">
            <Switch />
          </Form.Item>
          <Form.Item name="judge_enabled" valuePropName="checked" label="启用审核">
            <Switch />
          </Form.Item>
          <Form.Item name="greeting_ai_enabled" valuePropName="checked" label="招呼语由 AI 生成">
            <Switch />
          </Form.Item>
        </Space>
        <Form.Item name="persona_chat_model" label="人设模型"><Input /></Form.Item>
        <Form.Item name="deepseek_api_key" label="DeepSeek 密钥（留空则不修改）">
          <Input.Password placeholder="只写入本机 .env，不会显示已保存的密钥" />
        </Form.Item>
        <Form.Item name="greetings_txt" label="招呼语文件"><Input /></Form.Item>
        <Form.Item name="persona_profile_json" label="人设方案包路径"><Input /></Form.Item>
        <Form.Item name="ocr_backend" label="OCR 后端">
          <Select
            options={[
              { value: "auto", label: "自动" },
              { value: "cpu", label: "CPU" },
              { value: "paddle_cuda", label: "NVIDIA GPU" },
            ]}
          />
        </Form.Item>
        {PERSONA_FIELDS.map(([key, hint]) => (
          <Form.Item key={key} name={["persona_texts", key]} label={`${key}（${hint}）`}>
            <Input.TextArea rows={4} />
          </Form.Item>
        ))}
        <Space>
          <Button type="primary" onClick={save}>保存</Button>
          <Button
            onClick={async () => {
              try {
                await api.resetVision(deviceIp, form.getFieldValue("app_name"));
                message.success("视觉识别次数已重置");
              } catch (error) {
                message.error(error.message);
              }
            }}
          >
            重置视觉识别
          </Button>
        </Space>
      </Form>
    </Card>
  );
}
