export function formatTime(value) {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  const pad = (number) => String(number).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

export const STATUS_LABELS = {
  issued: "未绑定",
  active: "使用中",
  expired: "已过期",
  revoked: "已作废",
};

export const STATUS_COLORS = {
  issued: "default",
  active: "green",
  expired: "orange",
  revoked: "red",
};

export const BIND_LABELS = {
  machine: "机器标识",
  ip: "仅 IP",
};

export const ACTION_LABELS = {
  "customer.create": "新建客户",
  "customer.update": "更新客户",
  "key.issue": "发放卡密",
  "key.renew": "续期",
  "key.revoke": "作废",
  "key.unbind": "解绑",
};
