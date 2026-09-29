import { useEffect, useState } from "react";
import { Card } from "antd";
import { api } from "../api";
import { STATUS_LABELS } from "../api";

export default function MonitorPage() {
  const [frames, setFrames] = useState([]);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const timer = window.setInterval(() => setTick((value) => value + 1), 2000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    api.monitor().then((data) => setFrames(data.frames || [])).catch(() => {});
  }, [tick]);

  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
      {frames.map((frame) => (
        <Card key={frame.ip} title={`${frame.ip} · ${STATUS_LABELS[frame.status] || frame.status}`}>
          {frame.has_image ? (
            <img
              alt={frame.ip}
              className="max-h-[520px] w-full object-contain"
              src={`/api/monitor/${encodeURIComponent(frame.ip)}.jpg?t=${tick}`}
            />
          ) : (
            <div className="py-16 text-center text-slate-500">等待这台设备截图</div>
          )}
          {frame.label ? <div className="mt-2 text-slate-500">{frame.label}</div> : null}
        </Card>
      ))}
      {frames.length === 0 && <Card>还没有设备</Card>}
    </div>
  );
}
