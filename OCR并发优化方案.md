# OCR 并发优化方案

## 当前 OCR 版本

当前工程使用的是轻量移动端 OCR 模型，不是 server 大模型：

- PaddleOCR: 3.4.0
- PaddlePaddle: 3.2.2
- PaddleX: 3.4.3
- 模型: PP-OCRv5_mobile_det + PP-OCRv5_mobile_rec
- 默认模型档位: lite / mobile
- NVIDIA 后端: paddle_cuda，device=gpu:0
- 打包运行方式: subprocess

所以当前速度慢的主要原因不是模型太大，而是多设备同时跑时 OCR 推理基本仍然排队。

## 当前慢的原因

多设备并发时，机械臂动作、截图、AI 请求、UI 状态处理可以并行。

但 OCR 为了避免之前多线程并发调用 Paddle/DirectML 导致闪退，目前仍有全局预测锁保护。结果就是多台设备同时进入 OCR 步骤时，大概率会一张一张处理。

这提高了稳定性，但会造成明显排队，尤其是 5 台、10 台设备一起跑的时候。

## 显卡并发建议

### GTX 1060 5GB

建议配置：

```text
默认: 1
可测试: 2
不建议长期默认: 2
不建议: 3+
```

原因：

- 5GB 显存比较紧。
- PP-OCRv5 mobile 单进程占用不高，但多 OCR 子进程会分别加载 Paddle runtime、CUDA runtime、模型和推理 workspace。
- 单路占用低不代表 2/3 路一定线性安全。

实测判断：

```text
单路 OCR 显存长期低于 3.5GB: 可以短测 2
双路 OCR 显存接近 4.3GB+: 不建议长期用 2
出现 CUDA out of memory / OCR 超时 / fallback: 立刻降回 1
```

结论：1060 5GB 默认锁 1 最稳，2 只能作为客户机器实测后的可选值。

### RTX 3060 Ti 8GB

建议配置：

```text
默认: 1
推荐可用: 2
高级压测: 3
不建议默认: 3+
```

原因：

- 8GB 显存余量明显比 1060 5GB 好。
- 当前模型是 PP-OCRv5 mobile，单路占用通常不会太高。
- 开 2 路 OCR lane 很可能能明显减少 5 台设备时的 OCR 排队。

实测判断：

```text
单路 OCR 显存低于 1.5GB: 可以试 2
双路 OCR 显存低于 5.5GB: 2 可以长期用
三路 OCR 显存低于 7GB 且速度确实提升: 3 可以给高级用户用
显存接近 7.5GB: 不要再加
出现 CUDA OOM / OCR 超时 / fallback: 立即降并发
```

结论：3060 Ti 8GB 最值得测试并发 2。并发 3 可以压测，但不建议默认给普通客户。

## 推荐优化方向

最优方案是做：

```text
OCR 调度器 + OCR 子进程池 + OCR 缓存
```

核心原则：

- 主 worker 不直接调用 Paddle/DirectML OCR。
- 所有 OCR 请求进入统一队列。
- OCR 子进程独立加载 OCR runtime。
- UI 的 OCR 并发数量真正控制同时处理几张图。
- 某个 OCR 子进程崩溃，只重启这个子进程，不影响主程序和其他设备。

## 子进程池设计

新增模块建议：

```text
services/ocr_scheduler.py
ocr_worker_entry.py
runtime/tmp/ocr_requests/
runtime/tmp/ocr_results/
```

调度器负责：

- 启动 N 个 OCR 子进程。
- 维护 OCR 队列。
- 分发 OCR 任务。
- 收集 OCR 结果。
- 处理任务超时。
- OCR 子进程异常自动重启。
- 程序退出时清理子进程和临时文件。

OCR 子进程负责：

- 独立加载 PaddleOCR / DirectML / CPU OCR runtime。
- 接收 JSON 请求。
- 执行 OCR。
- 返回 JSON 结果。

## 缓存策略

增加短时内存缓存，减少重复识别。

缓存 key：

```text
image_hash + app_name + mode + region + ocr_config_signature
```

缓存建议：

```text
缓存时间: 60 秒
最大数量: 128 条
OCR 成功结果缓存
OCR 失败结果不缓存
```

收益：

- 同一轮里重复识别同一张图时直接复用。
- 异常恢复、确认界面、重复截图时减少 OCR 压力。
- 多设备排队时可以明显减负。

## 超时和取消

每个 OCR 任务必须支持：

- 排队超时。
- 推理超时。
- worker 停止时取消等待。
- OCR 子进程无响应自动重启。

建议默认：

```text
list OCR: 20 秒
dialogue OCR: 25 秒
points OCR: 15 秒
raw OCR: 20 秒
```

用户点击停止时：

- 当前设备不继续等待 OCR。
- OCR 子进程可以跑完当前任务，但结果丢弃。
- 后续机械臂动作必须停止。

## UI 显示建议

顶部 OCR 状态可以显示：

```text
OCR: GPU | 并发 2 | 队列 5
```

运行日志建议增加：

- OCR 入队：设备 IP / app / mode
- OCR 开始：lane 编号
- OCR 完成：耗时 / 是否命中缓存
- OCR 超时：任务 ID / mode
- OCR 子进程重启：lane 编号 / 原因

设备运行状态栏只显示简短状态：

```text
OCR等待中
OCR识别中
```

不要在设备状态栏显示过多 OCR 细节。

## 实施顺序

1. 新增 `services/ocr_scheduler.py` 和 `ocr_worker_entry.py`。
2. 先让 raw `vision.ocr.ocr()` 支持走调度器。
3. 再让 `vision.ocr_rules.run_ocr_mode()` 走调度器。
4. 接入短时缓存。
5. 接入 OCR 超时和取消。
6. UI 显示 OCR backend / 并发 / 队列。
7. 添加多设备 fake 压测。
8. 2 台真机测试。
9. 5 台真机测试。
10. 根据显卡档位设置默认推荐并发。

## 第一版建议范围

第一版先做：

- OCR 子进程池。
- OCR 并发数真实生效。
- OCR 缓存。
- OCR 超时。
- 基础日志。

先不做复杂 GPU 自动降级策略，保留现有降级逻辑。这样风险小，收益已经比较明显。

## 结论

当前 OCR 版本本身没有问题，已经是 PP-OCRv5 mobile 轻量模型。

当前需要优化的是调度方式，而不是换模型。

建议优先实现 OCR 子进程池，让 OCR 并发数量真正生效：

```text
1060 5GB: 默认 1，可测试 2
3060 Ti 8GB: 默认 1，推荐 2，可压测 3
```
