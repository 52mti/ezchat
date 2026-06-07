# OCR 单线程 GPU 提速方案

> 本方案只讨论 OCR 单线程本身的提速方式：OCR runtime、GPU 推理、预热、父子进程传图和 OCR 结果序列化。不讨论截图、机械臂动作、页面等待、模板匹配、AI/TTS 或业务流程优化。

## 1. 当前实现现状

当前工程 OCR 的主路径在以下文件中：

- `vision/ocr.py`：统一 OCR 入口，负责裁剪区域、归一化坐标、并发等待。
- `vision/ocr_runtime.py`：OCR 后端选择、Paddle/DirectML/CPU runtime、子进程管理。
- `ocr_subprocess_entry.py`：OCR 子进程入口。
- `main.py`：主程序启动时尝试预加载 OCR runtime。
- `workers/device_worker.py`：把全局 `ocr_max_concurrency` 写入当前设备 App 的 OCR 配置。

当前已实现的能力：

- 使用 PP-OCRv5 mobile 模型：
  - `PP-OCRv5_mobile_det`
  - `PP-OCRv5_mobile_rec`
- Paddle 相关版本：
  - `paddlepaddle / paddlepaddle-gpu == 3.2.2`
  - `paddleocr == 3.4.0`
  - `paddlex[ocr-core] == 3.4.3`
- 打包版默认通过独立 OCR 子进程运行：
  - `runtime/ocr_nvidia`
  - `runtime/ocr_cpu`
  - `runtime/ocr_directml`
- OCR 子进程正常情况下会常驻：
  - 父进程启动子进程后，通过 stdin/stdout 发请求。
  - 子进程里只初始化一次 OCR 引擎。
  - 后续 OCR 请求复用同一个 PaddleOCR/RapidOCR 引擎。
- OCR 异常时会关闭子进程：
  - 子进程返回错误、空输出、非法 JSON 时，父进程会关闭它。
  - 下一次 OCR 会重新拉起子进程，避免 PaddleX 重复初始化状态继续污染后续请求。
- PaddleOCR 额外功能默认关闭：
  - `use_doc_orientation_classify=False`
  - `use_doc_unwarping=False`
  - `use_textline_orientation=False`
- 主程序启动时会调用 `engine.preload()`：
  - 能提前初始化 OCR 引擎。
  - 能提前写入当前 OCR 状态，例如 UI 上显示 CPU/GPU。

当前还没有实现或需要补强的点：

- `preload()` 目前主要是初始化引擎，不等于真正跑了一次 OCR `predict()`。
- OCR 子进程启动环境里还没有设置 Paddle 显存分配参数。
- 每次 OCR 的耗时拆分还不够细，无法直接看出慢在传图、读图、推理还是结果序列化。
- 父子进程之间现在用临时 `.npy` 文件传图，稳定但有额外磁盘读写成本。

## 2. OCR 内部耗时拆分

只讨论 OCR 本身时，优化前应该先把 OCR 调用拆成下面几段：

```text
父进程准备图片耗时
父进程保存/传输图片耗时
OCR 子进程读图耗时
OCR predict 耗时
OCR 结果序列化耗时
OCR 规则后处理耗时
```

其中最核心的是 `predict`，也就是 OCR 模型推理。GPU 参数主要影响这一段，以及 Paddle runtime 的显存分配和 kernel 初始化开销。

## 3. 最优实施顺序

### 第一步：加 OCR 分段耗时日志

这是优先级最高的改动，因为没有分段耗时就无法判断 GPU 到底慢在哪里。

建议在调试日志里输出：

```text
OCR timing: backend=paddle_cuda device=gpu:0 mode=list total=830ms save=45ms request=770ms child_load=25ms predict=690ms serialize=8ms
```

需要记录的点：

- 父进程保存 `.npy` 的耗时。
- 父进程等待子进程响应的耗时。
- 子进程 `np.load()` 读图耗时。
- 子进程 `engine.predict()` 推理耗时。
- 子进程序列化结果耗时。
- OCR mode，例如 `list`、`dialogue`、`points`。
- OCR backend，例如 `paddle_cuda`、`cpu`、`onnx_directml`。
- OCR device，例如 `gpu:0`、`cpu`、`dml:0`。

这一步不改变业务行为，风险最低。

### 第二步：给 OCR 子进程加 Paddle 环境参数

当前子进程已经设置：

```text
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
PADDLE_PDX_CACHE_HOME=runtime/.paddlex
PADDLEX_HOME=runtime/.paddlex
```

建议新增：

```text
FLAGS_allocator_strategy=auto_growth
FLAGS_fraction_of_gpu_memory_to_use=0.6
```

含义：

- `auto_growth`：显存按需增长，降低一次性占用和显存碎片风险。
- `0.6`：默认允许 Paddle 使用约 60% 显存。

建议默认值：

```json
{
  "ocr_gpu_allocator_strategy": "auto_growth",
  "ocr_gpu_memory_fraction": 0.6
}
```

如果客户电脑只跑本程序，可以继续测试 `0.7`。

如果客户电脑还开直播、剪辑、游戏或其他占 GPU 的程序，建议降回 `0.5` 或更低。

注意：这个参数主要提升稳定性和减少显存分配抖动，不保证单次 OCR 一定大幅变快。

### 第三步：把预热从 preload 升级为 predict warmup

当前 `main.py` 启动时会调用 `engine.preload()`。

这个动作能初始化 PaddleOCR，但不能保证 CUDA kernel、模型推理路径和真实图片尺寸相关的缓存都提前完成。

建议新增真正的 warmup：

```text
启动 OCR 子进程
初始化 PaddleOCR
用固定尺寸空白图跑一次 predict
记录 warmup 是否成功
```

因为正式 OCR 图片尺寸不能缩小，所以 warmup 不应该影响正式 OCR 的输入尺寸。

可选策略：

- 如果能从 OCR 输入配置中拿到固定尺寸，就用同尺寸空白图 warmup。
- 如果拿不到尺寸，就只做当前 preload，不强行猜尺寸。
- 首次真实 OCR 完成后，后续就已经完成真实尺寸推理路径的预热。

推荐先做“可配置尺寸 warmup”，不要在没有尺寸时乱猜。

### 第四步：确认单线程常驻状态

当前 `get_ocr_runtime_engine()` 会按 OCR 相关配置缓存 runtime。

在常规配置不变的情况下，多台设备会复用同一个 OCR runtime 对象和同一个 OCR 子进程。

这符合“单线程 GPU OCR”的要求，但也意味着：

- 多台设备同时发 OCR 请求时，OCR 会排队。
- 不会同时跑两张图的 GPU OCR。
- 单线程方案的优化重点是让每一张图更快，而不是让多张图并行。

需要继续保证：

- 正常 OCR 成功后，子进程不退出。
- 日志里不要反复出现 OCR 初始化。
- 只有错误时才关闭子进程并重启。

### 第五步：测 `.npy` 传图成本，再决定是否改传输方式

当前流程：

```text
父进程 np.save(image)
父进程发送 image_path
子进程 np.load(image_path)
子进程 predict
父进程删除临时文件
```

优点：

- 稳定。
- 调试简单。
- 不容易因为 stdin 大数据破坏 JSON 协议。

缺点：

- 每次 OCR 都有磁盘写入和读取。
- 多设备排队时，临时文件 I/O 可能累计成明显耗时。

只有当分段日志证明 `save + child_load` 明显偏高时，才建议改成：

- stdin 二进制传图。
- 固定 mmap 文件复用。
- 共享内存。

目前不建议直接改，因为这会增加复杂度和出错面。

## 4. 不建议现在做的事

### 不缩小图片

不使用下面这类降尺寸参数：

```python
text_det_limit_side_len = 960
text_det_limit_type = "max"
```

原因：当前业务依赖小字、未读数量、列表内容和聊天内容，缩小图片可能影响识别准确率。

### 不盲目开启 TensorRT

TensorRT 对客户机器的 CUDA、cuDNN、显卡架构和 Paddle 版本要求更高。

风险：

- 有些客户机器直接启动失败。
- 第一次构建 engine 很慢。
- 不同显卡表现差异大。
- 打包和售后排查复杂度明显上升。

当前阶段不建议默认启用 TensorRT。

### 不通过增加 OCR 并发解决本问题

本方案讨论的是单线程 GPU OCR。

提高 `ocr_max_concurrency` 属于多图并行排队优化，会影响显存、稳定性和调度方式，不放在本方案里。

## 5. 推荐改动清单

优先做这几项：

1. 在 `vision/ocr_runtime.py` 和 `ocr_subprocess_entry.py` 增加 OCR 分段耗时。
2. 在 `SubprocessOcrRuntimeEngine._start_process()` 里设置 Paddle 显存环境变量。
3. 在 runtime 状态或日志中显示：
   - backend
   - device
   - execution_mode
   - allocator_strategy
   - gpu_memory_fraction
4. 把启动预热升级为可选 `predict warmup`。
5. 根据分段耗时决定是否优化 `.npy` 传图。

暂时不做：

- 不缩小图片。
- 不改 OCR 并发数量。
- 不默认启用 TensorRT。
- 不把 `backend` 强行固定成 `paddle_cuda`，除非客户机 GPU runtime 已经验证通过。

## 6. 建议日志格式

启动时输出一次：

```text
OCR runtime: backend=paddle_cuda device=gpu:0 execution_mode=subprocess model=PP-OCRv5_mobile allocator=auto_growth gpu_memory_fraction=0.6
```

每次 OCR 在调试模式下输出：

```text
OCR timing: mode=list total=830ms save=45ms request=770ms child_load=25ms predict=690ms serialize=8ms items=36
```

异常时输出：

```text
OCR error: backend=paddle_cuda device=gpu:0 action=predict error=...
```

这样能快速判断：

- 是否真的在用 GPU。
- 是否每次都在重新初始化。
- 是否慢在推理。
- 是否慢在父子进程传图。
- 是否慢在 OCR 规则后处理。

## 7. 预期收益

| 优化项 | 预期收益 | 风险 |
| --- | --- | --- |
| 分段耗时日志 | 定位瓶颈，避免误优化 | 低 |
| 子进程常驻确认 | 避免重复初始化 | 低 |
| Paddle 显存参数 | 稳定性提升，可能减少分配抖动 | 低到中 |
| predict warmup | 减少首轮 OCR 卡顿 | 低 |
| `.npy` 传图优化 | 只有 I/O 明显偏高时才有收益 | 中 |
| TensorRT | 可能更快，但兼容风险高 | 高 |

## 8. 当前结论

当前代码已经具备“单线程 OCR 子进程常驻”的基础，PP-OCRv5 mobile 也已经是轻量模型。

下一步最有价值的不是盲目改模型，而是：

```text
先加分段耗时日志
再加 Paddle 显存环境参数
再把 preload 升级为真正 predict warmup
最后根据数据决定是否优化传图方式
```

如果日志显示 `predict` 本身仍然很慢，再继续研究 Paddle 参数或 TensorRT。

如果日志显示慢在 `.npy` 传图或 OCR 规则后处理，就不要继续从 GPU 参数上硬找原因。
