# 机械臂脚本（jixiebijiaoben）

机械臂 + 手机自动化聊天系统：PC 端 PyQt6 界面统一管理多台设备，通过摄像头透视截图 + 模板匹配 + PaddleOCR 识别手机屏幕上的消息列表与聊天内容，调用云端大模型按人设生成回复，再由机械臂在手机屏幕上完成点击与输入，实现全自动聊天值守。

## 功能特性

- **多设备并行**：每台设备一个工作线程，UI 实时展示运行状态、聊天记录与错误事件
- **视觉识别**：摄像头透视校正取图、模板匹配定位界面、PaddleOCR 条带化识别（按 App 独立维护 OCR 规则）
- **坐标缓存**：列表/聊天区域按「设备 + App + 透视签名」四维缓存，越用越准，无需每轮全屏 OCR
- **人设 AI**：DeepSeek 官方直连生成人设回复（密钥本地保存，绝不入库）
- **TTS 语音**：piper 本地语音合成
- **卡密授权**：按设备授权，license 服务校验
- **自动更新**：蓝奏云 / 腾讯云 COS 分发更新包
- **一键标定**：UI 内置「视觉 + OCR 识别一次」，可视化窗口直接看到识别出的聊天区域

## 系统架构

```
main.py ──► ui/（PyQt6 界面）
              │
              ▼
        workers/（设备工作线程）
              │
              ▼
        apps/（业务 App 层：xinyu + 各 App 独立 OCR 规则）
              │
        ┌─────┴──────┬───────────┬──────────┐
        ▼            ▼           ▼          ▼
    vision/       services/    lgb/      jiekou/
  （视觉/OCR基础服务）（配置/AI/消息/更新）（机械臂）(接口参考🔒)
```

| 目录 | 职责 |
| --- | --- |
| `ui/` | PyQt6 主界面、聊天记录窗口、视觉标定窗口 |
| `workers/` | 每设备一线程的主循环（轮询截图 → 识别 → 回复） |
| `apps/` | 业务 App 层，当前为 `xinyu`（`wechat`/`huanyou` 为历史别名）；每个 App 独立维护 `ocr_rules.py` |
| `vision/` | 纯基础服务：透视截图、模板匹配、PaddleOCR、OCR 运行时 |
| `services/` | 配置管理、DeepSeek 人设、视觉对话识别客户端、TTS、消息、记忆、更新、卡密 |
| `auto_glm/` | AutoGLM / GPT Planner 规划模型接入 |
| `lgb/`、`jiekou/` | 🔒 只读保护区：机械臂控制与接口参考文档 |
| `docs/` | 模块文档索引（[入口](docs/README.md)） |

## 环境要求

- Windows 10/11（实测），Python **3.13**（3.12+ 应可）
- 机械臂控制盒（自带 HTTP 服务，接口见 `jiekou/接口说明.txt`）
- 手机支架 + 摄像头（用于透视取图）
- 可访问外网（DeepSeek / OpenRouter API）

## 快速开始

### 1. 安装依赖

```bash
git clone <仓库地址>
cd jixiebijiaoben
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> 首次运行 OCR 时 PaddleOCR 会自动下载模型，第一次识别耗时约 10 秒属正常冷启动。

### 2. 创建配置文件

```bash
copy config.json.example config.json
```

然后编辑 `config.json`，关键字段：

| 字段 | 说明 |
| --- | --- |
| `devices[].ip` | 机械臂控制盒的 IP（如 `192.168.1.108`） |
| `devices[].license_key` | 该设备的卡密（授权服务签发） |
| `devices[].selected_app` | 运行的 App，当前为 `xinyu` |
| `devices[].app_overrides.xinyu.persona` | 人设配置（`requestPersona` 为 Coze 人设应用 ID） |
| `global.license_api_url` | 卡密校验服务地址（内部部署） |

### 3. 配置 API Key（重点）

所有密钥都放在项目根目录的 **`.env/` 文件夹**中，该目录已被 `.gitignore` 排除，**永远不会被提交**。克隆后需要手工创建：

| 文件（`.env/` 下） | 内容格式 | 用途 | 去哪申请 | 必填？ |
| --- | --- | --- | --- | --- |
| `deepseek_api_key.txt` | 一行裸 key | 人设 AI 对话回复（DeepSeek 官方直连） | platform.deepseek.com | **跑人设必填** |
| `vision_dialogue_keys.json` | JSON（见下） | 视觉对话识别：截图 → 消息与 sender 判定（走 OpenRouter） | openrouter.ai | **视觉识别必填** |
| `autoglm.txt` 或 `zhipu.txt` | 一行裸 key | 智谱 AutoGLM 规划模型 | open.bigmodel.cn | 可选 |
| `gpt_planner.txt` | 一行裸 key | GPT Planner（OpenAI 兼容接口） | 按所用平台 | 可选 |
| `cos.txt` | `TENCENT_COS_SECRET_ID=...` 等多行 | 打包后上传腾讯云 COS | 腾讯云控制台 | 仅打包发布需要 |
| `lanzou.txt` | `LANZOU_USERNAME=...` 等多行 | 蓝奏云更新分发 | 蓝奏云账号 | 仅打包发布需要 |

各文件格式示例：

```
# .env/deepseek_api_key.txt —— 一行裸 key 即可
sk-xxxxxxxxxxxxxxxxxxxxxxxx

# .env/vision_dialogue_keys.json
{
  "openrouter_key": "sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxx"
}

# .env/cos.txt（KEY=value 形式）
TENCENT_COS_BUCKET=xxx
TENCENT_COS_SECRET_ID=xxx
TENCENT_COS_SECRET_KEY=xxx
TENCENT_COS_SESSION_TOKEN=xxx

# .env/lanzou.txt
LANZOU_USERNAME=xxx
LANZOU_PASSWORD=xxx
LANZOU_UPDATE_FOLDER_ID=xxx
LANZOU_UPDATE_FOLDER_NAME=xxx
```

说明：

- 环境变量优先：设置 `DEEPSEEK_API_KEY`、`OPENROUTER_API_KEY`（或 COS/Lanzou 的 `TENCENT_COS_SECRET_ID` 等）会优先于 `.env/` 文件。
- 密钥只写 `.env/`，**绝不写入 `config.json`、代码或任何会提交到 git 的文件**。
- 卡密（license）填在 `config.json` 的 `devices[].license_key`，`config.json` 同样不入库。

### 4. 启动

```bash
python main.py
```

## 首次使用流程

1. UI 左侧添加设备（IP 与 `config.json` 一致），选择 App（xinyu）
2. 手机摆上支架、点亮并停在 App 消息列表页
3. 点击「重置视觉识别」清除旧坐标缓存
4. 点击「视觉 + OCR 识别一次」：弹出可视化窗口，成功后直接在截图上标出识别到的聊天区域，坐标缓存自动保存
5. 启动工作线程，之后每轮自动「条带 OCR → 选会话 → 读消息 → 人设回复 → 机械臂发送」

## 测试

```bash
python -m pytest tests/ -q
```

## 打包发布

```bash
python build.py
```

> ⚠️ 打包必须在官方工程目录（本仓库根目录）执行，禁止从任何 worktree / 副本目录打包。

## 更多文档

模块级详细文档见 [docs/README.md](docs/README.md)（入口与打包、UI、设备线程、App 层、视觉层、服务层、机械臂接口、配置体系、测试、基础服务调用手册）。

## 重要约束

- `jiekou/` 与 `lgb/` 为只读保护区，只允许读取与调用，不允许修改其中文件（详见 `AGENTS.md`）。
- `.env/`、`config.json`、`runtime/`（日志/照片/错误快照等运行数据）均不入库。
