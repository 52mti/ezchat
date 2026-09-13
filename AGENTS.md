# Workspace Rules

These rules apply to the entire workspace.

## Protected Directories

- `/jiekou`
- `/lgb`

## Required Behavior

- Never modify, create, delete, rename, or move any files under `/jiekou` or `/lgb`.
- These two directories are read-only for Codex.
- Codex may only read files from `/jiekou` and `/lgb`, or call/use the existing code and interfaces found there.
- Any new code, scripts, notes, or temporary files must be created outside `/jiekou` and `/lgb`.
- If a task would require changing `/jiekou` or `/lgb`, stop and ask the user first instead of making the change.

## Packaging Rules (打包规则)

- All packaging/build operations (打包) MUST be performed inside the official project directory: `D:\Program Files (x86)\appkaifa\jixiebijiaoben`.
- NEVER package from a Codex worktree, agent worktree, or any other copy/checkout of the project (禁止使用 Codex worktree 中的工程副本打包).
- Before running any build/packaging command, verify the current working directory is the official directory above. If it is not, stop and ask the user first.
- Reason: a package was once built from a Codex worktree copy and shipped to a customer, which broke the customer's environment. This must never happen again.

## 数据分层规范（Data Layering Rules）

权威口径见 `docs/11-数据分层规范.md`；`build.py`（打包收录）与 `update_apply.py`（更新保护）与其互为镜像。任何新增落盘数据必须按以下三层归位：

- **L1 项目数据**（git 跟踪、随包下发）：App 源码、模板 `apps/<app>/templates/`、配置三件套默认值 `apps/<app>/config.json`——只由开发端修改，客户层在读写两端剥离开发域键。
- **L2 客户设置**（更新永不覆盖、永不删除）：根 `config.json`、`.env/`、`runtime/configs/<app>/`、`runtime/greetings/`——UI 或人工写入。
- **L3 客户运行数据**（更新永不触碰、删了可重建）：一律进 `runtime/`——坐标缓存 `runtime/coordinate_cache/`（四维键 = 机器哈希/设备IP/App/透视签名，透视签名变化全清）、聊天记录与记忆 `runtime/chat_history/`、`runtime/memory/`、截图 `runtime/photo/`、日志与错误 `runtime/logs/`、`runtime/errors/`、`runtime/error_incidents/`、TTS 缓存 `runtime/tts/`、临时文件 `runtime/tmp/`（lgb 取图必须显式传 `save_path=runtime/tmp/…`）。

硬性约束：

1. 除根 `config.json` 与 `.env` 外，**禁止写仓库根目录**。
2. 新增落盘路径前先过 docs/11 §1 的"判定四问"（要不要随包发/算不算客户资产/删了能否重建）。
3. `engine/`、`models/`、`runtime/`、`tts/piper/` 为更新器不可触碰目录；`jiekou/`、`lgb/` 只读（见上）。
4. 老布局数据（根 `photo/`、根 `outbox.json`）由程序自动迁入 `runtime/`，不要手工搬动。

