"""生成客户手动更新工具包。

历史上手拼的 ps1 出现过转义损坏（'\\\\' 进入 PowerShell 导致相对路径
首字母被吃），本工具是唯一的 kit 生成入口：
- apply_update.ps1 / 手动更新.bat 一律来自 update_orchestration 模板；
- 禁止手工编辑 kit 里的脚本，改模板必须改 update_orchestration.py。

用法（在官方工程目录内执行）：
    python make_manual_update_kit.py                      # 打包最新的更新包
    python make_manual_update_kit.py --zip dist/xxx.zip   # 指定更新/全量包
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from update_orchestration import write_manual_update_bat, write_orchestration_script


USAGE_TEXT = """手动更新说明
================

一、程序还能正常打开时（推荐）
  1. 关闭正在运行的机械臂脚本程序。
  2. 把本文件夹里的所有文件，复制到安装目录（jixiebijiaoben_launcher.exe
     所在的目录；如果你解压后看到本文件夹旁边还有一层 jixiebijiaoben
     文件夹，也可以直接把本文件夹放到那一层旁边，脚本会自动识别）。
  3. 双击 手动更新.bat，等待提示 Update finished。
  4. 重新打开机械臂脚本。

二、程序已经打不开 / 提示 OCR runtime not ready 时
  1. 关闭所有相关程序。
  2. 把压缩包里的 jixiebijiaoben_full_*.zip 直接解压到安装目录，
     选择"解压并替换"。
  3. 重新打开机械臂脚本。

注意：
- 不要单独解压 update_*.zip 到安装目录，必须通过 手动更新.bat 执行。
- 更新失败时，把安装目录 runtime\\errors\\update_apply_error.txt 发回来。
"""


def _latest_zip(dist_root: Path, pattern: str) -> Path:
    candidates = sorted(dist_root.glob(pattern))
    if not candidates:
        raise SystemExit(f"[error] no {pattern} found under {dist_root}")
    return candidates[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the customer manual update kit.")
    parser.add_argument("--zip", default="", help="update/full zip to ship; default: latest update zip in dist/")
    parser.add_argument("--name", default="", help="output kit name (without .zip)")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent
    dist_root = root / "dist"
    if args.zip:
        payload = (root / args.zip).resolve() if not Path(args.zip).isabsolute() else Path(args.zip)
    else:
        payload = _latest_zip(dist_root, "jixiebijiaoben_update_*.zip")
    if not payload.is_file():
        raise SystemExit(f"[error] payload zip not found: {payload}")

    is_full = "_full_" in payload.name
    default_name = "修复工具" if is_full else "手动更新工具"
    kit_name = args.name or default_name
    stage = dist_root / f"_kit_stage_{kit_name}"
    if stage.exists():
        import shutil

        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    # 模板生成 + 自校验（转义损坏会直接抛错，绝不会产出坏脚本）。
    write_orchestration_script(stage)
    write_manual_update_bat(stage)
    (stage / "使用说明.txt").write_text(USAGE_TEXT, encoding="utf-8-sig", newline="\r\n")

    timestamp = payload.stem.split("_")[-1]
    output = dist_root / f"{kit_name}_{timestamp}.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as kit:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                kit.write(path, path.name)
        kit.write(payload, payload.name)
    import shutil

    shutil.rmtree(stage)
    print(f"[done] kit={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
