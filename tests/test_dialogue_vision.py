"""用一张本地截图测试欢友视觉对话识别。

示例：
    python test_dialogue_vision.py "D:\\screens\\chat.jpg"
    python test_dialogue_vision.py "D:\\screens\\chat.jpg" --output result.json
    python test_dialogue_vision.py "D:\\screens\\chat.jpg" --mode normal

默认 mode=ensemble：直接分开请求三个 OpenRouter 分析模型，再请求汇总模型。
密钥只从项目 .env 或系统环境变量读取，不会写入结果文件。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from services.vision_dialogue_client import VisionDialogueClient, VisionDialogueError


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _load_config(root: Path, config_path: str | None) -> dict:
    path = Path(config_path) if config_path else root / "apps" / "huanyou" / "dialogue_llm_config.json"
    if not path.is_absolute():
        path = root / path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise VisionDialogueError(f"无法读取配置文件：{path}") from exc
    except json.JSONDecodeError as exc:
        raise VisionDialogueError(f"配置文件不是有效 JSON：{path}") from exc
    if not isinstance(payload, dict):
        raise VisionDialogueError(f"配置文件根节点必须是对象：{path}")
    return payload


def _run_test(root: Path, image_path: Path, mode: str, config_path: str | None) -> dict:
    config = _load_config(root, config_path)
    image = Image.open(image_path)
    image.load()
    client = VisionDialogueClient(workspace_root=root)
    started = time.perf_counter()
    if mode == "ensemble":
        prompt_path = config.get("prompt_path") or "openrouter/dialogue_vision_prompt.txt"
        reply = client._request_openrouter_ensemble(config, client._prompt_text(prompt_path), client._image_to_data_uri(image))
    else:
        reply = client.parse_dialogue(image, {"dialogue_llm": config})
    output = root / "runtime" / "vision_test_results" / f"{image_path.stem}_result.json"
    result = {"image": str(image_path), "mode": mode, "provider": reply.provider, "model": reply.model,
              "elapsed_seconds": round(time.perf_counter() - started, 3), "result": reply.payload}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"result": result, "output": output}


def _launch_gui(root: Path) -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    window = tk.Tk()
    window.title("欢友视觉对话识别测试")
    window.geometry("860x620")
    window.minsize(680, 460)
    path_var = tk.StringVar()
    mode_var = tk.StringVar(value="ensemble")
    status_var = tk.StringVar(value="请选择一张截图")

    ttk.Label(window, text="截图路径").pack(anchor="w", padx=14, pady=(14, 4))
    path_row = ttk.Frame(window)
    path_row.pack(fill="x", padx=14)
    ttk.Entry(path_row, textvariable=path_var).pack(side="left", fill="x", expand=True)

    def choose_file():
        selected = filedialog.askopenfilename(title="选择待识别截图", filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.webp"), ("所有文件", "*.*")])
        if selected:
            path_var.set(selected)
            status_var.set("已选择图片，可以开始识别")

    ttk.Button(path_row, text="选择照片", command=choose_file).pack(side="left", padx=(8, 0))
    options = ttk.Frame(window)
    options.pack(fill="x", padx=14, pady=10)
    ttk.Label(options, text="请求模式").pack(side="left")
    ttk.Combobox(options, textvariable=mode_var, values=("ensemble", "normal"), state="readonly", width=14).pack(side="left", padx=8)
    output_box = tk.Text(window, wrap="word", state="disabled", font=("Consolas", 10))
    output_box.pack(fill="both", expand=True, padx=14, pady=(0, 8))

    def write_output(value: str):
        output_box.configure(state="normal")
        output_box.delete("1.0", "end")
        output_box.insert("end", value)
        output_box.configure(state="disabled")

    def start():
        image_path = Path(path_var.get().strip()).expanduser()
        if not image_path.is_file():
            messagebox.showerror("路径错误", "请选择存在的图片文件")
            return
        status_var.set("识别中，请等待模型返回...")
        start_button.configure(state="disabled")

        def worker():
            try:
                result = _run_test(root, image_path.resolve(), mode_var.get(), None)
                message = f"识别完成\n结果文件：{result['output']}\n\n{json.dumps(result['result'], ensure_ascii=False, indent=2)}"
                window.after(0, lambda: (write_output(message), status_var.set("识别完成")))
            except Exception as exc:
                window.after(0, lambda: (write_output(f"识别失败：{type(exc).__name__}: {exc}"), status_var.set("识别失败")))
            finally:
                window.after(0, lambda: start_button.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    start_button = ttk.Button(options, text="开始识别", command=start)
    start_button.pack(side="left", padx=8)
    ttk.Label(window, textvariable=status_var).pack(anchor="w", padx=14, pady=(0, 12))
    window.mainloop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="测试欢友截图的视觉对话识别")
    parser.add_argument("image", nargs="?", help="待识别的本地图片路径；不填写则打开选择窗口")
    parser.add_argument("--mode", choices=("ensemble", "normal"), default="ensemble", help="ensemble=OpenRouter分开请求；normal=线上完整流程")
    parser.add_argument("--config", help="自定义 dialogue_llm_config.json 路径")
    parser.add_argument("--output", help="保存最终 JSON 的路径；默认保存到 runtime/vision_test_results/")
    args = parser.parse_args()

    root = _project_root()
    if not args.image:
        return _launch_gui(root)
    image_path = Path(args.image).expanduser()
    if not image_path.is_absolute():
        image_path = Path.cwd() / image_path
    image_path = image_path.resolve()
    if not image_path.is_file():
        print(f"错误：图片不存在：{image_path}", file=sys.stderr)
        return 2

    try:
        config = _load_config(root, args.config)
        image = Image.open(image_path)
        image.load()
        client = VisionDialogueClient(workspace_root=root)
        started = time.perf_counter()
        if args.mode == "ensemble":
            prompt_path = config.get("prompt_path") or "openrouter/dialogue_vision_prompt.txt"
            prompt = client._prompt_text(prompt_path)
            data_uri = client._image_to_data_uri(image)
            reply = client._request_openrouter_ensemble(config, prompt, data_uri)
        else:
            reply = client.parse_dialogue(image, {"dialogue_llm": config})
        elapsed = time.perf_counter() - started
    except (UnidentifiedImageError, OSError) as exc:
        print(f"错误：无法打开图片：{image_path}\n{exc}", file=sys.stderr)
        return 2
    except VisionDialogueError as exc:
        print(f"识别失败：{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"识别失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    result = {
        "image": str(image_path),
        "mode": args.mode,
        "provider": reply.provider,
        "model": reply.model,
        "elapsed_seconds": round(elapsed, 3),
        "result": reply.payload,
    }
    output = Path(args.output) if args.output else root / "runtime" / "vision_test_results" / f"{image_path.stem}_result.json"
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"识别完成：{elapsed:.2f} 秒")
    print(f"请求通道：{reply.provider}")
    print(f"汇总模型：{reply.model}")
    print(f"结果文件：{output}")
    print(json.dumps(reply.payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
