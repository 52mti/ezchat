"""准备打包用的引导资源：嵌入式 Python + get-pip.py。

运行一次后，仓库根会生成：

    _bootstrap/
    ├─ python/        # 解压好的嵌入式 Python（已开启 import site）
    └─ get-pip.py     # 官方 get-pip.py（自带离线 pip）

之后直接 `python build.py` 即可（build 默认从 `_bootstrap/` 取这两样）。

仅构建机使用，需联网。也可用 `--embed-zip` 指向本地已下载的嵌入式 zip 离线准备。
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_PY_VERSION = "3.13.1"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"


def _bootstrap_dir(root: Path) -> Path:
    return root / "_bootstrap"


def _embed_url(version: str) -> str:
    return f"https://www.python.org/ftp/python/{version}/python-{version}-embed-amd64.zip"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    with urllib.request.urlopen(url) as response, dest.open("wb") as handle:  # noqa: S310
        shutil.copyfileobj(response, handle)


def _enable_site(python_dir: Path) -> None:
    """取消嵌入式 `pythonXX._pth` 里 `import site` 的注释，否则 venv/pip 不可用。"""
    matched = False
    for pth in python_dir.glob("python*._pth"):
        matched = True
        lines = pth.read_text(encoding="utf-8").splitlines()
        changed = False
        for index, line in enumerate(lines):
            if line.strip() in {"#import site", "# import site"}:
                lines[index] = "import site"
                changed = True
        if changed:
            pth.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"[patch] 已开启 import site：{pth.name}")
    if not matched:
        print("[warn] 未找到 pythonXX._pth，无法确认 import site 是否开启")


def prepare_embed_python(root: Path, version: str, embed_zip: Path | None) -> Path:
    target = _bootstrap_dir(root) / "python"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    if embed_zip is not None:
        zip_path = embed_zip
        if not zip_path.is_file():
            raise FileNotFoundError(f"嵌入式 zip 不存在：{zip_path}")
    else:
        zip_path = _bootstrap_dir(root) / f"python-{version}-embed-amd64.zip"
        _download(_embed_url(version), zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target)
    print(f"[ok] 嵌入式 Python 解压到 {target}")

    _enable_site(target)
    if embed_zip is None:
        zip_path.unlink(missing_ok=True)
    return target


def prepare_get_pip(root: Path) -> Path:
    dest = _bootstrap_dir(root) / "get-pip.py"
    _download(GET_PIP_URL, dest)
    print(f"[ok] get-pip.py -> {dest}")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare bootstrap python + get-pip.py for packaging.")
    parser.add_argument("--python-version", default=DEFAULT_PY_VERSION,
                        help=f"嵌入式 Python 版本（默认 {DEFAULT_PY_VERSION}）")
    parser.add_argument("--embed-zip", type=Path, default=None,
                        help="本地已下载的 python-X.Y.Z-embed-amd64.zip（离线准备用）")
    parser.add_argument("--skip-get-pip", action="store_true", help="跳过下载 get-pip.py")
    args = parser.parse_args(argv)

    root = Path.cwd()
    prepare_embed_python(root, args.python_version, args.embed_zip)
    if not args.skip_get_pip:
        prepare_get_pip(root)

    print("\n引导资源已就绪，可直接运行：python build.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
