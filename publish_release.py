"""把最近一次本地构建的产物发布到 COS，不重新构建。

适用场景：build.py --skip-release-upload 验证通过后正式发布。
直接复用 dist/release.json 里记录的版本与文件名，避免重跑 build
导致 app_version 递增、包缓存失效、整套引擎重装。

用法：
    python publish_release.py               # 发布 dist 里最近一次构建
    python publish_release.py --no-force-full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import build


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish the latest local build to COS without rebuilding.")
    parser.add_argument(
        "--no-force-full",
        action="store_true",
        help="do not set requires_full_package (default sets it for engine-layout transitions)",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent
    dist = root / "dist"
    release_path = dist / build.RELEASE_FILE_NAME
    if not release_path.is_file():
        raise SystemExit("[error] dist/release.json not found; run build.py first")
    payload = json.loads(release_path.read_text(encoding="utf-8"))
    build._apply_local_env_files(root)

    app_version = str(payload.get("app_version") or "").strip()
    runtime_version = str(payload.get("runtime_version") or "").strip()
    model_version = str(payload.get("model_version") or "").strip()
    update_file_name = str(payload.get("update_file_name") or "").strip()
    if not (app_version and runtime_version and model_version and update_file_name):
        raise SystemExit(f"[error] incomplete release.json: {payload}")

    update_zip = dist / update_file_name
    if not update_zip.is_file():
        raise SystemExit(f"[error] update zip missing: {update_zip}")

    requires_full = not args.no_force_full
    print(f"[publish] app_version={app_version} runtime={runtime_version} model={model_version}")
    print(f"[publish] update={update_file_name} requires_full_package={requires_full}")

    release_manifest_url = build._publish_release_manifest(
        app_version=app_version,
        runtime_version=runtime_version,
        model_version=model_version,
        update_file_name=update_file_name,
        requires_full_package=requires_full,
        update_zip=update_zip,
    )
    print(f"[done] release(uploaded)={release_manifest_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
