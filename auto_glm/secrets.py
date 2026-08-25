"""Local secret loading for the AutoGLM integration.

Secrets remain in the workspace ``.env`` directory and are never logged.
"""

from __future__ import annotations

from pathlib import Path


def read_local_secret(workspace_root: Path, *names: str) -> str:
    env_dir = workspace_root / ".env"
    for name in names:
        path = env_dir / name
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            value = line.strip().lstrip("\ufeff")
            if not value or value.startswith("#"):
                continue
            if "=" in value:
                value = value.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                return value
    return ""
