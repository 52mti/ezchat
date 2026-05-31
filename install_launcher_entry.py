from __future__ import annotations

import sys

import install_runtime


def main() -> int:
    return int(install_runtime.main())


if __name__ == "__main__":
    raise SystemExit(main())
