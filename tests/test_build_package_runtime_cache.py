from __future__ import annotations

import zipfile

import build


def test_write_customer_update_archive_excludes_paddlex_cache(tmp_path):
    package_root = tmp_path / build.PACKAGE_NAME
    dist_root = tmp_path / "dist"
    model_dir = package_root / "runtime" / ".paddlex" / "official_models" / "PP-OCRv5_mobile_det"
    model_dir.mkdir(parents=True)
    (model_dir / "inference.json").write_text("{}", encoding="utf-8")
    (package_root / "main.py").write_text("print('ok')\n", encoding="utf-8")

    archive_path = build._write_customer_update_archive(package_root, dist_root)

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())

    assert f"{build.PACKAGE_NAME}/main.py" in names
    assert all("runtime/.paddlex/" not in name for name in names)
