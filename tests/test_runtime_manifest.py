from __future__ import annotations

from vision.ocr_runtime import build_runtime_package_specs, load_runtime_manifest


def test_runtime_manifest_exposes_shared_ocr_packages():
    manifest = load_runtime_manifest()
    shared_specs = [item.spec for item in manifest.shared_packages]

    assert "paddleocr==3.4.0" in shared_specs
    assert "paddlex[ocr-core]==3.4.3" in shared_specs


def test_runtime_package_plan_merges_shared_and_backend_packages():
    manifest = load_runtime_manifest()
    plan = build_runtime_package_specs(manifest, "cpu")
    specs = [item.spec for item in plan]

    assert specs[0] == "paddleocr==3.4.0"
    assert "paddlepaddle==3.2.2" in specs
