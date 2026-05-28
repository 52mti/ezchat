from __future__ import annotations

from datetime import datetime, timedelta

from services.license_client import LicenseClient


def test_license_client_not_configured_is_safe_failure():
    result = LicenseClient().verify("1.1.1.1", "key")

    assert result.ok is False
    assert result.status == "not_configured"


def test_license_offline_allows_unexpired_cached_license():
    future = (datetime.now() + timedelta(days=1)).isoformat(timespec="seconds")
    client = LicenseClient(api_url="http://127.0.0.1:1")

    result = client.verify_with_offline("1.1.1.1", "key", future)

    assert result.ok is True
    assert result.status == "offline"
