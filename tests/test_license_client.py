from __future__ import annotations

from datetime import datetime, timedelta

from services.license_client import LicenseClient


def test_license_client_not_configured_is_safe_failure():
    result = LicenseClient().verify("1.1.1.1", "key")

    assert result.ok is False
    assert result.status == "not_configured"


class _Session:
    def __init__(self) -> None:
        self.payload = None

    def post(self, url, json, timeout):
        self.payload = {"url": url, "json": json, "timeout": timeout}

        class _Response:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {"ok": True, "status": "ok", "message": "license verified", "expires_at": "2099-01-01T00:00:00+00:00"}

        return _Response()


def test_license_verify_sends_machine_id():
    session = _Session()
    client = LicenseClient(
        api_url="https://license.example.com/api/verify",
        project_secret="secret",
        session=session,
    )

    result = client.verify("10.0.0.8", "JY-KEY", machine_id="aabbccddeeff")

    assert result.ok is True
    assert session.payload["url"] == "https://license.example.com/api/verify"
    assert session.payload["json"]["machine_id"] == "aabbccddeeff"
    assert session.payload["json"]["device_ip"] == "10.0.0.8"
    assert session.payload["json"]["license_key"] == "JY-KEY"
    assert session.payload["json"]["project_secret"] == "secret"
    assert "project_uuid" not in session.payload["json"]


def test_license_offline_allows_unexpired_cached_license():
    future = (datetime.now() + timedelta(days=1)).isoformat(timespec="seconds")
    client = LicenseClient(api_url="http://127.0.0.1:1")

    result = client.verify_with_offline("1.1.1.1", "key", future)

    assert result.ok is True
    assert result.status == "offline"
