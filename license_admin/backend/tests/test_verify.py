from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.models import LicenseKey


VERIFY = {
    "project_secret": "project-secret",
    "device_ip": "10.0.0.8",
    "machine_id": "aabbccddeeff",
}


def _issue(client: TestClient, auth: dict[str, str], name: str = "甲客户") -> dict:
    customer = client.post("/api/admin/customers", headers=auth, json={"name": name})
    assert customer.status_code == 200, customer.text
    issued = client.post("/api/admin/keys", headers=auth, json={"customer_id": customer.json()["id"]})
    assert issued.status_code == 200, issued.text
    return issued.json()


def _verify(client: TestClient, code: str, **overrides):
    payload = {**VERIFY, "license_key": code, **overrides}
    return client.post("/api/verify", json=payload)


def test_verify_success_shape_and_same_machine(client: TestClient, auth: dict[str, str]):
    issued = _issue(client, auth)
    first = _verify(client, issued["code"], device_ip="10.0.0.8")
    assert first.status_code == 200
    body = first.json()
    assert set(body) == {"ok", "status", "message", "expires_at"}
    assert body["ok"] is True
    assert body["status"] == "ok"
    assert body["message"] == "license verified"
    datetime.fromisoformat(body["expires_at"])

    second = _verify(client, issued["code"], device_ip="10.0.0.9")
    assert second.json()["ok"] is True
    detail = client.get(f"/api/admin/keys/{issued['id']}", headers=auth)
    assert detail.json()["bind_mode"] == "machine"
    assert detail.json()["machine_id"] == "aabbccddeeff"
    assert detail.json()["last_device_ip"] == "10.0.0.9"
    assert detail.json()["status"] == "active"


def test_verify_rejects_other_machine(client: TestClient, auth: dict[str, str]):
    issued = _issue(client, auth)
    assert _verify(client, issued["code"]).json()["ok"] is True
    rejected = _verify(client, issued["code"], machine_id="112233445566")
    body = rejected.json()
    assert body["ok"] is False
    assert body["status"] == "device_mismatch"


def test_verify_ip_binding_upgrades_when_same_address_sends_machine_id(client: TestClient, auth: dict[str, str]):
    issued = _issue(client, auth)
    first = _verify(client, issued["code"], machine_id="", device_ip="10.1.1.1")
    assert first.json()["ok"] is True
    detail = client.get(f"/api/admin/keys/{issued['id']}", headers=auth).json()
    assert detail["bind_mode"] == "ip"

    upgraded = _verify(client, issued["code"], machine_id="aabbccddeeff", device_ip="10.1.1.1")
    assert upgraded.json()["ok"] is True
    detail = client.get(f"/api/admin/keys/{issued['id']}", headers=auth).json()
    assert detail["bind_mode"] == "machine"
    moved = _verify(client, issued["code"], machine_id="aabbccddeeff", device_ip="10.9.9.9")
    assert moved.json()["ok"] is True


def test_verify_rejects_missing_expired_revoked_and_bad_secret(client: TestClient, auth: dict[str, str], app):
    missing = _verify(client, "JY-NOPE-NOPE-NOPE")
    assert missing.json()["status"] == "invalid"

    issued = _issue(client, auth)
    bad_secret = _verify(client, issued["code"], project_secret="wrong")
    assert bad_secret.status_code == 200
    assert bad_secret.json()["status"] == "unauthorized"

    session = app.state.session_factory()
    key = session.get(LicenseKey, issued["id"])
    key.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
    session.commit()
    session.close()
    expired = _verify(client, issued["code"])
    assert expired.json()["status"] == "expired"

    session = app.state.session_factory()
    key = session.get(LicenseKey, issued["id"])
    key.expires_at = datetime.now(timezone.utc) + timedelta(days=5)
    key.revoked = True
    session.commit()
    session.close()
    revoked = _verify(client, issued["code"])
    assert revoked.json()["status"] == "revoked"


def test_verify_rate_limit(tmp_path):
    from app.config import Settings
    from app.main import create_app

    database = (tmp_path / "limited.sqlite").as_posix()
    settings = Settings(
        database_url=f"sqlite+pysqlite:///{database}",
        project_secret="project-secret",
        jwt_secret="test-jwt-secret-value-32-bytes-ok",
        admin_username="admin",
        admin_password="admin-pass-123",
        cookie_secure=False,
        verify_rate_limit=2,
        verify_rate_window_seconds=60,
        bcrypt_rounds=4,
    )
    with TestClient(create_app(settings)) as client:
        payload = {**VERIFY, "license_key": "JY-NOPE-NOPE-NOPE"}
        assert client.post("/api/verify", json=payload).status_code == 200
        assert client.post("/api/verify", json=payload).status_code == 200
        blocked = client.post("/api/verify", json=payload)
        assert blocked.status_code == 429
        assert blocked.json()["status"] == "rate_limited"
