from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.models import LicenseKey


def _customer(client: TestClient, auth: dict[str, str], name: str = "甲客户") -> dict:
    response = client.post("/api/admin/customers", headers=auth, json={"name": name, "contact": "13800000000"})
    assert response.status_code == 200, response.text
    return response.json()


def _issue(client: TestClient, auth: dict[str, str], customer_id: int) -> dict:
    response = client.post("/api/admin/keys", headers=auth, json={"customer_id": customer_id})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["code"].startswith("JY-")
    assert body["status"] == "issued"
    return body


def test_login_refresh_and_reject_missing_token(client: TestClient, auth: dict[str, str]):
    me = client.get("/api/admin/me", headers=auth)
    assert me.status_code == 200
    assert me.json()["username"] == "admin"
    assert client.get("/api/admin/stats").status_code == 401
    assert client.post("/api/admin/login", json={"username": "admin", "password": "nope"}).status_code == 401

    refreshed = client.post("/api/admin/refresh")
    assert refreshed.status_code == 200
    assert refreshed.json()["access_token"]
    client.post("/api/admin/logout")
    assert client.post("/api/admin/refresh").status_code == 401


def test_renew_keeps_remaining_days_and_restarts_after_expiry(client: TestClient, auth: dict[str, str], app):
    customer = _customer(client, auth)
    issued = _issue(client, auth, customer["id"])
    session = app.state.session_factory()
    key = session.get(LicenseKey, issued["id"])
    key.expires_at = datetime.now(timezone.utc) + timedelta(days=10)
    session.commit()
    session.close()

    renewed = client.post(f"/api/admin/keys/{issued['id']}/renew", headers=auth, json={"note": "本月"})
    assert renewed.status_code == 200, renewed.text
    new_expires = datetime.fromisoformat(renewed.json()["expires_at"])
    assert datetime.now(timezone.utc) + timedelta(days=39) < new_expires < datetime.now(timezone.utc) + timedelta(days=41)

    session = app.state.session_factory()
    key = session.get(LicenseKey, issued["id"])
    key.expires_at = datetime.now(timezone.utc) - timedelta(hours=2)
    session.commit()
    session.close()
    again = client.post(f"/api/admin/keys/{issued['id']}/renew", headers=auth, json={})
    restarted = datetime.fromisoformat(again.json()["expires_at"])
    assert datetime.now(timezone.utc) + timedelta(days=29) < restarted < datetime.now(timezone.utc) + timedelta(days=31)
    detail = client.get(f"/api/admin/keys/{issued['id']}", headers=auth).json()
    assert len(detail["renewals"]) == 2


def test_batch_renew_revoke_unbind_and_audit(client: TestClient, auth: dict[str, str]):
    first = _customer(client, auth, "客户一")
    second = _customer(client, auth, "客户二")
    key_a = _issue(client, auth, first["id"])
    key_b = _issue(client, auth, second["id"])
    batch = client.post("/api/admin/keys/renew", headers=auth, json={"ids": [key_a["id"], key_b["id"]]})
    assert batch.status_code == 200, batch.text
    assert [item["ok"] for item in batch.json()["results"]] == [True, True]

    bound = client.post(
        "/api/verify",
        json={
            "project_secret": "project-secret",
            "device_ip": "10.0.0.2",
            "license_key": key_a["code"],
            "machine_id": "001122334455",
        },
    )
    assert bound.json()["ok"] is True
    unbound = client.post(f"/api/admin/keys/{key_a['id']}/unbind", headers=auth)
    assert unbound.json()["status"] == "issued"
    rebound = client.post(
        "/api/verify",
        json={
            "project_secret": "project-secret",
            "device_ip": "10.0.0.3",
            "license_key": key_a["code"],
            "machine_id": "66778899aabb",
        },
    )
    assert rebound.json()["ok"] is True

    revoked = client.post(f"/api/admin/keys/{key_b['id']}/revoke", headers=auth)
    assert revoked.json()["status"] == "revoked"
    assert client.post(f"/api/admin/keys/{key_b['id']}/renew", headers=auth, json={}).status_code == 400
    rejected = client.post(
        "/api/verify",
        json={
            "project_secret": "project-secret",
            "device_ip": "10.0.0.4",
            "license_key": key_b["code"],
            "machine_id": "ccddeeff0011",
        },
    )
    assert rejected.json()["status"] == "revoked"

    audit = client.get("/api/admin/audit", headers=auth)
    actions = {item["action"] for item in audit.json()["items"]}
    assert {"customer.create", "key.issue", "key.renew", "key.unbind", "key.revoke"} <= actions

    updated = client.put(
        f"/api/admin/customers/{first['id']}",
        headers=auth,
        json={"name": "客户一改", "contact": "", "note": "备注"},
    )
    assert updated.json()["name"] == "客户一改"
    stats = client.get("/api/admin/stats", headers=auth).json()
    assert stats["customers"] == 2
    assert stats["active"] == 1
    assert stats["unbound"] == 0
