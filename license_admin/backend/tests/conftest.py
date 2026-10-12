from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import Settings
from app.main import create_app


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    database = (tmp_path / "license.sqlite").as_posix()
    return Settings(
        database_url=f"sqlite+pysqlite:///{database}",
        project_secret="project-secret",
        jwt_secret="test-jwt-secret-value-32-bytes-ok",
        admin_username="admin",
        admin_password="admin-pass-123",
        cookie_secure=False,
        verify_rate_limit=30,
        verify_rate_window_seconds=60,
        bcrypt_rounds=4,
    )


@pytest.fixture()
def app(settings: Settings):
    return create_app(settings)


@pytest.fixture()
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def auth(client: TestClient) -> dict[str, str]:
    response = client.post("/api/admin/login", json={"username": "admin", "password": "admin-pass-123"})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}
