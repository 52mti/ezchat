from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt

from app.config import Settings


REFRESH_COOKIE = "license_refresh"


def same_secret(left: str, right: str) -> bool:
    return hmac.compare_digest(
        hashlib.sha256(left.encode("utf-8")).digest(),
        hashlib.sha256(right.encode("utf-8")).digest(),
    )


def hash_password(password: str, rounds: int = 12) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=rounds)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def _encode(payload: dict[str, Any], settings: Settings) -> str:
    token = jwt.encode(payload, settings.jwt_secret, algorithm="HS256")
    if isinstance(token, bytes):
        return token.decode("utf-8")
    return token


def create_access_token(user_id: int, username: str, settings: Settings) -> str:
    now = datetime.now(timezone.utc)
    return _encode(
        {
            "sub": str(user_id),
            "username": username,
            "typ": "access",
            "exp": now + timedelta(minutes=settings.access_token_minutes),
        },
        settings,
    )


def create_refresh_token(user_id: int, settings: Settings) -> str:
    now = datetime.now(timezone.utc)
    return _encode(
        {
            "sub": str(user_id),
            "typ": "refresh",
            "exp": now + timedelta(days=settings.refresh_token_days),
        },
        settings,
    )


def decode_token(token: str, settings: Settings, expected_type: str) -> dict[str, Any]:
    payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    if payload.get("typ") != expected_type:
        raise jwt.InvalidTokenError("token type mismatch")
    return payload
