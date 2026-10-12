from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.licensing import verify_license
from app.rate_limit import RateLimiter
from app.schemas import VerifyBody
from app.security import same_secret


logger = logging.getLogger("license_admin.verify")
router = APIRouter()


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    parts = [part.strip() for part in forwarded.split(",") if part.strip()]
    if parts:
        return parts[-1]
    if request.client is None:
        return "unknown"
    return request.client.host


def _reject(status: str, message: str, expires_at: str = "", http_status: int = 200) -> JSONResponse:
    return JSONResponse(
        status_code=http_status,
        content={"ok": False, "status": status, "message": message, "expires_at": expires_at},
    )


@router.post("/api/verify")
def verify(body: VerifyBody, request: Request) -> JSONResponse:
    settings = request.app.state.settings
    limiter: RateLimiter = request.app.state.rate_limiter
    if not limiter.allow(client_ip(request)):
        return _reject("rate_limited", "请求过于频繁", http_status=429)

    configured = bool(settings.project_secret)
    secret_ok = same_secret(body.project_secret, settings.project_secret)
    if not configured or not secret_ok:
        logger.info("license verify status=unauthorized")
        return _reject("unauthorized", "授权校验未通过")

    session: Session = request.app.state.session_factory()
    try:
        outcome = verify_license(
            session,
            code=body.license_key,
            device_ip=body.device_ip,
            machine_id=body.machine_id,
        )
        if outcome.persist:
            session.commit()
        else:
            session.rollback()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    logger.info("license verify status=%s", outcome.status)
    return JSONResponse(
        content={
            "ok": outcome.ok,
            "status": outcome.status,
            "message": outcome.message,
            "expires_at": outcome.expires_at,
        }
    )
