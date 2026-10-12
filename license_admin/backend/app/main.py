from __future__ import annotations

from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.licensing import utcnow
from app.models import AdminUser, Base
from app.rate_limit import RateLimiter
from app.routers.admin import router as admin_router
from app.routers.verify import router as verify_router
from app.security import hash_password


def make_engine(database_url: str):
    if database_url.startswith("sqlite"):
        return create_engine(database_url, connect_args={"check_same_thread": False})
    return create_engine(database_url, pool_pre_ping=True)


def bootstrap_admin(session_factory, settings: Settings) -> None:
    session = session_factory()
    try:
        if session.query(AdminUser).count():
            return
        username = settings.admin_username.strip()
        password = settings.admin_password
        if not username and not password:
            return
        if not username or len(password) < 8:
            raise RuntimeError("ADMIN_USERNAME 不能为空，且 ADMIN_PASSWORD 至少 8 位")
        session.add(
            AdminUser(
                username=username,
                password_hash=hash_password(password, settings.bcrypt_rounds),
                created_at=utcnow(),
            )
        )
        session.commit()
    finally:
        session.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings()
    engine = make_engine(resolved.database_url)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    bootstrap_admin(session_factory, resolved)

    app = FastAPI(title="卡密管理")
    app.state.settings = resolved
    app.state.session_factory = session_factory
    app.state.rate_limiter = RateLimiter(resolved.verify_rate_limit, resolved.verify_rate_window_seconds)
    app.include_router(verify_router)
    app.include_router(admin_router)

    @app.get("/api/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    return app
