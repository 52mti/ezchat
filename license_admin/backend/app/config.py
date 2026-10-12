from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    project_secret: str = ""
    jwt_secret: str
    admin_username: str = ""
    admin_password: str = ""
    cookie_secure: bool = True
    access_token_minutes: int = 30
    refresh_token_days: int = 14
    verify_rate_limit: int = 60
    verify_rate_window_seconds: int = 60
    bcrypt_rounds: int = 12

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
