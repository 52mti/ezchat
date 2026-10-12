from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests


@dataclass(frozen=True)
class LicenseResult:
    ok: bool
    status: str
    message: str
    expires_at: str = ""
    raw: dict[str, Any] | None = None


class LicenseClient:
    def __init__(
        self,
        api_url: str = "",
        project_secret: str = "",
        timeout: float = 10.0,
        session: requests.Session | None = None,
    ) -> None:
        self.api_url = api_url
        self.project_secret = project_secret
        self.timeout = timeout
        self.session = session or requests.Session()

    def verify(self, device_ip: str, license_key: str, machine_id: str = "") -> LicenseResult:
        if not self.api_url:
            return LicenseResult(False, "not_configured", "license_api_url is not configured")
        if not license_key:
            return LicenseResult(False, "missing_key", "license_key is empty")

        payload = {
            "project_secret": self.project_secret,
            "device_ip": device_ip,
            "license_key": license_key,
            "machine_id": machine_id,
        }
        try:
            response = self.session.post(self.api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            return LicenseResult(False, "request_failed", str(exc))

        ok = bool(data.get("ok", data.get("success", False)))
        status = "ok" if ok else str(data.get("status", "invalid"))
        message = str(data.get("message", "license verified" if ok else "license rejected"))
        expires_at = str(data.get("expires_at", data.get("expire_time", "")))
        return LicenseResult(ok, status, message, expires_at, data)

    def verify_with_offline(
        self,
        device_ip: str,
        license_key: str,
        cached_expires_at: str = "",
        machine_id: str = "",
    ) -> LicenseResult:
        result = self.verify(device_ip, license_key, machine_id=machine_id)
        if result.ok:
            return result
        if result.status == "request_failed" and _is_future_time(cached_expires_at):
            return LicenseResult(
                True,
                "offline",
                "license server unavailable; using cached unexpired license",
                cached_expires_at,
                result.raw,
            )
        return result


def _is_future_time(value: str) -> bool:
    if not value:
        return False
    candidates = [value, value.replace("Z", "+00:00")]
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if parsed.tzinfo is not None:
            return parsed > datetime.now(parsed.tzinfo)
        return parsed > datetime.now()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt) > datetime.now()
        except ValueError:
            continue
    return False
