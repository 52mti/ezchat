from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from app.models import AuditLog, Customer, LicenseKey, Renewal


RENEW_DAYS = 30
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


@dataclass(frozen=True)
class VerifyOutcome:
    ok: bool
    status: str
    message: str
    expires_at: str = ""
    persist: bool = False


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def iso(value: datetime | None) -> str:
    if value is None:
        return ""
    return ensure_utc(value).replace(microsecond=0).isoformat()


def status_of(key: LicenseKey, now: datetime | None = None) -> str:
    moment = now or utcnow()
    if key.revoked:
        return "revoked"
    if ensure_utc(key.expires_at) <= moment:
        return "expired"
    if key.machine_id or key.bound_device_ip:
        return "active"
    return "issued"


def bind_mode(key: LicenseKey) -> str:
    if key.machine_id:
        return "machine"
    if key.bound_device_ip:
        return "ip"
    return ""


def serialize_key(key: LicenseKey, now: datetime | None = None) -> dict:
    customer_name = key.customer.name if key.customer is not None else ""
    return {
        "id": key.id,
        "code": key.code,
        "customer_id": key.customer_id,
        "customer_name": customer_name,
        "status": status_of(key, now),
        "bind_mode": bind_mode(key),
        "issued_at": iso(key.issued_at),
        "activated_at": iso(key.activated_at),
        "expires_at": iso(key.expires_at),
        "machine_id": key.machine_id,
        "bound_device_ip": key.bound_device_ip,
        "last_device_ip": key.last_device_ip,
        "last_verified_at": iso(key.last_verified_at),
    }


def serialize_renewal(row: Renewal) -> dict:
    username = row.operator.username if row.operator is not None else ""
    return {
        "id": row.id,
        "old_expires_at": iso(row.old_expires_at),
        "new_expires_at": iso(row.new_expires_at),
        "note": row.note,
        "operator": username,
        "created_at": iso(row.created_at),
    }


def serialize_audit(row: AuditLog) -> dict:
    username = row.operator.username if row.operator is not None else ""
    return {
        "id": row.id,
        "action": row.action,
        "target_type": row.target_type,
        "target_id": row.target_id,
        "summary": row.summary,
        "operator": username,
        "created_at": iso(row.created_at),
    }


def generate_code() -> str:
    groups = ["".join(secrets.choice(CODE_ALPHABET) for _ in range(4)) for _ in range(3)]
    return "JY-" + "-".join(groups)


def allocate_code(session: Session) -> str:
    for _ in range(8):
        code = generate_code()
        exists = session.query(LicenseKey.id).filter(LicenseKey.code == code).first()
        if exists is None:
            return code
    raise RuntimeError("无法生成卡密")


def record_audit(
    session: Session,
    operator_id: int,
    action: str,
    target_type: str,
    target_id: str,
    summary: str,
) -> None:
    session.add(
        AuditLog(
            operator_id=operator_id,
            action=action,
            target_type=target_type,
            target_id=target_id,
            summary=summary,
            created_at=utcnow(),
        )
    )


def issue_key(session: Session, customer: Customer, operator_id: int) -> LicenseKey:
    now = utcnow()
    key = LicenseKey(
        code=allocate_code(session),
        customer_id=customer.id,
        revoked=False,
        issued_at=now,
        activated_at=None,
        expires_at=now + timedelta(days=RENEW_DAYS),
        machine_id="",
        bound_device_ip="",
        last_device_ip="",
        last_verified_at=None,
        created_at=now,
    )
    session.add(key)
    session.flush()
    record_audit(
        session,
        operator_id,
        "key.issue",
        "license_key",
        str(key.id),
        f"为 {customer.name} 发放卡密 {key.code}",
    )
    return key


def renew_key(session: Session, key: LicenseKey, operator_id: int, note: str = "") -> LicenseKey:
    if key.revoked:
        raise ValueError("卡密已作废，不能续期")
    now = utcnow()
    old_expires = ensure_utc(key.expires_at)
    base = old_expires if old_expires > now else now
    new_expires = base + timedelta(days=RENEW_DAYS)
    key.expires_at = new_expires
    session.add(
        Renewal(
            license_key_id=key.id,
            operator_id=operator_id,
            old_expires_at=old_expires,
            new_expires_at=new_expires,
            note=note.strip(),
            created_at=now,
        )
    )
    record_audit(
        session,
        operator_id,
        "key.renew",
        "license_key",
        str(key.id),
        f"卡密 {key.code} 续期至 {iso(new_expires)}",
    )
    return key


def revoke_key(session: Session, key: LicenseKey, operator_id: int) -> None:
    if key.revoked:
        raise ValueError("卡密已作废")
    key.revoked = True
    record_audit(session, operator_id, "key.revoke", "license_key", str(key.id), f"作废卡密 {key.code}")


def unbind_key(session: Session, key: LicenseKey, operator_id: int) -> None:
    if key.revoked:
        raise ValueError("卡密已作废")
    if not key.machine_id and not key.bound_device_ip:
        raise ValueError("卡密尚未绑定设备")
    previous = key.machine_id or key.bound_device_ip
    key.machine_id = ""
    key.bound_device_ip = ""
    record_audit(
        session,
        operator_id,
        "key.unbind",
        "license_key",
        str(key.id),
        f"解绑卡密 {key.code}，原设备 {previous}",
    )


def verify_license(
    session: Session,
    *,
    code: str,
    device_ip: str,
    machine_id: str,
) -> VerifyOutcome:
    normalized = code.strip().upper()
    if not normalized:
        return VerifyOutcome(False, "invalid", "卡密无效")
    raw_machine = machine_id.strip().lower()
    if len(raw_machine) > 64:
        return VerifyOutcome(False, "invalid", "设备标识无效")
    ip = device_ip.strip()[:64]
    key = session.query(LicenseKey).filter(LicenseKey.code == normalized).one_or_none()
    if key is None:
        return VerifyOutcome(False, "invalid", "卡密无效")
    if key.revoked:
        return VerifyOutcome(False, "revoked", "卡密已作废", iso(key.expires_at))
    now = utcnow()
    if ensure_utc(key.expires_at) <= now:
        return VerifyOutcome(False, "expired", "卡密已过期", iso(key.expires_at))

    if key.machine_id:
        if raw_machine and raw_machine == key.machine_id:
            return _touch(key, ip, now)
        return VerifyOutcome(False, "device_mismatch", "卡密已绑定其他设备", iso(key.expires_at))

    if key.bound_device_ip:
        if ip != key.bound_device_ip:
            return VerifyOutcome(False, "device_mismatch", "卡密已绑定其他设备", iso(key.expires_at))
        if raw_machine:
            key.machine_id = raw_machine
            key.bound_device_ip = ""
        return _touch(key, ip, now)

    if raw_machine:
        key.machine_id = raw_machine
    elif ip:
        key.bound_device_ip = ip
    else:
        return VerifyOutcome(False, "invalid", "缺少设备标识")
    if key.activated_at is None:
        key.activated_at = now
    return _touch(key, ip, now)


def _touch(key: LicenseKey, ip: str, now: datetime) -> VerifyOutcome:
    key.last_device_ip = ip
    key.last_verified_at = now
    if key.activated_at is None:
        key.activated_at = now
    return VerifyOutcome(True, "ok", "license verified", iso(key.expires_at), persist=True)


def list_keys(
    session: Session,
    *,
    customer_id: int | None,
    status: str,
    expiring_within_days: int | None,
    q: str,
    page: int,
    page_size: int,
) -> tuple[list[LicenseKey], int]:
    query = session.query(LicenseKey).options(joinedload(LicenseKey.customer)).order_by(LicenseKey.id.desc())
    if customer_id:
        query = query.filter(LicenseKey.customer_id == customer_id)
    text = q.strip()
    if text:
        like = f"%{text}%"
        query = query.join(Customer).filter(or_(LicenseKey.code.ilike(like), Customer.name.ilike(like)))
    rows = query.all()
    now = utcnow()
    soon = now + timedelta(days=expiring_within_days) if expiring_within_days else None
    filtered: list[LicenseKey] = []
    for key in rows:
        if status and status_of(key, now) != status:
            continue
        if soon is not None:
            expires = ensure_utc(key.expires_at)
            if key.revoked or not (now < expires <= soon):
                continue
        filtered.append(key)
    start = (page - 1) * page_size
    return filtered[start : start + page_size], len(filtered)


def collect_stats(session: Session) -> dict[str, int]:
    now = utcnow()
    soon = now + timedelta(days=7)
    keys = session.query(LicenseKey).all()
    active = expiring = expired = unbound = 0
    for key in keys:
        current = status_of(key, now)
        if current == "active":
            active += 1
        elif current == "expired":
            expired += 1
        elif current == "issued":
            unbound += 1
        if current in {"active", "issued"} and ensure_utc(key.expires_at) <= soon:
            expiring += 1
    return {
        "active": active,
        "expiring_soon": expiring,
        "expired": expired,
        "unbound": unbound,
        "customers": session.query(Customer).count(),
    }
