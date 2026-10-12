from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from jwt import PyJWTError
from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from app.licensing import (
    collect_stats,
    iso,
    issue_key,
    list_keys,
    record_audit,
    renew_key,
    revoke_key,
    serialize_audit,
    serialize_key,
    serialize_renewal,
    unbind_key,
    utcnow,
)
from app.models import AdminUser, AuditLog, Customer, LicenseKey, Renewal
from app.schemas import BatchRenewBody, CustomerBody, IssueKeyBody, LoginBody, RenewBody
from app.security import (
    REFRESH_COOKIE,
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
)


router = APIRouter(prefix="/api/admin")


def get_session(request: Request):
    session: Session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def get_current_admin(
    request: Request,
    session: Session = Depends(get_session),
) -> AdminUser:
    authorization = request.headers.get("authorization", "")
    token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
    if not token:
        raise HTTPException(status_code=401, detail="未登录")
    user_id = _user_id(token, request, "access")
    user = session.get(AdminUser, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="未登录")
    return user


def _set_refresh_cookie(response: Response, request: Request, token: str) -> None:
    settings = request.app.state.settings
    response.set_cookie(
        key=REFRESH_COOKIE,
        value=token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        max_age=settings.refresh_token_days * 24 * 3600,
        path="/api/admin",
    )


def _clear_refresh_cookie(response: Response, request: Request) -> None:
    settings = request.app.state.settings
    response.delete_cookie(key=REFRESH_COOKIE, path="/api/admin", secure=settings.cookie_secure, samesite="lax")


@router.post("/login")
def login(body: LoginBody, request: Request, response: Response, session: Session = Depends(get_session)) -> dict:
    user = session.query(AdminUser).filter(AdminUser.username == body.username.strip()).one_or_none()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="用户名或口令错误")
    settings = request.app.state.settings
    _set_refresh_cookie(response, request, create_refresh_token(user.id, settings))
    return {
        "access_token": create_access_token(user.id, user.username, settings),
        "token_type": "bearer",
        "username": user.username,
    }


@router.post("/refresh")
def refresh(request: Request, response: Response, session: Session = Depends(get_session)) -> dict:
    token = request.cookies.get(REFRESH_COOKIE, "")
    if not token:
        raise HTTPException(status_code=401, detail="未登录")
    settings = request.app.state.settings
    user_id = _user_id(token, request, "refresh")
    user = session.get(AdminUser, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="未登录")
    _set_refresh_cookie(response, request, create_refresh_token(user.id, settings))
    return {
        "access_token": create_access_token(user.id, user.username, settings),
        "token_type": "bearer",
        "username": user.username,
    }


@router.post("/logout")
def logout(request: Request, response: Response) -> dict:
    _clear_refresh_cookie(response, request)
    return {"ok": True}


@router.get("/me")
def me(user: AdminUser = Depends(get_current_admin)) -> dict:
    return {"id": user.id, "username": user.username}


@router.get("/stats")
def stats(session: Session = Depends(get_session), _: AdminUser = Depends(get_current_admin)) -> dict:
    return collect_stats(session)


@router.get("/customers")
def list_customers(
    q: str = "",
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_session),
    _: AdminUser = Depends(get_current_admin),
) -> dict:
    query = session.query(Customer).order_by(Customer.id.desc())
    text = q.strip()
    if text:
        like = f"%{text}%"
        query = query.filter(or_name_contact(like))
    total = query.count()
    rows = query.offset((page - 1) * page_size).limit(page_size).all()
    return {"items": [_customer_dict(row) for row in rows], "total": total}


def or_name_contact(like: str):
    return or_(Customer.name.ilike(like), Customer.contact.ilike(like))


def _customer_dict(row: Customer) -> dict:
    return {
        "id": row.id,
        "name": row.name,
        "contact": row.contact,
        "note": row.note,
        "created_at": iso(row.created_at),
        "updated_at": iso(row.updated_at),
    }


def _user_id(token: str, request: Request, expected_type: str) -> int:
    try:
        payload = decode_token(token, request.app.state.settings, expected_type)
        return int(payload["sub"])
    except (PyJWTError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=401, detail="登录已过期") from exc


@router.post("/customers")
def create_customer(
    body: CustomerBody,
    session: Session = Depends(get_session),
    user: AdminUser = Depends(get_current_admin),
) -> dict:
    now = utcnow()
    customer = Customer(
        name=body.name.strip(),
        contact=body.contact.strip(),
        note=body.note.strip(),
        created_at=now,
        updated_at=now,
    )
    if not customer.name:
        raise HTTPException(status_code=400, detail="请填写客户名称")
    session.add(customer)
    session.flush()
    record_audit(session, user.id, "customer.create", "customer", str(customer.id), f"新建客户 {customer.name}")
    session.commit()
    return _customer_dict(customer)


@router.put("/customers/{customer_id}")
def update_customer(
    customer_id: int,
    body: CustomerBody,
    session: Session = Depends(get_session),
    user: AdminUser = Depends(get_current_admin),
) -> dict:
    customer = session.get(Customer, customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="客户不存在")
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="请填写客户名称")
    customer.name = name
    customer.contact = body.contact.strip()
    customer.note = body.note.strip()
    customer.updated_at = utcnow()
    record_audit(session, user.id, "customer.update", "customer", str(customer.id), f"更新客户 {customer.name}")
    session.commit()
    return _customer_dict(customer)


@router.get("/keys")
def get_keys(
    customer_id: int | None = None,
    status: str = "",
    expiring_within_days: int | None = Query(default=None, ge=1, le=365),
    q: str = "",
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_session),
    _: AdminUser = Depends(get_current_admin),
) -> dict:
    rows, total = list_keys(
        session,
        customer_id=customer_id,
        status=status,
        expiring_within_days=expiring_within_days,
        q=q,
        page=page,
        page_size=page_size,
    )
    return {"items": [serialize_key(row) for row in rows], "total": total}


@router.post("/keys")
def create_key(
    body: IssueKeyBody,
    session: Session = Depends(get_session),
    user: AdminUser = Depends(get_current_admin),
) -> dict:
    customer = session.get(Customer, body.customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="客户不存在")
    key = issue_key(session, customer, user.id)
    session.commit()
    session.refresh(key)
    key.customer = customer
    return serialize_key(key)


@router.get("/keys/{key_id}")
def get_key(
    key_id: int,
    session: Session = Depends(get_session),
    _: AdminUser = Depends(get_current_admin),
) -> dict:
    key = (
        session.query(LicenseKey)
        .options(joinedload(LicenseKey.customer))
        .filter(LicenseKey.id == key_id)
        .one_or_none()
    )
    if key is None:
        raise HTTPException(status_code=404, detail="卡密不存在")
    renewals = (
        session.query(Renewal)
        .options(joinedload(Renewal.operator))
        .filter(Renewal.license_key_id == key.id)
        .order_by(Renewal.id.desc())
        .all()
    )
    payload = serialize_key(key)
    payload["renewals"] = [serialize_renewal(row) for row in renewals]
    return payload


@router.post("/keys/{key_id}/renew")
def renew_one(
    key_id: int,
    body: RenewBody,
    session: Session = Depends(get_session),
    user: AdminUser = Depends(get_current_admin),
) -> dict:
    key = _load_key(session, key_id)
    try:
        renew_key(session, key, user.id, body.note)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session.commit()
    return serialize_key(key)


@router.post("/keys/renew")
def renew_many(
    body: BatchRenewBody,
    session: Session = Depends(get_session),
    user: AdminUser = Depends(get_current_admin),
) -> dict:
    results = []
    for key_id in body.ids:
        key = session.get(LicenseKey, key_id)
        if key is None:
            results.append({"id": key_id, "ok": False, "error": "卡密不存在"})
            continue
        if key.customer is None:
            key.customer = session.get(Customer, key.customer_id)
        try:
            renew_key(session, key, user.id, body.note)
        except ValueError as exc:
            results.append({"id": key_id, "ok": False, "error": str(exc)})
            continue
        results.append({"id": key_id, "ok": True, "expires_at": serialize_key(key)["expires_at"], "code": key.code})
    session.commit()
    return {"results": results}


@router.post("/keys/{key_id}/revoke")
def revoke_one(
    key_id: int,
    session: Session = Depends(get_session),
    user: AdminUser = Depends(get_current_admin),
) -> dict:
    key = _load_key(session, key_id)
    try:
        revoke_key(session, key, user.id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session.commit()
    return serialize_key(key)


@router.post("/keys/{key_id}/unbind")
def unbind_one(
    key_id: int,
    session: Session = Depends(get_session),
    user: AdminUser = Depends(get_current_admin),
) -> dict:
    key = _load_key(session, key_id)
    try:
        unbind_key(session, key, user.id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session.commit()
    return serialize_key(key)


@router.get("/audit")
def audit(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_session),
    _: AdminUser = Depends(get_current_admin),
) -> dict:
    total = session.query(AuditLog).count()
    rows = (
        session.query(AuditLog)
        .options(joinedload(AuditLog.operator))
        .order_by(AuditLog.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return {"items": [serialize_audit(row) for row in rows], "total": total}


def _load_key(session: Session, key_id: int) -> LicenseKey:
    key = (
        session.query(LicenseKey)
        .options(joinedload(LicenseKey.customer))
        .filter(LicenseKey.id == key_id)
        .one_or_none()
    )
    if key is None:
        raise HTTPException(status_code=404, detail="卡密不存在")
    return key
