from __future__ import annotations

from pydantic import BaseModel, Field


class VerifyBody(BaseModel):
    project_secret: str = ""
    device_ip: str = ""
    license_key: str = ""
    machine_id: str = ""


class LoginBody(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=200)


class CustomerBody(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    contact: str = Field(default="", max_length=200)
    note: str = Field(default="", max_length=2000)


class IssueKeyBody(BaseModel):
    customer_id: int


class RenewBody(BaseModel):
    note: str = Field(default="", max_length=500)


class BatchRenewBody(BaseModel):
    ids: list[int] = Field(min_length=1, max_length=500)
    note: str = Field(default="", max_length=500)
