"""Small Flask gateway for client rule-learning uploads.

Run on the server only. COS credentials must be supplied through environment
variables; this file is never included in the customer package.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from pathlib import PurePosixPath
from urllib.parse import quote, urlparse

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)
_LOCK = threading.RLock()
ALLOWED_APPS = {"xinyu", "huanyou"}
COS_BASE = os.environ.get(
    "RULE_LEARNING_COS_BASE",
    "https://jixiebi-1311918180.cos.ap-shanghai.myqcloud.com/rule-learning",
).rstrip("/")
COS_SECRET_ID = os.environ.get("TENCENT_COS_SECRET_ID", "").strip()
COS_SECRET_KEY = os.environ.get("TENCENT_COS_SECRET_KEY", "").strip()
COS_SESSION_TOKEN = os.environ.get("TENCENT_COS_SESSION_TOKEN", "").strip()


def _cos_headers(method: str, url: str, content_type: str) -> dict[str, str]:
    parsed = urlparse(url)
    headers = {"Host": parsed.netloc, "Content-Type": content_type}
    if method.upper() == "PUT":
        headers["x-cos-acl"] = "public-read"
    if COS_SESSION_TOKEN:
        headers["x-cos-security-token"] = COS_SESSION_TOKEN
    normalized = sorted((key.lower(), " ".join(value.strip().split())) for key, value in headers.items())
    header_list = ";".join(key for key, _ in normalized)
    header_pairs = "&".join(f"{key}={quote(value, safe='-_.~')}" for key, value in normalized)
    now = int(time.time())
    sign_time = f"{now};{now + 600}"
    http_string = "\n".join([method.lower(), parsed.path or "/", "", header_pairs, ""])
    digest = hashlib.sha1(http_string.encode()).hexdigest()
    string_to_sign = "\n".join(["sha1", sign_time, digest, ""])
    sign_key = hmac.new(COS_SECRET_KEY.encode(), sign_time.encode(), hashlib.sha1).hexdigest()
    signature = hmac.new(sign_key.encode(), string_to_sign.encode(), hashlib.sha1).hexdigest()
    headers["Authorization"] = (
        f"q-sign-algorithm=sha1&q-ak={COS_SECRET_ID}&q-sign-time={sign_time}"
        f"&q-key-time={sign_time}&q-header-list={header_list}&q-url-param-list=&q-signature={signature}"
    )
    return headers


def _cos_get(url: str) -> dict:
    response = requests.get(url, timeout=15)
    if response.status_code == 404:
        return {"schema_version": 1, "rules": []}
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {"schema_version": 1, "rules": []}


def _cos_put(url: str, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    headers = _cos_headers("PUT", url, "application/json; charset=utf-8")
    response = requests.put(url, data=body, headers=headers, timeout=15)
    response.raise_for_status()


@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "rule-learning"})


@app.post("/rule-learning/<app_name>/upload")
def upload(app_name: str):
    if app_name not in ALLOWED_APPS:
        return jsonify({"ok": False, "error": "unsupported app"}), 404
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or payload.get("app") != app_name:
        return jsonify({"ok": False, "error": "invalid payload"}), 400
    event_id = str(payload.get("event_id", "")).strip()
    keywords = payload.get("keywords")
    if not event_id or not isinstance(keywords, list) or len(keywords) > 100:
        return jsonify({"ok": False, "error": "invalid event"}), 400
    values = list(dict.fromkeys(" ".join(str(value or "").split()).strip() for value in keywords if str(value or "").strip()))
    if not values or any(len(value) > 200 for value in values):
        return jsonify({"ok": False, "error": "invalid keywords"}), 400
    event = {"schema_version": 1, "event_id": event_id, "app": app_name, "keywords": values,
             "client_id": str(payload.get("client_id", ""))[:64], "device_id": str(payload.get("device_id", ""))[:64],
             "source": str(payload.get("source", "vision_model"))[:40], "created_at": str(payload.get("created_at", ""))[:40]}
    try:
        with _LOCK:
            event_url = f"{COS_BASE}/{app_name}/messages/{quote(event_id, safe='-_.~')}.json"
            rules_url = f"{COS_BASE}/{app_name}/rules.json"
            # Idempotent event object prevents duplicate uploads from multiplying rules.
            existing_event = requests.get(event_url, timeout=15)
            if existing_event.status_code == 404:
                _cos_put(event_url, event)
            rules = _cos_get(rules_url)
            current = [str(item.get("keyword", "")).strip() for item in rules.get("rules", []) if isinstance(item, dict)]
            merged = list(dict.fromkeys([value for value in current if value] + values))
            _cos_put(rules_url, {"schema_version": 1, "app": app_name, "rules": [{"keyword": value} for value in merged]})
    except (OSError, requests.RequestException, ValueError) as exc:
        app.logger.exception("rule upload failed: %s", exc)
        return jsonify({"ok": False, "error": "storage unavailable"}), 503
    return jsonify({"ok": True, "app": app_name, "added": len(values)})


if __name__ == "__main__":
    app.run(host=os.environ.get("RULE_LEARNING_HOST", "127.0.0.1"), port=int(os.environ.get("RULE_LEARNING_PORT", "5004")))
