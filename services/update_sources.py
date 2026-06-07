from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import requests


DEFAULT_UPDATE_SHARE_URL = "https://liaotianbao.lanzouq.com/b006712hbg"
DEFAULT_FULL_PACKAGE_PAGE_URL = "https://1822169749.share.123865.com/123pan/0OrBjv-AoL7d"
DEFAULT_UPDATE_FILE_PREFIX = "jixiebijiaoben_update"
DEFAULT_UPDATE_FILE_SUFFIX = ".zip"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 20

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
}
_AJAX_HEADERS = {
    "X-Requested-With": "XMLHttpRequest",
}
_ACW_ARG1_RE = re.compile(r"var\s+arg1='([0-9A-F]+)'", re.IGNORECASE)
_FILEMORE_RE = re.compile(r"/filemoreajax\.php\?file=(\d+)")
_AJAX_DATA_BLOCK_RE = re.compile(
    r"filemoreajax\.php\?file=\d+.*?data\s*:\s*\{(.*?)\}\s*(?:,\s*dataType|\}\s*\)\s*;|\}\s*\)\s*)",
    re.IGNORECASE | re.DOTALL,
)
_AJAX_DATA_FIELD_RE = re.compile(r"'([^']+)'\s*:\s*([^,\n\r]+)")
_IFRAME_SRC_RE = re.compile(r'<iframe[^>]+src="([^"]+)"', re.IGNORECASE)
_ACW_PERMUTATION = (
    0x0F,
    0x23,
    0x1D,
    0x18,
    0x21,
    0x10,
    0x01,
    0x26,
    0x0A,
    0x09,
    0x13,
    0x1F,
    0x28,
    0x1B,
    0x16,
    0x17,
    0x19,
    0x0D,
    0x06,
    0x0B,
    0x27,
    0x12,
    0x14,
    0x08,
    0x0E,
    0x15,
    0x20,
    0x1A,
    0x02,
    0x1E,
    0x07,
    0x04,
    0x11,
    0x05,
    0x03,
    0x1C,
    0x22,
    0x25,
    0x0C,
    0x24,
)
_ACW_XOR_KEY = "3000176000856006061501533003690027800375"


class UpdateSourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class LanzouShareEntry:
    file_id: str
    name: str
    size: str
    icon: str
    detail_url: str


@dataclass(frozen=True)
class ResolvedUpdatePackage:
    share_url: str
    file_name: str
    detail_url: str
    frame_url: str
    ajax_url: str
    download_url: str
    size: str


def full_package_download_page_url() -> str:
    return DEFAULT_FULL_PACKAGE_PAGE_URL


def resolve_lanzou_update_package(
    *,
    share_url: str = DEFAULT_UPDATE_SHARE_URL,
    file_name: str = "",
    file_prefix: str = DEFAULT_UPDATE_FILE_PREFIX,
    file_suffix: str = DEFAULT_UPDATE_FILE_SUFFIX,
    timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    session: requests.Session | None = None,
) -> ResolvedUpdatePackage:
    http = session or requests.Session()

    share_response = _get_with_acw(http, share_url, timeout_seconds=timeout_seconds)
    entries = _load_share_entries(http, share_response, timeout_seconds=timeout_seconds)
    entry = _select_update_entry(entries, file_name=file_name, file_prefix=file_prefix, file_suffix=file_suffix)

    detail_response = _get_with_acw(
        http,
        entry.detail_url,
        timeout_seconds=timeout_seconds,
        referer=share_response.url,
    )
    frame_url = _extract_frame_url(detail_response.text, detail_response.url)

    frame_response = _get_with_acw(
        http,
        frame_url,
        timeout_seconds=timeout_seconds,
        referer=detail_response.url,
    )
    ajax_url, ajax_payload = _build_frame_ajax_request(frame_response.text, frame_response.url)
    ajax_response = http.post(
        ajax_url,
        data=ajax_payload,
        headers=_request_headers(frame_response.url, ajax=True),
        timeout=timeout_seconds,
    )
    _raise_for_status(ajax_response)

    try:
        payload = ajax_response.json()
    except json.JSONDecodeError as exc:
        raise UpdateSourceError("Lanzou update ajax response was not valid JSON.") from exc

    if str(payload.get("zt", "")) != "1":
        raise UpdateSourceError(f"Lanzou update ajax returned non-success payload: {payload!r}")

    domain = str(payload.get("dom", "") or "").strip().rstrip("/")
    path = str(payload.get("url", "") or "").strip()
    if not domain or not path:
        raise UpdateSourceError("Lanzou update ajax payload did not contain a usable download URL.")
    download_url = f"{domain}/file/{path.lstrip('/')}"

    return ResolvedUpdatePackage(
        share_url=share_url,
        file_name=entry.name,
        detail_url=entry.detail_url,
        frame_url=frame_url,
        ajax_url=ajax_url,
        download_url=download_url,
        size=entry.size,
    )


def _get_with_acw(
    session: requests.Session,
    url: str,
    *,
    timeout_seconds: int,
    referer: str | None = None,
) -> requests.Response:
    response = session.get(url, headers=_request_headers(referer), timeout=timeout_seconds)
    _raise_for_status(response)
    challenge = _ACW_ARG1_RE.search(response.text)
    if challenge:
        host = urlparse(url).hostname
        if not host:
            raise UpdateSourceError(f"Could not determine host for Lanzou challenge URL: {url}")
        session.cookies.set("acw_sc__v2", _solve_acw_sc_v2(challenge.group(1)), domain=host, path="/")
        response = session.get(url, headers=_request_headers(referer), timeout=timeout_seconds)
        _raise_for_status(response)
    return response


def _raise_for_status(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise UpdateSourceError(f"HTTP request failed for {response.url}: {exc}") from exc


def _request_headers(referer: str | None, *, ajax: bool = False) -> dict[str, str]:
    headers = dict(_DEFAULT_HEADERS)
    if ajax:
        headers.update(_AJAX_HEADERS)
    if referer:
        parsed = urlparse(referer)
        headers["Referer"] = referer
        headers["Origin"] = f"{parsed.scheme}://{parsed.netloc}"
    return headers


def _load_share_entries(
    session: requests.Session,
    share_response: requests.Response,
    *,
    timeout_seconds: int,
) -> list[LanzouShareEntry]:
    html_text = share_response.text
    folder_id_match = _FILEMORE_RE.search(html_text)
    if not folder_id_match:
        raise UpdateSourceError("Could not find Lanzou share folder ajax endpoint.")
    folder_id = folder_id_match.group(1)
    ajax_fields = _extract_share_ajax_fields(html_text)
    if not str(ajax_fields.get("uid", "") or "").strip():
        ajax_fields["uid"] = _extract_js_string(html_text, "uid", from_object=True)
    if not str(ajax_fields.get("puid", "") or "").strip():
        ajax_fields["puid"] = _extract_js_string(html_text, "puid", from_object=True)
    if not str(ajax_fields.get("t", "") or "").strip():
        ajax_fields["t"] = _extract_first_available_js_string(html_text, ("ib2801", "ib4d5t"))
    if not str(ajax_fields.get("k", "") or "").strip():
        ajax_fields["k"] = _extract_first_available_js_string(html_text, ("_h0k1u", "_h2p7m"))

    payload = {
        "lx": str(ajax_fields.get("lx", "2") or "2"),
        "fid": str(ajax_fields.get("fid", folder_id) or folder_id),
        "uid": str(ajax_fields.get("uid", "") or ""),
        "puid": str(ajax_fields.get("puid", "") or ""),
        "pg": "1",
        "rep": str(ajax_fields.get("rep", "0") or "0"),
        "t": str(ajax_fields.get("t", "") or ""),
        "k": str(ajax_fields.get("k", "") or ""),
        "up": str(ajax_fields.get("up", "1") or "1"),
        "vip": str(ajax_fields.get("vip", "0") or "0"),
        "webfoldersign": str(ajax_fields.get("webfoldersign", "") or ""),
    }
    ajax_url = urljoin(share_response.url, f"/filemoreajax.php?file={folder_id}")
    response = session.post(
        ajax_url,
        data=payload,
        headers=_request_headers(share_response.url, ajax=True),
        timeout=timeout_seconds,
    )
    _raise_for_status(response)

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise UpdateSourceError("Lanzou share ajax response was not valid JSON.") from exc

    if str(data.get("zt", "")) != "1":
        raise UpdateSourceError(f"Lanzou share ajax returned non-success payload: {data!r}")

    raw_entries = data.get("text", [])
    if not isinstance(raw_entries, list):
        raise UpdateSourceError("Lanzou share ajax payload did not contain a file list.")

    entries: list[LanzouShareEntry] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        file_id = str(item.get("id", "") or "").strip()
        file_name = html.unescape(str(item.get("name_all", "") or "").strip())
        if not file_id or not file_name:
            continue
        detail_url = file_id if file_id.startswith("http") else urljoin(share_response.url, "/" + file_id.lstrip("/"))
        entries.append(
            LanzouShareEntry(
                file_id=file_id,
                name=file_name,
                size=str(item.get("size", "") or "").strip(),
                icon=str(item.get("icon", "") or "").strip().lower(),
                detail_url=detail_url,
            )
        )
    if not entries:
        raise UpdateSourceError("Lanzou share ajax returned an empty file list.")
    return entries


def _extract_share_ajax_fields(html_text: str) -> dict[str, str]:
    block_match = _AJAX_DATA_BLOCK_RE.search(html_text)
    if not block_match:
        return {}
    block = block_match.group(1)
    values: dict[str, str] = {}
    for key, raw_value in _AJAX_DATA_FIELD_RE.findall(block):
        value = raw_value.strip()
        if value.endswith(","):
            value = value[:-1].strip()
        if value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            try:
                value = _extract_js_string(html_text, value)
            except UpdateSourceError:
                continue
        values[key] = html.unescape(value)
    return values


def _extract_first_available_js_string(html_text: str, names: tuple[str, ...]) -> str:
    for name in names:
        try:
            return _extract_js_string(html_text, name)
        except UpdateSourceError:
            continue
    return ""


def _select_update_entry(
    entries: list[LanzouShareEntry],
    *,
    file_name: str,
    file_prefix: str,
    file_suffix: str,
) -> LanzouShareEntry:
    target_name = str(file_name or "").strip().lower()
    if target_name:
        for entry in entries:
            if entry.name.lower() == target_name:
                return entry
        raise UpdateSourceError(f"Could not find the requested update file in Lanzou share: {file_name}")

    suffix = str(file_suffix or "").strip().lower()
    prefix = str(file_prefix or "").strip().lower()
    zip_entries = [entry for entry in entries if entry.name.lower().endswith(suffix)]
    if not zip_entries:
        raise UpdateSourceError(f"No Lanzou share entry ended with {file_suffix!r}.")

    preferred = [entry for entry in zip_entries if entry.name.lower().startswith(prefix)]
    candidates = preferred or zip_entries
    candidates.sort(key=lambda entry: entry.name.lower())
    return candidates[-1]


def _extract_frame_url(detail_html: str, detail_url: str) -> str:
    iframe_match = _IFRAME_SRC_RE.search(detail_html)
    if not iframe_match:
        raise UpdateSourceError("Could not find the Lanzou detail iframe.")
    return urljoin(detail_url, iframe_match.group(1))


def _build_frame_ajax_request(frame_html: str, frame_url: str) -> tuple[str, dict[str, str]]:
    file_match = re.search(r"/ajaxm\.php\?file=(\d+)", frame_html)
    ajaxdata = _extract_js_string(frame_html, "ajaxdata")
    wp_sign = _extract_js_string(frame_html, "wp_sign")
    if not file_match:
        raise UpdateSourceError("Could not find the Lanzou frame ajax endpoint.")
    file_id = file_match.group(1)
    ajax_url = urljoin(frame_url, f"/ajaxm.php?file={file_id}")
    payload = {
        "action": "downprocess",
        "websignkey": ajaxdata,
        "signs": ajaxdata,
        "sign": wp_sign,
        "websign": "",
        # The page loads kdns.js before issuing the ajax request, so the
        # browser-side runtime normally keeps `kdns = 1`.
        "kd": "1",
        "ves": "1",
    }
    return ajax_url, payload


def _extract_js_string(text: str, name: str, *, from_object: bool = False) -> str:
    if from_object:
        pattern = re.compile(rf"'{re.escape(name)}':'([^']*)'")
    else:
        pattern = re.compile(rf"var\s+{re.escape(name)}\s*=\s*'([^']*)'")
    match = pattern.search(text)
    if not match:
        raise UpdateSourceError(f"Could not find Lanzou javascript field: {name}")
    return match.group(1)


def _solve_acw_sc_v2(arg1: str) -> str:
    if len(arg1) < len(_ACW_PERMUTATION):
        raise UpdateSourceError("Unexpected Lanzou acw challenge length.")
    remapped = [""] * len(_ACW_PERMUTATION)
    for source_index, char in enumerate(arg1):
        for target_index, target in enumerate(_ACW_PERMUTATION):
            if target == source_index + 1:
                remapped[target_index] = char
                break
    merged = "".join(remapped)
    return "".join(
        f"{int(merged[index:index + 2], 16) ^ int(_ACW_XOR_KEY[index:index + 2], 16):02x}"
        for index in range(0, min(len(merged), len(_ACW_XOR_KEY)), 2)
    )
