from __future__ import annotations

import json

import pytest

from services import update_sources


class _CookieJar:
    def __init__(self) -> None:
        self.values: list[tuple[str, str, str | None, str | None]] = []

    def set(self, name: str, value: str, domain: str | None = None, path: str | None = None) -> None:
        self.values.append((name, value, domain, path))


class _FakeResponse:
    def __init__(self, url: str, text: str, *, status_code: int = 200) -> None:
        self.url = url
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise update_sources.requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return json.loads(self.text)


class _FakeSession:
    def __init__(self) -> None:
        self.cookies = _CookieJar()
        self._get_queue: dict[str, list[_FakeResponse]] = {}
        self._post_queue: dict[str, list[_FakeResponse]] = {}

    def queue_get(self, url: str, *responses: _FakeResponse) -> None:
        self._get_queue.setdefault(url, []).extend(responses)

    def queue_post(self, url: str, *responses: _FakeResponse) -> None:
        self._post_queue.setdefault(url, []).extend(responses)

    def get(self, url: str, headers=None, timeout=None):  # noqa: ANN001
        queue = self._get_queue.get(url, [])
        if not queue:
            raise AssertionError(f"Unexpected GET: {url}")
        return queue.pop(0)

    def post(self, url: str, data=None, headers=None, timeout=None):  # noqa: ANN001
        queue = self._post_queue.get(url, [])
        if not queue:
            raise AssertionError(f"Unexpected POST: {url}")
        return queue.pop(0)


def test_solve_acw_sc_v2_matches_expected_cookie():
    arg1 = "CDB2BE2DCD81465CC7281C15127B207906A52775"
    expected = "6a251ba7dca237c712fd76de5191b922ec3b6160"

    assert update_sources._solve_acw_sc_v2(arg1) == expected


def test_resolve_lanzou_update_package_walks_share_detail_and_ajax_chain():
    share_url = "https://liaotianbao.lanzouq.com/b006712hbg"
    detail_url = "https://liaotianbao.lanzouq.com/io9Bl3rbb3de"
    frame_url = "https://liaotianbao.lanzouq.com/fn?frame-token"
    filemore_url = "https://liaotianbao.lanzouq.com/filemoreajax.php?file=13580356"
    ajax_url = "https://liaotianbao.lanzouq.com/ajaxm.php?file=289469534"

    challenge_html = (
        "<html><script>var arg1='CDB2BE2DCD81465CC7281C15127B207906A52775';</script></html>"
    )
    share_html = """
<script type="text/javascript">
var ib2801 = '1780817496';
var _h0k1u = 'd23aa3bedb658a092356887b1c24e112';
function file(){
  $.ajax({
    url : '/filemoreajax.php?file=13580356',
    data : {
      'uid':'2680763',
      'puid':'UGJXNlw5VTEBYAdnBmFccANoUGJVMgc1BTpSZVA6W2IIMVQ1C3tQNlI0UzcAYlZnVTkHMgVs'
    }
  });
}
</script>
"""
    filemore_json = {
        "zt": 1,
        "info": "sucess",
        "text": [
            {
                "icon": "zip",
                "t": 0,
                "id": "io9Bl3rbb3dd",
                "name_all": "jixiebijiaoben_update_20260606_0904.zip",
                "size": "13.5 M",
            },
            {
                "icon": "zip",
                "t": 0,
                "id": "io9Bl3rbb3de",
                "name_all": "jixiebijiaoben_update_20260607_0904.zip",
                "size": "14.5 M",
            },
        ],
    }
    detail_html = '<html><iframe src="/fn?frame-token"></iframe></html>'
    frame_html = """
<script src="https://developer3.lanrar.com/file/kdns.js"></script>
<script type="text/javascript">
var wp_sign = 'UzVVaws6Dj8CCwQ7VGRTbwdvUm8CZAA2VWQBPlc_bAzFVYVsqDCUCawluUzJWOgAxUjBVYVY_bBjVVYwE2';
var ajaxdata = 'w9pk';
$.ajax({
  url : '/ajaxm.php?file=289469534',
  data : { 'action':'downprocess' }
});
</script>
"""
    ajax_json = {
        "zt": 1,
        "dom": "https://developer3.lanrar.com",
        "url": "?B2FUal1sDj8IAQc/example-token",
    }

    session = _FakeSession()
    session.queue_get(
        share_url,
        _FakeResponse(share_url, challenge_html),
        _FakeResponse(share_url, share_html),
    )
    session.queue_post(filemore_url, _FakeResponse(filemore_url, json.dumps(filemore_json)))
    session.queue_get(detail_url, _FakeResponse(detail_url, detail_html))
    session.queue_get(frame_url, _FakeResponse(frame_url, frame_html))
    session.queue_post(ajax_url, _FakeResponse(ajax_url, json.dumps(ajax_json)))

    resolved = update_sources.resolve_lanzou_update_package(session=session)

    assert resolved.share_url == share_url
    assert resolved.file_name == "jixiebijiaoben_update_20260607_0904.zip"
    assert resolved.detail_url == detail_url
    assert resolved.frame_url == frame_url
    assert resolved.ajax_url == ajax_url
    assert resolved.download_url == "https://developer3.lanrar.com/file/?B2FUal1sDj8IAQc/example-token"
    assert resolved.size == "14.5 M"
    assert session.cookies.values == [
        ("acw_sc__v2", "6a251ba7dca237c712fd76de5191b922ec3b6160", "liaotianbao.lanzouq.com", "/")
    ]


def test_full_package_download_page_url_matches_configured_manual_link():
    assert (
        update_sources.full_package_download_page_url()
        == "https://1822169749.share.123865.com/123pan/0OrBjv-AoL7d"
    )


def test_select_update_entry_falls_back_to_single_zip():
    entries = [
        update_sources.LanzouShareEntry(
            file_id="abc",
            name="custom_release.zip",
            size="1 M",
            icon="zip",
            detail_url="https://example.com/abc",
        )
    ]

    selected = update_sources._select_update_entry(
        entries,
        file_name="",
        file_prefix=update_sources.DEFAULT_UPDATE_FILE_PREFIX,
        file_suffix=update_sources.DEFAULT_UPDATE_FILE_SUFFIX,
    )

    assert selected.name == "custom_release.zip"


def test_select_update_entry_raises_when_no_zip_exists():
    entries = [
        update_sources.LanzouShareEntry(
            file_id="abc",
            name="readme.txt",
            size="1 K",
            icon="txt",
            detail_url="https://example.com/abc",
        )
    ]

    with pytest.raises(update_sources.UpdateSourceError):
        update_sources._select_update_entry(
            entries,
            file_name="",
            file_prefix=update_sources.DEFAULT_UPDATE_FILE_PREFIX,
            file_suffix=update_sources.DEFAULT_UPDATE_FILE_SUFFIX,
        )


def test_select_update_entry_prefers_exact_file_name_match():
    entries = [
        update_sources.LanzouShareEntry(
            file_id="older",
            name="jixiebijiaoben_update_20260606_0904.zip",
            size="1 M",
            icon="zip",
            detail_url="https://example.com/older",
        ),
        update_sources.LanzouShareEntry(
            file_id="newer",
            name="jixiebijiaoben_update_20260607_0904.zip",
            size="1 M",
            icon="zip",
            detail_url="https://example.com/newer",
        ),
    ]

    selected = update_sources._select_update_entry(
        entries,
        file_name="jixiebijiaoben_update_20260606_0904.zip",
        file_prefix=update_sources.DEFAULT_UPDATE_FILE_PREFIX,
        file_suffix=update_sources.DEFAULT_UPDATE_FILE_SUFFIX,
    )

    assert selected.file_id == "older"
