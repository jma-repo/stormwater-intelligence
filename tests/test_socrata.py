import pandas as pd
import pytest

from stormwater_intel import config as C
from stormwater_intel import fetch as F
from stormwater_intel import socrata as S


class _Resp:
    def __init__(self, status: int, text: str = ""):
        self.status_code, self.text, self.url, self.encoding = status, text, "https://example.test/x", None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise S.requests.HTTPError(f"HTTP {self.status_code}")


class _Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params or {}))
        return self.responses.pop(0)


def test_fetch_all_pages_until_a_short_page(monkeypatch):
    monkeypatch.setattr(S.time, "sleep", lambda s: None)
    sess = _Session([_Resp(200, "a,b\n1,x\n2,y\n"), _Resp(200, "a,b\n3,z\n")])
    df = S.fetch_all("abcd-1234", page_size=2, session=sess)
    assert df["a"].tolist() == ["1", "2", "3"]                  # everything stays a string
    assert [c["$offset"] for c in sess.calls] == [0, 2]
    assert sess.calls[0]["$order"] == ":id"                      # stable paging


def test_fetch_all_keeps_na_looking_codes_as_text(monkeypatch):
    monkeypatch.setattr(S.time, "sleep", lambda s: None)
    sess = _Session([_Resp(200, "comm_code\nNA\nN/A\n")])
    assert S.fetch_all("abcd-1234", session=sess)["comm_code"].tolist() == ["NA", "N/A"]


def test_get_text_retries_on_throttling_and_server_errors(monkeypatch):
    monkeypatch.setattr(S.time, "sleep", lambda s: None)
    sess = _Session([_Resp(429), _Resp(503), _Resp(200, "ok")])
    assert S.get_text(sess, "https://example.test") == "ok"
    assert len(sess.calls) == 3


def test_get_text_gives_up_after_the_retry_budget(monkeypatch):
    monkeypatch.setattr(S.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        S.get_text(_Session([_Resp(500)] * 3), "https://example.test", retries=3)


def test_fetch_311_requests_every_known_service_name_and_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "RAW_DIR", tmp_path)
    seen = {}

    def fake_fetch_all(dataset, **kwargs):
        seen.update(dataset=dataset, **kwargs)
        return pd.DataFrame({"service_name": ["WATS - Catch Basin Concerns"]})

    monkeypatch.setattr(F, "fetch_all", fake_fetch_all)
    out = F.fetch_311()
    assert out.exists() and seen["dataset"] == C.DATASET_311
    assert seen["select"].split(",") == C.COLS_311
    for name in C.SERVICE_TO_GROUP:
        assert f"'{name}'" in seen["where"]
    seen.clear()
    F.fetch_311()                                                # file exists -> no second download
    assert not seen
