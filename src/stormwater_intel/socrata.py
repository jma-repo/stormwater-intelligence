"""Tiny Socrata (SODA 2.x) client: paged CSV download with retries."""
from __future__ import annotations

import io
import logging
import time

import pandas as pd
import requests

from . import config as C

log = logging.getLogger(__name__)
USER_AGENT = "stormwater-intelligence-poc (open-data research script)"


def new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def get_text(session: requests.Session, url: str, params: dict | None = None,
             retries: int = 6, timeout: int = 180) -> str:
    """GET with exponential backoff on throttling / transient server errors."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code} from {r.url[:120]}")
            r.raise_for_status()
            r.encoding = "utf-8"
            return r.text
        except requests.RequestException as exc:  # noqa: PERF203
            last = exc
            wait = min(60, 2 ** attempt)
            log.warning("GET failed (%s); retry %d/%d in %ss", exc, attempt + 1, retries, wait)
            time.sleep(wait)
    raise RuntimeError(f"GET failed after {retries} attempts: {last}")


def resource_url(dataset: str, fmt: str = "csv") -> str:
    return f"{C.SOCRATA_HOST}/resource/{dataset}.{fmt}"


def fetch_all(dataset: str, *, select: str | None = None, where: str | None = None,
              order: str | None = ":id", page_size: int = 50_000, pause: float = 0.2,
              session: requests.Session | None = None) -> pd.DataFrame:
    """Download a whole dataset (or a filtered slice) as an all-string DataFrame."""
    s = session or new_session()
    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        params: dict[str, str | int] = {"$limit": page_size, "$offset": offset}
        if select:
            params["$select"] = select
        if where:
            params["$where"] = where
        if order:
            params["$order"] = order
        text = get_text(s, resource_url(dataset), params)
        page = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        log.info("%s offset=%d rows=%d", dataset, offset, len(page))
        if page.empty:
            break
        frames.append(page)
        if len(page) < page_size:
            break
        offset += page_size
        time.sleep(pause)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
