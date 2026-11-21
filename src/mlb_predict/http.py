"""A small, polite HTTP client.

Every request has a time limit, is retried a few times on temporary errors,
and is spaced out per host. Failures raise. Nothing here swallows an error.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import USER_AGENT

log = logging.getLogger(__name__)

TIMEOUT = (10, 90)  # seconds to connect, seconds to read
_MIN_INTERVAL = {  # minimum seconds between requests to one host
    "www.retrosheet.org": 1.0,
    "statsapi.mlb.com": 0.12,
    "site.api.espn.com": 0.12,
    "sports.core.api.espn.com": 0.12,
    "raw.githubusercontent.com": 0.3,
    "baseballsavant.mlb.com": 1.0,
}
_last_call: dict[str, float] = {}
_lock = threading.Lock()
_local = threading.local()


class FetchError(RuntimeError):
    """A download failed after retries."""


def _session() -> requests.Session:
    session = getattr(_local, "session", None)
    if session is None:
        session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
        _local.session = session
    return session


def _wait_turn(host: str) -> None:
    gap = _MIN_INTERVAL.get(host, 0.25)
    with _lock:
        now = time.monotonic()
        ready = max(now, _last_call.get(host, 0.0) + gap)
        _last_call[host] = ready
    delay = ready - now
    if delay > 0:
        time.sleep(delay)


def get_bytes(url: str, *, allow_404: bool = False) -> bytes | None:
    """Download ``url``. Returns ``None`` for a 404 only when ``allow_404`` is set."""
    host = urlparse(url).netloc
    last_error: Exception | None = None
    for attempt in range(4):
        _wait_turn(host)
        try:
            response = _session().get(url, timeout=TIMEOUT)
        except requests.RequestException as exc:  # connection reset, timeout
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
            continue
        if response.status_code == 404 and allow_404:
            return None
        if response.status_code == 200:
            return response.content
        last_error = FetchError(f"HTTP {response.status_code} for {url}")
        if response.status_code in (400, 401, 403, 404):
            break
        time.sleep(1.5 * (attempt + 1))
    raise FetchError(f"could not download {url}: {last_error}")


def get_json(url: str, *, cache: Path | None = None, refresh: bool = False, allow_404: bool = False):
    """Download and parse JSON, keeping a copy on disk when ``cache`` is given.

    A cached copy is reused unless ``refresh`` is true, so an interrupted
    download can be restarted without repeating finished requests.
    """
    if cache is not None and cache.exists() and not refresh:
        with cache.open("rb") as handle:
            return json.loads(handle.read().decode("utf-8"))
    body = get_bytes(url, allow_404=allow_404)
    if body is None:
        return None
    data = json.loads(body.decode("utf-8"))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(cache.suffix + ".part")
        tmp.write_bytes(body)
        tmp.replace(cache)
    return data


def download(url: str, dest: Path, *, refresh: bool = False, allow_404: bool = False) -> bool:
    """Download ``url`` to ``dest``. Returns False when the file does not exist upstream."""
    if dest.exists() and dest.stat().st_size > 0 and not refresh:
        return True
    body = get_bytes(url, allow_404=allow_404)
    if body is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(body)
    tmp.replace(dest)
    log.info("downloaded %s (%.1f MB)", dest.name, len(body) / 1e6)
    return True
