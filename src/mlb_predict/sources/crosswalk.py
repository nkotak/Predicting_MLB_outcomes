"""Player id crosswalk between Retrosheet and MLB.

Retrosheet and MLB each give every player their own id. The Chadwick Bureau
publishes a free register that lists both for each person. Using it means a
pitcher's history (Retrosheet) and his upcoming start (MLB) are joined by id
and never by name.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd

from ..config import ensure, raw_dir
from ..http import get_bytes

log = logging.getLogger(__name__)

URL = "https://raw.githubusercontent.com/chadwickbureau/register/master/data/people-{shard}.csv"
SHARDS = "0123456789abcdef"


def path() -> Path:
    return ensure(raw_dir() / "crosswalk") / "retro_to_mlb.csv"


def fetch(*, refresh: bool = False) -> pd.DataFrame:
    """Download the register and keep the rows that carry both ids."""
    if path().exists() and not refresh:
        return load()
    parts = []
    for shard in SHARDS:
        body = get_bytes(URL.format(shard=shard))
        frame = pd.read_csv(
            io.BytesIO(body),
            usecols=["key_mlbam", "key_retro", "name_first", "name_last"],
            dtype=str,
            keep_default_na=False,
        )
        parts.append(frame[(frame["key_mlbam"] != "") & (frame["key_retro"] != "")])
    table = pd.concat(parts, ignore_index=True)
    table["key_mlbam"] = table["key_mlbam"].str.replace(r"\.0$", "", regex=True)
    table = table.drop_duplicates("key_retro").sort_values("key_retro")
    table.to_csv(path(), index=False)
    log.info("player id crosswalk: %d people with both ids", len(table))
    return table


def load() -> pd.DataFrame:
    return pd.read_csv(path(), dtype=str, keep_default_na=False)


def retro_to_mlb() -> dict[str, str]:
    table = load()
    return dict(zip(table["key_retro"], table["key_mlbam"]))
