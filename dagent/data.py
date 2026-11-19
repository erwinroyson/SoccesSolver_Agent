"""Transfermarkt public dataset: download (cached), freshness probe, load."""
from __future__ import annotations
import shutil
from pathlib import Path
import pandas as pd
import requests
from . import config
from .store import tool


def _path(name: str) -> Path:
    """Accepts the downloaded .csv.gz, or a plain .csv placed manually in data/raw (e.g. from Kaggle)."""
    gz, plain = config.RAW_DIR / f"tm_{name}.csv.gz", config.RAW_DIR / f"tm_{name}.csv"
    return plain if plain.exists() and not gz.exists() else gz


def data_stamp() -> str:
    """Cheap 'did the data change?' probe: the HTTP validators of the players file."""
    r = requests.head(f"{config.TM_BASE_URL}/players.csv.gz", timeout=30, allow_redirects=True)
    r.raise_for_status()
    return r.headers.get("ETag") or r.headers.get("Last-Modified") or r.headers.get("Content-Length", "")


@tool("download")
def download(force: bool = False) -> list[str]:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    got = []
    for name in config.TM_FILES:
        dest = _path(name)
        if dest.exists() and not force:
            continue
        with requests.get(f"{config.TM_BASE_URL}/{name}.csv.gz", stream=True, timeout=120) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        tmp.replace(dest)
        got.append(name)
    return got


@tool("load")
def load() -> dict[str, pd.DataFrame]:
    for name in config.TM_FILES:
        if not _path(name).exists():
            raise FileNotFoundError(f"{_path(name)} missing - run `python -m dagent fetch`")
    players = pd.read_csv(_path("players"), usecols=["player_id", "name", "date_of_birth", "position", "sub_position", "foot"],
                          parse_dates=["date_of_birth"])
    vals = pd.read_csv(_path("player_valuations"), usecols=["player_id", "date", "market_value_in_eur"], parse_dates=["date"])
    apps = pd.read_csv(_path("appearances"), parse_dates=["date"],
                       usecols=["player_id", "player_club_id", "date", "competition_id", "goals", "assists",
                                "yellow_cards", "red_cards", "minutes_played"])
    return {"players": players, "valuations": vals.dropna().sort_values("date"), "appearances": apps}
