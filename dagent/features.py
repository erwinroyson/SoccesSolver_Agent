"""Player table as of a decision date. Only rows dated on or before that date are used (no look-ahead)."""
from __future__ import annotations
from datetime import date, timedelta
import numpy as np
import pandas as pd
from . import config
from .spec import pos_group
from .store import tool

LEAGUE_RE = r"^[A-Z]{1,3}1$"  # Transfermarkt top-flight ids: ES1, GB1, L1, IT1, FR1, ...


def league_strength(feat: pd.DataFrame) -> pd.Series:
    """Proxy for league strength: log of the mean market value of that league's regulars."""
    return np.log1p(feat[feat.minutes >= 900].groupby("league").value.mean())


@tool("build_features")
def build_features(tm: dict, as_of: date) -> pd.DataFrame:
    ts = pd.Timestamp(as_of)
    a = tm["appearances"]
    a = a[(a.date > ts - timedelta(days=config.WINDOW_DAYS)) & (a.date <= ts)]
    g = a.groupby("player_id").agg(minutes=("minutes_played", "sum"), goals=("goals", "sum"), assists=("assists", "sum"),
                                   yellow=("yellow_cards", "sum"), red=("red_cards", "sum"))
    # main league = the domestic top flight where he played most minutes; club = the one of his latest appearance
    dom = a[a.competition_id.str.match(LEAGUE_RE, na=False)]
    if len(dom):
        g["league"] = dom.groupby(["player_id", "competition_id"]).minutes_played.sum().reset_index() \
                         .sort_values("minutes_played").drop_duplicates("player_id", keep="last") \
                         .set_index("player_id").competition_id
    else:
        g["league"] = None
    g["club_id"] = a.sort_values("date").drop_duplicates("player_id", keep="last").set_index("player_id").player_club_id
    # latest market value on or before the date (cost proxy)
    v = tm["valuations"]
    v = v[v.date <= ts].drop_duplicates("player_id", keep="last").set_index("player_id")
    g["value"], g["value_date"] = v.market_value_in_eur, v.date
    g = g.join(tm["players"].set_index("player_id")[["name", "date_of_birth", "position", "sub_position", "foot"]], how="inner")
    g["age"] = (ts - g.date_of_birth).dt.days / 365.25
    fresh = (ts - g.value_date).dt.days <= config.VALUE_MAX_AGE_DAYS
    g = g[g.value.notna() & fresh & (g.minutes > 0)].reset_index().rename(columns={"index": "player_id"})
    g["group"] = g.sub_position.map(lambda p: pos_group(p) if isinstance(p, str) else "att")
    return g
