"""TEST FIXTURES ONLY. The agent never uses synthetic data at runtime; these frames exist so unit tests run offline,
and have the same schema as the Transfermarkt public dataset tables the loader returns."""
import numpy as np, pandas as pd, pytest
from dagent import config

POS = ["Left-Back", "Right-Back", "Centre-Back", "Central Midfield", "Centre-Forward"]


@pytest.fixture(autouse=True)
def tmp_state(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "state.db")
    monkeypatch.setattr(config, "OUTBOX", tmp_path / "outbox")


def make_tm(seed=0, per_pos=70, seasons=range(2016, 2024)):
    rng = np.random.default_rng(seed)
    players, vals, apps = [], [], []
    pid = 0
    for pos in POS:
        for _ in range(per_pos):
            pid += 1
            q = rng.normal()                                    # latent quality (hidden from the agent)
            lg = rng.choice(["ES1", "GB1", "NL1"], p=[.4, .35, .25])
            dob = pd.Timestamp("1995-01-01") + pd.Timedelta(days=int(rng.integers(0, 365 * 8)))
            players.append((pid, f"Player {pid}", dob, "Defender", pos, "right"))
            for s in seasons:
                base = 10 ** (6.0 + 0.35 * q + rng.normal(0, .15))
                vals.append((pid, pd.Timestamp(f"{s}-06-15"), base))
                p_play = 1 / (1 + np.exp(-(q * 1.2 - 0.2)))
                for g in range(34):
                    d = pd.Timestamp(f"{s}-08-15") + pd.Timedelta(days=int(g * 8.5))
                    if rng.random() < p_play:
                        m = int(rng.integers(50, 91))
                        rate = (0.05 + 0.04 * max(q, -1)) * (0.3 if "Back" in pos else 1.0)
                        apps.append((pid, 1000 + pid % 12, d, lg, int(rng.poisson(rate)), int(rng.poisson(rate / 2)),
                                     int(rng.random() < .1), 0, m))
    return {
        "players": pd.DataFrame(players, columns=["player_id", "name", "date_of_birth", "position", "sub_position", "foot"]),
        "valuations": pd.DataFrame(vals, columns=["player_id", "date", "market_value_in_eur"]).sort_values("date"),
        "appearances": pd.DataFrame(apps, columns=["player_id", "player_club_id", "date", "competition_id", "goals", "assists",
                                                   "yellow_cards", "red_cards", "minutes_played"]),
    }


@pytest.fixture(scope="session")
def tm():
    return make_tm()
