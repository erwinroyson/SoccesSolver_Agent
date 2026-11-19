import itertools, json
from datetime import date
import numpy as np, pandas as pd
from dagent import config, store
from dagent.decide import decide
from dagent.features import build_features
from dagent.optimize import solve
from dagent.scoring import load_learned, weights_for
from dagent.spec import ProblemSpec


def test_solver_matches_bruteforce():
    rng = np.random.default_rng(1)
    s, c, club = rng.normal(size=12), rng.uniform(1, 10, 12), rng.integers(0, 3, 12)
    got = solve(s, c, club, n=3, budget=15, max_per_club=2)
    best = max((x for x in itertools.combinations(range(12), 3)
                if c[list(x)].sum() <= 15 and max(np.bincount(club[list(x)])) <= 2), key=lambda x: s[list(x)].sum())
    assert got == sorted(best)


def test_no_lookahead(tm):
    d = date(2019, 7, 1)
    a = build_features(tm, d).sort_values("player_id").reset_index(drop=True)
    future = {**tm, "appearances": tm["appearances"].copy()}
    future["appearances"].loc[future["appearances"].date > pd.Timestamp(d), "minutes_played"] = 9999   # tamper with the future
    b = build_features(future, d).sort_values("player_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a[["player_id", "minutes", "goals", "value"]], b[["player_id", "minutes", "goals", "value"]])


def test_decide_respects_constraints(tm):
    spec = ProblemSpec(position="Left-Back", n_players=2, total_budget_eur=4e6, max_age=28, as_of=date(2020, 7, 1))
    r = decide(spec, tm, draws=40)
    assert len(r["picks"]) == 2 and sum(p["market_value_eur"] for p in r["picks"]) <= 4e6
    assert all(p["age"] <= 28 and p["source"] for p in r["picks"])
    assert r["confidence"]["label"] in ("low", "medium", "high") and 0 <= r["confidence"]["rank_stability"] <= 1


def test_old_learned_weights_file_with_style_key_still_loads(tmp_state):
    f = config.DATA_DIR / "learned_weights.json"
    f.write_text(json.dumps({"weights": {"def": {"production": .3, "availability": .6, "discipline": .1, "style": 0.0}}}))
    w = load_learned()
    assert set(w["def"]) == {"production", "availability", "discipline"}
    spec = ProblemSpec(position="Left-Back", total_budget_eur=3e6)
    assert abs(sum(weights_for(spec, w).values()) - 1) < 1e-9


def test_queue_gates_irreversible_actions(tmp_state):
    pid = store.add_proposal("notify", "x", {"body": "hello"})
    assert not (config.OUTBOX / f"proposal_{pid}.md").exists()          # nothing is sent while pending
    store.resolve_proposal(pid, "approve")
    assert (config.OUTBOX / f"proposal_{pid}.md").read_text() == "hello"
    pid2 = store.add_proposal("notify", "y", {"body": "no"})
    store.resolve_proposal(pid2, "reject")
    assert not (config.OUTBOX / f"proposal_{pid2}.md").exists()


def test_edit_then_approve_sends_the_edited_text(tmp_state):
    pid = store.add_proposal("notify", "x", {"body": "draft"})
    store.resolve_proposal(pid, "approve", {"body": "edited by a human"})
    assert (config.OUTBOX / f"proposal_{pid}.md").read_text() == "edited by a human"
