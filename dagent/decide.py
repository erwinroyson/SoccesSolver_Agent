"""The decision procedure: problem spec + data -> optimised, source-cited recommendation with a confidence level. No LLM in here."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config
from .features import build_features
from .optimize import top_sets, stability, label
from .scoring import component_table, weights_for, score, load_learned
from .spec import ProblemSpec, COMPONENTS

POOL_CAP = 1000


def candidate_pool(feat: pd.DataFrame, spec: ProblemSpec, w: dict) -> pd.DataFrame:
    comp = component_table(feat, spec)
    comp["score"] = score(comp, w)
    ok = (comp.value <= spec.total_budget_eur) & comp.age.between(spec.min_age, spec.max_age)
    return comp[ok].sort_values("score", ascending=False).head(POOL_CAP).reset_index(drop=True)


def decide(spec: ProblemSpec, tm: dict, draws: int = 200) -> dict:
    as_of = spec.as_of or tm["appearances"].date.max().date()
    feat = build_features(tm, as_of)
    learned = load_learned()
    w = weights_for(spec, learned)
    pool = candidate_pool(feat, spec, w)
    if len(pool) < spec.n_players:
        raise ValueError(f"Only {len(pool)} candidates satisfy the constraints; relax budget, age or minutes.")
    cost, club = pool.value.to_numpy(float), pool.club_id.fillna(-1).to_numpy()
    sets = top_sets(pool.score.to_numpy(), cost, club, spec.n_players, spec.total_budget_eur, spec.max_per_club)
    if not sets:
        raise ValueError("No feasible selection under the constraints.")
    st = stability(pool[COMPONENTS].to_numpy(), np.array([w[c] for c in COMPONENTS]), pool.minutes.to_numpy(), cost, club,
                   spec.n_players, spec.total_budget_eur, spec.max_per_club, draws=draws)
    best = sets[0]
    rank_p = float(np.mean([st["in_topk"].get(i, 0.0) for i in best]))

    def card(i: int) -> dict:
        r = pool.loc[i]
        return {"player_id": int(r.player_id), "name": r["name"], "position": r.sub_position, "age": round(float(r.age), 1),
                "league": r.league, "club_id": None if pd.isna(r.club_id) else int(r.club_id),
                "market_value_eur": float(r.value), "minutes_12m": float(r.minutes), "goals": int(r.goals),
                "assists": int(r.assists), "score": round(float(r.score), 3),
                "contributions": {c: round(float(r[c] * w[c]), 3) for c in COMPONENTS},
                "optimal_in": round(st["in_set"].get(i, 0.0), 2), "top_k_in": round(st["in_topk"].get(i, 0.0), 2),
                "confidence": label(st["in_topk"].get(i, 0.0)), "source": config.TM_SOURCE}
    return {
        "as_of": str(as_of), "weights": {k: round(v, 3) for k, v in w.items()},
        "picks": [card(i) for i in best],
        "confidence": {"label": label(rank_p), "rank_stability": round(rank_p, 2), "top_k": st["k"],
                       "meaning": f"share of 200 perturbed scenarios in which the picks stayed within the top {st['k']} candidates (not a rank)"},
        "alternatives": [[pool.loc[i, "name"] for i in s] for s in sets[1:]],
        "coverage": {"candidates": int(len(pool)),
                     "reference_players": int(((feat.sub_position == spec.position) & (feat.minutes >= spec.min_minutes)).sum())},
        "assumptions": spec.assumptions + [
            "Cost = Transfermarkt market value at the decision date (no fee or wage data in public sources).",
            f"League adjustment: production scaled by exp({config.LEAGUE_ALPHA} x league-strength gap); strength = log mean value of regulars.",
            "Weights learned walk-forward from past outcomes (data/learned_weights.json)." if learned
            else "Weights are hand-set priors; run `python -m dagent learn` to fit them on past outcomes.",
            "No event data (passes, tackles, ...): playing style cannot be scored, so stylistic requests are not applied.",
        ],
        "sources": [config.TM_SOURCE],
    }
