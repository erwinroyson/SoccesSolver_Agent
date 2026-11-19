"""Walk-forward weight learning: fit the score weights on PAST outcomes only, evaluate on LATER years.
For a test date, a training window counts only if its whole 365-day outcome had finished before that date."""
from __future__ import annotations
import json
from datetime import date, timedelta
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from . import config, backtest
from .features import build_features
from .scoring import component_table, WEIGHTS
from .spec import ProblemSpec, COMPONENTS, pos_group
from .store import tool

TRAIN_POS = ["Left-Back", "Right-Back", "Centre-Back", "Defensive Midfield", "Central Midfield", "Attacking Midfield",
             "Left Winger", "Right Winger", "Centre-Forward"]
GROUPS = ("def", "mid", "att")      # goalkeepers keep the hand-set weights


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and sd == sd else s * 0.0


@tool("training_table")
def training_table(tm: dict, as_of: date, blend: bool = True, max_value: float = 15e6) -> pd.DataFrame:
    """One row per regular player in a past window: his three score ingredients (X) and what happened next (y).
    y = z(next-year minutes), averaged with z(log value growth) when `blend`."""
    try:
        feat = build_features(tm, as_of)
    except Exception:
        return pd.DataFrame()               # window before the dataset has coverage
    if len(feat) < 200:
        return pd.DataFrame()
    mins, nxt = backtest.outcomes(tm, as_of)
    parts = []
    for pos in TRAIN_POS:
        try:
            comp = component_table(feat, ProblemSpec(position=pos, total_budget_eur=1e12, as_of=as_of))
        except ValueError:
            continue
        comp = comp[comp.value <= max_value]          # learn from the market the agent actually shops in
        if len(comp) < 30:
            continue
        m = comp.player_id.map(lambda p: float(mins.get(p, 0.0)))
        g = np.log(comp.player_id.map(nxt) / comp.value)
        t = comp[COMPONENTS].copy()
        t["y"] = pd.concat([_z(m)] + ([_z(g)] if blend else []), axis=1).mean(axis=1)
        t["group"] = pos_group(pos)
        parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fit_weights(tbl: pd.DataFrame, min_rows: int = 150):
    """Non-negative ridge regression of the outcome on the ingredients, per position group. Weights = coefficients / their sum."""
    out, diag = {}, {}
    for grp in GROUPS:
        sub = tbl[tbl.group == grp].dropna(subset=["y"]) if len(tbl) else tbl
        if len(sub) < min_rows:
            out[grp], diag[grp] = dict(WEIGHTS[grp]), {"rows": int(len(sub)), "note": "too little data, hand-set weights kept"}
            continue
        X, y = sub[COMPONENTS].to_numpy(float), sub.y.to_numpy(float)
        lam = 0.01 * len(sub)
        coef, _ = nnls(np.vstack([X, np.sqrt(lam) * np.eye(len(COMPONENTS))]), np.concatenate([y, np.zeros(len(COMPONENTS))]))
        if coef.sum() <= 1e-9:
            out[grp], diag[grp] = dict(WEIGHTS[grp]), {"rows": int(len(sub)), "note": "no signal found, hand-set weights kept"}
            continue
        out[grp] = {c: float(v / coef.sum()) for c, v in zip(COMPONENTS, coef)}
        diag[grp] = {"rows": int(len(sub)), "corr_score_vs_outcome": round(float(np.corrcoef(X @ coef, y)[0, 1]), 3)}
    return out, diag


def run(tm: dict, test_years: list[int], train_start: int = 2013, blend: bool = True, **backtest_kwargs):
    """Returns (cases, summary, weights_per_test_date, final_weights, diagnostics)."""
    last = tm["appearances"].date.max()
    dates = [date(y, 7, 1) for y in range(train_start, last.year + 1) if pd.Timestamp(date(y, 7, 1)) + timedelta(days=365) <= last]
    tables = {d: training_table(tm, d, blend) for d in dates}
    learned, diags = {}, {}
    for y in test_years:
        td = date(y, 7, 1)
        used = [d for d in dates if pd.Timestamp(d) + timedelta(days=365) <= pd.Timestamp(td) and len(tables[d])]
        if used:
            learned[td], diags[td] = fit_weights(pd.concat([tables[d] for d in used], ignore_index=True))
            diags[td]["train_dates"] = [str(d) for d in used]
    cases, summ = backtest.run(tm, sorted(learned), learned=learned, **backtest_kwargs)
    allt = [t for t in tables.values() if len(t)]
    final, fdiag = fit_weights(pd.concat(allt, ignore_index=True)) if allt else (None, {})
    return cases, summ, learned, final, {"by_test_date": diags, "final": fdiag, "train_windows": [str(d) for d in dates]}


def save(final: dict, meta: dict):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    (config.DATA_DIR / "learned_weights.json").write_text(json.dumps({"weights": final, "meta": meta}, indent=2), encoding="utf-8")
