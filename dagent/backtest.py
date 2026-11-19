"""Walk-forward backtest: at each past decision date the agent sees only earlier data; outcomes come from the following 365 days."""
from __future__ import annotations
from datetime import date, timedelta
import numpy as np
import pandas as pd
from .decide import candidate_pool
from .features import build_features
from .optimize import stability, top_sets
from .scoring import weights_for
from .spec import ProblemSpec, COMPONENTS

POSITIONS = ["Left-Back", "Right-Back", "Centre-Back", "Central Midfield", "Centre-Forward"]
METRICS = ("minutes", "starter", "value_growth")


def outcomes(tm: dict, as_of: date):
    """What happened in the 365 days after as_of: minutes per player and his latest market value."""
    ts, end = pd.Timestamp(as_of), pd.Timestamp(as_of) + timedelta(days=365)
    a, v = tm["appearances"], tm["valuations"]
    mins = a[(a.date > ts) & (a.date <= end)].groupby("player_id").minutes_played.sum()
    nxt = v[(v.date > ts) & (v.date <= end)].drop_duplicates("player_id", keep="last").set_index("player_id").market_value_in_eur
    return mins, nxt


def _greedy(order, cost, n, budget):
    out, spent = [], 0.0
    for i in order:
        if spent + cost[i] <= budget:
            out.append(int(i)); spent += cost[i]
        if len(out) == n:
            break
    return out


def _metrics(idx, pool, mins, nxt):
    if not idx:
        return dict.fromkeys(METRICS, np.nan)
    pid = pool.player_id.iloc[idx].to_numpy()
    m = np.array([mins.get(p, 0.0) for p in pid])          # a player missing from next year's data counts as 0 minutes
    g = [np.log(nxt[p] / v) for p, v in zip(pid, pool.value.iloc[idx]) if p in nxt.index and v > 0 and nxt[p] > 0]
    return {"minutes": m.mean(), "starter": (m >= 1500).mean(), "value_growth": np.mean(g) if g else np.nan}


def run(tm, as_ofs, positions=None, budgets=(3e6, 10e6), n=3, max_age=28, draws=60, seed=0, n_random=50, learned=None):
    """learned: {date: {group: weights}}. If given, `agent` uses them and `agent_prior` uses the hand-set weights."""
    rng, last, rows = np.random.default_rng(seed), tm["appearances"].date.max(), []
    for d in as_ofs:
        if pd.Timestamp(d) + timedelta(days=365) > last:
            continue                                        # outcome year not fully observed yet
        feat = build_features(tm, d)
        mins, nxt = outcomes(tm, d)
        for pos in positions or POSITIONS:
            for b in budgets:
                spec = ProblemSpec(position=pos, n_players=n, total_budget_eur=b * n, max_age=max_age, as_of=d)
                w_prior = weights_for(spec)
                w = weights_for(spec, learned.get(d)) if learned else w_prior
                try:
                    pool = candidate_pool(feat, spec, w)
                except ValueError:
                    continue
                if len(pool) < max(30, n):
                    continue
                cost, club, budget = pool.value.to_numpy(float), pool.club_id.fillna(-1).to_numpy(), spec.total_budget_eur
                best = top_sets(pool.score.to_numpy(), cost, club, n, budget, spec.max_per_club, k=1)
                if not best:
                    continue
                st = stability(pool[COMPONENTS].to_numpy(), np.array([w[c] for c in COMPONENTS]), pool.minutes.to_numpy(),
                               cost, club, n, budget, spec.max_per_club, draws=draws, seed=int(rng.integers(1e9)))
                conf = float(np.mean([st["in_topk"].get(i, 0.0) for i in best[0]]))
                methods = {"agent": best[0],
                           "top_value": _greedy(np.argsort(-cost), cost, n, budget),
                           "most_minutes": _greedy(np.argsort(-pool.minutes.to_numpy()), cost, n, budget)}
                if learned:
                    sp = pool[COMPONENTS].to_numpy() @ np.array([w_prior[c] for c in COMPONENTS])
                    prior = top_sets(sp, cost, club, n, budget, spec.max_per_club, k=1)
                    if prior:
                        methods["agent_prior"] = prior[0]
                key = {"date": d, "position": pos, "budget_per_player": b, "confidence": conf}
                for name, idx in methods.items():
                    rows.append({**key, "method": name, **_metrics(idx, pool, mins, nxt)})
                rnd = [_metrics(_greedy(rng.permutation(len(pool)), cost, n, budget), pool, mins, nxt) for _ in range(n_random)]
                rows.append({**key, "method": "random", **{k: float(np.nanmean([r[k] for r in rnd])) for k in METRICS}})
    df = pd.DataFrame(rows)
    return df, summarise(df, rng)


def _year_ci(diff: pd.Series, rng, B=4000):
    """95% interval that resamples WHOLE YEARS (cases within a year share players and market conditions)."""
    years = [d.year for d in diff.index.get_level_values("date")]
    d = pd.DataFrame({"d": diff.to_numpy(float), "y": years}).dropna()
    if d.y.nunique() < 2:
        return (np.nan, np.nan), "n/a"
    g = d.groupby("y").d.agg(["sum", "count"])
    sm, ct = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), (B, len(g)))
    lo, hi = np.percentile(sm[idx].sum(1) / ct[idx].sum(1), [2.5, 97.5])
    return (float(lo), float(hi)), f"{int((d.groupby('y').d.mean() > 0).sum())}/{len(g)}"


def summarise(df: pd.DataFrame, rng) -> dict:
    if df.empty:
        return {"cases": 0}
    key = ["date", "position", "budget_per_player"]
    piv = {m: df[df.method == m].set_index(key) for m in df.method.unique()}
    out = {"cases": int(len(piv["agent"])), "by_method": {}, "agent_minus_baseline": {}, "by_confidence": {}}
    for m, t in piv.items():
        out["by_method"][m] = {k: round(float(t[k].mean()), 3) for k in METRICS}
    for m in [x for x in ("agent_prior", "top_value", "most_minutes", "random") if x in piv]:
        entry = {}
        for k in METRICS:
            diff = piv["agent"][k] - piv[m][k]
            (lo, hi), better = _year_ci(diff, rng)
            entry[k] = {"diff": round(float(diff.mean()), 3), "ci95_by_year": [round(lo, 3), round(hi, 3)], "years_agent_better": better}
        out["agent_minus_baseline"][m] = entry
    a = piv["agent"].copy()
    a["bucket"] = pd.cut(a.confidence, [-.01, .5, .8, 1.0], labels=["low", "medium", "high"])
    out["by_confidence"] = {str(k): {"cases": int(len(g)), "mean_minutes": round(float(g.minutes.mean()), 1)}
                            for k, g in a.groupby("bucket", observed=True)}
    return out


def format_summary(s: dict) -> str:
    if not s.get("cases"):
        return "no cases"
    L = [f"{s['cases']} out-of-sample cases", "", f"{'method':<14}{'next-yr minutes':>16}{'played>=1500':>14}{'value growth':>14}"]
    for m, v in s["by_method"].items():
        L.append(f"{m:<14}{v['minutes']:>16.0f}{v['starter']:>13.0%} {v['value_growth']:>+13.2f}")
    L += ["", "Agent minus baseline (interval resamples whole years; 'yrs' = years in which the agent was ahead):"]
    for m, e in s["agent_minus_baseline"].items():
        parts = []
        for k, lab in (("minutes", "minutes"), ("starter", "starter share"), ("value_growth", "value growth")):
            b, f = e[k], ("{:+.0f}" if k == "minutes" else "{:+.2f}")
            parts.append(f"{lab} {f.format(b['diff'])} [{f.format(b['ci95_by_year'][0])}, {f.format(b['ci95_by_year'][1])}] ({b['years_agent_better']} yrs)")
        L.append(f"  vs {m:<13}" + " | ".join(parts))
    L += ["", "Outcome by confidence bucket: " + ", ".join(f"{k}: {v['cases']} cases -> {v['mean_minutes']:.0f} min" for k, v in s["by_confidence"].items())]
    return "\n".join(L)
