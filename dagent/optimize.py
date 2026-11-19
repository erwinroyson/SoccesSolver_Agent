"""Choosing players is an integer program; confidence comes from re-solving it under perturbed inputs."""
from __future__ import annotations
from collections import Counter
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from .store import tool


def solve(score, cost, club, n, budget, max_per_club, banned=None):
    """Maximise total score with exactly n players, total cost <= budget, at most max_per_club from one club.
    `banned` = sets already found (adds a cut so the next-best set is returned)."""
    N = len(score)
    A, lo, hi = [np.ones(N), cost], [n, 0], [n, budget]
    for c in np.unique(club):
        row = (club == c).astype(float)
        if row.sum() > max_per_club:
            A.append(row); lo.append(0); hi.append(max_per_club)
    for S in banned or []:
        row = np.zeros(N); row[list(S)] = 1
        A.append(row); lo.append(0); hi.append(len(S) - 1)
    res = milp(c=-np.asarray(score), constraints=LinearConstraint(np.vstack(A), lo, hi), integrality=np.ones(N), bounds=Bounds(0, 1))
    return sorted(int(i) for i in np.flatnonzero(res.x > 0.5)) if res.success else None


@tool("solve")
def top_sets(score, cost, club, n, budget, max_per_club, k=3):
    sets, banned = [], []
    for _ in range(k):
        s = solve(score, cost, club, n, budget, max_per_club, banned)
        if not s:
            break
        sets.append(s); banned.append(set(s))
    return sets


@tool("confidence")
def stability(comp_matrix, base_w, minutes, cost, club, n, budget, max_per_club, draws=200, seed=0) -> dict:
    """Re-run the decision `draws` times with perturbed weights and noisy scores (more noise for players with few minutes).
    Returns how often each player is in the optimal set and how often he stays in the top-K of the ranking."""
    rng = np.random.default_rng(seed)
    noise = 0.3 * np.sqrt(900 / np.maximum(minutes, 90))
    K = max(5, 3 * n)
    in_set, in_topk = Counter(), Counter()
    for _ in range(draws):
        sc = comp_matrix @ rng.dirichlet(base_w * 20 + 1e-3) + rng.normal(0, noise)
        in_topk.update(int(i) for i in np.argsort(-sc)[:K])
        s = solve(sc, cost, club, n, budget, max_per_club)
        if s:
            in_set.update(s)
    return {"in_set": {i: c / draws for i, c in in_set.items()}, "in_topk": {i: c / draws for i, c in in_topk.items()}, "k": K}


def label(p: float) -> str:
    return "high" if p >= 0.8 else "medium" if p >= 0.5 else "low"
