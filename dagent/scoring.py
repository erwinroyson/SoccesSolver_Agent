"""Score = weighted sum of three z-scored ingredients: production, availability, discipline."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from . import config
from .features import league_strength
from .spec import ProblemSpec, pos_group, COMPONENTS

# Hand-set starting weights per position group. `python -m dagent learn` replaces them with weights fitted on past outcomes.
WEIGHTS = {"def": {"production": 0.23, "availability": 0.54, "discipline": 0.23},
           "mid": {"production": 0.38, "availability": 0.46, "discipline": 0.16},
           "att": {"production": 0.64, "availability": 0.29, "discipline": 0.07},
           "gk": {"production": 0.00, "availability": 0.88, "discipline": 0.12}}


def _z(x: pd.Series, ref: pd.Series) -> pd.Series:
    sd = ref.std(ddof=0)
    return (x - ref.mean()) / sd if sd and not np.isnan(sd) else x * 0.0


def component_table(feat: pd.DataFrame, spec: ProblemSpec) -> pd.DataFrame:
    """z-scores for everyone who plays the requested position, measured against ALL regulars in that position
    (not just the affordable ones), so a score never depends on the budget."""
    ref = feat[(feat.sub_position == spec.position) & (feat.minutes >= spec.min_minutes)].copy()
    if len(ref) < 20:
        raise ValueError(f"Only {len(ref)} {spec.position}s with >= {spec.min_minutes} min in the window: too little evidence.")
    strength = league_strength(feat)
    target = strength.get(spec.target_league, strength.mean()) if spec.target_league else strength.mean()
    # production in a weaker league is discounted, in a stronger league slightly boosted
    adj = np.exp(config.LEAGUE_ALPHA * (ref.league.map(strength).fillna(strength.min()) - target))
    m = ref.minutes
    prod = (ref.goals + 0.7 * ref.assists) / m * 90 * adj
    cards = -(ref.yellow + 3 * ref.red) / m * 90
    s = m / (m + config.SHRINK_MIN)        # few minutes -> pull towards the position average (a lucky streak is not skill)
    prod_s, cards_s = s * prod + (1 - s) * prod.mean(), s * cards + (1 - s) * cards.mean()
    comp = pd.DataFrame({"production": _z(prod_s, prod_s), "availability": _z(np.log(m), np.log(m)),
                         "discipline": _z(cards_s, cards_s)}, index=ref.index)
    return ref.join(comp)


def load_learned() -> dict | None:
    """Weights saved by `python -m dagent learn`: {group: {component: weight}}, or None if never run."""
    f = config.DATA_DIR / "learned_weights.json"
    if not f.exists():
        return None
    w = json.loads(f.read_text())["weights"]
    return {g: {c: ws.get(c, 0.0) for c in COMPONENTS} for g, ws in w.items()}   # tolerate files from older versions


def weights_for(spec: ProblemSpec, learned: dict | None = None) -> dict[str, float]:
    grp = pos_group(spec.position)
    w = dict(learned[grp]) if learned and grp in learned else dict(WEIGHTS[grp])
    for k, mult in spec.weight_overrides.items():
        w[k] *= mult
    tot = sum(w.values()) or 1.0
    return {k: v / tot for k, v in w.items()}


def score(comp: pd.DataFrame, w: dict[str, float]) -> pd.Series:
    return sum(comp[k] * w[k] for k in COMPONENTS)
