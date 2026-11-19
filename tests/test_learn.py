from datetime import date, timedelta
import pandas as pd
from dagent import learn


def test_walkforward_never_trains_on_unfinished_outcomes(tm):
    cases, summ, learned, final, diag = learn.run(tm, [2020, 2021, 2022], train_start=2017, n=3, draws=20, n_random=20)
    assert learned
    for td, d in diag["by_test_date"].items():
        for tr in d["train_dates"]:      # the whole outcome year of every training window ended on/before the test date
            assert pd.Timestamp(tr) + timedelta(days=365) <= pd.Timestamp(td)
    for grp in learned.values():
        for g in ("def", "mid", "att"):
            assert abs(sum(grp[g].values()) - 1) < 1e-6 and all(v >= 0 for v in grp[g].values())
    assert "agent_prior" in summ["by_method"]


def test_fit_recovers_planted_signal(tm):
    tbl = pd.concat([learn.training_table(tm, date(y, 7, 1), blend=False) for y in (2017, 2018, 2019, 2020, 2021)], ignore_index=True)
    w, diag = learn.fit_weights(tbl)
    assert w["def"]["availability"] > 0.3 and diag["def"]["corr_score_vs_outcome"] > 0.2
