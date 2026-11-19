from datetime import date
from dagent import backtest


def test_backtest_runs_and_agent_beats_random_on_planted_signal(tm):
    df, s = backtest.run(tm, [date(2018, 7, 1), date(2019, 7, 1), date(2020, 7, 1), date(2021, 7, 1)],
                         positions=["Left-Back", "Centre-Forward"], budgets=(3e6, 8e6), n=3, draws=20, n_random=20)
    assert s["cases"] >= 6
    e = s["agent_minus_baseline"]["random"]["minutes"]
    assert e["diff"] > 0 and e["ci95_by_year"][0] <= e["diff"] <= e["ci95_by_year"][1]
    assert "vs random" in backtest.format_summary(s)
