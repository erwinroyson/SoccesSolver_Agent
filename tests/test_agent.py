from datetime import date
from dagent import agent, store
from dagent.spec import ProblemSpec


def test_ask_end_to_end_with_stubbed_llm(tm, monkeypatch):
    spec = ProblemSpec(position="Left-Back", total_budget_eur=3e6, as_of=date(2021, 7, 1))
    monkeypatch.setattr(agent, "parse_request", lambda t: (spec, 10, 5))      # no network in tests
    monkeypatch.setattr(agent, "explain", lambda r: ("because", 20, 7))
    res = agent.ask("a left back under 3M", draws=30, tm=tm)
    assert res["explanation"] == "because" and res["picks"]
    pending = store.pending()
    assert [p["kind"] for p in pending] == ["watch_mandate"]                   # follow-up action is only PROPOSED
    store.resolve_proposal(pending[0]["id"], "approve")
    with store.db() as c:
        assert c.execute("SELECT active FROM mandates").fetchone()[0] == 1
        assert c.execute("SELECT tokens_in, tokens_out FROM runs").fetchone()[:] == (30, 12)
        assert c.execute("SELECT count(*) FROM calls WHERE tool='confidence'").fetchone()[0] >= 1


def test_asking_the_same_request_twice_does_not_duplicate_proposals(tm, monkeypatch):
    spec = ProblemSpec(position="Left-Back", total_budget_eur=3e6, as_of=date(2021, 7, 1))
    monkeypatch.setattr(agent, "parse_request", lambda t: (spec, 0, 0))
    monkeypatch.setattr(agent, "explain", lambda r: ("ok", 0, 0))
    a, b = agent.ask("same", draws=20, tm=tm), agent.ask("same", draws=20, tm=tm)
    assert a["mandate_id"] == b["mandate_id"] and len(store.pending()) == 1
