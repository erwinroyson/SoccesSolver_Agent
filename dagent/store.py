"""Persistent memory (SQLite): mandates, runs, tool-call log, and the human approval queue."""
from __future__ import annotations
import json, sqlite3, time, functools
from contextlib import contextmanager
from datetime import datetime, timezone
from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS mandates(id INTEGER PRIMARY KEY, text TEXT, spec TEXT, active INTEGER DEFAULT 0,
  last_run_id INTEGER, last_data_stamp TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY, mandate_id INTEGER, text TEXT, spec TEXT, result TEXT,
  tokens_in INTEGER DEFAULT 0, tokens_out INTEGER DEFAULT 0, cost_usd REAL DEFAULT 0, created TEXT);
CREATE TABLE IF NOT EXISTS proposals(id INTEGER PRIMARY KEY, kind TEXT, title TEXT, payload TEXT,
  status TEXT DEFAULT 'pending', mandate_id INTEGER, created TEXT, decided TEXT);
CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY, run_id INTEGER, tool TEXT, ms REAL, ok INTEGER,
  summary TEXT, ts TEXT);
"""
_current_run: list[int | None] = [None]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def db():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def set_run(run_id: int | None):
    _current_run[0] = run_id


def tool(name: str):
    """Decorator: every tool call is logged (duration, success, short summary)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **k):
            t0, ok, out = time.time(), 1, None
            try:
                out = fn(*a, **k)
                return out
            except Exception:
                ok = 0
                raise
            finally:
                try:
                    summ = (f"rows={len(out)}" if hasattr(out, "__len__") else str(type(out).__name__))
                    with db() as c:
                        c.execute("INSERT INTO calls(run_id,tool,ms,ok,summary,ts) VALUES(?,?,?,?,?,?)",
                                  (_current_run[0], name, (time.time() - t0) * 1000, ok, summ, now()))
                except Exception:
                    pass
        return wrap
    return deco


def add_run(mandate_id, text, spec: dict, result: dict, tin=0, tout=0) -> int:
    cost = tin / 1e6 * config.PRICE_IN + tout / 1e6 * config.PRICE_OUT
    with db() as c:
        cur = c.execute("INSERT INTO runs(mandate_id,text,spec,result,tokens_in,tokens_out,cost_usd,created) "
                        "VALUES(?,?,?,?,?,?,?,?)",
                        (mandate_id, text, json.dumps(spec), json.dumps(result, default=str), tin, tout, cost, now()))
        return cur.lastrowid


def add_proposal(kind: str, title: str, payload: dict, mandate_id=None) -> int:
    with db() as c:
        cur = c.execute("INSERT INTO proposals(kind,title,payload,mandate_id,created) VALUES(?,?,?,?,?)",
                        (kind, title, json.dumps(payload, default=str), mandate_id, now()))
        return cur.lastrowid


def add_mandate(text: str, spec: dict) -> int:
    with db() as c:
        cur = c.execute("INSERT INTO mandates(text,spec,created) VALUES(?,?,?)", (text, json.dumps(spec), now()))
        return cur.lastrowid


# ---- human-in-the-loop queue: nothing irreversible runs without an explicit approve ----
import json  # noqa: E402


def pending():
    with db() as c:
        return [dict(r) for r in c.execute("SELECT * FROM proposals WHERE status='pending' ORDER BY id")]


def resolve_proposal(pid: int, action: str, edit: dict | None = None) -> str:
    assert action in ("approve", "reject")
    with db() as c:
        r = c.execute("SELECT * FROM proposals WHERE id=?", (pid,)).fetchone()
        if not r or r["status"] != "pending":
            return "not pending"
        payload = json.loads(r["payload"])
        if edit:
            payload.update(edit)
            c.execute("UPDATE proposals SET payload=? WHERE id=?", (json.dumps(payload), pid))
        status = "approved" if action == "approve" else "rejected"
        c.execute("UPDATE proposals SET status=?, decided=? WHERE id=?", (status, now(), pid))
        mandate_id = r["mandate_id"]
    if action == "approve":
        return _execute(r["kind"], payload, mandate_id, pid)
    return "rejected"


def _execute(kind: str, payload: dict, mandate_id, pid: int) -> str:
    if kind == "watch_mandate":                      # reversible, but still gated: starts autonomous re-runs
        with db() as c:
            c.execute("UPDATE mandates SET active=1 WHERE id=?", (mandate_id,))
        return f"mandate {mandate_id} is now watched"
    if kind == "notify":                             # 'send' = write to the outbox (the real send is out of scope)
        config.OUTBOX.mkdir(parents=True, exist_ok=True)
        f = config.OUTBOX / f"proposal_{pid}.md"
        f.write_text(payload.get("body", ""), encoding="utf-8")
        return f"written to {f}"
    return "unknown kind"


def finish_run(run_id: int, result: dict, tin: int, tout: int):
    cost = tin / 1e6 * config.PRICE_IN + tout / 1e6 * config.PRICE_OUT
    with db() as c:
        c.execute("UPDATE runs SET result=?, tokens_in=?, tokens_out=?, cost_usd=? WHERE id=?",
                  (json.dumps(result, default=str), tin, tout, cost, run_id))
