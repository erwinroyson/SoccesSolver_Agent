"""The agent loop: ask (parse -> decide -> explain -> remember -> propose) and watch (re-run when the data changes)."""
from __future__ import annotations
import json, time
from datetime import date
from . import data, store
from .decide import decide
from .llm import explain, parse_request


def ask(text: str, draws: int = 200, as_of: date | None = None, tm=None, mandate_id=None) -> dict:
    spec, tin, tout = parse_request(text)                    # LLM job 1: sentence -> formal problem
    if as_of:
        spec.as_of = as_of
    tm = tm if tm is not None else data.load()
    run_id = store.add_run(mandate_id, text, spec.model_dump(mode="json"), {})
    store.set_run(run_id)                                    # tool calls below are logged against this run
    try:
        res = decide(spec, tm, draws)                        # the decision: scoring + optimiser + confidence (no LLM)
        narrative, t2i, t2o = explain(res)                   # LLM job 2: result -> prose
    finally:
        store.set_run(None)
    res["explanation"] = narrative
    store.finish_run(run_id, res, tin + t2i, tout + t2o)
    if mandate_id is None:                                   # first time we see this request: offer to keep watching it
        with store.db() as c:
            row = c.execute("SELECT id FROM mandates WHERE text=? ORDER BY id LIMIT 1", (text,)).fetchone()
        if row:                                              # same request asked again: reuse it, don't pile up proposals
            mandate_id = row["id"]
            with store.db() as c:
                c.execute("UPDATE mandates SET last_run_id=? WHERE id=?", (run_id, mandate_id))
        else:
            mandate_id = store.add_mandate(text, spec.model_dump(mode="json"))
            store.add_proposal("watch_mandate", f"Keep watching: {text}", {"text": text}, mandate_id)
            with store.db() as c:
                c.execute("UPDATE mandates SET last_run_id=? WHERE id=?", (run_id, mandate_id))
    res["run_id"], res["mandate_id"] = run_id, mandate_id
    return res


def _names(res: dict) -> list[str]:
    return [p["name"] for p in res["picks"]]


def watch_once(force: bool = False) -> list[str]:
    """Re-run every approved mandate if the data changed (or force). Queue a proposal if the recommendation moved."""
    with store.db() as c:
        mandates = [dict(r) for r in c.execute("SELECT * FROM mandates WHERE active=1")]
    if not mandates:
        return ["no watched mandates yet (approve a 'watch_mandate' proposal first)"]
    stamp = data.data_stamp()
    log = []
    for m in mandates:
        if not force and stamp == m["last_data_stamp"]:
            log.append(f"mandate {m['id']}: data unchanged, skipped")
            continue
        data.download(force=True)
        tm = data.load()
        with store.db() as c:
            prev = c.execute("SELECT result FROM runs WHERE id=?", (m["last_run_id"],)).fetchone()
        before = _names(json.loads(prev["result"])) if prev and prev["result"] else []
        res = ask(m["text"], tm=tm, mandate_id=m["id"])
        with store.db() as c:
            c.execute("UPDATE mandates SET last_run_id=?, last_data_stamp=? WHERE id=?", (res["run_id"], stamp, m["id"]))
        if _names(res) != before:
            body = f"# Shortlist changed for: {m['text']}\n\nBefore: {', '.join(before) or 'n/a'}\nNow: {', '.join(_names(res))}\n\n{res['explanation']}\n"
            pid = store.add_proposal("notify", f"Shortlist changed: {m['text']}", {"body": body}, m["id"])
            log.append(f"mandate {m['id']}: recommendation changed -> proposal {pid} awaiting approval")
        else:
            log.append(f"mandate {m['id']}: re-run, recommendation unchanged")
    return log


def watch_loop(hours: float):
    while True:
        print("\n".join(watch_once()), flush=True)
        time.sleep(hours * 3600)
