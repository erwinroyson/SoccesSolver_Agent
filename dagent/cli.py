from __future__ import annotations
import argparse, json, os
from datetime import date, datetime
from . import config, store


def _print(res: dict):
    print(f"\nRECOMMENDED (as of {res['as_of']}, budget applies to the combined market value)")
    print(f"  {'player':<26}{'age':>5}{'league':>8}{'value EUR':>12}{'minutes':>9}{'score':>7}{'in top-K':>10}")
    for p in res["picks"]:
        print(f"  {p['name'][:25]:<26}{p['age']:>5.0f}{str(p['league']):>8}{p['market_value_eur']:>12,.0f}{p['minutes_12m']:>9.0f}"
              f"{p['score']:>7.2f}{p['top_k_in']:>10.0%}")
    if res["alternatives"]:
        print("  next-best sets: " + "; ".join(", ".join(a) for a in res["alternatives"]))
    print("\n" + res["explanation"])
    print(f"\nConfidence: {res['confidence']}\nWeights: {res['weights']}\nCoverage: {res['coverage']}")
    print("Assumptions:\n  - " + "\n  - ".join(res["assumptions"]))
    print(f"\n(run {res['run_id']}; a 'watch' proposal is waiting in `queue`)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dagent")
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fetch", help="download the Transfermarkt data"); f.add_argument("--force", action="store_true")
    a = sp.add_parser("ask"); a.add_argument("text"); a.add_argument("--json", action="store_true")
    a.add_argument("--as-of", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date())
    sp.add_parser("queue")
    ap_ = sp.add_parser("approve", help="approve a proposal (optionally edit its text first)")
    ap_.add_argument("id", type=int); ap_.add_argument("--edit-body", help="replace the notification text before approving")
    sp.add_parser("reject").add_argument("id", type=int)
    b = sp.add_parser("backtest"); b.add_argument("--years", default="2017-2023"); b.add_argument("--n", type=int, default=3)
    l = sp.add_parser("learn", help="learn the weights walk-forward and test them on later years")
    l.add_argument("--test-years", default="2019-2023"); l.add_argument("--target", choices=["blend", "minutes"], default="blend")
    l.add_argument("--n", type=int, default=3); l.add_argument("--no-save", action="store_true")
    w = sp.add_parser("watch"); w.add_argument("--once", action="store_true"); w.add_argument("--force", action="store_true")
    w.add_argument("--hours", type=float, default=6)
    sp.add_parser("log"); sp.add_parser("llmcheck", help="test the LLM connection")
    args = ap.parse_args(argv)

    if args.cmd == "fetch":
        from . import data
        print("downloaded:", data.download(args.force) or "already cached")
    elif args.cmd == "ask":
        from . import agent
        res = agent.ask(args.text, as_of=args.as_of)
        print(json.dumps(res, indent=2, default=str)) if args.json else _print(res)
    elif args.cmd == "queue":
        rows = store.pending()
        for p in rows:
            print(f"[{p['id']}] {p['kind']}: {p['title']}")
        if not rows:
            print("(empty)")
    elif args.cmd in ("approve", "reject"):
        edit = {"body": args.edit_body} if getattr(args, "edit_body", None) else None
        print(store.resolve_proposal(args.id, args.cmd, edit))
    elif args.cmd == "backtest":
        from . import backtest, data
        y0, y1 = map(int, args.years.split("-"))
        df, summ = backtest.run(data.load(), [date(y, 7, 1) for y in range(y0, y1 + 1)], n=args.n)
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.DATA_DIR / "backtest_cases.csv", index=False)
        print(backtest.format_summary(summ))
    elif args.cmd == "learn":
        from . import backtest, data, learn
        y0, y1 = map(int, args.test_years.split("-"))
        cases, summ, learned, final, diag = learn.run(data.load(), list(range(y0, y1 + 1)), blend=args.target == "blend", n=args.n)
        print("WEIGHTS LEARNED FOR EACH TEST YEAR (trained only on earlier outcomes)")
        for d, grp in learned.items():
            print(f"  {d}: " + " | ".join(f"{g}: " + ", ".join(f"{c[:4]}={x:.2f}" for c, x in ws.items()) for g, ws in grp.items() if g != "gk"))
        print("\nOUT-OF-SAMPLE BACKTEST (agent = learned weights, agent_prior = hand-set weights)")
        print(backtest.format_summary(summ))
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cases.to_csv(config.DATA_DIR / "learn_cases.csv", index=False)
        if final and not args.no_save:
            learn.save(final, {"target": args.target, "diagnostics": diag["final"]})
            print("\nsaved data/learned_weights.json (used by `ask` from now on)")
            print("FINAL WEIGHTS:", json.dumps({g: {c: round(x, 2) for c, x in ws.items()} for g, ws in final.items()}))
    elif args.cmd == "watch":
        from . import agent
        if args.once or args.force:
            print("\n".join(agent.watch_once(force=args.force)))
        else:
            agent.watch_loop(args.hours)
    elif args.cmd == "log":
        with store.db() as c:
            print("RUNS")
            for r in c.execute("SELECT id,text,tokens_in,tokens_out,round(cost_usd,5) cost_usd,created FROM runs ORDER BY id DESC LIMIT 10"):
                print(dict(r))
            print("TOOL CALLS")
            for r in c.execute("SELECT tool, count(*) n, round(avg(ms)) avg_ms, sum(1-ok) failures FROM calls GROUP BY tool"):
                print(dict(r))
    elif args.cmd == "llmcheck":
        import requests
        from . import llm
        prov = llm.provider()
        print("provider:", prov)
        if prov is None:
            return print("No API key found. Put GEMINI_API_KEY=... (no comment on the line) in .env")
        cands = [os.getenv("LLM_MODEL") or (config.GEMINI_MODEL if prov == "gemini" else config.OPENAI_MODEL)]
        if prov == "gemini":
            try:
                r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                                 params={"key": os.environ["GEMINI_API_KEY"], "pageSize": 200}, timeout=30)
                r.raise_for_status()
                names = [m["name"].removeprefix("models/") for m in r.json().get("models", [])
                         if "generateContent" in m.get("supportedGenerationMethods", [])]
                flash = [n for n in names if "flash" in n and not any(x in n for x in ("tts", "image", "live"))]
                print("flash models your key can call:", flash or names[:15])
                cands += [n for n in flash if n not in cands][:8]
            except Exception as e:
                print(f"could not list models: {type(e).__name__}: {str(e)[:300]} (invalid key?)")
        for m in dict.fromkeys(cands):
            os.environ["LLM_MODEL"] = m
            try:
                out = llm.chat([{"role": "user", "content": 'Reply with exactly this JSON: {"ok": true}'}], 0.0, json_mode=True)
                print(f"  {m}: OK -> {out[0][:60]!r}\n\nPut this in .env:  LLM_MODEL={m}")
                return
            except Exception as e:
                print(f"  {m}: FAILED -> {type(e).__name__}: {str(e)[:200]}")
        print("\nNo model worked. Check the key at aistudio.google.com/apikey.")
