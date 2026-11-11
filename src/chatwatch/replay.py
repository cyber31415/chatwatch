"""Replay collected raw logs through the engine, in global time order.

    python -m chatwatch.replay data/raw/2026-10-20 [--channels a,b] [--limit N]
        [--alerts] [--ml] [--dump out.jsonl]

Used for development, the evaluation (evaluate.py) and the dashboard's replay
mode. Lines are merged across channel files by receive time.
"""
import argparse
import glob
import heapq
from collections import deque
import json
import os
import sys
import time

from . import irc


def iter_events(paths, channels=None):
    """Merge several collector files into one time-ordered event stream."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.log")) + glob.glob(os.path.join(p, "*.log.gz")))
        else:
            files += sorted(glob.glob(p))
    if channels:
        files = [f for f in files if os.path.basename(f).split(".")[0] in channels]
    files = [f for f in files if not os.path.basename(f).startswith("_server")]
    streams = [irc.read_raw(f) for f in files]
    # two collector runs may overlap (a handoff): drop events seen already.
    # Duplicates arrive within seconds of each other, so a bounded memory is enough.
    seen, order = set(), deque()
    for ms, line in heapq.merge(*streams, key=lambda x: x[0]):
        ev = irc.to_event(line, ms)
        if not ev:
            continue
        key = irc.dedup_key(ev)
        if key in seen:
            continue
        seen.add(key)
        order.append(key)
        if len(order) > 500_000:
            seen.discard(order.popleft())
        yield ev


def batch_gc():
    """For a batch replay (a job or tool that reads hours or days of chat): collect, freeze what
    exists (models, lexicons, caches: never garbage) and make collections rarer. A replay builds
    millions of long-lived objects the default thresholds rescan again and again: 14% of a
    1.5M-event replay, growing with the state (docs/review/performance.md P6). The output is
    identical (same record hash with default, raised and no collection)."""
    import gc
    gc.collect()
    gc.freeze()
    gc.set_threshold(50_000, 20, 100)


def message_records(eng, events, n=0, skip=0, before=lambda: None):
    """(event, record) for the engine's message records number skip+1 .. skip+n, in replay order
    (n=0: all after skip). Every event goes through the engine, the skipped ones too, so the engine
    state is the same as for a run that counts from the start. tools/calm_threshold.py matches a
    threshold on the first n calm messages (its noise check uses a random calm sample disjoint from
    them, so it is not measured on the messages it was tuned on; learning M4).
    before() runs before each event (calm_threshold clears its capture there)."""
    seen = 0
    for ev in events:
        if n and seen >= skip + n:
            break
        before()
        r = eng.process(ev)
        if r and r.get("type") == "msg":
            seen += 1
            if seen > skip:
                yield ev, r


def open_source(name, ml=None):
    """(engine, events) for a judged set's source, by name: a dashboard session
    (data/sessions/<name>/, restored from its start state when saved) or a set
    built from the roster capture (data/evalsets/<name>/meta.json: a raw time
    window, replayed with a fresh engine from its warm-up start)."""
    import pickle
    from .emotes import EmoteSets
    from .engine import Engine
    S = os.path.join("data/sessions", name) + "/"
    em = EmoteSets("data/emotes", fetch=False)
    if os.path.isdir(S):
        meta = json.load(open(S + "meta.json")) if os.path.exists(S + "meta.json") else {}
        if os.path.exists(S + "state_at_start.pkl"):
            eng = pickle.load(open(S + "state_at_start.pkl", "rb")).upgrade()
            eng.ml, eng.emotes = ml, em
        else:
            eng = Engine(ml=ml, emotes=em, legacy=False)
        if meta.get("pseudo"):
            return eng, iter_events([meta["source"]], set(meta["channels"]))
        return eng, iter_events(sorted(glob.glob(S + "raw/*")))
    meta = json.load(open(os.path.join("data/evalsets", name, "meta.json")))
    files = irc.raw_files(meta["warmup_from"], meta["t1"])

    def events():
        for ev in iter_events(files):
            if ev["ts"] > meta["t1"] * 1000:
                return
            if ev["ts"] >= meta["warmup_from"] * 1000:
                yield ev
    return Engine(ml=ml, emotes=em, legacy=False), events()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--channels", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--alerts", action="store_true", help="print alerts as they fire")
    ap.add_argument("--ml", action="store_true", help="load the local toxicity model")
    ap.add_argument("--dump", help="write every engine output record to this JSONL")
    a = ap.parse_args()

    from .engine import Engine
    ml = None
    if a.ml:
        from .models import Combined
        ml = Combined()
    from .emotes import EmoteSets
    eng = Engine(ml=ml, emotes=EmoteSets("data/emotes"))
    chans = {c.strip().lower() for c in a.channels.split(",") if c.strip()} or None
    out = open(a.dump, "w") if a.dump else None
    t0, n = time.time(), 0
    for ev in iter_events(a.paths, chans):
        rec = eng.process(ev)
        n += 1
        if rec and out:
            out.write(json.dumps(rec, default=list) + "\n")
        if a.alerts and rec:
            for al in rec.get("alerts", ()):
                print(f"[{al['channel']}] {al['kind']:<11} {al.get('display', '')}: "
                      f"{al.get('text', '')[:110]!r} bf={al.get('badfaith', '')} "
                      f"{[r['reason'] for r in al.get('reasons', [])][:3]}")
        if a.limit and n >= a.limit:
            break
    dt = time.time() - t0
    print(f"\n{n:,} events in {dt:.1f}s ({n / max(dt, 1e-9):,.0f}/s)", file=sys.stderr)
    print(json.dumps(dict(eng.stats.most_common()), indent=1), file=sys.stderr)


if __name__ == "__main__":
    main()
