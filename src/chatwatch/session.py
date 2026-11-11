"""Full logging of every live run, so it can be re-evaluated later.

Each live `serve` run creates data/sessions/<UTC start>_<channels>/:

  meta.json            start/end, command line, engine fingerprint, thresholds,
                       models, the owner's marked bots at the start; at the end
                       "stats" (this session's counters) and "stats_lifetime" (the saved state's running totals)
  src_snapshot.tar.gz  the exact code that ran (re-run any old session as it was; the owner's
                       corrections and roles are not replayed: compare_session counts the
                       chatters they touched apart, audits LB2-14 and LB3-4)
  state_at_start.pkl   the saved state the run started from; only the newest
                       STATE_COPIES_KEPT sessions keep theirs (older copies are
                       deleted at the next Start; meta.json keeps their sha256)
  raw/<day>/<ch>.log   every IRC line received, mod actions included
  outputs.jsonl        everything the tool produced: per-message reads ("bot_msg",
                       no read, for a chatter the owner marked a bot), alerts,
                       LLM reviews, mod actions as seen, the owner's corrections
                       and role changes, room snapshots (30 s)

Sessions from before 2026-10-23 may also hold feedback.jsonl (the card's verdict
buttons, since removed: no human-feedback labelling).

tools/compare_session.py re-runs a session's raw chat through the current
engine and diffs it against outputs.jsonl.
"""
import datetime as dt
import glob
import hashlib
import io
import json
import os
import shutil
import tarfile
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE = os.path.join(ROOT, "data", "state", "engine.pkl")
STATE_COPIES_KEPT = 3      # sessions that keep their state_at_start.pkl (audit LB-8: 19 copies held 402 MB)
# the owner's message corrections (the dashboard's "reclassify"), one line each: an audit trail, and the list of
# messages the set builders leave out. Never a label: nothing that trains or evaluates reads it.
OVERRIDES = os.path.join(ROOT, "data", "sessions", "owner_overrides.jsonl")


def fingerprint():
    """Short hash of the engine source + dashboard, same scheme as the eval freezes."""
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(ROOT, "src", "chatwatch", "*.py"))):
        h.update(hashlib.sha256(open(p, "rb").read()).hexdigest().encode() + b"  " + p.encode())
    return h.hexdigest()[:16]


class Session:
    def __init__(self, channels, argv, eng, base=None, ai=None):
        """Creates the session folder and its files at once (an OSError here is a failed Start). The code
        snapshot and the state copy are written by a worker thread, off the event loop (LB-8). ai: the run's AI as
        the dashboard chose it (Hub.session_ai: mode, live model, reviewer, caps); without it (an old caller) the
        model is CW_LLM_MODEL's, as before."""
        start = dt.datetime.now(dt.timezone.utc)
        name = start.strftime("%Y%m%d-%H%M%S") + "_" + "-".join(channels)[:80]
        self.base = base or os.path.join(ROOT, "data", "sessions")
        self.dir = os.path.join(self.base, name)
        os.makedirs(os.path.join(self.dir, "raw"), exist_ok=True)
        from . import archetypes, engine
        self.meta = {
            "started": start.isoformat(), "channels": channels, "argv": argv,
            "engine_fingerprint": fingerprint(),
            "thresholds": {"alert_badfaith": engine.ALERT_BADFAITH, "alert_first_msg": engine.ALERT_FIRST_MSG,
                           "alert_budget": eng.alert_budget, "damping": archetypes.DAMPING,
                           "evidence_half_life_days": engine.EVIDENCE_HALF_LIFE_DAYS},
            "ml": type(eng.ml).__name__ if eng.ml else None,
            # T5 (sweep 2026-11-09): the model the run actually calls (the page's choice), None when the AI is off
            "llm_model": (ai or {}).get("live") if ai is not None else os.environ.get("CW_LLM_MODEL", "gemma4mtp:31b-n3-text"),
            "ai": ai,
            "state_users_at_start": len(eng.users),
            # the user ids the owner had marked a bot when the run started (roles.py; changes are logged as
            # "owner_role"): compare_session counts their alerts apart (marked_bots, audit LB3-4). Never scored with
            "owner_bots_at_start": sorted(getattr(eng, "owner_bots", ()) or ()),
        }
        self.stats_at_start = dict(eng.stats)      # meta "stats" is this session's share of the totals (LB-15)
        self._write_meta()
        from .collect import RawSink
        self.raw = RawSink(os.path.join(self.dir, "raw"))
        self.out = open(os.path.join(self.dir, "outputs.jsonl"), "a", buffering=1 << 16)
        self.n = 0
        self._prep = threading.Thread(target=self._prepare, daemon=True, name="cw-session")
        self._prep.start()
        print(f"[session] logging to {os.path.relpath(self.dir, ROOT)}", flush=True)

    def _prepare(self):
        for step in (self._snapshot_src, self._state_at_start, self._drop_old_state_copies):
            try:
                step()
            except Exception as e:                     # the session still logs; the meta says what is missing
                self.meta.setdefault("prepare_errors", []).append(f"{step.__name__}: {type(e).__name__}: {e}")
        try:
            self._write_meta()                         # the state sha and removed copies, before the end
        except OSError:
            pass

    def _snapshot_src(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as t:
            for sub in ("src/chatwatch", "web"):
                t.add(os.path.join(ROOT, sub), arcname=sub,
                      filter=lambda ti: None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti)
        open(os.path.join(self.dir, "src_snapshot.tar.gz"), "wb").write(buf.getvalue())

    def _state_at_start(self):
        """The per-user state the run started from, so a re-run starts equal (replay.open_source,
        tools/compare_session.py). Every Stop saves a new state, so it is copied (the earlier "link
        when unchanged" path never fired: LB-8)."""
        if not os.path.exists(STATE):
            return
        dst = os.path.join(self.dir, "state_at_start.pkl")
        with open(STATE, "rb") as f:          # one handle: an autosave renaming a new state in meanwhile
            h = hashlib.sha256()               # cannot make the hash and the copy differ
            with open(dst, "wb") as out:
                for b in iter(lambda: f.read(1 << 20), b""):
                    h.update(b)
                    out.write(b)
            self.meta["state_sha256"] = h.hexdigest()

    def _drop_old_state_copies(self):
        """Keep state_at_start.pkl in the newest STATE_COPIES_KEPT sessions only: older copies hold
        chatters that pruning (S7) and Engine.forget have removed (docs/RESPONSIBLE_USE.md)."""
        with_copy = sorted(d for d in glob.glob(os.path.join(self.base, "*"))
                           if os.path.exists(os.path.join(d, "state_at_start.pkl")))
        for d in with_copy[:-STATE_COPIES_KEPT] if len(with_copy) > STATE_COPIES_KEPT else []:
            if d != self.dir:
                os.remove(os.path.join(d, "state_at_start.pkl"))
                self.meta.setdefault("state_copies_removed", []).append(os.path.basename(d))

    def _write_meta(self):
        json.dump(self.meta, open(os.path.join(self.dir, "meta.json"), "w"), indent=1, default=str)

    def raw_line(self, chan, line, ms):
        self.raw.write(chan, line, ms)

    def log(self, rec):
        r = dict(rec)
        r.pop("alerts", None)          # logged as their own records
        self.out.write(json.dumps(r, default=_jsonable, ensure_ascii=False) + "\n")
        self.n += 1

    def flush(self):
        self.out.flush()
        self.raw.flush()

    def close(self, stats):
        self.flush()
        self.raw.close()
        self.out.close()
        self._prep.join(timeout=120)
        here = {k: v - self.stats_at_start.get(k, 0) for k, v in stats.items()
                if isinstance(v, (int, float)) and v != self.stats_at_start.get(k, 0)}
        self.meta.update(ended=dt.datetime.now(dt.timezone.utc).isoformat(), records_logged=self.n,
                         stats=here, stats_lifetime=dict(stats))
        self._write_meta()
        print(f"[session] closed {os.path.relpath(self.dir, ROOT)} ({self.n:,} records)", flush=True)


def _jsonable(o):
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


# A chatter's line in outputs.jsonl: "msg", or "bot_msg" when the owner had marked them a bot (live only, roles.py:
# not scored, so the record carries no read). Every reader that picks chat lines for labels or their context (silver
# labels, silverctx) takes both, so a role never decides which messages are labelled or what a window holds (audit
# LB3-7): the lines are the same with or without a role.
CHAT_TYPES = ("msg", "bot_msg")


def chat_lines(path):
    """The chat lines of one outputs.jsonl, in the order they were logged, whatever role their chatter had."""
    with open(path) as f:
        return [r for r in map(json.loads, f) if r.get("type") in CHAT_TYPES]


def marked_bots(session_dir):
    """User ids the owner had marked a bot at some point of this session: at its start (meta "owner_bots_at_start")
    or later ("owner_role" records). compare_session counts their alerts apart, as owner-caused (audit LB3-4); no
    replay is scored with them."""
    out = set()
    try:
        meta = json.load(open(os.path.join(session_dir, "meta.json")))
        out |= {str(u) for u in meta.get("owner_bots_at_start") or ()}
    except (OSError, ValueError, AttributeError):
        pass
    p = os.path.join(session_dir, "outputs.jsonl")
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                if '"owner_role"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("type") == "owner_role" and r.get("role") == "bot":
                    out.add(str(r.get("user_id")))
    return out


def log_override(rec, path=None):
    """Append one owner correction to OVERRIDES (ids, levels and times only: no message text)."""
    path = path or OVERRIDES
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def owner_override_ids(path=None):
    """Message ids the owner reclassified by hand. The set builders leave them out, so a judged set never
    holds a message whose live class the owner set."""
    path = path or OVERRIDES
    ids = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                ids.add(str(json.loads(line)["msg_id"]))
            except (ValueError, KeyError, TypeError):
                continue                         # a torn line: the rest still counts
    return ids


def forget_overrides(uids, go, path=None):
    """Lines of OVERRIDES about these user ids (and, when go, the file rewritten without them)."""
    path = path or OVERRIDES
    if not os.path.exists(path):
        return 0
    keep, n = [], 0
    for line in open(path):
        try:
            hit = str(json.loads(line).get("user_id")) in uids
        except (ValueError, AttributeError, TypeError):      # a torn line, or one that is not an object (LB2-16)
            hit = False
        n += hit
        if not hit:
            keep.append(line)
    if n and go:
        tmp = path + ".forget.tmp"
        with open(tmp, "w") as f:
            f.writelines(keep)
        os.replace(tmp, path)
    return n
