"""Opt-out: forget chatters in every saved copy of the dashboard's state (docs/RESPONSIBLE_USE.md).

    bin/cw python -m chatwatch.forget 123456789 [987654321 ...]          # dry run: what would change
    bin/cw python -m chatwatch.forget 123456789 --go                      # do it

Covers data/state/engine.pkl, engine.pkl.prev and every data/sessions/*/state_at_start.pkl, with
Engine.forget (profiles, name indexes, register sums, alerts, the rooms' recent lines and memberships, and the
messages kept behind a listing, which go with the profile), plus the owner's corrections
(data/sessions/owner_overrides.jsonl) and roles (data/sessions/owner_roles.json and its .prev) about them.
A copy that cannot be read, and the copies set aside in data/state (engine.pkl.unreadable-*, unfinished
engine.pkl.*tmp*) that may name them, are reported, to delete by hand; forget goes on past them (audit LB4-3).
Each file is rewritten through a temporary file. Refuses while data/state/dashboard.lock exists: a
running dashboard would save its own copy over the result.

It does not edit chat logs: data/raw (the research corpus; frozen test sets hash its windows) and each
session's raw/ and outputs.jsonl keep the chatter's lines. It counts them, so you can delete those
session folders (or their raw/) when they are no longer needed (docs/LIVE_RUNS.md "Retention").
"""
import argparse
import glob
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def state_files(root=ROOT):
    st = os.path.join(root, "data", "state", "engine.pkl")
    return [p for p in [st, st + ".prev"] + sorted(glob.glob(os.path.join(root, "data", "sessions", "*",
                                                                          "state_at_start.pkl")))
            if os.path.exists(p)]


def set_aside(root=ROOT):
    """State copies serve.py set aside (engine.pkl.unreadable-*) or left half-written (*.tmp*): never rewritten."""
    st = os.path.join(root, "data", "state", "engine.pkl")
    return sorted(set(glob.glob(glob.escape(st) + ".unreadable-*") + glob.glob(glob.escape(st) + "*.tmp*")))


class Unreadable(Exception):
    pass


def forget_in(path, uids, go):
    """Profiles removed in one pickled state (and the file rewritten when go and any were). Unreadable when the file
    cannot be loaded (a torn copy)."""
    try:
        with open(path, "rb") as f:
            eng = pickle.load(f).upgrade()       # older states lack fields forget() reads
    except Exception as e:
        raise Unreadable(f"{type(e).__name__}: {e}") from e
    logins = {u.login for (_, uid), u in eng.users.items() if uid in uids}
    n = sum(eng.forget(uid) for uid in uids)
    if n and go:
        eng.risk = eng.persist = eng.ml = eng.emotes = None   # as the dashboard saves it
        eng.__dict__.pop("_notice_p", None)
        eng.__dict__.pop("_by_ch", None)
        tmp = path + ".forget.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(eng, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    return n, logins


def log_mentions(logins, root=ROOT):
    """Session folders whose raw/ or outputs.jsonl mention one of these logins: {folder: lines}."""
    out = {}
    if not logins:
        return out
    needles = [f":{x}!".encode() for x in logins] + [f'"login": "{x}"'.encode() for x in logins]
    for d in sorted(glob.glob(os.path.join(root, "data", "sessions", "*"))):
        n = 0
        for p in glob.glob(os.path.join(d, "raw", "*", "*")) + [os.path.join(d, "outputs.jsonl")]:
            if os.path.isfile(p):
                with open(p, "rb") as f:
                    n += sum(1 for line in f if any(x in line for x in needles))
        if n:
            out[os.path.basename(d)] = n
    return out


def main(argv=None, root=ROOT):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("user_ids", nargs="+")
    ap.add_argument("--go", action="store_true", help="rewrite the files (default: dry run)")
    a = ap.parse_args(argv)
    uids = {str(u) for u in a.user_ids}
    if a.go and os.path.exists(os.path.join(root, "data", "state", "dashboard.lock")):
        sys.exit("refused: data/state/dashboard.lock exists. Stop the dashboard first (it would save over this).")
    logins = set()
    for p in state_files(root):
        try:
            n, ls = forget_in(p, uids, a.go)
        except Unreadable as e:     # a torn copy: the others are still done (audit LB4-3)
            print(f"{os.path.relpath(p, root)}: could not be read ({e}); remove it by hand if it may name them")
            continue
        logins |= ls
        print(f"{os.path.relpath(p, root)}: {n} profile(s)" + (" removed" if a.go and n else " would be removed" if n else ""))
    for p in set_aside(root):
        with open(p, "rb") as f:
            raw = f.read()
        if any(u.encode() in raw for u in uids):
            print(f"{os.path.relpath(p, root)}: a set-aside copy that may name them; remove it by hand (not edited)")
    from .session import forget_overrides
    n = forget_overrides(uids, a.go, os.path.join(root, "data", "sessions", "owner_overrides.jsonl"))
    if n:
        print(f"data/sessions/owner_overrides.jsonl: {n} correction(s)" + (" removed" if a.go else " would be removed"))
    from .roles import forget as forget_roles
    n = forget_roles(uids, a.go, os.path.join(root, "data", "sessions", "owner_roles.json"))
    if n:
        print(f"data/sessions/owner_roles.json: {n} role(s) you set" + (" removed" if a.go else " would be removed"))
    for d, n in log_mentions(logins, root).items():
        print(f"data/sessions/{d}: {n} logged line(s) mention them (raw/, outputs.jsonl); not edited")
    if not a.go:
        print("dry run: add --go to rewrite the state files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
