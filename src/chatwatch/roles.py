"""The owner's roles for chatters, set and unset by hand on the dashboard's user card (owner, 2026-11-01).

    "ideally the ability to label certain chatters with roles ... primarily i am thinking of bots ... the manual
    determination for a role should be something i do and not automated ... id want to avoid putting a role like
    this on mods because id worry it excludes them for the same characterization that everyone gets"

The only role is "bot": while it is set, the live engine treats the chatter like the reviewed bot list (not scored,
profiled or listed; Engine.owner_bots). Nothing suggests or sets a role automatically, and there is no "mod" role:
mods are held to the same standard as everyone (docs/DESIGN.md "Absolute standard").

LIVE ONLY. serve.py reads this file for a live run and writes it when the owner changes a role (a run that saves
its state); a replay never reads it. No training, silver, evaluation, replay, layer-cache or coldstore path reads
it (tests/test_owner_features.py checks the sources). Putting a login on assets/bots.json, the reviewed list that
training reads too, is a separate step the owner asks for.

The file: data/sessions/owner_roles.json, beside the owner's corrections, {"roles": {user id: {"role", "login",
"display", "set"}}}, written with fileio.atomic_json (the old copy kept as .prev). `python -m chatwatch.forget`
removes a chatter's entry.
"""
import os
import time

from . import fileio

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROLES = os.path.join(ROOT, "data", "sessions", "owner_roles.json")
KINDS = ("bot",)


def load(path=None, log=print):
    """{user id: entry} from the file (or its .prev); {} when there is none. fileio.Corrupt when neither loads."""
    path = path or ROLES
    try:
        obj = fileio.load_json(path, log=log, what="roles")
    except FileNotFoundError:
        return {}
    roles = (obj or {}).get("roles") if isinstance(obj, dict) else None
    if not isinstance(roles, dict):
        raise fileio.Corrupt(path, ["no \"roles\" object"])
    return {str(k): v for k, v in roles.items() if isinstance(v, dict) and v.get("role") in KINDS}


def save(roles, path=None):
    """Write every role (the whole file, through a checked temporary file; the old one becomes .prev)."""
    path = path or ROLES
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fileio.atomic_json(path, {"roles": roles}, keep_prev=True, indent=1, sort_keys=True)


def entry(role, login, display):
    return {"role": role, "login": login, "display": display, "set": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def bots(roles):
    """The user ids with the owner's "bot" role, for Engine.owner_bots."""
    return frozenset(uid for uid, e in roles.items() if e.get("role") == "bot")


def _read_one(p):
    """{user id: entry} of one copy (the file or its .prev, not falling back to the other); Corrupt when it cannot be
    read."""
    import json
    try:
        with open(p) as f:
            obj = json.load(f)
    except (OSError, ValueError) as e:
        raise fileio.Corrupt(p, [str(e)])
    roles = obj.get("roles") if isinstance(obj, dict) else None
    if not isinstance(roles, dict):
        raise fileio.Corrupt(p, ["no \"roles\" object"])
    return {str(k): v for k, v in roles.items()}


def forget(uids, go, path=None, log=print):
    """How many of these user ids have a role in the file or its .prev (and, when go, each copy that names them
    rewritten without them). A copy that cannot be read, and the owner_roles.json[.prev].unreadable-* copies serve.py sets
    aside, are reported (by name) for removal by hand: they cannot be rewritten safely (audit LB3-6)."""
    import glob
    path = path or ROLES
    found = set()
    for p in (path, path + fileio.PREV):
        if not os.path.exists(p):
            continue
        try:
            roles = _read_one(p)
        except fileio.Corrupt as e:
            log(f"{os.path.basename(p)}: could not be read ({e}); remove it by hand if it may name them")
            continue
        gone = {u for u in roles if u in uids}
        found |= gone
        if gone and go:
            fileio.atomic_json(p, {"roles": {u: e for u, e in roles.items() if u not in uids}}, indent=1, sort_keys=True)
    for p in sorted(glob.glob(glob.escape(path) + "*.unreadable-*")):      # the file's and the .prev's (LB5-2)
        with open(p, "rb") as f:
            raw = f.read()
        if any(f'"{u}"'.encode() in raw for u in uids):
            log(f"{os.path.basename(p)}: an unreadable copy that names them; remove it by hand (it is not edited)")
    return len(found)
