"""Parse Twitch IRC lines (IRCv3 tags) into plain event dicts.

Only what the analyzer needs is normalised; the full tag dict is kept on every
event so nothing is lost. Raw lines are what the collector stores, so this
parser can change without re-collecting anything.
"""
import os
import re

_TAG_UNESCAPE = {r"\:": ";", r"\s": " ", r"\\": "\\", r"\r": "\r", r"\n": "\n"}
_ESC = re.compile(r"\\[:s\\rn]")
_LINE = re.compile(r"^(?::(?P<prefix>\S+) )?(?P<cmd>\S+)(?: (?P<params>.*))?$")


def _unescape(v):
    return _ESC.sub(lambda m: _TAG_UNESCAPE[m.group(0)], v)


def parse_tags(raw):
    tags = {}
    for part in raw.split(";"):
        k, _, v = part.partition("=")
        tags[k] = _unescape(v)
    return tags


def parse_badges(s):
    """'subscriber/12,vip/1' -> {'subscriber': '12', 'vip': '1'}"""
    out = {}
    for b in (s or "").split(","):
        if b:
            k, _, v = b.partition("/")
            out[k] = v
    return out


def parse_emotes(s, text):
    """IRC emotes tag -> list of (emote_id, start, end, name). Positions are
    in Unicode code points, which is what Python str indexing uses."""
    out = []
    for grp in (s or "").split("/"):
        if not grp:
            continue
        eid, _, spans = grp.partition(":")
        for span in spans.split(","):
            a, _, b = span.partition("-")
            if a.isdigit() and b.isdigit():
                a, b = int(a), int(b)
                out.append((eid, a, b, text[a:b + 1]))
    out.sort(key=lambda e: e[1])
    return out


def parse_line(line):
    """Return (tags, prefix, command, params list) or None."""
    line = line.rstrip("\r\n")
    if not line:
        return None
    tags = {}
    if line.startswith("@"):
        raw, _, line = line.partition(" ")
        tags = parse_tags(raw[1:])
    m = _LINE.match(line)
    if not m:
        return None
    params, trailing = [], None
    rest = m.group("params") or ""
    if rest.startswith(":"):
        trailing, rest = rest[1:], ""
    elif " :" in rest:
        rest, _, trailing = rest.partition(" :")
    params = rest.split() if rest else []
    if trailing is not None:
        params.append(trailing)
    return tags, m.group("prefix") or "", m.group("cmd"), params


def parse_flags(s):
    """Undocumented AutoMod-style span labels (research/01 §2.2):
    'start-end:A.6/P.6,...' -> {'A': 6, 'P': 6} (max level per category).
    A = aggression, I = identity, P = profanity, S = sexual; levels 0-7."""
    out = {}
    for span in (s or "").split(","):
        _, _, labels = span.partition(":")
        for lab in labels.split("/"):
            cat, _, lvl = lab.partition(".")
            if cat and lvl.isdigit():
                out[cat] = max(out.get(cat, 0), int(lvl))
    return out


def parse_flag_spans(s):
    """Like parse_flags, but keeps where each label applies:
    [(start, end, {"A": 6, ...}), ...] (indices into the message text)."""
    out = []
    for span in (s or "").split(","):
        rng, _, labels = span.partition(":")
        a, _, b = rng.partition("-")
        if not (a.isdigit() and b.isdigit()):
            continue
        levels = {}
        for lab in labels.split("/"):
            cat, _, lvl = lab.partition(".")
            if cat and lvl.isdigit():
                levels[cat] = max(levels.get(cat, 0), int(lvl))
        out.append((int(a), int(b), levels))
    return out


def _int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def to_event(line, recv_ms=None):
    """Normalise one raw line into an event dict, or None for noise.

    kinds: msg, clearchat, clearmsg, usernotice, roomstate, notice
    """
    if "\x00" in line:          # a NUL-filled spot left by a lost write on the share (N1): never a real line
        return None             # (IRC forbids NUL); the cut-off line around it is dropped, not mis-read
    p = parse_line(line)
    if not p:
        return None
    tags, prefix, cmd, params = p
    chan = params[0].lstrip("#") if params and params[0].startswith("#") else ""
    ts = _int(tags.get("tmi-sent-ts")) or recv_ms or 0
    base = {"ts": ts, "channel": chan, "room_id": tags.get("room-id", "")}

    if cmd == "PRIVMSG":
        text = params[1] if len(params) > 1 else ""
        action = False
        if text.startswith("\x01ACTION ") and text.endswith("\x01"):
            text, action = text[8:-1], True
        badges = parse_badges(tags.get("badges"))
        login = prefix.split("!", 1)[0]
        return {**base, "kind": "msg",
                "id": tags.get("id", ""),
                "user_id": tags.get("user-id", ""),
                "login": login,
                "display": tags.get("display-name") or login,
                "text": text, "action": action,
                "badges": badges,
                "badge_info": parse_badges(tags.get("badge-info")),
                "color": tags.get("color", ""),
                "emotes": parse_emotes(tags.get("emotes"), text),
                "first_msg": tags.get("first-msg") == "1",
                "returning": tags.get("returning-chatter") == "1",
                "mod": tags.get("mod") == "1" or "moderator" in badges,
                "broadcaster": "broadcaster" in badges,
                "vip": "vip" in badges,
                "sub": tags.get("subscriber") == "1",
                "bits": _int(tags.get("bits")),
                "reply_to_login": tags.get("reply-parent-user-login", ""),
                "reply_to_id": tags.get("reply-parent-user-id", ""),
                "reply_to_msg": tags.get("reply-parent-msg-body", ""),
                "source_room": tags.get("source-room-id", ""),
                "automod": parse_flags(tags.get("flags")),
                "automod_spans": parse_flag_spans(tags.get("flags")),
                "tags": tags}
    if cmd == "CLEARCHAT":
        target = params[1] if len(params) > 1 else ""
        return {**base, "kind": "clearchat",
                "target_login": target,
                "target_user_id": tags.get("target-user-id", ""),
                # absent duration = permanent ban; absent target = /clear
                "duration": _int(tags.get("ban-duration"), -1) if target else None}
    if cmd == "CLEARMSG":
        return {**base, "kind": "clearmsg",
                "target_login": tags.get("login", ""),
                "target_msg_id": tags.get("target-msg-id", ""),
                "text": params[1] if len(params) > 1 else ""}
    if cmd == "USERNOTICE":
        return {**base, "kind": "usernotice",
                "id": tags.get("id", ""),
                "msg_id": tags.get("msg-id", ""),
                "user_id": tags.get("user-id", ""),
                "login": tags.get("login", ""),
                "text": params[1] if len(params) > 1 else "",
                "system_msg": tags.get("system-msg", ""),
                "viewers": _int(tags.get("msg-param-viewerCount")),
                "tags": tags}
    if cmd == "ROOMSTATE":
        return {**base, "kind": "roomstate", "tags": tags}
    if cmd == "NOTICE":
        return {**base, "kind": "notice", "msg_id": tags.get("msg-id", ""),
                "text": params[-1] if params else ""}
    return None


def notice_as_msg(ev):
    """The chatter-written text of a USERNOTICE (a resub or sub message, a viewer milestone, an
    announcement, ...) as a chat message event, or None when it carries none. Only the text the
    chatter typed (the trailing parameter) is the message; Twitch's own wording (system-msg: "X
    subscribed for 12 months") is not. The event keeps "notice": its msg-id."""
    text = ev.get("text") or ""
    if not text.strip() or not ev.get("login"):
        return None
    tags = ev.get("tags") or {}
    badges = parse_badges(tags.get("badges"))
    return {"ts": ev["ts"], "channel": ev["channel"], "room_id": ev.get("room_id", ""), "kind": "msg",
            "id": ev.get("id") or tags.get("id", ""), "user_id": ev.get("user_id", ""), "login": ev["login"],
            "display": tags.get("display-name") or ev["login"], "text": text, "action": False,
            "badges": badges, "badge_info": parse_badges(tags.get("badge-info")), "color": tags.get("color", ""),
            "emotes": parse_emotes(tags.get("emotes"), text), "first_msg": False, "returning": False,
            "mod": tags.get("mod") == "1" or "moderator" in badges, "broadcaster": "broadcaster" in badges,
            "vip": "vip" in badges, "sub": tags.get("subscriber") == "1", "bits": 0,
            "reply_to_login": "", "reply_to_id": "", "reply_to_msg": "", "source_room": tags.get("source-room-id", ""),
            "automod": parse_flags(tags.get("flags")), "automod_spans": parse_flag_spans(tags.get("flags")),
            "tags": tags, "notice": ev.get("msg_id", "")}


def parsed_lines(lines, since_ms=0):
    """(line number, event or None) for raw-file lines "<ms> <irc line>"; lines
    before since_ms and unparsable ones give None (so callers keep their counts)."""
    for i, row in enumerate(lines):
        ms, _, line = row.partition(" ")
        yield i, (to_event(line, int(ms)) if ms.isdigit() and int(ms) >= since_ms else None)


def raw_slices(todo, since_ms=0):
    """The unread part of each raw file (todo: {path: (start, end)}) in path
    order, as one stream, so look-ahead scoring (models.Ahead) runs across files:
    (path, cut, n_lines, i, event or None) per line, then (path, cut, n_lines,
    None, None) when that file's slice is done; cut = bytes of complete lines."""
    for path in sorted(todo):
        start, end = todo[path]
        with open(path, "rb") as fh:
            fh.seek(start)
            blob = fh.read(end - start)
        cut = blob.rfind(b"\n") + 1
        # split on "\n" only, as read_raw does: str.splitlines() also splits on U+2028/2029, U+0085 and
        # \x0b-\x0c, \x1c-\x1e inside a message (audit_20261101 PB3-8). A UTF-8 sequence never holds byte 0x0A.
        lines = [x.decode("utf-8", errors="replace") for x in blob[:cut].split(b"\n")[:-1]]
        for i, ev in parsed_lines(lines, since_ms):
            yield path, cut, len(lines), i, ev
        yield path, cut, len(lines), None, None


def channel_of_path(path):
    """'data/raw/2026-10-21/somechannel.r20261021T0455.log' -> 'somechannel'."""
    return os.path.basename(path).split(".")[0]


def dedup_key(ev):
    """Identity of an event, so overlapping collector runs can be merged."""
    k = ev["kind"]
    if k == "msg":
        return ("m", ev.get("id") or (ev["channel"], ev["ts"], ev.get("login"), ev.get("text")))
    if k == "clearmsg":
        return ("d", ev.get("target_msg_id"))
    if k == "clearchat":
        return ("c", ev["channel"], ev["ts"], ev.get("target_login"))
    if k == "usernotice":
        return ("u", ev.get("tags", {}).get("id") or (ev["channel"], ev["ts"], ev.get("msg_id")))
    return (k, ev.get("channel"), ev["ts"])


def read_raw(path):
    """Yield (recv_ms, line) from a collector file ('<ms> <raw line>'). A line holding NUL bytes (a spot a
    lost write on the share left zero-filled, docs/review/audit_20261029b N1) is skipped whole: it is the cut
    end of one line, the zeros and the cut start of another (tools/raw_integrity.py reports them)."""
    import gzip
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="replace") as f:
        for row in f:
            if "\x00" in row:
                continue
            ms, _, line = row.partition(" ")
            if ms.isdigit():
                yield int(ms), line.rstrip("\n")


def raw_files(since=None, until=None, root="data/raw"):
    """Collector files that can hold chat between since and until (unix s).
    Selected by their UTC day folder (the collector writes one file per channel
    per UTC day), never by modification time: on the network share a file being
    appended to keeps its old mtime until it is closed."""
    import glob as _glob
    import os as _os
    import time as _time
    lo = _time.strftime("%Y-%m-%d", _time.gmtime(since)) if since else ""
    hi = _time.strftime("%Y-%m-%d", _time.gmtime(until)) if until else "9999"
    out = []
    for d in sorted(_glob.glob(_os.path.join(root, "*"))):
        day = _os.path.basename(d)
        if _os.path.isdir(d) and lo <= day <= hi:
            out += [p for p in _glob.glob(_os.path.join(d, "*.log")) if not _os.path.basename(p).startswith("_")]
    return sorted(out)
