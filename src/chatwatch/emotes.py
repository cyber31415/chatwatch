"""Third-party emote sets (7TV, BTTV, FFZ), global + per channel.

First-party Twitch emotes arrive with exact positions in the IRC `emotes` tag,
so only third-party codes need a lookup. Sets are cached on disk per room id
(data/emotes/<room_id>.json) and refreshed after REFRESH_S, also during a long live run
(audit LB4-9). The room id comes from the ROOMSTATE / PRIVMSG `room-id` tag, so no Twitch API call is needed.

    es = EmoteSets("data/emotes")
    es.codes(room_id)    # frozenset of emote codes (case-sensitive)

A provider that fails (a timeout, a server error) is not cached as empty: its codes from the
previous file are kept and the room is fetched again after RETRY_S. A 404 means the channel has no
set there, which is a real empty set. Files are written through a temporary file and a rename, since
every pipeline job and replay reads this folder (audit LB-9). The cache format gained "by_src" (the
codes per provider); "codes" (their union) is what readers use, as before.
A cache file that cannot be written keeps the codes just fetched (the room is fetched and written again
after RETRY_S); a retry time left by an earlier process is kept in this run; a failed global fetch is
retried too (audit LB2-15).
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request

REFRESH_S = 6 * 3600
RETRY_S = 15 * 60          # a failed provider is fetched again this soon
UA = {"User-Agent": "chatwatch/0.9 (local moderation tool)"}

GLOBAL = {
    "7tv": "https://7tv.io/v3/emote-sets/global",
    "bttv": "https://api.betterttv.net/3/cached/emotes/global",
    "ffz": "https://api.frankerfacez.com/v1/set/global",
}
CHANNEL = {
    "7tv": "https://7tv.io/v3/users/twitch/{id}",
    "bttv": "https://api.betterttv.net/3/cached/users/twitch/{id}",
    "ffz": "https://api.frankerfacez.com/v1/room/id/{id}",
}
NOT_THERE = {}             # a 404: the channel has no set with this provider


def _get(url, timeout=8):
    """The provider's JSON, NOT_THERE for a 404, or None when the fetch failed."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return NOT_THERE if e.code == 404 else None
    except Exception:
        return None


def _codes(src, d):
    """Pull emote codes out of each provider's response shape."""
    out = set()
    if not d:
        return out
    if src == "7tv":
        es = d.get("emote_set", d) if isinstance(d, dict) else {}
        for e in (es or {}).get("emotes") or []:
            if e.get("name"):
                out.add(e["name"])
    elif src == "bttv":
        if isinstance(d, list):
            out |= {e["code"] for e in d if e.get("code")}
        else:
            for k in ("channelEmotes", "sharedEmotes"):
                out |= {e["code"] for e in d.get(k) or [] if e.get("code")}
    elif src == "ffz":
        for s in (d.get("sets") or {}).values():
            out |= {e["name"] for e in s.get("emoticons") or [] if e.get("name")}
    return out


def _read(path):
    """A cache file, or None when it is missing or unreadable (a torn or corrupt file is refetched)."""
    try:
        with open(path) as f:
            d = json.load(f)
        set(d["codes"])
        return d
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write(path, d):
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, path)


class EmoteSets:
    def __init__(self, cache_dir="data/emotes", fetch=True, background=False):
        self.dir = cache_dir
        self.fetch = fetch
        self.background = background
        self.sets = {}
        self.pending = set()
        self.lock = threading.Lock()
        self.errors = 0               # fills that failed (the room then uses the global codes)
        self.retry = {}               # room id -> when to fetch again (a provider failed; live runs refill)
        self.refresh = {}             # key -> when its set is REFRESH_S old (a long live run refills it; audit LB4-9)
        if fetch:
            os.makedirs(cache_dir, exist_ok=True)
        self.global_codes = frozenset(self._load_or_fetch("global", GLOBAL))

    def _load_or_fetch(self, key, urls, rid=None):
        """The codes for one cache key; sets self.retry[key] when a provider failed."""
        path = os.path.join(self.dir, f"{key}.json")
        d = _read(path)
        if d is not None and (not self.fetch or time.time() < d.get("retry_at", d.get("fetched", 0) + REFRESH_S)):
            if self.fetch and "retry_at" in d:        # an earlier process's failed fetch: retry it in this run too (LB2-15)
                self.retry[key] = d["retry_at"]
            self.refresh[key] = d.get("fetched", 0) + REFRESH_S
            return set(d["codes"])
        if not self.fetch:
            return set() if d is None else set(d["codes"])
        old = (d or {}).get("by_src") or {}
        codes, per, by_src, failed = set(), {}, {}, []
        for src, url in urls.items():
            got = _get(url.format(id=rid))
            if got is None:                       # failed: keep what the last good fetch had
                failed.append(src)
                c = set(old[src]) if src in old else (set(d["codes"]) if d is not None and not old else set())
            else:
                c = _codes(src, got)
            per[src] = len(c)
            by_src[src] = sorted(c)
            codes |= c
        now = time.time()
        out = {"fetched": now, "counts": per, "codes": sorted(codes), "by_src": by_src}
        self.refresh[key] = now + REFRESH_S
        if failed:
            out.update(failed=failed, retry_at=now + RETRY_S)
            self.retry[key] = now + RETRY_S
        try:
            _write(path, out)
        except OSError:                           # the share full or flaky: the codes just fetched still count (LB2-15)
            self.errors += 1
            self.retry[key] = now + RETRY_S       # fetched and written again later
        return codes

    def _fill(self, rid):
        try:
            codes = self._load_or_fetch(rid, CHANNEL, rid)
        except Exception:                         # never leave the room pending for good (LB-9)
            codes = set()
            self.errors += 1
            self.retry[rid] = time.time() + RETRY_S
        with self.lock:
            self.sets[rid] = frozenset(codes | self.global_codes)
            self.pending.discard(rid)

    def _refill_global(self):
        """A failed global fetch, fetched again (live runs; LB2-15): the new codes join every room's set."""
        try:
            g = frozenset(self._load_or_fetch("global", GLOBAL))
        except Exception:
            self.errors += 1
            self.retry["global"] = time.time() + RETRY_S
            g = self.global_codes
        with self.lock:
            self.global_codes = g | self.global_codes
            self.sets = {rid: s | g for rid, s in self.sets.items()}
            self.pending.discard("global")

    def _due(self, key):
        """A provider failed for this key earlier and its retry time has come, or its set is REFRESH_S old (live runs
        only)."""
        now = time.time()
        return self.background and self.fetch and (now >= self.retry.get(key, now + 1)
                                                   or now >= self.refresh.get(key, now + 1))

    def codes(self, room_id):
        if self._due("global"):
            with self.lock:
                if "global" not in self.pending:
                    self.retry.pop("global", None)
                    self.refresh.pop("global", None)
                    self.pending.add("global")
                    threading.Thread(target=self._refill_global, daemon=True).start()
        if not room_id:
            return self.global_codes
        s = self.sets.get(room_id)
        if s is not None:
            if self._due(room_id):
                with self.lock:                   # a provider failed earlier, or the set is old: fetch again, keep these meanwhile
                    if room_id not in self.pending:
                        self.retry.pop(room_id, None)
                        self.refresh.pop(room_id, None)
                        self.pending.add(room_id)
                        threading.Thread(target=self._fill, args=(room_id,), daemon=True).start()
            return s
        if self.background:
            # live: never block the message loop on HTTP; use globals meanwhile
            with self.lock:
                if room_id not in self.pending:
                    self.pending.add(room_id)
                    threading.Thread(target=self._fill, args=(room_id,), daemon=True).start()
            return self.global_codes
        self._fill(room_id)
        return self.sets[room_id]
