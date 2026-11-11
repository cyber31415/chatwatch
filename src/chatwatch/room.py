"""Per-channel rolling state: what the room is like right now.

Everything about a single message is relative to its room. "who asked" in a
heated debate stream is ordinary; the same line as the first thing anyone has
said in a calm cozy stream is an instigation. This module keeps the cheap,
trailing statistics that make that distinction:

  mood / friction  share of recent messages that are positive vs friction
  first-timers     per-minute count/share of first-ever chatters (raids,
                   brigades)
  shapes           copypasta (shared culture, many established users) vs
                   coordinated templates (same text from several new accounts)
  banned           recent timeouts/bans, for ban-evasion lookalike names
  char model       a trailing character-trigram model of the room, so each
                   message gets a surprisal ("how odd is this here")
"""
import bisect
import math
import re
import time
from collections import Counter, deque

FRICTION_WINDOW_S = 90         # "right now"
BASELINE_WINDOW_S = 20 * 60    # "usual for this stream"
SHAPE_WINDOW_S = 180
CHAR_MODEL_SIZE = 3000         # messages in the trailing char model


class CharModel:
    """Trailing character-trigram model with add-k smoothing.

    Surprisal is bits per character of a normalised message given the last
    CHAR_MODEL_SIZE messages of the room. Odd messages (another language,
    random keyboard mash, a wall of copypasta nobody has posted, a topic far
    from the room's vocabulary) score high.
    """

    def __init__(self, size=CHAR_MODEL_SIZE, k=0.1):
        self.size, self.k = size, k
        self.tri, self.bi = Counter(), Counter()
        self.buf = deque()
        self.vocab = 60

    @staticmethod
    def _grams(s):
        s = "\x02" + s[:200] + "\x03"
        return [s[i:i + 3] for i in range(len(s) - 2)]

    def surprisal(self, s):
        g = self._grams(s)
        if not g or not self.buf:
            return None
        bits = 0.0
        for t in g:
            bits -= math.log2((self.tri[t] + self.k) / (self.bi[t[:2]] + self.k * self.vocab))
        return bits / len(g)

    def add(self, s):
        g = self._grams(s)
        self.buf.append(g)
        for t in g:
            self.tri[t] += 1
            self.bi[t[:2]] += 1
        if len(self.buf) > self.size:
            for t in self.buf.popleft():
                self.tri[t] -= 1
                self.bi[t[:2]] -= 1
                if self.tri[t] <= 0:
                    del self.tri[t]
                if self.bi[t[:2]] <= 0:
                    del self.bi[t[:2]]

    def ready(self):
        return len(self.buf) >= 200


STOP = set("""a an the and or but so to of in on at for with is are was were be been it its
this that i you u he she we they me my your ur im i'm its it's do does did not no yes
just like lol what why how when where who can will would should could have has had get
got go going if then than there here up down out about all any some one oh ok okay
too very really much more now dont don't is isn't""".split())
TERM = re.compile(r"[a-z][a-z']+")


def terms_of(norm):
    """Unigrams + bigrams of content words, for learning moderated topics."""
    ws = [w for w in TERM.findall(norm) if w not in STOP and len(w) > 2]
    return set(ws) | {f"{a} {b}" for a, b in zip(ws, ws[1:])}


class ModLearner:
    """Learns, live, what this channel's mods act on.

    Each mod action (timeout / ban / deletion) is credited to the content
    terms of the target's last messages. A term becomes a *moderated topic*
    once several different people were actioned after using it and it is
    far more common among actioned messages than in the room overall
    (e.g. a streamer who times out every "elden ring" mention). Uses only
    actions that already happened, so it is safe to evaluate prospectively.
    """
    MIN_USERS = 3
    MIN_LIFT = 8.0

    def __init__(self):
        self.base = Counter()
        self.n_msgs = 0
        self.hits = Counter()
        self.users = {}
        self.n_actions = 0
        self.last_terms = {}          # login -> terms of last 2 messages
        self.topics = {}              # term -> (users, lift)

    def observe(self, login, norm):
        t = terms_of(norm)
        self.base.update(t)
        self.n_msgs += 1
        prev = self.last_terms.pop(login, None)     # re-insert = most recent last
        self.last_terms[login] = (prev[1] if prev else set(), t)
        if len(self.last_terms) > 20000:
            for k in list(self.last_terms)[:5000]:
                del self.last_terms[k]
        if len(self.base) > 200000:
            self.base = Counter(dict(self.base.most_common(100000)))
        return t

    def action(self, login):
        lt = self.last_terms.get(login.lower())
        if not lt:
            return
        terms = lt[0] | lt[1]
        self.n_actions += 1
        for t in terms:
            self.hits[t] += 1
            self.users.setdefault(t, set()).add(login.lower())
            nu = len(self.users[t])
            if nu >= self.MIN_USERS:
                lift = ((self.hits[t] + 0.5) / (self.n_actions + 1)) / \
                       ((self.base[t] + 0.5) / (self.n_msgs + 1))
                if lift >= self.MIN_LIFT:
                    self.topics[t] = (nu, round(lift, 1))
                else:
                    self.topics.pop(t, None)

    def matches(self, terms):
        return sorted((t for t in terms if t in self.topics),
                      key=lambda t: -self.topics[t][0])


class Room:
    def __init__(self, channel, broadcaster_names=()):
        self.channel = channel
        self.room_id = ""
        # how chat refers to the streamer: login plus any hand-added nicknames
        self.broadcaster_names = {channel.lower(), *[n.lower() for n in broadcaster_names]}
        # two trailing windows with running sums, so every query is O(1)
        self.short = _Window(FRICTION_WINDOW_S)
        self.long = _Window(BASELINE_WINDOW_S)
        self.minutes = deque()           # (minute, msgs, first_timers, friction)
        self.shapes = {}                 # shape -> deque[(ts, login, is_new)]
        self.banned = deque()            # (ts, login, duration)
        self.char = CharModel()
        self.surprisals = deque(maxlen=2000)
        self._sur_sorted = None          # the same values sorted (oddness_pct); rebuilt when None
        self.n_msgs = 0
        self.roomstate = {}
        self.raid_until = 0
        self.last_raid = None
        self.started = None
        self.modlearn = ModLearner()
        self.context = deque(maxlen=60)  # (ts, login, text) for reviews / UI
        self.chants = {}                  # chant key -> deque[(ts, login)] (last 2 min)
        self.divisive = deque(maxlen=200)  # (ts, login) of divisive-topic messages
        from .vocab import RoomVocab
        self.vocab = RoomVocab()           # coded-vocabulary profile of this room
        self.arrivals = deque()           # (ts, login, new_account, random_name, friction)
        self.pile = deque()               # (ts, first_msg, mod_action, hostile_first) for pile_on
        self._pile_times = None           # sorted entry times per kind (pile_on); rebuilt when None
        self.pile_alerted_until = 0
        self.wave_alerted_until = 0

    # ------------------------------------------------------------ updates
    def _trim(self, now):
        while self.banned and now - self.banned[0][0] > 3 * 3600:
            self.banned.popleft()

    def add_message(self, ts, login, *, friction, positive, first, shape, is_new, norm):
        now = ts / 1000
        if self.started is None:
            self.started = now
        s = self.char.surprisal(norm) if self.char.ready() else None
        self.short.add(now, friction > 0, positive)
        self.long.add(now, friction > 0, positive)
        if s is not None:
            srt = self._sorted_surprisals()
            if len(self.surprisals) == self.surprisals.maxlen:
                del srt[bisect.bisect_left(srt, self.surprisals[0])]
            self.surprisals.append(s)
            bisect.insort(srt, s)
        if len(norm) >= 2:
            self.char.add(norm)
        self.n_msgs += 1
        minute = int(now // 60)
        if not self.minutes or self.minutes[-1][0] != minute:
            self.minutes.append([minute, 0, 0, 0])
            while len(self.minutes) > 60:
                self.minutes.popleft()
        m = self.minutes[-1]
        m[1] += 1
        m[2] += int(first)
        m[3] += int(friction > 0)
        if shape and len(shape) >= 8:
            dq = self.shapes.setdefault(shape, deque())
            dq.append((now, login, is_new))
            while dq and now - dq[0][0] > SHAPE_WINDOW_S:
                dq.popleft()
            if len(self.shapes) > 5000:   # keep memory flat in huge chats
                for k in [k for k, v in self.shapes.items() if not v or now - v[-1][0] > SHAPE_WINDOW_S]:
                    del self.shapes[k]
        self._trim(now)
        return s

    WAVE_WINDOW_S = 30
    WAVE_MIN = 6

    def add_arrival(self, ts, login, new_account, random_name, friction):
        """A first-time chatter's first message. Hate raids (2021) were median
        32 bots in <16 s, 99.6% random alphanumeric names, 25% created within 2
        days (research/02 #10): a burst of *new accounts* arriving together."""
        now = ts / 1000
        self.arrivals.append((now, login, new_account, random_name, friction))
        while self.arrivals and now - self.arrivals[0][0] > self.WAVE_WINDOW_S:
            self.arrivals.popleft()

    def newcomer_wave(self, ts):
        """Returns a description if new accounts are arriving in a burst, else None.
        A declared raid (USERNOTICE) explains a burst of *established* accounts,
        not one of day-old accounts."""
        now = ts / 1000
        if now < self.wave_alerted_until:
            return None
        recent = [a for a in self.arrivals if now - a[0] <= self.WAVE_WINDOW_S]
        new = [a for a in recent if a[2]]
        if len(new) < self.WAVE_MIN:
            return None
        self.wave_alerted_until = now + 120
        return {"new_accounts": len(new), "first_timers": len(recent),
                "random_names": sum(1 for a in new if a[3]),
                "with_friction": sum(1 for a in new if a[4]),
                "declared_raid": self.is_raid_window(ts),
                "logins": [a[1] for a in new][:12]}

    def chant_count(self, key, ts, login):
        """How many *other* people said the same short thing in the last 2 min."""
        if not key:
            return 0
        now = ts / 1000
        dq = self.chants.get(key)
        if not dq:
            return 0
        return len({l for t, l in dq if now - t <= 120 and l != login})

    def add_chant(self, key, ts, login):
        if not key or len(key) > 60:
            return
        now = ts / 1000
        dq = self.chants.setdefault(key, deque(maxlen=200))
        dq.append((now, login))
        if len(self.chants) > 20000:
            for k in [k for k, v in self.chants.items() if not v or now - v[-1][0] > 120]:
                del self.chants[k]

    def add_divisive(self, ts, login):
        self.divisive.append((ts / 1000, login))

    def topic_in_room(self, ts, login, window_s=300, min_others=3):
        """Were *other* people already on divisive topics in the last 5 min?
        Then politics here is the conversation, not an injection."""
        now = ts / 1000
        others = {l for t, l in self.divisive if now - t <= window_s and l != login}
        return len(others) >= min_others

    def add_ban(self, ts, login, duration):
        self.banned.append((ts / 1000, login.lower(), duration))

    def add_mod_action(self, login):
        self.modlearn.action(login)

    def add_raid(self, ts, from_login, viewers):
        self.raid_until = ts / 1000 + 300
        self.last_raid = {"from": from_login, "viewers": viewers, "ts": ts}

    # ------------------------------------------------------------- queries
    def friction_now(self, ts):
        """(share of friction messages in the last 90 s, usual share, n)."""
        now = ts / 1000
        self.short.trim(now)
        self.long.trim(now)
        return _friction(self.short.n, self.short.fr, self.long.n, self.long.fr)

    def heated(self, ts):
        """True when the room is currently much more combative than usual —
        friction then is partly contagion, not individual intent. The absolute
        floor (20%) stops an hour-long brigade from becoming its own baseline."""
        return _heated(*self.friction_now(ts))

    PILE_WINDOW_S = 600

    @staticmethod
    def _pile_kinds(p):
        """Which counts a pile entry is in: messages, first-timer messages, mod actions, hostile first messages."""
        return (not p[2], bool(p[1]), bool(p[2]), bool(p[3]))

    def _pile_sorted(self):
        """The pile's entry times, sorted, per kind: pile_on counts its window by
        binary search instead of rescanning 10 minutes of chat on every message
        (half the replay time in big chats). Entries can be out of time order
        (server timestamps, overlapping collector files), so sorted lists, not
        running sums."""
        if self._pile_times is None:
            self._pile_times = ([], [], [], [])
            for p in self.pile:
                for lst, k in zip(self._pile_times, self._pile_kinds(p)):
                    if k:
                        bisect.insort(lst, p[0])
        return self._pile_times

    def note_pile(self, ts, *, first=False, action=False, hostile_first=False):
        now = ts / 1000
        times = self._pile_sorted()
        p = (now, first, action, hostile_first)
        self.pile.append(p)
        for lst, k in zip(times, self._pile_kinds(p)):
            if k:
                bisect.insort(lst, now)
        while self.pile and now - self.pile[0][0] > self.PILE_WINDOW_S:
            old = self.pile.popleft()
            for lst, k in zip(times, self._pile_kinds(old)):
                if k:
                    del lst[bisect.bisect_left(lst, old[0])]

    def _pile_count(self, lst, now):
        """How many times t in the sorted list have now - t <= window (the same test
        as a scan; it holds from some index on, so binary search for it)."""
        lo, hi = 0, len(lst)
        while lo < hi:
            mid = (lo + hi) // 2
            if now - lst[mid] <= self.PILE_WINDOW_S:
                hi = mid
            else:
                lo = mid + 1
        return len(lst) - lo

    def pile_on(self, ts):
        """A sustained influx of first-time chatters that mods are fighting
        (or that is hostile): the brigade pattern, as opposed to a bot raid
        (newcomer_wave) or a friendly raid (USERNOTICE). Over the last 10 min:
        >= 10% of messages from first-timers, >= 10 first-timer messages, and
        >= 20% of first-timer messages hostile. Mod actions are reported as
        context only: what a community acts on is its own rule (sometimes
        viewpoint), not evidence of a brigade (docs/DESIGN.md)."""
        now = ts / 1000
        msgs, firsts, actions, hostile = (self._pile_count(lst, now) for lst in self._pile_sorted())
        if msgs < 30 or firsts < 10 or firsts / msgs < 0.10:
            return None
        if hostile / firsts < 0.20:
            return None
        return {"first_timer_share": round(firsts / msgs, 3), "first_timer_msgs": firsts,
                "mod_actions": actions, "hostile_first_msgs": hostile, "window_min": self.PILE_WINDOW_S // 60}

    def shape_stats(self, shape, ts, login):
        """How many *other* people posted this same shape recently, and how
        many of those were new accounts."""
        dq = self.shapes.get(shape)
        if not dq:
            return 0, 0
        others = {(l, new) for t, l, new in dq if l != login}
        users = {l for l, _ in others}
        new_users = {l for l, new in others if new}
        # a raid/bot template is *mostly* new accounts; crowd copypasta in a
        # big channel always contains a few, which must not count
        if len(new_users) < 0.6 * len(users):
            return len(users), 0
        return len(users), len(new_users)

    def first_timer_surge(self, ts):
        """Share and count of first-time chatters in the current minute (none once chat has gone
        quiet past it: the last recorded minute is not this one, LB5-3)."""
        if not self.minutes or self.minutes[-1][0] != int(ts / 1000 // 60):
            return 0.0, 0
        m = self.minutes[-1]
        return (m[2] / m[1] if m[1] else 0.0), m[2]

    def oddness_pct(self, s):
        """Percentile of a surprisal value among the room's recent messages."""
        if s is None or len(self.surprisals) < 200:
            return None
        below = bisect.bisect_left(self._sorted_surprisals(), s)     # = how many are < s
        return below / len(self.surprisals)

    def _sorted_surprisals(self):
        if self._sur_sorted is None or len(self._sur_sorted) != len(self.surprisals):
            self._sur_sorted = sorted(self.surprisals)
        return self._sur_sorted

    def lookalike_of_banned(self, login, ts):
        """A new name that closely resembles someone banned/long-timed-out in
        the last 3 hours: classic ban evasion ('troll123' -> 'troll1234')."""
        login = login.lower()
        stem = re.sub(r"[\d_]+$", "", login)
        for t, b, dur in reversed(self.banned):
            if b == login or (dur is not None and 0 <= dur < 600):
                continue          # short timeouts rarely prompt a new account
            bstem = re.sub(r"[\d_]+$", "", b)
            if len(stem) >= 4 and stem == bstem:
                return b
            if len(login) >= 6 and _edit_distance_le(login, b, 2):
                return b
        return None

    def is_raid_window(self, ts):
        return ts / 1000 < self.raid_until

    def snapshot(self, ts, full=True):
        """The room right now. full=False: only what the dashboard page shows (rate, first-timers, raid);
        the room profile (friction, mood, room state, moderated topics, vocabulary) is for the session
        log, every 30 s, not for the page every 2 s (audit, live_frontend section 4).
        Read-only: the live page asks at wall time, which can be ahead of a message still in flight,
        and that message must find the windows as the chat clock left them (LB5-1). Nothing is trimmed;
        the windows are read as a trim at ts would leave them."""
        now = ts / 1000
        n, fr, pos = self.short.view(now)
        share, count = self.first_timer_surge(ts)
        rate = round(n * 60 / FRICTION_WINDOW_S)
        lean = {"channel": self.channel, "msgs_per_min": rate, "first_timers_this_min": count,
                "raid": self.last_raid if self.is_raid_window(ts) else None}
        if not full:
            return lean
        cur, base, _ = _friction(n, fr, *self.long.view(now)[:2])
        return {"channel": self.channel, "msgs_per_min": rate,
                "friction_now": round(cur, 3), "friction_usual": round(base, 3),
                "heated": _heated(cur, base, n), "mood": round((pos - fr) / n if n else 0.0, 3),
                "first_timers_this_min": count, "first_timer_share": round(share, 3),
                "raid": self.last_raid if self.is_raid_window(ts) else None,
                "roomstate": self.roomstate, "messages_seen": self.n_msgs,
                "moderated_topics": sorted(self.modlearn.topics.items(),
                                           key=lambda kv: -kv[1][0])[:6],
                "vocabulary": [v for v in self.vocab.profile(12) if v["tier"] != "T3"][:6]}


class _Window:
    __slots__ = ("span", "q", "n", "fr", "pos")

    def __init__(self, span):
        self.span, self.q = span, deque()
        self.n = self.fr = 0
        self.pos = 0.0

    def add(self, now, friction, positive):
        self.q.append((now, friction, positive))
        self.n += 1
        self.fr += friction
        self.pos += positive
        self.trim(now)

    def trim(self, now):
        q = self.q
        while q and now - q[0][0] > self.span:
            _, f, p = q.popleft()
            self.n -= 1
            self.fr -= f
            self.pos -= p

    def view(self, now):
        """(n, fr, pos) as trim(now) would leave them, without changing the window."""
        n, fr, pos = self.n, self.fr, self.pos
        for t, f, p in self.q:
            if now - t <= self.span:
                break
            n, fr, pos = n - 1, fr - f, pos - p
        return n, fr, pos


def _friction(n, fr, long_n, long_fr):
    """(friction share of the 90 s window, usual share, n). "Usual" = the part of the long window
    *before* the current 90 s, so a burst can't raise its own baseline; default when there's no history."""
    n_before = long_n - n
    base = (long_fr - fr) / n_before if n_before >= 20 else 0.03
    return (fr / n if n else 0.0), base, n


def _heated(cur, base, n):
    return n >= 8 and (cur >= max(0.12, 2.5 * base) or cur >= 0.20)


def _edit_distance_le(a, b, k):
    """True if Levenshtein(a, b) <= k (banded, early exit)."""
    if abs(len(a) - len(b)) > k:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        lo = max(1, i - k)
        hi = min(len(b), i + k)
        if lo > 1:
            cur[lo - 1] = k + 1
        best = cur[lo - 1]
        for j in range(lo, hi + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != b[j - 1]))
            best = min(best, cur[j])
        for j in range(hi + 1, len(b) + 1):
            cur[j] = k + 1
        if best > k:
            return False
        prev = cur
    return prev[len(b)] <= k
