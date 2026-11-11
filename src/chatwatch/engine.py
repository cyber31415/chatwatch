"""The live engine: events in, per-message reads and per-user profiles out.

    eng = Engine()
    for ev in events:                 # irc.to_event dicts, in time order
        out = eng.process(ev)         # dict for msg events (for the UI), or None
    eng.profile("channel", "user_id") # full card for one chatter

Two layers per chatter, kept separate on purpose:

  first impression  archetype posterior (archetypes.py), from message #1,
                    with reasons; this is what answers "what might they be?"
  register          the handoff's speech-style model (legacy_style.py),
                    meaningful after ~100-300 words; "what are they like?"

Trait vs state: each user keeps a lifetime mean of bad-faith evidence and a
recent mean, so a regular having a bad day reads differently from a newcomer
whose first act is a provocation.
"""
import bisect
import copy
import json
import re
import os
import time
from collections import Counter, deque

from . import archetypes as A
from . import legacy_style, risk, signals
from .toxicity import message_toxicity

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "assets", "baseline.json")

# Set from four judged sets by replaying the real alert logic (tools/eval_alerts.py,
# THRESHOLDS; FINDINGS "alert threshold"): pooled precision / recall 0.42 / 0.75 at
# 0.5, 0.46 / 0.72 at 0.7, 0.46 / 0.61 at 0.8. First messages keep the lower bar
# (0.35: precision 0.68, recall 0.93 on chatters judged on their first message).
ALERT_BADFAITH = 0.7        # posterior mass on bad-faith archetypes
ALERT_FIRST_MSG = 0.35      # lower bar for someone's very first message
ML_FIRST_N = 8              # ML-score a user's first N messages ...
SUSPECT_LOOKBACK_S = 20     # callouts attach to a suspect message this recent
OOC_MIN_MSGS = 40           # "out of character" needs a lifetime to compare to
EVIDENCE_HALF_LIFE_DAYS = 7
# S7 (live review H1, P4): the dashboard drops chatters idle this long (their evidence is down to
# 1/19.7 by then), at load and every autosave, so its state stops growing without bound
PRUNE_IDLE_DAYS = 30
# Owner standard (docs/DESIGN.md "Absolute standard"): a one-off insult from
# someone otherwise taking part is toxic but "mild", not bad faith. An alert
# needs the shape of bad faith: severe content, a pattern, a bad-faith type, or
# nothing but provocation. False = the old behaviour (score alone), for evaluation.
ALERT_NEEDS_SHAPE = True
# ... and at least one toxic message: alerts on chatters with none were all good
# faith or mild (docs/FINDINGS 2026-10-25, passed its rule on capture_20261025T0819).
ALERT_NEEDS_TOXIC = True
# "Likely to keep it up": cold-start model of who STAYS toxic, from what the
# engine sees in their first messages (docs/COLDSTART.md; forward-tested on 474k
# unseen snapshots: within-channel AUC 0.662 vs 0.624 for the bad-faith score)
PERSIST_PATH = "data/models/persist_lr.json"
PERSIST_UP_TO = 10          # the last of coldstart.CHECKPOINTS: past it the persist model is not shown (audit LB4-6)
TYPES_PATH = "data/models/types.json"      # data-driven behaviour types (tools/fit_types.py)
TYPE_FROM_N = 5
_TYPES = None


def _types():
    global _TYPES
    if _TYPES is None:
        _TYPES = False
        if os.path.exists(TYPES_PATH):
            import numpy as np
            t = json.load(open(TYPES_PATH))
            _TYPES = {"f": t["features"], "mu": np.array(t["mean"]), "sd": np.array(t["scale"]),
                      "c": np.array(t["centers"]), "names": t["names"]}
    return _TYPES
# Owner standard: mods and chat are members of the community, and their reactions
# reflect that community's rules (sometimes viewpoint, FINDINGS "mod actions").
# Timeouts, bans, deletions and call-outs are shown as context; they are not
# evidence about the person. Mods/VIPs/the broadcaster are held to the same
# standard as everyone else (their role is shown).
REACTIONS_AS_EVIDENCE = False
SEVERE_KINDS = {"hate", "threat", "sexual", "harassment"}
SHAPE_TYPES = {"spammer", "injector", "concern_troll"}

# Heads-up panel (docs/DASHBOARD.md; set from judged chatters by tools/eval_panel.py).
# One card per person, two levels; a single toxic message is a chip in chat, not a card.
# Shown as "likely" (red, key 'act') and "possible" (amber, key 'watch'): a read, never an
# instruction; the person watching decides (owner, 2026-10-25).
#   act:   a slur or dehumanising term, or a threat / sexual / harassment cue at high level
#          (keywords and coded vocabulary, or the LLM's re-read; not the model's identity score)
#   watch: a bad-faith alert, or 3+ toxic messages in 10 minutes. With the local LLM on
#          (PERSON_REVIEW): the LLM's read of the person says bad faith (docs/FINDINGS
#          2026-10-25 "bad-faith read"); alerts and toxic counts only ask for that read.
# Judged chatters (4 sets, LLM re-reads applied): act 68% bad faith, 88% bad or mild, finds
# 72% of bad faith; watch-only 27% bad, 73% bad or mild (a heads-up, not a verdict).
# "Likely to keep it up" never makes 'act' on its own: on 39 channels it rose with the
# room (political chats), judged good faith in 14 of 40 cases (FINDINGS "heads-up panel").
NOTICE_PERSIST_WATCH = 0.3
NOTICE_RECENT_MS = 10 * 60 * 1000
NOTICE_RECENT_N = 3            # 2 in 10 min added ~30 people per channel-hour, mostly good faith or mild
NOTICE_SHOW_MS = {"act": 30 * 60 * 1000, "watch": 15 * 60 * 1000}   # after the last message
# Who the local LLM reads as a person (the pre-filter; it found 91% of judged bad-faith
# chatters): a toxic message, a bad-faith alert, or a bad-faith score this high.
REVIEW_BF = 0.35
REVIEW_MAX = 3                 # reads per person: on becoming a candidate, then on new toxic messages
# A queued read that has not come back in this long was lost (the queue holds at most 200 reads, a
# few seconds each): the chatter is a candidate again. Restores D7 after a restart or a lost read.
REVIEW_PENDING_MS = 10 * 60 * 1000
# Per-chatter context lists for the user card (LB-19): mod actions seen, and reactions. Callouts are kept
# whatever the cap: their count feeds the cold-start risk vector (risk.py "callouts").
MOD_ACTIONS_KEPT = 50
REACTIONS_KEPT = 100
# The watchlist's evidence (owner, 2026-11-01; docs/DASHBOARD.md "Why someone is listed"): per chatter, every message
# whose read is or was toxic, or that the owner corrected, kept past the 40 recent messages (oldest dropped first), so
# a listing can still be corrected after its message leaves u.msgs. The same dicts as in u.msgs while they are there,
# so a re-read or a correction updates both.
EVIDENCE_KEPT = 100
# A standing flag keeps the card (owner, 2026-11-07: "if a user says even just one thing that is flagged and review
# maintains that it should be flagged then they will always be on the watchlist"): _notice_reasons reads the evidence
# too, so a slur or severe line that no review has cleared still counts after it leaves the 40 recent messages (before
# this, it dropped out of the rule there: Route B, data/study/person_level/DEV.md). A cleared line (a re-read to
# "none", the owner's "Not toxic") stays listed for correction and counts for nothing; "N toxic messages in 10 min" is
# by time anyway. When the evidence is full, the oldest line that is not a standing slur or severe flag goes first.
FLAGS_PERSIST = True
_HATE_WORDS = ("hate", "slur", "dehuman", "racis", "homophob", "transphob", "antisemit")


_LV = {"none": 0, "low": 1, "medium": 2, "high": 3}


def _note_reaction(u, item):
    """Append to u.reactions, dropping the oldest non-callout entry past REACTIONS_KEPT (LB-19)."""
    u.reactions.append(item)
    if len(u.reactions) > REACTIONS_KEPT:
        i = next((j for j, r in enumerate(u.reactions) if r[1] != "callout"), None)
        if i is not None:
            del u.reactions[i]


def _keep_evidence(u, m):
    """Keep one of the chatter's message dicts (the object in u.msgs) in their evidence, once (EVIDENCE_KEPT). The
    deque is made on the first one: most chatters never have a toxic message."""
    ev = u.evidence
    if ev is None:
        ev = u.evidence = deque(maxlen=EVIDENCE_KEPT)
    elif any(x is m for x in ev):
        return
    if FLAGS_PERSIST and ev.maxlen and len(ev) >= ev.maxlen:
        # full: drop the oldest line that is not a standing slur or severe flag (all flags: the oldest, as before)
        drop = next((x for x in ev if _tox_class(x.get("toxicity")) not in ("hate", "severe")), None)
        if drop is not None:
            ev.remove(drop)
    ev.append(m)


def _owner(t):
    """The owner corrected this message by hand (the dashboard's "reclassify"): their class stands, for
    the live watchlist only. Nothing that trains or evaluates reads it (docs/DASHBOARD.md)."""
    return bool(t) and t.get("source") == "owner"


def _auto_summary(auto):
    """What an owner correction's tooltip says the tool read (the level it counted at, its kinds, its source)."""
    return {"level": _level(auto), "kinds": list(auto.get("kinds") or ()), "source": auto.get("source", "layer")}


def _counted_kinds(t):
    """The kinds a message's read adds to the chatter's toxic_kinds (the card's "toxic.kinds"): its kinds when
    it counts as toxic, else none. Under S1 a keyword slur the LLM re-read without a kind keeps the layer's kinds."""
    if _level(t) == "none":
        return []
    kinds = list(t.get("kinds") or ())
    return (kinds or list((t.get("layer") or {}).get("kinds") or ())) if _layer_slur(t) else kinds


def _layer_slur(t):
    """S1: an LLM re-read of a message the keyword layer read as sure hate (keywords or coded
    vocabulary). The absolute standard: the re-read cannot clear the slur."""
    return (t.get("source") == "llm" and "hate" in ((t.get("layer") or {}).get("sure") or ()))


def _level(t):
    """The level a toxicity read counts at (panel, toxic_n). The LLM's level, except that under S1 a
    re-read of a keyword slur counts at least at the layer's level."""
    lv = (t or {}).get("level", "none")
    if _owner(t):
        return lv
    if t and _layer_slur(t):
        ll = t["layer"].get("level", "none")
        return ll if _LV.get(ll, 0) > _LV.get(lv, 0) else lv
    return lv


def _sure_hate(t):
    """Hate from the keyword layer's sure list (keywords, coded vocabulary), not only from an LLM read."""
    if _owner(t):
        return "hate" in (t.get("kinds") or ())     # the owner said slur or hate: it stands like a keyword slur
    src = (t.get("layer") or {}) if t.get("source") == "llm" else t
    return "hate" in set(src.get("sure", src.get("kinds") or ()))


def _tox_class(t):
    """'hate', 'severe' (high-level threat / sexual / harassment) or None for one toxicity read.
    The LLM's re-read, when there is one, replaces the fast layer's (and can clear it), except a
    keyword slur (S1)."""
    if _owner(t):
        if t.get("level", "none") == "none":
            return None
        kinds = set(t.get("kinds") or ())
        return "hate" if "hate" in kinds else "severe" if t["level"] == "high" and kinds & {"threat", "sexual", "harassment"} else None
    if t and _layer_slur(t):
        return "hate"
    if not t or t.get("level", "none") == "none":
        return None
    if t.get("source") == "llm":
        kinds = " ".join(t.get("kinds") or ()).lower()
        return "hate" if any(w in kinds for w in _HATE_WORDS) else "severe" if t["level"] == "high" else None
    sure = set(t.get("sure", t.get("kinds") or ()))      # older records: no "sure" list
    if "hate" in sure:
        return "hate"
    if t["level"] == "high" and sure & {"threat", "sexual", "harassment"}:
        return "severe"
    return None

BEHAVIOUR_TAGS = [  # handoff 01_METHODS §2 rules, over running ratios
    ("conversationalist", lambda b: b["avg_len"] >= 55),
    ("one-liner", lambda b: b["avg_len"] <= 25),
    ("emote-forward", lambda b: b["emotes_per_msg"] >= 1.5),
    ("asker", lambda b: b["question_rate"] >= 0.12),
    ("socialiser", lambda b: b["mention_rate"] >= 0.06),
    ("channel-games", lambda b: b["cmd_rate"] >= 0.20),
    ("shouter", lambda b: b["caps_rate"] >= 0.25),
]


_ARCHIVE_EMOTES = None


def _archive_emotes():
    """The emote vocabulary the frozen baseline was built with (loaded once)."""
    global _ARCHIVE_EMOTES
    if _ARCHIVE_EMOTES is None:
        import json
        _ARCHIVE_EMOTES = frozenset(json.load(open(os.path.join(HERE, "assets", "emote_vocab.json")))["emotes"])
    return _ARCHIVE_EMOTES


class User:
    __slots__ = ("uid", "login", "display", "channel", "first_seen", "last_seen",
                 "n", "saw_first_msg", "msgs", "msg_ev", "acct_ev", "acct_why",
                 "react_ev", "reactions", "why", "shapes", "friction_msgs",
                 "instigated", "joined", "mod_actions", "peak_bf", "alerted",
                 "bf_sum", "bf_recent", "last_f", "badges", "sub_months", "color", "agg", "held",
                 "toxic_n", "toxic_kinds", "vocab", "review", "evidence", "owner_cleared")

    def __init__(self, uid, login, channel, ts):
        self.uid, self.login, self.display, self.channel = uid, login, login, channel
        self.first_seen = self.last_seen = ts
        self.n = 0
        self.saw_first_msg = False
        self.msgs = deque(maxlen=40)
        self.msg_ev = dict.fromkeys(A.NAMES, 0.0)
        self.acct_ev = dict.fromkeys(A.NAMES, 0.0)
        self.acct_why = []
        self.react_ev = dict.fromkeys(A.NAMES, 0.0)
        self.reactions = []
        self.why = {}                 # (archetype, reason) -> summed weight
        self.shapes = deque(maxlen=12)
        self.friction_msgs = self.instigated = self.joined = 0
        self.mod_actions = []
        self.peak_bf = 0.0
        self.alerted = False
        self.held = False             # crossed the threshold while the room was over budget
        self.toxic_n = 0              # messages read as toxic (level >= low), whoever they target
        self.toxic_kinds = Counter()
        self.review = None            # the local LLM's read of the person (person_review)
        self.evidence = None          # deque of message dicts behind a listing (EVIDENCE_KEPT), made when first needed
        self.owner_cleared = False    # an owner correction took their alert back (no toxic message left): restored
        #                               when a correction gives them one again (apply_owner_override, audit LB3-1)
        from .vocab import PersonVocab
        self.vocab = PersonVocab()
        self.bf_sum = 0.0
        self.bf_recent = deque(maxlen=10)
        self.last_f = None
        self.badges, self.sub_months, self.color = {}, 0, ""
        self.agg = risk.new_agg()

    def logscore(self):
        pri = A.prior_logscore()
        return {a: pri[a] + self.acct_ev[a] + A.DAMPING * A.saturate(a, self.msg_ev[a]) + self.react_ev[a]
                for a in A.NAMES}

    def post(self):
        return A.posterior(self.logscore())

    def badfaith(self, post=None):
        post = post or self.post()
        return sum(post[a] for a in A.BAD)

    def add_reason(self, a, w, reason):
        k = (a, reason)
        self.why[k] = self.why.get(k, 0.0) + w
        if len(self.why) > 200:       # keep the strongest reasons only
            for k2 in sorted(self.why, key=lambda k: abs(self.why[k]))[:50]:
                del self.why[k2]


class IdIndex:
    """Where a user-id sits among every id seen: Twitch ids are assigned
    roughly in creation order, so the top percentiles are the newest accounts.
    No API needed; it calibrates itself from the channels being watched."""

    def __init__(self):
        self.ids = []
        self.seen = set()

    def add(self, uid):
        if uid and uid.isdigit() and uid not in self.seen:
            self.seen.add(uid)
            bisect.insort(self.ids, int(uid))

    def newness(self, uid):
        if not uid or not uid.isdigit() or len(self.ids) < 500:
            return None
        return bisect.bisect_left(self.ids, int(uid)) / len(self.ids)


class Engine:
    # The owner's "bot" role (roles.py; owner 2026-11-01): user ids the live dashboard does not score, profile or list,
    # like the reviewed bot list (is_bot). Empty here: serve.py sets it for a live run only, and a saved state never
    # carries it (serve._detached, upgrade), so no replay, training or evaluation path sees it.
    owner_bots = frozenset()

    def __init__(self, ml=None, emotes=None, nicknames=None, legacy=True,
                 learn_from_mods=True, alert_budget=20):
        """ml: optional object with .score(list[str]) -> list[dict] (models.py)
        emotes: optional EmoteSets (emotes.py) for third-party emote codes
        nicknames: {channel: [names chat uses for the broadcaster]}"""
        from .room import Room
        self._Room = Room
        self.ml = ml
        self.emotes = emotes
        self.nicknames = nicknames or {}
        self.rooms = {}
        self.users = {}               # (channel, uid) -> User
        self.by_login = {}            # (channel, login) -> uid
        self.names = {}               # channel -> set of sender logins
        self.suspects = {}            # channel -> deque[(ts, uid, login)]
        self.ids = IdIndex()
        self.legacy = {} if legacy else None
        # live: a timeout is evidence about someone. Evaluating against mod
        # actions: it must not be, or the score would contain its own label.
        self.learn_from_mods = learn_from_mods
        # Twitch's developer agreement forbids building databases "to collect
        # information about Twitch's end users": profiles stay per channel (no cross-channel tracking)
        # alerts per channel per hour before only the strongest cases get through
        # (research/02 #3: set thresholds by alert budget, not by AUC)
        self.alert_budget = alert_budget
        self.alert_times = {}
        self.alerts = deque(maxlen=500)
        # "will act" (risk_lr): no longer scored here (audit LB4-7: nothing read it); still attached, as the
        # replay loops (coldstart, coldstore, evaluate) switch it off and serve.py detaches it to save
        self.risk = risk.RiskModel()
        self.persist = risk.RiskModel(PERSIST_PATH)
        self.room_tox = {}              # channel -> [deque[(ts, toxic)], toxic count] over risk.ROOM_WINDOW_MS
        self.room_actions = Counter()   # channel -> mod actions seen
        self.room_msgs = Counter()
        self.stats = Counter()
        # set by the dashboard when the local LLM reads people (serve --llm-review):
        # 'watch' then comes from that read instead of alerts and toxic counts
        self.person_review = False

    # ----------------------------------------------------------- helpers
    def room(self, ch):
        r = self.rooms.get(ch)
        if r is None:
            r = self.rooms[ch] = self._Room(ch, self.nicknames.get(ch, ()))
            self.names[ch] = set()
            self.suspects[ch] = deque(maxlen=50)
            if self.legacy is not None:
                self.legacy[ch] = legacy_style.Characterizer(BASELINE)
                self.legacy[ch].emotes = set(_archive_emotes())
        return r

    def _user(self, ch, uid, login, ts):
        key = (ch, uid or login)
        u = self.users.get(key)
        if u is None:
            u = self.users[key] = User(uid or login, login, ch, ts)
            idx = self.__dict__.get("_by_ch")
            if idx is not None:
                idx.setdefault(ch, {})[key[1]] = u
        return u

    def channel_users(self, ch):
        """One channel's chatters in the order of self.users, from a per-channel index, so the
        room snapshot every 2 s does not scan every saved chatter (performance P7). The index is
        rebuilt when its size and self.users disagree: state saved without it, or chatters
        deleted from self.users directly."""
        idx = self.__dict__.get("_by_ch")
        if idx is None or sum(map(len, idx.values())) != len(self.users):
            idx = self._by_ch = {}
            for (c, uid), u in self.users.items():
                idx.setdefault(c, {})[uid] = u
        return list(idx.get(ch, {}).values())

    def drop_users(self, keys):
        """Delete chatters by (channel, uid) key, index included, and the hybrid reviewer's mark on them (PH4/PH5, audit
        2026-11-10: forget drops everything held about the user; a chatter pruned after 30 idle days who returns is
        reviewed on new content, their message count starting again from 1)."""
        idx = self.__dict__.get("_by_ch")
        rv = self.__dict__.get("hybrid_reviewed")
        for k in keys:
            u = self.users.pop(k, None)
            if u is not None and idx is not None:
                idx.get(k[0], {}).pop(k[1], None)
            if rv:
                rv.pop(k, None)

    def _alert(self, kind, ch, ts, **kw):
        a = {"type": "alert", "kind": kind, "channel": ch, "ts": ts, **kw}
        self.alerts.append(a)
        self.stats["alert_" + kind] += 1
        return a

    # ----------------------------------------------------------- events
    def process(self, ev):
        k = ev["kind"]
        if k == "msg":
            return self._on_msg(ev)
        ch = ev.get("channel", "")
        if not ch:
            return None
        room = self.room(ch)
        if ev.get("room_id"):
            room.room_id = ev["room_id"]
        if k == "clearchat" and ev["target_login"]:
            return self._on_modaction(ev, room)
        if k == "clearmsg":
            return self._on_modaction(ev, room)
        if k == "usernotice":
            if ev["msg_id"] == "raid":
                room.add_raid(ev["ts"], ev["login"], ev["viewers"])
                return {"type": "room_event", "kind": "raid", "channel": ch, "ts": ev["ts"],
                        "from": ev["login"], "viewers": ev["viewers"]}
            if ev["msg_id"] in ("sub", "resub", "subgift", "submysterygift") and ev["user_id"]:
                u = self.users.get((ch, ev["user_id"]))
                if u and ev["user_id"] not in self.owner_bots:      # a marked bot is not profiled (audit LB3-5)
                    u.msg_ev["supporter"] += 1.0
                    u.add_reason("supporter", 1.0, ev["msg_id"])
            m = _scored_notice(ev)
            return self._on_msg(m) if m else None
        if k == "roomstate":
            room.roomstate.update({t: v for t, v in ev["tags"].items()
                                   if t in ("followers-only", "slow", "emote-only", "subs-only", "r9k")})
        return None

    def _on_modaction(self, ev, room):
        ch, login = ev["channel"], ev["target_login"].lower()
        uid = ev.get("target_user_id") or self.by_login.get((ch, login))
        if ev["kind"] == "clearmsg":
            kind, detail = "deleted", ev.get("text", "")[:80]
        elif ev["duration"] is None or ev["duration"] < 0:
            kind, detail = "ban", "permanent"
        else:
            kind, detail = "timeout", f"{ev['duration']}s"
        room.add_mod_action(login)
        room.note_pile(ev["ts"], action=True)
        self.room_actions[ch] += 1
        if kind != "deleted":
            room.add_ban(ev["ts"], login, None if kind == "ban" else ev["duration"])
        self.stats["mod_" + kind] += 1
        # the owner marked them a bot (live only): the room still sees the action (context), their profile does not
        # change while the role is set (audit LB3-5)
        u = self.users.get((ch, uid)) if uid and uid not in self.owner_bots else None
        if u is None:
            return {"type": "modaction", "channel": ch, "ts": ev["ts"], "login": login,
                    "kind": kind, "detail": detail, "known": False}
        if self.learn_from_mods and REACTIONS_AS_EVIDENCE:
            w, reason = A.reaction_evidence(kind, detail)
            for a, x in w.items():
                u.react_ev[a] += x
                u.add_reason(a, x, reason)
        u.mod_actions.append((ev["ts"], kind, detail))
        del u.mod_actions[:-MOD_ACTIONS_KEPT]
        _note_reaction(u, (ev["ts"], kind, detail))
        bf = u.badfaith()
        u.peak_bf = max(u.peak_bf, bf)
        return {"type": "modaction", "channel": ch, "ts": ev["ts"], "login": login,
                "user_id": u.uid, "kind": kind, "detail": detail, "known": True,
                "badfaith": round(bf, 3), "msgs_seen": u.n}

    def _on_msg(self, ev):
        ch, ts = ev["channel"], ev["ts"]
        login = ev["login"].lower()
        uid = ev["user_id"] or login
        if is_bot(login):
            return None
        if uid in self.owner_bots:
            # the owner marked them a bot (live only): not scored or profiled, like the reviewed list; the line still
            # shows in the page's chat, tagged, so the role can be seen and undone there (docs/DASHBOARD.md "Roles").
            # KeywordLayer has no such set: replays never have one
            return {"type": "bot_msg", "channel": ch, "ts": ts, "id": ev["id"], "user_id": uid, "login": login,
                    "display": ev["display"], "color": ev["color"], "text": ev["text"]}
        room = self.room(ch)
        if ev.get("room_id"):
            room.room_id = ev["room_id"]
        self.ids.add(ev["user_id"])
        self.names[ch].add(login)
        self.by_login[(ch, login)] = uid
        u = self._user(ch, uid, login, ts)
        # let a bad moment fade: message/reaction evidence halves every week
        # of absence, so one heated night doesn't follow someone forever
        gap_days = (ts - u.last_seen) / 86_400_000
        if gap_days > 0.5:
            k = 0.5 ** (gap_days / EVIDENCE_HALF_LIFE_DAYS)
            for a in A.NAMES:
                u.msg_ev[a] *= k
                u.react_ev[a] *= k
        u.display, u.last_seen = ev["display"], ts
        u.badges, u.color = ev["badges"], ev["color"]

        # KeywordLayer (below the class) repeats the steps up to here that the toxicity read
        # depends on, for replays without ML: keep the two in step
        third = self.emotes.codes(room.room_id) if self.emotes else frozenset()
        f = signals.extract(ev, room, third_party=third, names=self.names[ch],
                            prev_shapes=u.shapes, account_newness=self.ids.newness(ev["user_id"]))
        u.sub_months = f["sub_months"]
        if f["first_msg"]:
            u.saw_first_msg = True

        # optional ML scores, only where they can change a decision
        # every message with 2+ words: ~2 ms each, and the distilled toxicity
        # head needs the embedding (was: first 8 messages / cue hits only)
        if self.ml is not None and f["n_words"] >= 2 and not f["command"]:
            try:
                f.update(self.ml.score_one(ev["text"]))
                self.stats["ml_scored"] += 1
            except Exception as e:      # never let a model crash the stream
                self.stats["ml_error"] += 1
                f["ml_error"] = str(e)

        # oddness from the room's char model, computed before adding this msg
        s = room.char.surprisal(f["norm"]) if room.char.ready() else None
        f["oddness"] = room.oddness_pct(s)

        room.vocab.observe(login, f["vocab"])
        u.vocab.add(f["vocab"], room.vocab)
        tox = message_toxicity(f)
        if tox["level"] != "none":
            u.toxic_n += 1
            u.toxic_kinds.update(tox["kinds"])
            u.owner_cleared = False            # a new toxic message: the usual alert path decides again (LB3-1)
        rt =self.room_tox.setdefault(ch, [deque(), 0])
        rt[0].append((ts, tox["level"] != "none"))
        rt[1] += tox["level"] != "none"
        self.room_context(ch, ts)                  # prune the window as it goes

        # account evidence is refreshed (not accumulated) on every message
        u.acct_ev, u.acct_why = A.account_evidence(f)
        ev_m, why_m = A.message_evidence(f)
        # repeated messages from one person aren't independent evidence; the
        # first one is taken at face value (stored pre-divided so logscore's
        # uniform DAMPING factor leaves it whole)
        k = 1 / A.DAMPING if u.n == 0 else 1.0
        for a, w in ev_m.items():
            u.msg_ev[a] += k * w
        for a, w, reason in why_m:
            u.add_reason(a, (1.0 if u.n == 0 else A.DAMPING) * w, reason)
        bf_ev = max(ev_m[a] for a in A.BAD)
        u.bf_sum += bf_ev
        u.bf_recent.append(bf_ev)
        if f["friction"]:
            u.friction_msgs += 1
            if f["room_heated"]:
                u.joined += 1
            else:
                u.instigated += 1
        if f["shape"]:
            u.shapes.append(f["shape"])
        risk.update(u.agg, f, bf_ev, tox)
        u.n += 1
        u.last_f = f

        positive = 1.0 if (f["positive"] or f["families"].get("benign")) and not f["friction"] else 0.0
        room.add_message(ts, login, friction=f["friction"] + f["families"].get("creep", 0),
                         positive=positive, first=f["first_msg"], shape=f["shape"],
                         is_new=(f["account_age_days"] if f["account_age_days"] is not None else 9e9) < 7,
                         norm=f["norm"])

        room.modlearn.observe(login, f["norm"])
        room.add_chant(f["chant_key"], ts, login)
        if f["families"].get("injection"):
            room.add_divisive(ts, login)
        room.context.append((ts, ev["display"], ev["text"][:300]))
        if self.legacy is not None:
            lg = self.legacy[ch]
            lg.emotes.update(e[3] for e in ev["emotes"])
            lg.emotes.update(third)
            lg.observe(login, ev["text"], user_id=uid)

        # reactions: a callout lands on whoever it answers
        out_events = []
        callout = f["cues"].get("callout_mods") or (
            f["cues"].get("callout_social") if f["target_chatter"] else None)
        if callout:
            tgt = self._callout_target(ch, ev, f, uid, room)
            if tgt is not None:
                if REACTIONS_AS_EVIDENCE:
                    w, reason = A.reaction_evidence("callout", callout)
                    for a, x in w.items():
                        tgt.react_ev[a] += x
                        tgt.add_reason(a, x, reason)
                _note_reaction(tgt, (ts, "callout", f"{login}: {ev['text'][:60]}"))
                self.stats["callouts_attributed"] += 1

        post = u.post()
        bf = sum(post[a] for a in A.BAD)
        u.peak_bf = max(u.peak_bf, bf)
        suspicious = bool(f["friction"] or f["families"].get("creep") or f["families"].get("spam")
                          or f["families"].get("injection") or bf_ev >= 1.0)
        if suspicious:
            self.suspects[ch].append((ts, uid, login))

        flags = []
        if f["first_msg"]:
            flags.append("first")
        if f["evasion"]:
            flags.append("evasion")
        if f["template_new_accounts"] >= 2:
            flags.append("template")
        elif f["copypasta_others"] >= 2:
            flags.append("copypasta")
        if f["lookalike_of_banned"]:
            flags.append("lookalike")
        if f["oddness"] is not None and f["oddness"] >= 0.98:
            flags.append("odd")
        if f["room_heated"]:
            flags.append("heated-room")
        if f["moderated_topics"]:
            flags.append("moderated-topic")
        if f["ascii_art"]:
            flags.append("ascii-art")
        if (f["account_age_days"] or 9e9) < 7:
            flags.append("new-account")
        if any(v >= 5 for k, v in f["automod"].items() if k in "AIS"):
            flags.append("twitch-flag")
        if any(h["distress"] for h in f["vocab"]):
            flags.append("distress")
        elif any(h["tier"] != "T3" for h in f["vocab"]):
            flags.append("coded-vocab")
        ooc = self._out_of_character(u)
        if ooc:
            flags.append("out-of-character")

        top = sorted(post.items(), key=lambda kv: -kv[1])[:3]
        rec = {"type": "msg", "channel": ch, "ts": ts, "id": ev["id"], "user_id": uid,
               "login": login, "display": ev["display"], "color": ev["color"],
               "text": ev["text"], "badges": list(ev["badges"]), "n": u.n,
               "badfaith": round(bf, 3), "top": [(a, round(p, 3)) for a, p in top],
               "certainty": round(A.entropy_certainty(post), 3), "flags": flags,
               "reasons": [r for _, _, r in sorted(why_m, key=lambda x: -abs(x[1]))[:3]],
               "msg_badfaith_ev": round(bf_ev, 2), "toxicity": tox}
        if f.get("ml_toxicity") is not None:
            rec["tox"] = round(f["ml_toxicity"], 3)
        if ev.get("notice"):
            rec["notice"] = ev["notice"]              # S5: resub / milestone / announcement text
        u.msgs.append({"id": ev["id"], "ts": ts, "text": ev["text"], "badfaith": rec["badfaith"],
                       "ev": rec["msg_badfaith_ev"], "flags": flags, "reasons": rec["reasons"],
                       "toxicity": tox})
        if tox["level"] != "none":
            _keep_evidence(u, u.msgs[-1])

        # alerts: crossing a threshold, not every message after it
        thr = ALERT_FIRST_MSG if u.n == 1 and f["first_msg"] else ALERT_BADFAITH
        newish = u.saw_first_msg or (f["account_age_days"] if f["account_age_days"] is not None else 9e9) < 30
        priority = "high" if (newish or f["evasion"] or "slur" in f["cues"] or "threat" in f["cues"]
                                or "sexual_threat" in f["cues"]) else "normal"
        if bf >= thr and not u.alerted and (u.toxic_n or not ALERT_NEEDS_TOXIC) and not self._mild_only(u, top):
            times = self.alert_times.setdefault(ch, deque())
            while times and ts - times[0] > 3_600_000:
                times.popleft()
            over = self.alert_budget and len(times) >= self.alert_budget
            if not over or (bf >= 0.85 and priority == "high"):
                u.alerted = True
                times.append(ts)
                out_events.append(self._alert(
                    "badfaith", ch, ts, user_id=uid, login=login, display=ev["display"],
                    badfaith=round(bf, 3), top=rec["top"], text=ev["text"], n=u.n,
                    first=f["first_msg"], priority=priority, over_budget=bool(over),
                    reasons=self._top_reasons(u, 4), persist=(self._persist_of(u) or {}).get("p"),
                    role=("broadcaster" if "broadcaster" in ev["badges"] else "mod" if "moderator" in ev["badges"]
                          else "VIP" if "vip" in ev["badges"] else None)))
            elif not u.held:
                u.held = True
                self.stats["alerts_held_by_budget"] += 1
        elif u.alerted and bf < 0.25:
            u.alerted = False         # re-arm if they settle down
        room.note_pile(ts, first=f["first_msg"],
                       hostile_first=f["first_msg"] and bf_ev >= 1.0)
        pile = room.pile_on(ts)
        if pile and ts / 1000 >= room.pile_alerted_until:
            room.pile_alerted_until = ts / 1000 + 900
            out_events.append(self._alert("pileon", ch, ts, **pile))
        if f["pile_on"]:
            flags.append("pile-on")
        if f["first_msg"]:
            room.add_arrival(ts, login, (f["account_age_days"] if f["account_age_days"] is not None else 9e9) < 7,
                             f["name_randomness"] >= 0.5, bool(f["friction"] or f["families"].get("creep")))
            wave = room.newcomer_wave(ts)
            if wave:
                out_events.append(self._alert("wave", ch, ts, **wave))
        if f["template_new_accounts"] >= 3:
            out_events.append(self._alert("coordinated", ch, ts, text=ev["text"],
                                          accounts=f["template_new_accounts"] + 1))
        if f["lookalike_of_banned"] and u.n == 1:
            out_events.append(self._alert("lookalike", ch, ts, user_id=uid, login=login,
                                          resembles=f["lookalike_of_banned"], text=ev["text"]))
        self.stats["msgs"] += 1
        self.room_msgs[ch] += 1
        if out_events:
            rec["alerts"] = out_events
        return rec

    def emote_names(self, ch):
        """Every emote code known for a channel (first-party seen in tags + 7TV/BTTV/FFZ)."""
        r = self.rooms.get(ch)
        known = set(self.emotes.codes(r.room_id)) if (self.emotes and r) else set()
        if self.legacy is not None and ch in self.legacy:
            known |= {e for e in self.legacy[ch].emotes if not e.islower()} - set(_archive_emotes())
        return frozenset(known)

    def _mild_only(self, u, top):
        """True when the person's record is a one-off insult from someone otherwise
        taking part: toxic, not bad faith (no alert; a second hostile message,
        severe content or a bad-faith type changes that)."""
        if not ALERT_NEEDS_SHAPE:
            return False
        msgs = list(u.msgs)
        hostile = sum(1 for m in msgs if m["ev"] >= 1.0)
        # S1 (audit LB-12): an LLM re-read cannot clear a keyword or coded-vocabulary slur, here as on the panel
        severe = any(_layer_slur(m["toxicity"]) or (m["toxicity"]["level"] != "none"
                     and SEVERE_KINDS & set(m["toxicity"].get("kinds", ()))) for m in msgs)
        if severe or hostile >= 2 or (top and top[0][0] in SHAPE_TYPES):
            return False
        if hostile == len(msgs):
            return False                       # nothing but provocation
        if hostile == 1:
            self.stats["alerts_mild_insult"] += 1
            return True
        return False                           # score without a hostile message: keep the old behaviour

    def room_context(self, ch, now=None):
        """The room's last 20 minutes: share of toxic messages and pace."""
        rt = self.room_tox.get(ch)
        if not rt or not rt[0]:
            return {"toxic_share": 0.0, "msgs_per_min": 0.0}
        q = rt[0]
        now = now or q[-1][0]
        while q and now - q[0][0] > risk.ROOM_WINDOW_MS:
            rt[1] -= q.popleft()[1]
        return {"toxic_share": rt[1] / max(len(q), 1), "msgs_per_min": len(q) / (risk.ROOM_WINDOW_MS / 60000)}

    def room_action_rate(self, ch):
        """Mod actions per 100 messages so far in this room (smoothed)."""
        return 100 * (self.room_actions[ch] + 1) / (self.room_msgs[ch] + 200)

    def _callout_target(self, ch, ev, f, speaker_uid, room):
        """Who is this 'mods' / 'who asked' / 'ratio' aimed at? A reply or
        @mention says so explicitly; otherwise the most recent suspicious
        message from someone else in the last 20 s, if there is exactly one
        candidate (ambiguous = attribute to no one)."""
        tgt_login = f["target_chatter"]
        if tgt_login:
            uid = self.by_login.get((ch, tgt_login))
            return self.users.get((ch, uid)) if uid else None
        now = ev["ts"]
        if room.short.n * 60 / 90 > 60:
            return None               # too fast to tell who it answers
        cands = {uid for t, uid, _ in self.suspects[ch]
                 if now - t <= SUSPECT_LOOKBACK_S * 1000 and uid != speaker_uid}
        if len(cands) == 1:
            return self.users.get((ch, cands.pop()))
        return None

    def _out_of_character(self, u):
        if u.n < OOC_MIN_MSGS or len(u.bf_recent) < 5:
            return False
        life = u.bf_sum / u.n
        recent = sum(u.bf_recent) / len(u.bf_recent)
        return recent >= 0.8 and recent >= 4 * max(life, 0.1)

    def _persist_of(self, u):
        """"Keeps it up", only up to the message counts persist_lr is fitted and forward-tested at (audit LB4-6)."""
        if not getattr(self, "persist", None) or not self.persist.ok or u.n > PERSIST_UP_TO:
            return None
        p, top = self.persist.score(risk.vector(u, self.room_action_rate(u.channel), self.room_context(u.channel)))
        return {"p": round(p, 3), "why": top}

    def _type_of(self, u):
        """The data-driven behaviour type, from message TYPE_FROM_N on (None before)."""
        t = _types()
        if not t or u.n < TYPE_FROM_N:
            return None
        import numpy as np
        v = risk.vector(u, self.room_action_rate(u.channel), self.room_context(u.channel))
        z = (np.array([v.get(k, 0.0) for k in t["f"]]) - t["mu"]) / t["sd"]
        return t["names"][int(((t["c"] - z) ** 2).sum(1).argmin())]

    @staticmethod
    def _top_reasons(u, k=6):
        items = sorted(u.why.items(), key=lambda kv: -kv[1])
        out, seen = [], set()
        for (a, reason), w in items:
            if A.FAMILY[a] == "badfaith" and w > 0 and reason not in seen:
                seen.add(reason)
                out.append({"archetype": A.LABEL[a], "reason": reason, "weight": round(w, 2)})
            if len(out) >= k:
                break
        return out

    def upgrade(self):
        """Bring state pickled by an older version up to the current code: any
        attribute, slot or aggregate key added since is filled from a fresh
        default. Without this, new code crashes on old saved state."""
        fresh = Engine(legacy=self.legacy is not None)
        for k, v in vars(fresh).items():
            if not hasattr(self, k):
                setattr(self, k, v)
        self.__dict__.pop("owner_bots", None)       # the owner's roles are the live run's, never the saved state's
        for k in ("global_users", "cross_channel"):   # cross-channel tracking, removed (never switched on)
            self.__dict__.pop(k, None)
        if getattr(self, "risk", None) is None:     # detached by save_engine()
            self.risk = risk.RiskModel()
        if getattr(self, "persist", None) is None:
            self.persist = risk.RiskModel(PERSIST_PATH)
        from .room import Room
        for ch, r in self.rooms.items():
            proto = Room(ch, self.nicknames.get(ch, ()))
            for k, v in vars(proto).items():
                if not hasattr(r, k):
                    setattr(r, k, v)
            for k, v in vars(proto.modlearn).items():
                if not hasattr(r.modlearn, k):
                    setattr(r.modlearn, k, v)
        blank = User("0", "x", "x", 0)
        agg_keys = risk.new_agg()
        for u in self.users.values():
            missing = set()
            for slot in User.__slots__:
                try:
                    getattr(u, slot)
                except AttributeError:
                    missing.add(slot)
                    setattr(u, slot, copy.deepcopy(getattr(blank, slot)))
            for k, v in agg_keys.items():
                u.agg.setdefault(k, v)
            if not REACTIONS_AS_EVIDENCE:      # evidence from mod/chat reactions saved by older versions
                u.react_ev = dict.fromkeys(A.NAMES, 0.0)
            for a in A.NAMES:                 # archetypes added since
                u.msg_ev.setdefault(a, 0.0)
                u.acct_ev.setdefault(a, 0.0)
                u.react_ev.setdefault(a, 0.0)
            # a take-back saved on the message by the previous version (LB2-2) moves to the chatter (LB3-1)
            if "owner_cleared" in missing and u.toxic_n <= 0 and not u.alerted:
                u.owner_cleared = any("owner_cleared_alert" in m for m in list(u.msgs) + list(u.evidence or ()))
            if u.review and u.review.get("pending"):     # its queue died with the process that saved it: the read
                u.review = {**u.review, "pending": False, "lost": True}     # is lost (LB-3, audit LB2-3)
        return self

    def wants_person_review(self, rec):
        """Should the local LLM read this chatter now? On becoming a candidate (a toxic
        message, a bad-faith alert, or a bad-faith score >= REVIEW_BF), then again on
        each new toxic message, at most REVIEW_MAX times."""
        u = self.users.get((rec["channel"], rec["user_id"]))
        if u is None:
            return False
        r = getattr(u, "review", None)
        if r and (r.get("reads", 0) >= REVIEW_MAX or _pending(r, rec["ts"])):
            return False
        toxic_now = (rec.get("toxicity") or {}).get("level", "none") != "none"
        alerted = any(a["kind"] == "badfaith" for a in rec.get("alerts", ()))
        if r is None:
            return toxic_now or alerted or rec.get("badfaith", 0) >= REVIEW_BF
        return toxic_now

    def wants_person_review_llm(self, ch, uid, tox, ts):
        """S11 (RULE_PHASE3): an LLM re-read (tox, the message's new read) that gives the chatter
        LLM-only hate (an LLM hate read on a line with no keyword or coded-vocabulary slur) asks for a
        person read, at most REVIEW_MAX reads and none while one is pending, so the read S11 needs
        arrives (publish() queues reads on messages only, and a re-read of an unflagged line comes
        after its message)."""
        if not getattr(self, "person_review", False):
            return False
        if _tox_class(tox) != "hate" or _sure_hate(tox):
            return False
        u = self.users.get((ch, uid))
        if u is None:
            return False
        r = getattr(u, "review", None)
        return not (r and (r.get("reads", 0) >= REVIEW_MAX or _pending(r, ts)))

    def mark_review_pending(self, ch, uid, ts=None):
        """The chatter's read is queued (at ts, default their last message)."""
        u = self.users.get((ch, uid))
        if u is not None:
            r = getattr(u, "review", None) or {"reads": 0}
            u.review = {**r, "pending": True, "queued": u.last_seen if ts is None else ts}

    def release_person_review(self, ch, uid):
        """A queued read that will not run (the dashboard's Stop, or a read that returned after it): no longer
        pending. No read is spent and it is not an error, but the read is lost: until one comes back the older
        rule judges the chatter, as for a failed read (LB-3 "whose read failed or is lost"; audit LB2-3)."""
        u = self.users.get((ch, uid))
        if u is not None and getattr(u, "review", None) and u.review.get("pending"):
            u.review = {**u.review, "pending": False, "lost": True}

    def apply_person_review(self, ch, uid, review, ts):
        """Store the local LLM's read of a person (llm.review). Returns True when it
        changes whether they are on the panel."""
        u = self.users.get((ch, uid))
        if u is None:
            return False
        old = getattr(u, "review", None) or {}
        was = _review_bad(u)
        if review.get("error"):
            u.review = {**old, "pending": False, "failed": True}       # LB-3: no read came back
            self.stats["llm_person_errors"] += 1
            return False
        u.review = {"label": review.get("label"), "rationale": review.get("rationale", ""), "rules": review.get("rules", []),
                    "n": u.n, "ts": ts, "reads": old.get("reads", 0) + 1, "pending": False}
        self.stats["llm_person_reads"] += 1
        self.stats[f"llm_person_{review.get('label')}"] += 1
        return was != _review_bad(u)

    def apply_review(self, ch, uid, review, ts):
        """Fold a policy-LLM verdict into a user's read. Returns an alert dict if
        it pushes them over the threshold for the first time, else None."""
        u = self.users.get((ch, uid))
        if u is None or review.get("error"):
            return None
        w, reason = A.llm_evidence(review)
        if not w:
            return None
        for a, x in w.items():
            u.react_ev[a] += x
            u.add_reason(a, x, reason)
        _note_reaction(u, (ts, "llm", f"{review.get('label')}: {review.get('rationale', '')[:120]}"))
        bf = u.badfaith()
        u.peak_bf = max(u.peak_bf, bf)
        thr = ALERT_FIRST_MSG if u.saw_first_msg and u.n <= 3 else ALERT_BADFAITH
        # the adopted rule (FINDINGS 2026-10-25, audit LB-13): a bad-faith alert needs a toxic message, here too
        if bf >= thr and not u.alerted and (u.toxic_n or not ALERT_NEEDS_TOXIC):
            u.alerted = True
            post = u.post()
            top = sorted(post.items(), key=lambda kv: -kv[1])[:3]
            last = u.msgs[-1]["text"] if u.msgs else ""
            return self._alert("badfaith", ch, ts, user_id=u.uid, login=u.login, display=u.display,
                               badfaith=round(bf, 3), top=[(a, round(p, 3)) for a, p in top], text=last,
                               n=u.n, first=u.saw_first_msg, priority="high", over_budget=False,
                               via="llm-screen", reasons=self._top_reasons(u, 4))
        return None

    LV = {"none": 0, "low": 1, "medium": 2, "high": 3}
    LV_SCORE = {"none": 0.0, "low": 0.4, "medium": 0.6, "high": 0.85}

    def apply_llm_toxicity(self, ch, uid, msg_id, res):
        """Replace one message's toxicity read with the LLM's (which reads
        meaning: FINDINGS 3b). Returns a msg_update record for the UI, or None."""
        u = self.users.get((ch, uid))
        # a line still held: in the recent messages, else (FLAGS_PERSIST) behind a listing, so a review can still clear
        # a flag that left the 40 recent messages before its re-read came back
        m = (self.held_message(ch, uid, msg_id) if FLAGS_PERSIST
             else next((x for x in (u.msgs if u else []) if x.get("id") == msg_id), None))
        if m is None or res.get("toxicity") not in self.LV:
            return None
        old = m.get("toxicity") or {"level": "none", "kinds": []}
        lvl = res["toxicity"]
        tgt = {"chatter": "a chatter", "third_party": "third party"}.get(res.get("target", ""), res.get("target", ""))
        # the owner's correction stands: the re-read only updates the automatic read kept under it
        auto = (m.get("toxicity_auto") or {"level": "none", "kinds": []}) if _owner(old) else old
        # S1: a second re-read keeps the keyword layer's read as its layer (not the first re-read)
        layer = auto["layer"] if auto.get("source") == "llm" and auto.get("layer") else auto
        new = {"score": self.LV_SCORE[lvl], "level": lvl, "kinds": [res["kind"]] if res.get("kind") else [],
               "target": tgt if lvl != "none" else "", "source": "llm", "layer": layer}
        if _owner(old):
            # the correction keeps standing; its tooltip ("the tool read it as ...") follows the new automatic read,
            # which is what "Back to automatic" would restore (audit LB2-8)
            m["toxicity_auto"] = new
            m["toxicity"] = {**old, "auto": _auto_summary(new)}
            self.stats["llm_tox_under_owner"] += 1
            return {"type": "msg_update", "channel": ch, "user_id": uid, "id": msg_id, "toxicity": m["toxicity"]}
        # toxic_n counts at _level: the LLM's level, or under S1 at least a keyword slur's layer level
        if (_level(old) == "none") != (_level(new) == "none"):
            u.toxic_n += 1 if _level(new) != "none" else -1
            if _level(new) != "none":
                u.owner_cleared = False        # toxic again by the LLM's read: the usual alert path decides (LB3-1)
        # the card's kinds count each message once, by its current read, as for corrections (audit LB3-2)
        u.toxic_kinds.subtract(_counted_kinds(old))
        u.toxic_kinds.update(_counted_kinds(new))
        m["toxicity"] = new
        if _level(new) != "none":
            _keep_evidence(u, m)
        self.stats["llm_tox_" + ("changed" if old["level"] != lvl else "same")] += 1
        return {"type": "msg_update", "channel": ch, "user_id": uid, "id": msg_id, "toxicity": new}

    OWNER_KINDS = ("hate", "threat", "sexual", "harassment")
    # the menu's choices, (level, kind); (None, None) is "Back to automatic" (audit LB2-6: no other pair)
    OWNER_CHOICES = frozenset({(None, None), ("none", None), ("low", None), ("medium", None)}
                              | {("high", k) for k in OWNER_KINDS})

    @staticmethod
    def owner_same(t, level, kind=None):
        """The message already carries exactly this correction (a repeated click: nothing to change)."""
        return (level is not None and _owner(t) and t.get("level") == level
                and list(t.get("kinds") or ()) == ([kind] if kind and level != "none" else []))

    def apply_owner_override(self, ch, uid, msg_id, level, kind=None):
        """The owner reclassifies one message (the dashboard's menu): level none/low/medium/high, with a kind
        (hate, threat, sexual, harassment) for the severe choices; level None goes back to the automatic read.
        Only the menu's pairs are accepted (OWNER_CHOICES).
        The correction stands over the keyword layer, the model and any LLM re-read (a later re-read only
        updates the automatic read kept under it), so it decides the live watchlist through _level,
        _tox_class and _sure_hate. It is never a label: no training, silver or evaluation path reads it
        (docs/DASHBOARD.md "Reclassifying a message"). Returns a msg_update record, or None when the message
        is no longer held (held_message: the recent messages and the evidence). The same correction twice changes
        nothing. A message corrected while only in the evidence has the same effects (toxic_n, toxic_kinds, the
        alert); the watchlist rule reads u.msgs alone, so it changes the listing only through those."""
        u = self.users.get((ch, uid))
        m = self.held_message(ch, uid, msg_id)
        if m is None or (level, kind) not in self.OWNER_CHOICES:
            return None
        old = m.get("toxicity") or {"level": "none", "kinds": []}
        if self.owner_same(old, level, kind):
            self.stats["owner_overrides_repeated"] += 1
            return {"type": "msg_update", "channel": ch, "user_id": uid, "id": m.get("id"), "toxicity": old}
        auto = (m.get("toxicity_auto") or {"level": "none", "kinds": []}) if _owner(old) else old
        if level is None:
            if not _owner(old):
                return None
            new = auto
            m.pop("toxicity_auto", None)
        else:
            new = {"score": self.LV_SCORE[level], "level": level, "kinds": [kind] if kind and level != "none" else [],
                   "target": auto.get("target", "") if level != "none" else "", "source": "owner",
                   "auto": _auto_summary(auto)}
            m["toxicity_auto"] = auto
        if (_level(old) == "none") != (_level(new) == "none"):
            u.toxic_n += 1 if _level(new) != "none" else -1
        # the card's kinds count each message once, by its current read: the old read's kinds go, the new one's
        # come (audit LB2-7: repeated corrections and undo used to add up). Display only (profile "toxic.kinds")
        u.toxic_kinds.subtract(_counted_kinds(old))
        u.toxic_kinds.update(_counted_kinds(new))
        # ALERT_NEEDS_TOXIC (FINDINGS 2026-10-25; audit LB2-2): a bad-faith alert needs a toxic message. A correction
        # that leaves the chatter with none takes the alert back (re-armed by the usual path); a correction that gives
        # them a toxic message again, whichever message it is, restores it (audit LB3-1: it used to wait for the very
        # message whose correction took it back). A new toxic message from chat or the LLM hands it to the usual path
        if ALERT_NEEDS_TOXIC:
            lifted = _level(old) == "none" and _level(new) != "none"
            if u.toxic_n <= 0 and u.alerted:
                u.alerted = False
                u.owner_cleared = True
                self.stats["owner_alerts_cleared"] += 1
            elif u.toxic_n > 0 and lifted and u.owner_cleared:
                u.alerted, u.owner_cleared = True, False
                self.stats["owner_alerts_restored"] += 1
        m["toxicity"] = new
        _keep_evidence(u, m)                   # listed in the dropdown from now on, "Not toxic" included, to undo it
        self.stats["owner_overrides" if level is not None else "owner_overrides_undone"] += 1
        return {"type": "msg_update", "channel": ch, "user_id": uid, "id": m.get("id"), "toxicity": new}

    def held_message(self, ch, uid, msg_id):
        """One of a chatter's messages still held: in their recent messages (u.msgs), else in their evidence."""
        u = self.users.get((ch, uid))
        if u is None:
            return None
        mid = str(msg_id)
        for src in (u.msgs, getattr(u, "evidence", None) or ()):
            for x in src:
                if str(x.get("id")) == mid:
                    return x
        return None

    def evidence(self, ch, uid):
        """The messages behind a chatter's listing, oldest first, for the dashboard's dropdown (EVIDENCE_KEPT): their
        evidence, plus a toxic or corrected message in u.msgs that is not in it (a state saved before it existed).
        "counted": the watchlist rule reads it now (toxic; with FLAGS_PERSIST at any age, before it only among the recent
        messages); "older": it has left the recent messages and is listed so it can still be corrected. None for an
        unknown chatter."""
        u = self.users.get((ch, uid))
        if u is None:
            return None
        recent = {id(m) for m in u.msgs}
        seen, out = set(), []
        extra = [m for m in u.msgs if _level(m.get("toxicity")) != "none" or _owner(m.get("toxicity"))]
        for m in list(getattr(u, "evidence", None) or ()) + extra:
            if id(m) in seen:
                continue
            seen.add(id(m))
            toxic = _level(m.get("toxicity")) != "none"
            out.append({"id": m.get("id"), "ts": m.get("ts"), "text": m.get("text", ""), "toxicity": m.get("toxicity"),
                        "flags": m.get("flags", []), "counted": toxic and (FLAGS_PERSIST or id(m) in recent),
                        "older": id(m) not in recent})
        out.sort(key=lambda d: d["ts"] or 0)
        return out

    def forget(self, uid):
        """Opt-out / deletion request: drop everything held about one user-id: their profiles in every
        channel, the name indexes, register sums and panel cache, their alerts, and the room structures
        that keep their login or display name (recent lines for the LLM, coded-vocabulary membership,
        mod-topic learning, recent bans, shapes, chants, arrivals, divisive topics). The account-age
        index drops the id. Returns how many per-channel profiles were removed. The file on disk changes
        when the state is saved again (python -m chatwatch.forget does it for the saved copies)."""
        uid = str(uid)
        gone = [k for k in self.users if k[1] == uid]
        names = {}                                       # channel -> lower-case login and display names
        for k in gone:
            u = self.users[k]
            names.setdefault(k[0], set()).update(n.lower() for n in (u.login, u.display) if n)
            if self.by_login.get((k[0], u.login)) == uid:
                del self.by_login[(k[0], u.login)]
            self.drop_users([k])
            if self.legacy is not None and k[0] in self.legacy:
                self.legacy[k[0]].profiles.pop(uid, None)
            self.__dict__.get("_notice_p", {}).pop(k, None)
        for dq in self.suspects.values():
            for item in [x for x in dq if x[1] == uid]:
                dq.remove(item)
        everyone = set().union(*names.values()) if names else set()
        kept = [a for a in self.alerts if str(a.get("user_id", "")) != uid
                and str(a.get("login", "")).lower() not in everyone and str(a.get("resembles", "")).lower() not in everyone]
        for a in kept:
            if a.get("logins"):
                a["logins"] = [x for x in a["logins"] if x.lower() not in everyone]
        self.alerts = deque(kept, maxlen=self.alerts.maxlen)
        for ch, ls in names.items():
            self.names.get(ch, set()).difference_update(ls)
            r = self.rooms.get(ch)
            if r is None:
                continue
            r.context = deque((x for x in r.context if str(x[1]).lower() not in ls), maxlen=r.context.maxlen)
            r.vocab.chatters.difference_update(ls)
            for s in r.vocab.users.values():
                s.difference_update(ls)
            for login in ls:
                r.modlearn.last_terms.pop(login, None)
            for s in r.modlearn.users.values():
                s.difference_update(ls)
            r.banned = deque(x for x in r.banned if x[1] not in ls)
            r.divisive = deque((x for x in r.divisive if x[1] not in ls), maxlen=r.divisive.maxlen)
            r.arrivals = deque(x for x in r.arrivals if x[1] not in ls)
            for key, dq in list(r.shapes.items()):
                r.shapes[key] = deque(x for x in dq if x[1] not in ls)
            for key, dq in list(r.chants.items()):
                r.chants[key] = deque((x for x in dq if x[1] not in ls), maxlen=dq.maxlen)
        if uid in self.ids.seen:
            self.ids.seen.discard(uid)
            i = bisect.bisect_left(self.ids.ids, int(uid))
            if i < len(self.ids.ids) and self.ids.ids[i] == int(uid):
                del self.ids.ids[i]
        return len(gone)

    def prune(self, now, idle_days=PRUNE_IDLE_DAYS):
        """S7 (live review H1): drop chatters idle for more than idle_days (last message before
        now - idle_days; now in ms): their profile, name index, the channel's sender names, legacy
        register profile, room-vocabulary membership and panel cache; the account-age index (IdIndex)
        is rebuilt from the chatters kept. Called at load and from the autosave, never per message.
        Returns how many per-channel profiles were dropped."""
        cut = now - idle_days * 86_400_000
        gone = [k for k, u in self.users.items() if u.last_seen < cut]
        if not gone:
            return 0
        logins = {}
        for k in gone:
            u = self.users[k]
            logins.setdefault(k[0], set()).add(u.login)
            if self.by_login.get((k[0], u.login)) == k[1]:
                del self.by_login[(k[0], u.login)]
            if self.legacy is not None and k[0] in self.legacy:
                self.legacy[k[0]].profiles.pop(str(k[1]), None)
            self.__dict__.get("_notice_p", {}).pop(k, None)
        self.drop_users(gone)
        for ch, ls in logins.items():
            ls -= {u.login for u in self.channel_users(ch)}      # a login someone kept still uses
            self.names.get(ch, set()).difference_update(ls)
            r = self.rooms.get(ch)
            if r is not None:
                r.vocab.chatters.difference_update(ls)
                for s in r.vocab.users.values():
                    s.difference_update(ls)
        idx = IdIndex()
        for (_, uid) in self.users:
            idx.add(str(uid))
        self.ids = idx
        self.stats["pruned_chatters"] += len(gone)
        return len(gone)

    # ----------------------------------------------------------- queries
    def profile(self, ch, uid, since=None):
        """since: the start of the live run (ms), as for notices()."""
        u = self.users.get((ch, uid))
        if u is None:
            return None
        post = u.post()
        bf = sum(post[a] for a in A.BAD)
        items = sorted(u.why.items(), key=lambda kv: -abs(kv[1]))
        reasons = [{"archetype": A.LABEL[a], "family": A.FAMILY[a], "reason": r, "weight": round(w, 2)}
                   for (a, r), w in items[:12]]
        acct = [{"archetype": A.LABEL[a], "reason": r, "weight": w} for a, w, r in u.acct_why]
        reg = None
        if self.legacy is not None:
            reg = self.legacy[ch].profile(uid)
            if reg:
                b = reg.pop("behaviour")
                reg["tags"] = [t for t, rule in BEHAVIOUR_TAGS if rule(b)]
        persist = self._persist_of(u)
        nr = _notice_reasons(u, (persist or {}).get("p") if u.toxic_n else None, max(getattr(self, "clock", 0) or 0, u.last_seen),
                             getattr(self, "person_review", False), since=since)
        notice = ({"level": nr[0], "why": nr[1],
                   "owner": any(_owner(m.get("toxicity")) and _level(m["toxicity"]) != "none" for m in u.msgs)}
                  if nr and u.uid not in self.owner_bots else None)
        return {
            "user_id": u.uid, "login": u.login, "display": u.display, "channel": ch,
            "first_seen": u.first_seen, "last_seen": u.last_seen, "messages": u.n,
            "first_msg_seen": u.saw_first_msg, "badges": list(u.badges),
            "sub_months": u.sub_months,
            "account_age_days": (u.last_f or {}).get("account_age_days"),
            "archetypes": [{"key": a, "label": A.LABEL[a], "family": A.FAMILY[a],
                            "desc": A.DESC[a], "p": round(post[a], 3)}
                           for a in sorted(post, key=lambda a: -post[a])],
            "badfaith": round(bf, 3), "peak_badfaith": round(u.peak_bf, 3),
            "certainty": round(A.entropy_certainty(post), 3),
            "verdict": verdict(post, u.n),
            "notice": notice,
            "reasons": reasons, "account_reasons": acct,
            "toxic": {"messages": u.toxic_n, "of": u.n, "kinds": [kv for kv in u.toxic_kinds.most_common() if kv[1] > 0][:5]},
            "vocabulary": u.vocab.markers(), "distress_notes": u.vocab.distress,
            "friction": {"messages": u.friction_msgs, "instigated": u.instigated,
                         "joined_heated_room": u.joined},
            "reactions": u.reactions[-20:],
            "register": reg,
            "persist": persist,
            "behaviour_type": self._type_of(u),
            "recent_messages": list(u.msgs)[-25:],
        }

    def notices(self, ch, ts, us=None, limit=40, since=None):
        """The heads-up panel: people worth a look right now, one entry each,
        'act' before 'watch', most recent first; plus who dropped off this run
        (since: the run's start, ms; earlier saved history is not listed)."""
        us = us if us is not None else self.channel_users(ch)
        live, earlier = [], []
        bots = self.owner_bots
        for u in us:
            if not u.toxic_n and not u.alerted and not _review_bad(u):
                continue
            if bots and u.uid in bots:         # the owner's bot role: never listed while it is set
                continue
            p = None
            if u.toxic_n:                      # cached per message count: it only changes when they post
                cache = self.__dict__.setdefault("_notice_p", {})
                c = cache.get((ch, u.uid))
                if c is None or c[0] != u.n:
                    c = cache[(ch, u.uid)] = (u.n, (self._persist_of(u) or {}).get("p"))
                p = c[1]
            r = _notice_reasons(u, p, ts, getattr(self, "person_review", False), since=since)
            if r is None:
                continue
            level, why, worst, sev = r
            item = {"user_id": u.uid, "display": u.display, "level": level, "why": why, "n": u.n, "severity": sev,
                    "toxic": u.toxic_n, "persist": p, "last_seen": u.last_seen, "since": worst["ts"],
                    "quote": worst["text"][:160], "first": u.saw_first_msg and u.n <= 3,
                    # the owner's own correction is part of why they are listed: the card says "set by you" (N-4)
                    "owner": any(_owner(m.get("toxicity")) and _level(m["toxicity"]) != "none" for m in u.msgs),
                    "quote_owner": _owner(worst.get("toxicity")),
                    "review": ({k: u.review.get(k) for k in ("label", "rationale")}
                               if getattr(self, "person_review", False) and getattr(u, "review", None) and u.review.get("label") else None),
                    "role": ("broadcaster" if "broadcaster" in u.badges else "mod" if "moderator" in u.badges
                             else "VIP" if "vip" in u.badges else None)}
            if ts - u.last_seen <= NOTICE_SHOW_MS[level]:
                live.append(item)
            elif since is None or u.last_seen >= since:
                earlier.append(item)
        counts = {"act": sum(d["level"] == "act" for d in live), "watch": sum(d["level"] == "watch" for d in live)}
        # people posting the same thing (a copypasta with a slur, a chant) become one card; each still counts
        groups = {}
        for d in live:
            groups.setdefault(_same_text(d["quote"]) or d["user_id"], []).append(d)
        cards = []
        for g in groups.values():
            if len(g) == 1:
                cards.append(g[0])
                continue
            g.sort(key=lambda d: (d["level"] != "act", -d["severity"]))
            cards.append({**g[0], "group": [{"user_id": d["user_id"], "display": d["display"], "level": d["level"],
                                     "why": d["why"], "owner": d["owner"]} for d in g],
                          "owner": any(d["owner"] for d in g), "severity": max(d["severity"] for d in g) + len(g), "since": max(d["since"] for d in g)})
        # within a level: most severe first; recent activity breaks ties
        cards.sort(key=lambda d: (d["level"] != "act", -d["severity"], -d["since"]))
        earlier.sort(key=lambda d: (d["level"] != "act", -d["last_seen"]))
        return {"items": cards[:limit], "counts": counts, "more": max(0, len(cards) - limit)}, earlier[:30]

    def room_snapshot(self, ch, ts=None, since=None, full=True):
        """since: the start of the dashboard's run (ms); "chatters" and the earlier list then
        cover this run only, not the saved history (display only: levels are unchanged).
        full=False leaves out the room profile the page does not show (Room.snapshot)."""
        r = self.rooms.get(ch)
        if r is None:
            return None
        ts = ts or int(time.time() * 1000)
        snap = r.snapshot(ts, full=full)
        us = self.channel_users(ch)
        snap["chatters"] = len(us) if since is None else sum(u.last_seen >= since for u in us)
        # first-timers this session (owner 2026-11-08): chatters whose first message ever in this channel (Twitch's
        # first-msg tag) came since the run started; with no start, all such chatters seen
        snap["first_timers_session"] = sum(1 for u in us if u.saw_first_msg and (since is None or u.first_seen >= since))
        snap["notices"], snap["notices_earlier"] = self.notices(ch, ts, us, since=since)
        return snap


class KeywordLayer:
    """The keyword layer's toxicity score, exactly as Engine(legacy=False, ml=None) computes it,
    without the engine (docs/review/performance.md P1, P5). With no ML models, message_toxicity
    reads only the text, its tags and three pieces of history: the logins that have chatted in the
    channel (text.tokenize names: is a word a chatter's name?), the channel's nicknames for the
    broadcaster, and the room id (third-party emotes). This keeps exactly those and repeats the
    steps Engine.process / _on_msg take before the toxicity read; everything else the engine does
    per event (users, room windows, archetypes) is skipped: about 97% of a replay's work.

        kl = KeywordLayer(emotes)
        for ev in events:                      # every event, in replay order
            if kl.observe(ev) and wanted(ev):  # a message the engine would score
                score = kl.score(ev)           # = eng.process(ev)["toxicity"]["score"]

    It sits next to _on_msg so tools/distill.py's layer-cache fingerprint covers both. A change
    that makes the layer read more of the engine's state must change this too:
    tests/test_learning_tools.py compares the two on a slice of real chat."""

    def __init__(self, emotes=None, nicknames=None):
        from .room import Room
        self._Room = Room
        self.emotes = emotes
        self.nicknames = nicknames or {}
        self.rooms = {}
        self.names = {}               # channel -> set of sender logins (Engine.names)

    def _room(self, ch):
        r = self.rooms.get(ch)
        if r is None:
            r = self.rooms[ch] = self._Room(ch, self.nicknames.get(ch, ()))
            self.names[ch] = set()
        return r

    def observe(self, ev):
        """Update the history as Engine.process does; True for a message the engine scores."""
        ch = ev.get("channel", "")
        if ev["kind"] != "msg" and ch:
            room = self._room(ch)
            if ev.get("room_id"):
                room.room_id = ev["room_id"]
        m = ev if ev["kind"] == "msg" else _scored_notice(ev) if ch else None
        if m is None:
            return False
        login = m["login"].lower()
        if is_bot(login):
            return False
        room = self._room(ch)
        if m.get("room_id"):
            room.room_id = m["room_id"]
        self.names[ch].add(login)
        return True

    def features(self, ev):
        """signals.extract for an observed message (only its toxicity inputs match the engine's)."""
        if ev["kind"] != "msg":
            ev = _scored_notice(ev)
        room = self.rooms[ev["channel"]]
        third = self.emotes.codes(room.room_id) if self.emotes else frozenset()
        return signals.extract(ev, room, third_party=third, names=self.names[ev["channel"]])

    def score(self, ev):
        return message_toxicity(self.features(ev))["score"]

    def process(self, ev):
        """The layer score of a message event (None for others and for skipped bots)."""
        return self.score(ev) if self.observe(ev) else None


def _pending(review, ts):
    """A person read is queued and not yet lost (REVIEW_PENDING_MS)."""
    return bool(review.get("pending")) and ts - review.get("queued", 0) <= REVIEW_PENDING_MS


def _review_bad(u):
    r = getattr(u, "review", None)
    return bool(r) and r.get("label") == "bad_faith"


def _read_failed(u, now):
    """The chatter has no person read to go by: their last read failed (the LLM down or timing out, the queue
    full), it was lost (dropped at Stop, or pending when the state was saved: audit LB2-3), or it has been pending
    past REVIEW_PENDING_MS. A read they already have keeps counting."""
    r = getattr(u, "review", None) or {}
    return not r.get("label") and bool(r.get("failed") or r.get("lost") or (r.get("pending") and now - r.get("queued", 0) > REVIEW_PENDING_MS))


def _notice_reasons(u, persist, now, reviewed=False, since=None):
    """(level, [plain reasons], the message that put them there, severity) for one chatter, or None.
    reviewed: the local LLM reads people, so its read makes 'watch': bad faith only (DB, 2026-11-02; a mild read with
    2+ toxic in 10 min or an alert made 'watch' before it).
    since: the start of the live run (ms); S12: content from before it never makes 'act' by itself."""
    if reviewed and _read_failed(u, now):
        reviewed = False       # LB-3 (DASHBOARD.md): no person read to go by, so the older rule applies to them
    msgs = list(u.msgs)
    if FLAGS_PERSIST and getattr(u, "evidence", None):
        held = {id(m) for m in msgs}
        older = sorted((m for m in u.evidence if id(m) not in held), key=lambda m: m.get("ts") or 0)
        msgs = older + msgs                       # the flags behind a listing that left the recent messages, oldest first
    tox = [m for m in msgs if _level(m.get("toxicity")) != "none"]
    if not tox and not u.alerted and not (reviewed and _review_bad(u)):
        return None
    cls = [(m, _tox_class(m["toxicity"])) for m in tox]
    hate = [m for m, c in cls if c == "hate"]
    severe = [m for m, c in cls if c == "severe"]
    recent = [m for m in tox if now - m["ts"] <= NOTICE_RECENT_MS]
    # S12: every act-making message is older than this run: at most 'watch', marked "earlier session".
    # Keyword and coded-vocabulary slurs are exempt (RULE_PHASE3: slurs are never traded, rule 8.5): a
    # chatter with one, in this run or before it, keeps 'act'. So S12 covers severe content and LLM-only hate
    earlier = bool(since is not None and (hate or severe)
                   and not any(_sure_hate(m["toxicity"]) for m in hate + severe)
                   and all(m["ts"] < since for m in hate + severe))
    tag = " (earlier session)" if earlier else ""
    why, level = [], None
    if hate:
        why.append("slur or dehumanising language" + (f" ({len(hate)} messages)" if len(hate) > 1 else "") + tag)
    if severe:
        why.append("severe: " + ", ".join(sorted({k for m in severe for k in m["toxicity"].get("kinds") or ["high"]}))[:60]
                   + tag)
    if why and not earlier:
        level = "act"
    if reviewed:
        # D7 (FINDINGS 2026-10-26, passed on capture_20261026T0231): the LLM's read makes 'watch'. DB (data/study/
        # bf_db/RULE_DB.md, passed on capture_20261102T0000_db): only a bad-faith read; a mild read with 2+ toxic in
        # 10 min or an alert no longer makes a card (71 not bad faith removed per 6 bad faith lost); both still ask
        # for a read
        rv = getattr(u, "review", None) or {}
        note = f" ({rv['rationale'][:80]})" if rv.get("rationale") else ""
        if _review_bad(u):
            why.append("LLM read: bad faith" + note)
        if not why:
            return None
        # A4h (FINDINGS 2026-10-27, bf_design/RULE_ACT_TIER.md): severe content without a slur is 'act'
        # only when the LLM reads them as at least mild; otherwise they stay listed at 'watch'.
        # Slurs stay 'act' whatever the read, and a pending read keeps 'act'.
        # S11 (RULE_PHASE3): hate from the LLM's re-read alone (no keyword or coded-vocabulary slur) is
        # treated like severe content under A4h: a read below mild (good faith, unclear) makes it
        # 'watch'; a chatter with no read, or a pending first one, keeps 'act', as A4h's code does for
        # severe. Keyword and coded-vocabulary slurs stay 'act' whatever the read
        if (level == "act" and not any(_sure_hate(m["toxicity"]) for m in hate)
                and rv.get("label") in ("good_faith", "unclear")):
            level = None
        if len(recent) >= 2:
            why.append(f"{len(recent)} toxic messages in 10 min")
    elif len(recent) >= NOTICE_RECENT_N:
        why.append(f"{len(recent)} toxic messages in 10 min")
    elif len(recent) >= 2 and why:
        why.append(f"{len(recent)} toxic messages in 10 min")       # context on someone already listed
    if why and tox and persist is not None and persist >= NOTICE_PERSIST_WATCH:
        why.append(f"may keep it up ({persist:.0%})")              # context only, never a trigger
    if u.alerted and (why or not reviewed):
        top = max(u.post().items(), key=lambda kv: kv[1])[0]
        why.append(f"bad-faith pattern ({A.LABEL[top].lower()})" if A.FAMILY[top] == "badfaith" else "bad-faith signs")
    if not why:
        return None
    level = level or "watch"
    worst = (hate or severe or recent or tox or msgs)[-1]
    # severity, for ordering within a level: repeated slurs and severe content first
    sev = 3 * len(hate) + 2 * len(severe) + len(recent)
    return level, why, worst, sev


def _same_text(t):
    return re.sub(r"\W+", " ", t.lower()).strip()[:80]


def verdict(post, n):
    """One honest sentence: what we think, and how sure we can be."""
    ranked = sorted(post.items(), key=lambda kv: -kv[1])
    (a1, p1), (a2, p2) = ranked[0], ranked[1]
    bf = sum(post[a] for a in A.BAD)
    evidence = ("first impression" if n <= 1 else f"{n} messages")
    if p1 >= 0.6:
        head = f"Likely {A.LABEL[a1].lower()}"         # a read, never a verdict (rule 8.9)
    elif p1 - p2 < 0.1:
        head = f"{A.LABEL[a1]} or {A.LABEL[a2]}"
    else:
        head = f"Probably {A.LABEL[a1].lower()}"
    tail = ""
    if bf >= 0.5:                                      # a model posterior: signs, never a percentage of certainty
        tail = "; strong bad-faith signs"
    elif bf >= 0.25:
        tail = "; some bad-faith signs"
    return f"{head} ({evidence}){tail}"


def _scored_notice(ev):
    """S5: the chatter-written text of a USERNOTICE as a message to score (irc.notice_as_msg), or None."""
    if ev.get("kind") != "usernotice" or ev.get("msg_id") in ("raid", "unraid"):
        return None
    from .irc import notice_as_msg
    return notice_as_msg(ev)


# S4: the bots the engine does not score are exactly the reviewed list (legacy_style.is_bot), not every
# login ending in "bot", which hid human chatters whose names end in "bot"
BOTS_PATH = legacy_style.BOTS_PATH
is_bot = legacy_style.is_bot
