#!/usr/bin/env python3
"""(Copied unchanged from the handoff realtime_starter/chatter_scorer.py.)
Incremental chatter characterizer, scored against a frozen baseline.

Feed it (user, message) pairs as they arrive; ask for a profile at any time.
Pure Python, no dependencies. Tokenisation mirrors the offline psych.py so the
frozen baseline stays valid — change one and you must rebuild the other.

    ch = Characterizer("baseline.json", emote_vocab="emote_vocab.json")
    ch.observe("someuser", "i think the monster is behind the door lol",
               user_id="12345")          # IRC `user-id` tag; survives renames
    print(ch.profile("12345"))

Live-specific choices (see 02_REALTIME_DESIGN.md):
  - rates are shrunk toward the cohort mean with PRIOR_WORDS pseudo-words,
    so a chatter with 15 words isn't labelled from 15 words of noise
  - every profile carries a confidence tier from its word count, and the
    distance margin to the runner-up style; both thresholds come from
    replaying the archive (validate_against_archive.py)
"""
import json
import os
import re
from collections import Counter

# Replaying the archive, a 50-word prior gave a small gain under 200 words and
# nothing above; 300 cost up to 8 points of agreement. Keep it light.
PRIOR_WORDS = 50

# Share of 1,843 archive chatters whose label at N words already matched their
# final label: 100w 53%, 200w 64%, 300w 70%, 1000w 75%, 2000w 80% (chance ~25%).
# Median messages to reach 100 / 300 / 1000 words: 20 / 57 / 204.
CONFIDENCE = [(1000, "high"), (300, "medium"), (100, "low"), (0, "insufficient")]

KNOWN_BOTS = {"nightbot", "streamelements", "streamlabs", "moobot", "fossabot",
              "sery_bot", "wizebot", "own3d", "commanderroot", "soundalerts",
              "kofistreambot", "pretzelrocks", "lolrankbot", "buttsbot",
              "revlobot", "fourthwallhq", "streamcaptainbot", "botisimo",
              "phantombot", "streamlootsbot",
              # found in the capture (docs/FINDINGS 2026-10-25): template / game posts
              "potatbotat", "jbotbtw", "puptime",
              "botrixoficial", "chetbotwow"}
BOTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "bots.json")
_BOTS = None


def is_bot(login):
    """A command or utility bot (login lower-cased) that is not scored or profiled: exactly the reviewed
    list (assets/bots.json, one reason per entry) plus KNOWN_BOTS (RULE_PHASE3 S4). A login ending in
    "bot" is not enough: it hid human chatters whose names end in "bot"."""
    global _BOTS
    if _BOTS is None:
        _BOTS = frozenset(b["login"] for b in json.load(open(BOTS_PATH))["bots"]) | KNOWN_BOTS
    return login in _BOTS

TOKEN = re.compile(r"[a-z][a-z0-9']*")
URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@([A-Za-z0-9_]+)")
CAMEL = re.compile(r"^[a-z]+[A-Z]\w*$|^[A-Z][a-z]+[A-Z]\w*$")
DIGITCODE = re.compile(r"^[A-Za-z]+\d+[A-Za-z0-9]*$")
SYSTEM = re.compile(
    r"^\s*\S+\s+subscribed\s+(?:at\s+Tier\s+\d|with\s+Prime)[^!.]*[.!]"
    r"(?:\s*They(?:'ve)?\s+subscribed\s+for\s+\d+\s+months?"
    r"(?:,\s*currently\s+on\s+a\s+\d+\s+month\s+streak)?\s*!)?"
    r"|^\s*\S+\s+gifted\s+a\s+Tier\s+\d\s+sub\s+to\s+\S+!"
    r"(?:\s*They\s+have\s+given\s+[\d,]+\s+Gift\s+Subs?\s+in\s+the\s+channel!)?"
    r"|^\s*\S+\s+is\s+gifting\s+\d+\s+Tier\s+\d\s+Subs?\s+to\s+\S+?\s+community!"
    r"(?:\s*They(?:'ve)?\s+gifted\s+a\s+total\s+of\s+[\d,]+\s+in\s+the\s+channel!)?"
    r"|^\s*\S+\s+is\s+continuing\s+the\s+Gift\s+Sub\s+they\s+got\s+from\s+\S+!"
    r"|^\s*\S+\s+converted\s+from\s+a\s+Prime\s+sub\s+to\s+a\s+Tier\s+\d\s+sub!",
    re.I)

# all-lowercase third-party emotes found by hand in the archive (nonwords.py)
LOWER_EMOTES = {
    "ajscage", "badger", "beep", "bling", "blip", "bork", "choo", "clappi",
    "corgo", "croissant", "diggy", "ding", "doo", "dun", "dunka", "glorp",
    "hee", "hopewillnotbecancelled", "iamclap", "kpopdance", "natti", "nom",
    "pew", "plop", "rainbowsparkle", "toot", "xdx",
}


class Baseline:
    def __init__(self, path):
        b = json.load(open(path))
        self.cats = b["lexicon"]["all_cats"]
        self.word2cats = {}
        for cat, words in b["lexicon"]["categories"].items():
            for w in set(words.split()):
                self.word2cats.setdefault(w, []).append(cat)
        self.phrases = b["lexicon"]["phrases"]
        self.lexicon_words = set(self.word2cats)
        self.cat_stats = b["cat_stats"]
        self.dims = b["dimensions"]
        self.dim_order = b["dim_order"]
        self.dim_meta = b["dimension_meta"]
        self.quantiles = b["dim_quantiles"]
        self.clip = b["clip"]
        self.styles = b["styles"]


class ChatterProfile:
    __slots__ = ("words", "msgs", "word_msgs", "emotes", "cats", "chars",
                 "questions", "caps", "commands", "mentions", "emote_tokens")

    def __init__(self):
        self.words = self.msgs = self.word_msgs = self.emotes = 0
        self.chars = self.questions = self.caps = self.commands = 0
        self.mentions = self.emote_tokens = 0
        self.cats = Counter()


class Characterizer:
    def __init__(self, baseline_path, emote_vocab=None, prior_words=PRIOR_WORDS):
        self.b = Baseline(baseline_path)
        self.prior_words = prior_words
        self.emotes = set()
        if emote_vocab:
            self.emotes = set(json.load(open(emote_vocab))["emotes"])
        # Other chatters' names are not vocabulary. Live, the roster is simply
        # everyone seen so far; add nicknames for the broadcaster by hand.
        self.names = set()
        self.profiles = {}

    # ---------------------------------------------------------------- tokens
    def is_emote(self, tok):
        return (tok in self.emotes or tok.lower() in LOWER_EMOTES
                or CAMEL.match(tok) or DIGITCODE.match(tok)
                or (tok.isupper() and len(tok) > 2 and tok.isalpha()))

    def add_name(self, name):
        n = (name or "").lower()
        # a name that is also a lexicon word ("hi", "you") stays a word
        if n and n not in self.b.lexicon_words:
            self.names.add(n)

    def tally(self, body):
        """Same rules as offline psych.tally: (n_words, Counter, n_emotes)."""
        raw = MENTION.sub(" ", URL.sub(" ", SYSTEM.sub(" ", body)))
        words, n_emotes = [], 0
        for tok in raw.split():
            bare = tok.strip(".,!?;:'\"()[]…").lstrip("@")
            if not bare:
                continue
            if self.is_emote(bare):
                n_emotes += 1
                continue
            if bare.lower() in self.names:
                continue
            words.extend(w for w in TOKEN.findall(bare.lower())
                         if not any(ch.isdigit() for ch in w))
        if not words:
            return 0, None, n_emotes
        cats = Counter()
        for w in words:
            for c in self.b.word2cats.get(w, ()):
                cats[c] += 1
        low = " ".join(words)
        for cat, phrases in self.b.phrases.items():
            for ph in phrases:
                if ph in low:
                    cats[cat] += 1
        return len(words), cats, n_emotes

    # --------------------------------------------------------------- ingest
    def observe(self, login, body, user_id=None):
        """Ingest one message. Returns the user's profile, or None if skipped.

        Profiles are keyed by user_id when given (stable across renames),
        otherwise by lowercased login.
        """
        u = (login or "").lower()
        if not u or is_bot(u) or not body:
            return None
        # only senders join the name roster. Adding @-targets too looks
        # harmless but lets "@here" / "@everybody" / typos delete real words
        # from everyone's denominator — that moved rates by up to 16 per 100.
        self.add_name(u)
        key = str(user_id) if user_id else u
        p = self.profiles.get(key)
        if p is None:
            p = self.profiles[key] = ChatterProfile()

        # behaviour counters use every message, commands included
        p.msgs += 1
        p.chars += len(body)
        p.mentions += len(MENTION.findall(body))
        if "?" in body:
            p.questions += 1
        letters = [c for c in body if c.isalpha()]
        if len(letters) >= 5 and sum(c.isupper() for c in letters) / len(letters) > 0.7:
            p.caps += 1
        if body.startswith("!"):
            p.commands += 1
            return p

        nw, cats, ne = self.tally(body)
        p.emote_tokens += ne
        if nw:
            # offline, emote-only messages never reached the language rates;
            # matching that keeps emote_fluency comparable to the baseline
            p.word_msgs += 1
            p.words += nw
            p.emotes += ne
            p.cats.update(cats)
        return p

    # --------------------------------------------------------------- scoring
    def rates(self, p):
        """Per-100-word rates, shrunk toward the cohort mean."""
        k = self.prior_words
        out = {}
        for c in self.b.cats:
            mu = self.b.cat_stats[c]["mean"]
            out[c] = (p.cats.get(c, 0) + k * mu / 100) / (p.words + k) * 100
        mu = self.b.cat_stats["__emote__"]["mean"]
        tot = p.emotes + p.words
        out["__emote__"] = (p.emotes + k * mu / 100) / (tot + k) * 100
        return out

    def dimensions(self, p):
        r = self.rates(p)
        dims = {}
        for d, parts in self.b.dims.items():
            z = sum(w * (r[c] - self.b.cat_stats[c]["mean"]) / self.b.cat_stats[c]["sd"]
                    for c, w in parts)
            dims[d] = z / sum(w for _, w in parts)
        return dims

    def percentile(self, d, v):
        q = self.b.quantiles[d]
        lo = sum(1 for x in q if x <= v)
        return max(0, min(100, lo - 1))

    def style(self, dims):
        z = [max(-self.b.clip, min(self.b.clip, dims[d])) for d in self.b.dim_order]
        dist = {s: sum((a - b) ** 2 for a, b in zip(z, v["centroid"]))
                for s, v in self.b.styles.items()}
        ranked = sorted(dist, key=dist.get)
        # margin between best and runner-up: small = ambiguous label
        return ranked[0], ranked[1], dist[ranked[1]] - dist[ranked[0]]

    def profile(self, key):
        """key: the user_id passed to observe(), else the lowercased login."""
        p = self.profiles.get(str(key or "").lower())
        if p is None:
            return None
        dims = self.dimensions(p)
        best, runner_up, margin = self.style(dims)
        conf = next(lbl for n, lbl in CONFIDENCE if p.words >= n)
        pct = {d: self.percentile(d, v) for d, v in dims.items()}
        top2 = sorted(dims, key=lambda d: -abs(dims[d]))[:2]
        n = max(p.msgs, 1)
        return {
            "user": key, "messages": p.msgs, "words": p.words,
            "confidence": conf,
            "style": best, "runner_up": runner_up, "margin": round(margin, 3),
            "summary": ", ".join(f"{_adverb(pct[d])} {self.b.dim_meta[d]['label'].lower()}"
                                 for d in top2),
            "dim_pct": pct,
            "behaviour": {
                "avg_len": round(p.chars / n, 1),
                "question_rate": round(p.questions / n, 3),
                "caps_rate": round(p.caps / n, 3),
                "cmd_rate": round(p.commands / n, 3),
                "mention_rate": round(p.mentions / n, 3),
                "emotes_per_msg": round(p.emote_tokens / n, 3),
            },
        }


def _adverb(v):
    if v >= 90:
        return "very high"
    if v >= 70:
        return "high"
    if v <= 10:
        return "very low"
    if v <= 30:
        return "low"
    return "average"


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    ch = Characterizer(os.path.join(here, "assets", "baseline.json"))
    demo = [
        ("alice", "hi everyone!! good evening chat <3"),
        ("bob", "i think the monster is behind the door, maybe go back?"),
        ("alice", "welcome back @bob how are you"),
        ("bob", "you should grab the shotgun before going upstairs"),
        ("carol", "LUL LUL KEKW"),
        ("bob", "probably the key is in the other room, not sure though"),
    ]
    for u, m in demo:
        ch.observe(u, m)
    for u in ("alice", "bob", "carol"):
        print(json.dumps(ch.profile(u), indent=1))
