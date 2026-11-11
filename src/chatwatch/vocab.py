"""Coded / subcultural vocabulary: message hits, person markers, room profile.

Implements research/04_coded_language.md. The lexicon itself is data
(assets/coded_lexicon.json) so it can be reviewed and updated without code.

  message   T1 / U / T1-violent terms raise toxicity; first-person suicide
            vocabulary becomes a separate "possible distress" note
  person    "uses <family> vocabulary" only by BREADTH: >= 3 distinct T2/C
            terms of one family in >= 2 messages, or T1/U in >= 3 messages,
            or 1 T1/U + 2 distinct T2/C. A term the whole room uses (> 1% of
            chatters) counts 0.25 toward a person (it says more about the room)
  room      per term: distinct chatters, uses, rate per 10k messages

Vocabulary shows exposure to a discourse, not belief (research/04 #1). The
UI says "uses vocabulary from...", never "is an incel".
"""
import json
import os
import re
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

# quoting, asking about a term, counter-speech: the word is mentioned, not used
EXCLUDE = re.compile(
    r"(?:\bcalled\s+(?:her|him|them|me|us|you)\s+an?\b|\bwhat\s+(?:does|is|'s)\b|\bwhats\b|\bmeaning\s+of\b|"
    r"\bstop\s+(?:saying|calling)\b|\bdon'?t\s+say\b|\bwhy\s+(?:do|would)\s+(?:you|u|people)\s+say\b|"
    r"\bis\s+a\s+slur\b|\bthat'?s\s+a\s+slur\b)")
FIRST_PERSON = re.compile(r"\b(?:i|im|i'm|i'll|ill|me|my)\b")
VIOLENT_TIERS = {"T1-violent"}
TOXIC_TIERS = {"T1", "U", "T1-violent"}
MARKER_TIERS = {"T2", "C"}
ROOM_NORM_SHARE = 0.01


class Lexicon:
    def __init__(self, path=os.path.join(HERE, "assets", "coded_lexicon.json")):
        d = json.load(open(path))
        self.version = d["version"]
        self.terms = [{**t, "rx": re.compile(t["regex"], re.I)} for t in d["terms"]]

    def match(self, norm):
        """Hits in a normalised message, after the exclusion layer."""
        hits = []
        for t in self.terms:
            m = t["rx"].search(norm)
            if not m:
                continue
            before = norm[max(0, m.start() - 40): m.start()]
            quoted = norm[:m.start()].count('"') % 2 == 1
            if EXCLUDE.search(before) or quoted or (norm.strip().endswith("?") and len(norm.split()) <= 6
                                                    and EXCLUDE.search(norm)):
                continue
            distress = t["distress_if_first_person"] and bool(FIRST_PERSON.search(norm))
            hits.append({"id": t["id"], "family": t["family"], "tier": t["tier"], "weight": t["weight"],
                         "text": m.group(0), "distress": distress})
        return hits


LEX = Lexicon()


class RoomVocab:
    """Per-channel term usage (the room-level profile), saved with the engine across runs."""

    def __init__(self):
        self.users = defaultdict(set)     # term id -> logins
        self.uses = Counter()
        self.chatters = set()
        self.msgs = 0

    def observe(self, login, hits):
        self.msgs += 1
        self.chatters.add(login)
        for h in hits:
            self.users[h["id"]].add(login)
            self.uses[h["id"]] += 1

    def share(self, term):
        return len(self.users.get(term, ())) / max(len(self.chatters), 1)

    def is_room_norm(self, term):
        return len(self.users.get(term, ())) >= 10 and self.share(term) > ROOM_NORM_SHARE

    def profile(self, k=8):
        tiers = {t["id"]: (t["family"], t["tier"]) for t in LEX.terms}
        rows = []
        # a term id no longer in the lexicon (a saved state outlives a lexicon edit) is skipped (LB4-1)
        known = [kv for kv in self.users.items() if kv[0] in tiers]
        for term, users in sorted(known, key=lambda kv: -len(kv[1]))[:k]:
            fam, tier = tiers[term]
            rows.append({"term": term, "family": fam, "tier": tier, "chatters": len(users),
                         "share": round(len(users) / max(len(self.chatters), 1), 4),
                         "per_10k": round(self.uses[term] * 10000 / max(self.msgs, 1), 1),
                         "room_norm": self.is_room_norm(term)})
        return rows


class PersonVocab:
    """Per-user accumulation and the breadth rule."""
    __slots__ = ("terms", "t1_msgs", "msgs_with_marker", "distress")

    def __init__(self):
        self.terms = defaultdict(dict)    # family -> term -> weight
        self.t1_msgs = Counter()          # family -> messages with a T1/U term
        self.msgs_with_marker = Counter() # family -> messages with any T2/C term
        self.distress = 0

    def add(self, hits, room):
        fams_t1, fams_t2 = set(), set()
        for h in hits:
            if h["distress"]:
                self.distress += 1
                continue
            if h["tier"] == "T3":
                continue
            # fixed standard: a term weighs the same however common it is in this
            # room (docs/DESIGN.md, "Absolute standard"); room_norm is still reported
            w = h["weight"]
            fam = self.terms[h["family"]]
            fam[h["id"]] = max(fam.get(h["id"], 0.0), w)
            if h["tier"] in TOXIC_TIERS:
                fams_t1.add(h["family"])
            elif h["tier"] in MARKER_TIERS:
                fams_t2.add(h["family"])
        for f in fams_t1:
            self.t1_msgs[f] += 1
        for f in fams_t2:
            self.msgs_with_marker[f] += 1

    def markers(self):
        out = []
        tiers = {t["id"]: t["tier"] for t in LEX.terms}
        for fam, terms in self.terms.items():
            terms = {t: w for t, w in terms.items() if t in tiers}     # ids gone from the lexicon (LB4-1)
            t2 = sum(w for t, w in terms.items() if tiers[t] in MARKER_TIERS)
            t1 = [t for t in terms if tiers[t] in TOXIC_TIERS]
            t1_weight = max((terms[t] for t in t1), default=0)
            show = ((t2 >= 3 and self.msgs_with_marker[fam] >= 2)
                    or (self.t1_msgs[fam] >= 3 and t1_weight >= 1.0)
                    or (t1 and t2 >= 2))
            if show:
                out.append({"family": fam, "terms": sorted(terms), "breadth": round(t2 + (1 if t1 else 0), 2),
                            "t1_messages": self.t1_msgs[fam]})
        return out
