"""Per-message feature extraction.

`extract()` turns one chat message (already parsed by irc.to_event) plus its
room into a flat dict of features. Everything downstream (archetypes, risk,
the dashboard's "why") reads these features, so each one is named for what a
mod would recognise.
"""
import json
import os
import re
from collections import Counter

from . import accounts, cues, irc, text
from .room import terms_of
from . import vocab

HERE = os.path.dirname(os.path.abspath(__file__))


class Lexicon:
    """The handoff's 22-category lexicon, loaded from the frozen baseline so
    both layers count the same words. Twitch emotes are mapped to categories
    explicitly (fixes handoff lesson #1)."""

    def __init__(self, path=os.path.join(HERE, "assets", "baseline.json")):
        b = json.load(open(path))["lexicon"]
        self.cats = b["all_cats"]
        self.word2cats = {}
        for cat, words in b["categories"].items():
            for w in set(words.split()):
                self.word2cats.setdefault(w, []).append(cat)
        self.phrases = b["phrases"]
        self.words = set(self.word2cats)

    def tally(self, words, emotes=()):
        c = Counter()
        for w in words:
            for cat in self.word2cats.get(w, ()):
                c[cat] += 1
        joined = " ".join(words)
        for cat, phs in self.phrases.items():
            for ph in phs:
                if ph in joined:
                    c[cat] += 1
        for e in emotes:
            cat = cues.EMOTE_CATEGORIES.get(e)
            if cat:
                c[cat] += 1
        return c


LEX = Lexicon()

SECOND_PERSON = {"you", "u", "ur", "your", "youre", "you're", "ya", "yours", "yourself"}
COMMAND = re.compile(r"^!\w+")
NAME_DIGITS = re.compile(r"^[a-z_]+?\d{4,}$|^[a-z]+_?[a-z]+\d{3,}$")
VOWELS = set("aeiou")
# explicit outsiders only: on Twitch "he/she" usually means the streamer
THIRD_PERSON = re.compile(r"\b(?:(?:those|these|the|all|some|other) (?:guys|people|dudes|kids|players|ppl|folks|devs|streamers|fans)|"
                          r"they|them|theyre|they're)\b")
MENTION_STRIP = re.compile(r"@\w+")


def name_randomness(login):
    """0..1: how machine-generated a login looks (long digit tails,
    consonant runs, underscores-as-padding). Throwaway accounts often have
    'adjective_noun1234' or keyboard-mash names; real ones rarely do."""
    l = login.lower()
    score = 0.0
    if NAME_DIGITS.match(l):
        score += 0.5
    letters = re.sub(r"[^a-z]", "", l)
    if letters:
        runs = re.findall(r"[^aeiou]{5,}", letters)
        if runs:
            score += 0.3
        if sum(c in VOWELS for c in letters) / len(letters) < 0.2:
            score += 0.2
    if l.count("_") >= 2:
        score += 0.1
    return min(1.0, score)


def automod_levels(ev, emotes=frozenset()):
    """Twitch flags levels per category. Spans on the "-a" form of the n-word
    count like any other since the owner's 2026-10-25 decision that it is a
    slur (they were skipped before: FINDINGS §8). A span that is exactly one of
    the message's emotes and a "schizo" reaction emote (SCHIZO, a 7TV emote) is not
    counted: judges called 19 of 19 such flags not toxic (FINDINGS 2026-10-27). Other
    emote names stay: "hesRetarded", "Cucklord" are insults whatever their form."""
    spans = ev.get("automod_spans")
    if not spans:
        return ev.get("automod", {})
    out = {}
    for a, b, levels in spans:
        span = ev["text"][a:b + 1].strip()
        if span in emotes and "schizo" in span.lower():
            continue
        for c, l in levels.items():
            out[c] = max(out.get(c, 0), l)
    return out


def extract(ev, room, *, third_party=frozenset(), names=frozenset(),
            prev_shapes=(), account_newness=None):
    """Features for one message.

    ev:   msg event from irc.to_event
    room: the channel's Room (state *before* this message)
    prev_shapes: shape keys of this user's recent messages (for self-repeats)
    account_newness: 0..1 percentile of the user's id among ids seen (1 = newest)
    """
    raw = ev["text"]
    norm = text.normalize(raw)
    plain = text.plain_lower(raw)
    tok = text.tokenize(raw, ev.get("emotes", ()), third_party, names,
                        protected=LEX.words)
    hits, evaded = cues.match(norm, plain)
    ev_sig = text.evasion_signals(raw)
    cats = LEX.tally(tok.words, tok.emotes)
    nw = len(tok.words)
    shape = text.shape_key(raw)

    fam = Counter(cues.CUE_FAMILY[h] for h in hits)
    bcast = room.broadcaster_names
    # a generic "you" deep in a long sentence ("as if you move abroad...") isn't
    # address; short messages, a leading "you", or "this <insult>" are
    words = tok.words
    second = bool(set(words) & SECOND_PERSON) and (len(words) <= 8 or words[0] in SECOND_PERSON)
    addresses_streamer = bool(
        second or set(tok.names) & bcast
        or re.search(r"\bthis (?:asshole|clown|idiot|bozo|loser|fraud|bum|dude|guy|kid|man|woman|girl)\b", norm)
        or set(tok.mentions) & bcast or re.search(r"\bstreamer\b", norm)
        or ev.get("reply_to_login", "").lower() in bcast)
    target_chatter = None
    rl = ev.get("reply_to_login", "").lower()
    if rl and rl not in bcast:
        target_chatter = rl
    elif tok.mentions:
        others = [m for m in tok.mentions if m not in bcast]
        target_chatter = others[0] if others else None

    # Who is the negativity about? An @ or reply often just *addresses* someone
    # ("@bob that's sweat mode, those guys never touch grass" answers bob and
    # mocks third parties). It is aimed at the @'d person only when the hostile
    # part speaks to them ("you...") or the message is a bare short jab.
    third_person = bool(THIRD_PERSON.search(MENTION_STRIP.sub(" ", norm)))
    second = bool(set(tok.words) & SECOND_PERSON)
    aimed_at_chatter = bool(target_chatter and (second or (not third_person and nw <= 5)))
    about_third_party = third_person and not second and not addresses_streamer

    others, new_others = room.shape_stats(shape, ev["ts"], ev["login"]) if shape else (0, 0)
    mocking = [e for e in tok.emotes if e in cues.MOCKING_EMOTES]
    caps = text.caps_ratio(raw)
    friction_n = fam.get("friction", 0)

    f = {
        # form
        "n_words": nw,
        "n_emotes": len(tok.emotes),
        "emote_only": nw == 0 and len(tok.emotes) > 0,
        "length": len(raw),
        "question": "?" in raw,
        "starts_you": bool(re.match(r"\s*(?:@\w+[ ,]+)?(?:you|u|ur|your|youre|you're|why do you|why are you|why don'?t you)\b", norm)),
        "caps": caps >= 0.7 and len(raw) >= 8,
        "has_url": bool(tok.urls),
        "command": bool(COMMAND.match(raw)),
        "non_latin": text.non_latin_share(raw),
        "action": ev.get("action", False),
        # lexicon categories (counts)
        "cats": dict(cats),
        "negative": cats.get("negemo", 0) + cats.get("anger", 0) + cats.get("swear", 0),
        "positive": cats.get("posemo", 0) + cats.get("laugh", 0) + cats.get("polite", 0),
        # cues
        "cues": hits,
        "families": dict(fam),
        "friction": friction_n,
        "mocking_emotes": mocking,
        # targeting
        "addresses_streamer": addresses_streamer,
        "target_chatter": target_chatter,
        "aimed_at_chatter": aimed_at_chatter,
        "second_person": second,
        "about_third_party": about_third_party,
        "is_reply": bool(ev.get("reply_to_login")),
        # evasion: deliberately disguised words or invisible tricks
        "evaded_cues": evaded,
        "evasion": bool(evaded) or ev_sig["zero_width"] > 0 or ev_sig["mixed_script_words"] > 0
                   or ev_sig["zalgo"],
        "evasion_detail": ev_sig,
        # shared text
        "shape": shape,
        "copypasta_others": others,          # same text from other people recently
        "template_new_accounts": new_others,  # ...of whom were brand-new accounts
        "self_repeat": shape in prev_shapes if shape else False,
        "chant_key": text.chant_key(raw),
        "crowd_echo": room.chant_count(text.chant_key(raw), ev["ts"], ev["login"]),
        # metadata (IRC tags)
        "first_msg": ev.get("first_msg", False),
        "returning": ev.get("returning", False),
        "sub_months": irc._int(ev.get("badge_info", {}).get("subscriber")),
        "has_badges": bool(ev.get("badges")),
        "privileged": ev.get("mod") or ev.get("vip") or ev.get("broadcaster"),
        "default_color": not ev.get("color"),
        "bits": ev.get("bits", 0),
        "name_randomness": name_randomness(ev["login"]),
        "account_newness": account_newness,
        "account_age_days": accounts.age_days(ev.get("user_id"), ev["ts"]),
        # Twitch's own AutoMod-style span labels, free in the IRC flags tag
        "automod": automod_levels(ev, frozenset(tok.emotes)),
        # room context at the moment of the message
        "room_heated": room.heated(ev["ts"]),
        "topic_in_room": room.topic_in_room(ev["ts"], ev["login"]) if fam.get("injection") else False,
        "room_raid": room.is_raid_window(ev["ts"]),
        "pile_on": bool(room.pile_on(ev["ts"])) if ev.get("first_msg") else False,
        "lookalike_of_banned": room.lookalike_of_banned(ev["login"], ev["ts"]) if ev.get("first_msg") else None,
        "norm": norm,
        # coded / subcultural vocabulary (research/04), after the exclusion layer
        "vocab": vocab.LEX.match(norm),
        "ascii_art": sum(1 for c in raw if 0x2800 <= ord(c) <= 0x28FF or 0x2500 <= ord(c) <= 0x259F) >= 20,
        # topics this channel's mods have been timing people out for today
        "moderated_topics": room.modlearn.matches(terms_of(norm)),
    }
    f["targeted_negative"] = bool(
        (friction_n or f["negative"]) and (addresses_streamer or aimed_at_chatter))
    return f
