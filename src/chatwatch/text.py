"""Text normalisation and tokenisation for live chat.

Two jobs:

1. Split a message into emotes / names / words without the casing heuristic
   that broke the batch project (lesson #1 in the handoff: LUL, KEKW and every
   shouted ALL-CAPS word were swallowed as emotes). Emotes are identified only
   from the IRC `emotes` tag (exact positions) and the channel's third-party
   sets (7TV/BTTV/FFZ). ALL-CAPS words stay words.

2. Produce a *normalised* form that undoes common filter evasion (leetspeak,
   homoglyphs, zero-width characters, zalgo, spaced-out letters). The
   difference between the raw and normalised forms is itself evidence:
   ordinary chatters don't disguise their words.
"""
import re
import unicodedata

URL = re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|net|org|gg|tv|io|ly|me|xyz|shop|store|live|link|site|ru|cc)(?:/\S*)?\b", re.I)
MENTION = re.compile(r"@([A-Za-z0-9_]{2,25})")
WORD = re.compile(r"[a-z][a-z']*")

ZERO_WIDTH = re.compile("[​-‏⁠-⁤﻿­͏᠎]")
# Twitch clients append U+E0000 "tag" chars or this invisible to bypass the
# duplicate-message filter; that is normal chat client behaviour, not evasion.
DUP_BYPASS = re.compile("[\U000e0000-\U000e007f͏]")

# Small confusables map: the Cyrillic/Greek letters that look Latin and are
# actually used to sneak words past filters.
HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "ɡ": "g", "һ": "h", "ӏ": "l",
    "к": "k", "м": "m", "т": "t", "в": "b", "н": "h", "п": "n",
    "α": "a", "ε": "e", "ο": "o", "ρ": "p", "τ": "t", "υ": "u", "ν": "v",
    "ι": "i", "κ": "k", "χ": "x", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z",
    "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P",
    "Τ": "T", "Χ": "X", "Υ": "Y", "А": "A", "В": "B", "Е": "E", "К": "K",
    "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
})
QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'", "\u201c": '"', "\u201d": '"', "`": "'"})
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
                      "7": "t", "@": "a", "$": "s", "!": "i", "|": "l"})
INNER_LEET = re.compile(r"(?<=[a-z])[0-9@$!|]+(?=[a-z])")
REPEAT = re.compile(r"(.)\1{2,}")
SPACED = re.compile(r"\b(?:[a-z][ ._\-*]){2,}[a-z]\b")


def _latin_script(ch):
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:
        return "UNKNOWN"


def evasion_signals(text):
    """What a raw message does that only someone dodging a filter would do."""
    zw = len(ZERO_WIDTH.findall(DUP_BYPASS.sub("", text)))
    combining = sum(1 for c in text if unicodedata.combining(c))
    mixed_words = 0
    for tok in text.split():
        scripts = {_latin_script(c) for c in tok if c.isalpha()}
        if "LATIN" in scripts and scripts & {"CYRILLIC", "GREEK"}:
            mixed_words += 1
    nfkc_changed = unicodedata.normalize("NFKC", text) != text and any(
        0xFF01 <= ord(c) <= 0xFF5E or 0x1D400 <= ord(c) <= 0x1D7FF for c in text)
    return {
        "zero_width": zw,
        "zalgo": combining >= 6,
        "mixed_script_words": mixed_words,
        "fancy_unicode": nfkc_changed,      # fullwidth / math-bold "fonts"
    }


def normalize(text):
    """Lower-case, de-obfuscated form for pattern matching (not for display)."""
    t = DUP_BYPASS.sub("", text)
    t = ZERO_WIDTH.sub("", t)
    t = unicodedata.normalize("NFKC", t).translate(HOMOGLYPHS).translate(QUOTES)
    t = re.sub(r"\s+", " ", t)
    t = "".join(c for c in unicodedata.normalize("NFD", t)
                if not unicodedata.combining(c))
    t = t.lower()
    # leetspeak only where a symbol sits between letters ("id1ot", "f4g",
    # "sh!t"), so "1v1", "3am", "gg2" and "@name" survive as written
    t = INNER_LEET.sub(lambda m: m.group(0).translate(LEET), t)
    t = SPACED.sub(lambda m: re.sub(r"[ ._\-*]", "", m.group(0)), t)
    return REPEAT.sub(r"\1\1", t)


def plain_lower(text):
    """Lower-cased with only invisible characters removed; the 'as typed' form."""
    t = ZERO_WIDTH.sub("", DUP_BYPASS.sub("", text)).translate(QUOTES)
    return re.sub(r"\s+", " ", t).lower()


def chant_key(text):
    """Key for short crowd chants ('egg', 'L', 'PartyHat Happy anniversary!'):
    normalised words only, so 'Egg', 'egg', 'EGG!!' match."""
    return " ".join(re.findall(r"[a-z0-9]+", normalize(text)))[:60]


class Tokens:
    __slots__ = ("words", "emotes", "names", "mentions", "urls", "n_tokens")

    def __init__(self):
        self.words, self.emotes, self.names = [], [], []
        self.mentions, self.urls = [], []
        self.n_tokens = 0


STRIP = ".,!?;:'\"()[]{}…*~"


def tokenize(text, emote_spans=(), third_party=frozenset(), names=frozenset(),
             protected=frozenset()):
    """Split a message into words / emotes / names.

    emote_spans: [(id, start, end, name)] from the IRC tag
    third_party: set of 7TV/BTTV/FFZ emote codes (case-sensitive) for the channel
    names:       lower-cased logins of people who have *sent* messages
    protected:   lexicon words that must never be treated as a name
    """
    out = Tokens()
    # blank out first-party emotes by position so their text can't be read as words
    if emote_spans:
        chars = list(text)
        for _, a, b, name in emote_spans:
            out.emotes.append(name)
            for i in range(a, min(b + 1, len(chars))):
                chars[i] = " "
        text = "".join(chars)
    out.urls = URL.findall(text)
    text = URL.sub(" ", text)
    out.mentions = [m.lower() for m in MENTION.findall(text)]
    text = MENTION.sub(" ", text)
    for tok in DUP_BYPASS.sub("", text).split():
        out.n_tokens += 1
        bare = tok.strip(STRIP)
        if not bare:
            continue
        if bare in third_party:
            out.emotes.append(bare)
            continue
        low = bare.lower()
        if low in names and low not in protected:
            out.names.append(low)
            continue
        out.words.extend(WORD.findall(low))
    out.n_tokens += len(emote_spans)
    return out


def caps_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 5:
        return 0.0
    return sum(c.isupper() for c in letters) / len(letters)


NON_LATIN = re.compile(r"[Ѐ-ӿ؀-ۿऀ-ॿ฀-๿"
                       r"぀-ヿ一-鿿가-힯]")


def non_latin_share(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if NON_LATIN.match(c)) / len(letters)


def shape_key(text):
    """Key for spotting copy-pasted / templated messages: lower-case letters
    only, digits collapsed, so 'buy viewers 123' ~ 'BUY VIEWERS 456'."""
    t = re.sub(r"\d+", "#", normalize(text))
    t = re.sub(r"[^a-z# ]", "", t)
    return re.sub(r"\s+", " ", t).strip()
