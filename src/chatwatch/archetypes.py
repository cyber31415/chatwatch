"""What kind of chatter might this be? A belief that sharpens per message.

The register styles from the handoff need ~300 words before they mean much.
This layer answers from the first message: it keeps, per user, a probability
over *archetypes* (what someone is doing in chat, not who they are), starting
from base rates and updated by every message.

Model
  log P(a | messages) = log prior(a) + account(a) + evidence(a, m1)
                        + DAMPING * sum_{m>1} evidence(a, m)

  evidence(a, m) is a sum of hand-set weights over the message's features
  (signals.extract). It is a log-likelihood ratio against an ordinary message,
  so "lol" or "what game is this" moves bad-faith archetypes by ~0.
  DAMPING < 1 because messages from one person aren't independent: five
  copies of the same complaint are not five times the evidence.

The weights are priors from reasoning, not fitted. evaluate.py measures them
against what mods actually did (timeouts, bans, deletions) and fit_risk.py
learns a separate calibrated risk score from the same features. Keep the two:
the archetypes explain, the risk score ranks.
"""
import math

from .cues import CUE_FAMILY as CUE_FAMILY_OF, CUE_REASON

DAMPING = 0.8

# name, family, label, short description shown in the UI
ARCHETYPES = [
    ("ordinary",      "benign",   "Ordinary chatter",   "plain on-topic chat"),
    ("reactor",       "benign",   "Reactor / hype",     "emotes, hype, copypasta with the crowd"),
    ("newcomer",      "benign",   "Friendly newcomer",  "greets, says they're new"),
    ("asker",         "benign",   "Curious asker",      "asks genuine questions"),
    ("supporter",     "benign",   "Supporter",          "compliments, bits, subs"),
    ("backseater",    "mild",     "Backseater",         "unsolicited directions / advice"),
    ("provocateur",   "badfaith", "Provocateur",        "baits, dismisses or insults the streamer or chat"),
    ("concern_troll", "badfaith", "Concern troll",      "hostility wrapped in politeness or 'just asking'"),
    ("injector",      "badfaith", "Topic injector",     "drops divisive / drama topics into a room that wasn't on them"),
    ("harasser",      "badfaith", "Harasser / creep",   "targets a person: sexual remarks, slurs, threats"),
    ("spammer",       "badfaith", "Spam / scam / bot",  "promotion, scams, templates, repetition"),
]
NAMES = [a[0] for a in ARCHETYPES]
FAMILY = {a[0]: a[1] for a in ARCHETYPES}
LABEL = {a[0]: a[2] for a in ARCHETYPES}
DESC = {a[0]: a[3] for a in ARCHETYPES}
BAD = [a for a in NAMES if FAMILY[a] == "badfaith"]

# Base rates for a first message in a typical channel. Bad-faith archetypes sum
# to ~0.13: deliberately higher than most channels' true rate so a few cues are
# enough to surface someone for a look (a false "watch" is cheap; this tool
# never acts on its own).
PRIOR = {
    "ordinary": 0.31, "reactor": 0.22, "newcomer": 0.10, "asker": 0.09,
    "supporter": 0.06, "backseater": 0.06,
    "provocateur": 0.06, "concern_troll": 0.025, "injector": 0.02,
    "harasser": 0.025, "spammer": 0.03,
}

BENIGN = [a for a in NAMES if FAMILY[a] == "benign"]
# A bad-faith cue is evidence *against* the benign archetypes too: a friendly
# newcomer almost never opens with "who even watches this guy". Applied once
# per message per family (log-likelihood ratio, benign vs reference).
FAMILY_BENIGN_PENALTY = {"friction": 1.0, "creep": 1.5, "spam": 1.5, "injection": 0.7}

# cue -> {archetype: weight}
CUE_W = {
    "insult_target":     {"provocateur": 1.8, "harasser": 0.4},
    "insult_chat":       {"provocateur": 1.8},
    "you_are_x":         {"provocateur": 1.3, "harasser": 0.8},
    "insult_word":       {"provocateur": 1.8, "harasser": 0.4},
    "appearance_insult": {"provocateur": 1.6, "harasser": 0.6},
    "dehumanize":        {"harasser": 2.4, "injector": 0.6},
    "insinuation":       {"harasser": 2.4, "provocateur": 0.6},
    "ban_evasion_claim": {"provocateur": 2.5, "harasser": 1.5},
    "threat":            {"harasser": 3.0, "provocateur": 0.5},
    "sexual_threat":     {"harasser": 3.0, "provocateur": 0.5},
    "dismiss_stream":    {"provocateur": 2.4},
    "l_streamer":        {"provocateur": 1.5},
    "viewbot_claim":     {"provocateur": 1.6},
    "mockery":           {"provocateur": 0.9, "reactor": 0.2},
    "hedged_attack":     {"concern_troll": 1.5, "provocateur": 0.3},
    "loaded_question":   {"concern_troll": 1.7, "provocateur": 0.4, "asker": -0.6},
    "comparison":        {"provocateur": 1.1, "concern_troll": 0.3},
    "leaving_threat":    {"provocateur": 0.9},
    "swear_at":          {"provocateur": 1.8, "harasser": 1.0},
    "mod_complaint":     {"provocateur": 0.8, "concern_troll": 0.4},
    "divisive_politics": {"injector": 1.7},
    "divisive_religion": {"injector": 1.5, "spammer": 0.3},
    "drama_bait":        {"injector": 1.1, "concern_troll": 0.4},
    "appearance_sexual": {"harasser": 2.4},
    "slur":              {"harasser": 2.4, "provocateur": 0.6},
    "promo":             {"spammer": 2.2},
    "scam":              {"spammer": 3.0},
    "newcomer":          {"newcomer": 2.2, "provocateur": -0.4},
    "greeting":          {"newcomer": 0.9, "ordinary": 0.3, "provocateur": -0.3},
    "honest_question":   {"asker": 1.8},
    "support":           {"supporter": 1.6, "provocateur": -0.6, "concern_troll": -0.3},
    "hype":              {"reactor": 1.0},
    "backseat":          {"backseater": 2.4},
    # callout cues describe someone *else*; they are applied to the target
    # via reaction evidence, never to the speaker
    "callout_mods":      {},
    "callout_social":    {},
}


def message_evidence(f):
    """Log-evidence per archetype from one message's features, plus the list
    of (archetype, weight, reason) contributions for explanation."""
    ev = dict.fromkeys(NAMES, 0.0)
    why = []

    def add(a, w, reason):
        ev[a] += w
        if abs(w) >= 0.3:
            why.append((a, round(w, 2), reason))

    # trash talk about third parties ("those guys never touch grass") isn't
    # aimed at the room: friction cues count for much less
    third = 0.4 if f.get("about_third_party") else 1.0
    for cue, matched in f["cues"].items():
        mult = min(2.0, 1 + 0.5 * matched.count(" | "))
        if CUE_FAMILY_OF.get(cue) == "friction":
            mult *= third
        for a, w in CUE_W.get(cue, {}).items():
            # the cue in words, as mods read it, not its key (audit LF4-6)
            add(a, w * mult if w > 0 else w, f"{CUE_REASON.get(cue, cue)}: “{matched}”")
    # ...scaled by the strongest cue of that family in this message, so a lone
    # "cope" (weak, everyday banter) barely dents the benign readings
    strength = {}
    for cue, matched in f["cues"].items():
        fam = CUE_FAMILY_OF.get(cue)
        w = max([w for w in CUE_W.get(cue, {}).values()] + [0])
        if fam == "friction":
            w *= third
        strength[fam] = max(strength.get(fam, 0), w)
    for fam, pen in FAMILY_BENIGN_PENALTY.items():
        if strength.get(fam):
            for a in BENIGN:
                ev[a] -= pen * min(1.0, strength[fam] / 2.0)

    if f["command"]:
        return ev, why

    # form
    if f["emote_only"]:
        add("reactor", 1.0, "emote-only")
    elif f["n_words"] and not f["cues"]:
        add("ordinary", 0.35, "plain message")
    if f["cats"].get("laugh") and not f["friction"]:
        add("reactor", 0.4, "laughing along")
    if f["question"] and not f["friction"] and not f["cues"].get("loaded_question"):
        add("asker", 0.5, "asks a question")
    if f["cats"].get("advice", 0) >= 1 and not f["friction"]:
        add("backseater", 0.3, "advice words")
    if f["bits"]:
        add("supporter", 1.5, f"cheered {f['bits']} bits")
        add("provocateur", -0.5, "cheered bits")

    # targeting & tone
    if f["targeted_negative"] and not f["friction"]:
        add("provocateur", 0.6, "negative words aimed at someone")
    if f["mocking_emotes"] and (f["friction"] or f["addresses_streamer"]):
        add("provocateur", 0.5, "mocking emote: " + " ".join(f["mocking_emotes"][:3]))
    if f["caps"] and (f["friction"] or f["negative"]):
        add("provocateur", 0.4, "shouting while negative")
    if f.get("aimed_at_chatter") and (f["friction"] or f["families"].get("creep")):
        add("harasser", 1.0, f"aimed at chatter @{f['target_chatter']}")
    if f["cues"].get("hedged_attack") and (f["negative"] or f["friction"] > 1):
        add("concern_troll", 0.8, "hedge followed by criticism")
    tox = f.get("ml_toxicity")
    if tox is not None:
        # Jigsaw-trained: flags praise ("you killed it" = 0.8) and banter, so
        # it counts fully only when aimed at someone (research/03 §1.3)
        if tox >= 0.5:
            aimed = f["addresses_streamer"] or f.get("aimed_at_chatter")
            add("provocateur", (1.2 if aimed else 0.7) * tox,
                f"toxicity model {tox:.2f}" + ("" if aimed else " (not aimed)"))
        if f.get("ml_identity_attack", 0) >= 0.3:
            add("harasser", 3.0 * f["ml_identity_attack"], f"identity attack {f['ml_identity_attack']:.2f}")
        if f.get("ml_threat", 0) >= 0.3:
            add("harasser", 2.5 * f["ml_threat"], f"threat {f['ml_threat']:.2f}")
        if f.get("ml_sexual", 0) >= 0.4:
            add("harasser", 2.0 * f["ml_sexual"], f"sexual {f['ml_sexual']:.2f}")

    # Twitch's own span labels (flags tag); profanity alone is just chat
    am = f.get("automod") or {}
    if am.get("A", 0) >= 5:
        add("provocateur", 0.25 * am["A"] - 0.5, f"Twitch hostility label A.{am['A']}")
    if am.get("I", 0) >= 5:
        add("harasser", 0.3 * am["I"] - 0.6, f"Twitch identity label I.{am['I']}")
    if am.get("S", 0) >= 5 and (f["addresses_streamer"] or f["target_chatter"]):
        add("harasser", 0.2 * am["S"], f"Twitch sexual label S.{am['S']} aimed at someone")

    # semantic similarity to example messages (T1b): only counts when the
    # message is clearly closer to an archetype's examples than to ordinary
    # chat, and only where no cue already fired for that archetype
    proto = f.get("proto")
    if proto and f["n_words"] >= 3:
        ben = proto.get("_benign", 0.0)
        cued = {a for c in f["cues"] for a, w in CUE_W.get(c, {}).items() if w > 0}
        for a in ("provocateur", "concern_troll", "injector", "harasser", "spammer"):
            sim = proto.get(a, 0.0)
            if sim >= 0.45 and sim - ben >= 0.12 and a not in cued:
                add(a, min(1.0, 5.0 * (sim - ben)), f"reads like {LABEL[a].lower()} examples (sim {sim:.2f})")

    # evasion is rare in good faith
    if f["evaded_cues"]:
        for a in ("provocateur", "harasser", "spammer"):
            add(a, 1.0, "disguised words: " + ", ".join(CUE_REASON.get(c, c) for c in f["evaded_cues"]))
    elif f["evasion"]:
        add("spammer", 0.6, "invisible / look-alike characters")
        add("harasser", 0.3, "invisible / look-alike characters")

    # shared text: crowd copypasta is culture; a template from new accounts isn't.
    # Chanting harmless things along with the room ("egg", "L") is participation;
    # chanting a slur, hate or a threat is not excused by the crowd doing it too
    # (fixed standard, docs/DESIGN.md "Absolute standard")
    toxic_words = bool(f["cues"].get("slur") or f["cues"].get("threat") or f["cues"].get("sexual_threat")
                       or any(h["tier"] in ("T1", "U", "T1-violent") and not h["distress"] for h in f.get("vocab", ())))
    if f.get("crowd_echo", 0) >= 3 and not f["template_new_accounts"] and not toxic_words:
        # the policy's own exclusion: chanting along with the room ("egg",
        # "PartyHat Happy anniversary", "L") is participation, not an attack
        for a in BAD:
            ev[a] -= 1.5
        add("reactor", 1.0, f"echoes what {f['crowd_echo']} others just said")
    if f["template_new_accounts"] >= 2:
        add("spammer", 2.0, f"same text from {f['template_new_accounts']} other new accounts")
    elif f["copypasta_others"] >= 2:
        add("reactor", 0.8, f"copypasta with {f['copypasta_others']} others")
    if f["self_repeat"]:
        # repeating a link is how spam looks; repeating your own taunt or joke is
        # flooding, judged by what it says (judges, one judged session: R5 means
        # promotion, scams, bots, templated text across accounts)
        if f["has_url"]:
            add("spammer", 0.7, "repeats own link")
        elif f["n_words"] >= 5:
            add("spammer", 0.25, "repeats own message")
        else:
            add("reactor", 0.2, "repeats a short reaction")
    if f["has_url"]:
        add("spammer", 1.4 if f["first_msg"] else 0.6, "posts a link")

    # topics mods have been acting on are shown as context (flag "moderated-topic"),
    # never as evidence: what a community moderates can be viewpoint (owner standard)
    if f["ascii_art"]:
        add("reactor", 0.4, "ASCII art")
        add("spammer", 0.4, "ASCII art")

    # oddness only matters when combined with something else
    # Oddness is shown as a flag but adds no bad-faith weight: newcomers
    # naturally sound unlike the room ("No Country for Old Members", research/02
    # #8) and perplexity-style signals penalise non-native writers.
    odd = f.get("oddness")
    if f["families"].get("injection") and f.get("topic_in_room"):
        # other people were already on it: this is the conversation, not an injection
        add("injector", -1.8, "topic already being discussed in the room")

    # pragmatic openers (Conversations Gone Awry, research/02 #9): a direct
    # question that opens with "you..." tends to precede derailment; greetings
    # and gratitude precede civil exchanges. Small weights; they only tip.
    if f["starts_you"] and f["question"]:
        add("concern_troll", 0.35, "opens with a direct 'you...?' question")
        add("provocateur", 0.2, "opens with a direct 'you...?' question")
    if f["cats"].get("polite") or f["cues"].get("greeting"):
        for a in ("provocateur", "concern_troll"):
            ev[a] -= 0.3

    # context: friction in a calm room is instigation, in a heated room it's
    # partly contagion (Cheng et al. 2017)
    # (no discount for joining a heated room: the fixed standard applies to everyone)
    if f["friction"] and not f["room_heated"]:
        add("provocateur", 0.3, "starts friction in a calm room")
    return ev, why


def account_evidence(f):
    """Per-user evidence from metadata, applied once (refreshed as tags change)."""
    ev = dict.fromkeys(NAMES, 0.0)
    why = []

    def add(a, w, reason):
        ev[a] += w
        why.append((a, round(w, 2), reason))

    # mods, VIPs and the broadcaster are held to the same standard as everyone else
    # (owner standard); their role is shown on alerts, not used as evidence
    sm = f["sub_months"]
    if sm >= 12:
        for a in BAD:
            add(a, -1.2, f"subscribed {sm} months")
    elif sm >= 3:
        for a in BAD:
            add(a, -0.6, f"subscribed {sm} months")
    if f["first_msg"]:
        add("newcomer", 0.8, "first message in this channel")
        add("spammer", 0.4, "first message in this channel")
    age = f.get("account_age_days")
    if age is not None:
        # 10.9% of first-time chatters vs 1.0% of all chatters had accounts
        # under 30 days in the research/01 probe; throwaways cluster at hours
        if age < 1:
            for a, w in (("spammer", 1.4), ("provocateur", 0.8), ("harasser", 0.8), ("concern_troll", 0.4)):
                add(a, w, f"account created {age * 24:.0f} h ago")
        elif age < 7:
            for a, w in (("spammer", 0.9), ("provocateur", 0.5), ("harasser", 0.5), ("concern_troll", 0.3)):
                add(a, w, f"account {age:.0f} days old")
        elif age < 30:
            for a, w in (("spammer", 0.4), ("provocateur", 0.2), ("harasser", 0.2)):
                add(a, w, f"account {age:.0f} days old")
        elif age > 5 * 365:
            for a in BAD:
                add(a, -0.3, f"account {age / 365:.0f} years old")
    if f["name_randomness"] >= 0.5 and not f["has_badges"]:
        add("spammer", 0.6, "machine-like username")
    # a name like a recently banned one is shown (flag, chip, room warning) but is no evidence: it would be a mod
    # action used as evidence about another person (BASELINE 8.1, audit LB-7)
    if f["room_raid"] and f["first_msg"]:
        add("reactor", 0.6, "arrived with a raid")
    if f.get("pile_on") and f["first_msg"]:
        # the base rate itself has changed: a sustained flood of hostile
        # newcomers means a first message is more likely part of it
        for a, w in (("provocateur", 0.7), ("harasser", 0.7), ("concern_troll", 0.4), ("injector", 0.3)):
            add(a, w, "arrived during a pile-on")
    return ev, why


def reaction_evidence(kind, detail=""):
    """Evidence about a user from how others / mods react to them."""
    if kind == "callout":
        return {"provocateur": 0.7, "concern_troll": 0.4, "harasser": 0.4,
                "spammer": 0.3}, f"chat called them out: “{detail}”"
    if kind == "deleted":
        return {"provocateur": 1.2, "harasser": 1.2, "spammer": 1.2,
                "injector": 0.8, "concern_troll": 0.6}, "a mod deleted their message"
    if kind == "timeout":
        # short timeouts are often bot filters (caps, links) or channel games
        secs = int(detail.rstrip("s") or 0) if detail.rstrip("s").isdigit() else 600
        k = 0.4 if secs <= 60 else 0.8 if secs < 600 else 1.0
        return {a: k * w for a, w in {"provocateur": 1.5, "harasser": 1.5, "spammer": 1.5,
                "injector": 1.0, "concern_troll": 0.8}.items()}, f"timed out ({detail})"
    if kind == "ban":
        return {a: 3.0 for a in BAD}, "banned"
    return {}, ""


BENIGN_CAP = 2.0


def saturate(a, x):
    """Benign evidence saturates; bad-faith evidence keeps adding up.

    Someone who has said "lol" ten times is no less likely to troll than
    someone who said it three times, but every insult is new information.
    Without this, a run of trivial friendly messages would cancel out real
    hostility that comes after it."""
    if FAMILY[a] == "benign" and x > 0:
        return BENIGN_CAP * math.tanh(x / BENIGN_CAP)
    return x


RULE_ARCHETYPE = {"R1": "provocateur", "R2": "concern_troll", "R3": "injector", "R4": "harasser", "R5": "spammer"}


def llm_evidence(review):
    """Evidence from the policy LLM's verdict (llm.review). Its value depends on
    the base rate: in a pile-on it finds 3x the bad faith of the cues at the same
    precision; on normal channels it over-flags (FINDINGS §1c), so it is only
    *screened* into reads during pile-ons and otherwise shown as a second opinion."""
    lab, conf = review.get("label"), review.get("confidence", "low")
    k = {"high": 1.0, "medium": 0.8, "low": 0.5}.get(conf, 0.5)
    if lab == "bad_faith":
        arcs = [RULE_ARCHETYPE[r] for r in review.get("rules", []) if r in RULE_ARCHETYPE] or ["provocateur"]
        return {a: 2.5 * k for a in arcs}, f"policy LLM: bad faith ({', '.join(review.get('rules', [])) or '?'}, {conf})"
    if lab == "good_faith":
        return {a: -1.0 * k for a in BAD}, f"policy LLM: good faith ({conf})"
    return {}, ""


def posterior(logscore):
    m = max(logscore.values())
    ex = {a: math.exp(v - m) for a, v in logscore.items()}
    z = sum(ex.values())
    return {a: v / z for a, v in ex.items()}


def prior_logscore():
    return {a: math.log(PRIOR[a]) for a in NAMES}


def entropy_certainty(post):
    """1 - normalised entropy: 0 = no idea, 1 = certain."""
    h = -sum(p * math.log(p) for p in post.values() if p > 0)
    return 1 - h / math.log(len(post))
