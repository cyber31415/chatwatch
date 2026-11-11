"""Is this message toxic? Answered per message, whoever it's about.

Two different questions are kept apart on purpose:

  message toxicity  (this module)  "those guys never touch grass" is toxic
                                   mockery even though it targets nobody in chat
  person bad faith  (archetypes)   is this chatter here mainly to provoke or
                                   derail? third-party trash talk counts less

Each message gets a level (none / low / medium / high), the kinds of toxicity
found, and the target (streamer / chatter / third party / general). Mods can
then filter on toxicity regardless of intent, and on intent regardless of
wording.
"""

# cue -> (kind, strength 0..1)
CUE_KIND = {
    "insult_target": ("insult", 0.7), "insult_chat": ("insult", 0.7), "you_are_x": ("insult", 0.7),
    "insult_word": ("insult", 0.6), "appearance_insult": ("insult", 0.6),
    "dismiss_stream": ("dismissive", 0.5), "l_streamer": ("dismissive", 0.4),
    "viewbot_claim": ("dismissive", 0.4), "leaving_threat": ("dismissive", 0.3),
    "mockery": ("mockery", 0.4), "comparison": ("dismissive", 0.3),
    "hedged_attack": ("passive-aggressive", 0.3), "loaded_question": ("passive-aggressive", 0.3),
    "mod_complaint": ("dismissive", 0.2),
    "swear_at": ("hostile", 0.8),
    "slur": ("hate", 1.0), "dehumanize": ("hate", 0.9),
    "insinuation": ("harassment", 0.9), "threat": ("threat", 1.0), "sexual_threat": ("threat", 1.0),
    "appearance_sexual": ("sexual", 0.8),
    "ban_evasion_claim": ("evasion", 0.5),
}


import json as _json
import os as _os

_HEAD = None
# Tuned against three judged sets together (tools/tune_toxicity.py; FINDINGS "fast
# layer"): three judged gold sets. Mean precision 0.74 -> 0.86
# at the same F1 (0.76). Precision first: the LLM reads what this layer flags or
# is unsure about (serve.py) and recovers most of the recall.
PROFANITY_SCORE = 0.2        # swearing with nobody insulted: noted, not flagged (rubric: not toxic)
HEAD_T = None                # override the learned head's threshold (tuning only)
HEAD_T_MIN = 0.5             # floor under the model file's own threshold
INSULT_NEEDS_AIM = True      # the model's insult score counts only when aimed at someone


def _head():
    """The distilled toxicity head (tools/distill.py): trained on the LLM's
    silver labels, validated on judges' gold across channels (FINDINGS 3c)."""
    global _HEAD
    if _HEAD is None:
        path = _os.environ.get("CW_TOX_HEAD", "data/models/tox_head.json")
        _HEAD = load_head(_json.load(open(path))) if _os.path.exists(path) else False
    return _HEAD


def load_head(h):
    """A head file's dict (logistic or kind "mlp") in the form head_logit scores."""
    import numpy as np
    out = {"mu": np.array(h["mean"]), "sd": np.array(h["scale"]), "keys": h.get("tox_keys"),
           "t": h.get("threshold", 0.9)}
    if h.get("kind") == "mlp":               # head_mlp/RULE_POOLED.md: passed 2026-10-26
        out["nets"] = [[(np.array(W), np.array(b)) for W, b in zip(n["W"], n["b"])] for n in h["nets"]]
        out["shift"] = h["shift"]
    else:
        out["w"], out["b"] = np.array(h["coef"]), h["bias"]
    return out


def head_logit(h, X):
    """Logit of a loaded head (_head() form) for rows of standardisable features.
    MLP: mean over the ensemble's nets of relu(z W0 + b0) W1 + b1, plus the calibration shift."""
    import numpy as np
    z = (np.asarray(X, float) - h["mu"]) / h["sd"]
    if "nets" not in h:
        return z @ h["w"] + h["b"]
    out = 0
    for net in h["nets"]:
        a = z
        for k, (W, b) in enumerate(net):
            a = a @ W + b
            if k < len(net) - 1:
                a = np.maximum(a, 0)
        out = out + a[..., 0]
    return out / len(h["nets"]) + h["shift"]


def head_probability(f, layer_score):
    h = _head()
    if not h or f.get("ml_emb") is None:
        return None
    import numpy as np
    x = np.concatenate([f["ml_emb"], [f.get(k) or 0.0 for k in h["keys"]], [layer_score]])
    return float(1 / (1 + np.exp(-head_logit(h, x[None, :])[0])))


# N1 (data/study/model_threat_n1/RULE_N1.md): the model's threat score does not count when the threat is narrated
# about someone other than the speaker: "SHES GONNA KILL YOU", "this boss gonna kill you", "he wants you dead",
# "you said you were gonna kill him", "its not gonna hurt you". It still counts whenever the speaker or a proxy
# is the agent (the threat cue's speaker forms, "me and X gonna", "my boys"), the message names something
# real-world (irl, your house, where you live, dox, swat) or holds a plain threat ("i kill you", kys, "i hope
# you die"). The regexes are the dev study's N1 round 2 (data/scratch/modelthreatdev/mt_variants.py) as is.
import re as _re
from . import cues as _cues
_N1_KEEP = [
    _re.compile(_cues._SPEAKER, _re.I),
    _re.compile(r"\b(?:i'?m|im|i am|i|we|we're|we are|(?<!they )(?<!you )(?<!u )(?<!he )(?<!she )(?<!it )were|we r|"
                r"i'?ma|imma|ima|am a|me and \w+|me n \w+|me & \w+)\s(?:gonna|going to|gon|gunna|finna|about to|"
                r"bout to|will|tryna|trying to)\b|\bme to (?:kill|beat|hurt|shoot|stab|murder|end|smoke|jump)\b|"
                r"\b(?:my|our)\b", _re.I),
    _re.compile(r"\b(?:irl|in real life|tonight|your (?:house|home|address|family|kids|mom|mum)|outside your|"
                r"where (?:you|u) live|dox\w*|swat\w*)\b", _re.I),
    _re.compile(r"\bi(?:'?d| would| kill| murder| shoot)\b.*\b(?:you|u)\b|\bkys\b|\bkill (?:yo)?urself\b|"
                r"\bkill your ?self\b|\bgo die\b|\bi hope (?:you|u) die\b|\bi'?ll kill\b", _re.I)]
_N1_NOT_SUBJ = (r"i|im|i'm|ima|imma|we|were|we're|ill|i'll|we'll|well|you|u|ur|youre|you're|your|ya|yall|y'all|"
                r"lol|lmao|lmfao|xd|haha|ok|okay|yes|yeah|yea|no|nah|and|but|so|or|then|just|now|not|is|are|its?|"
                r"r|me|am")
_N1_FUT = r"(?:gonna|going to|gon|gunna|finna|about to|bout to|abt to|will|'ll|ll|wants to|want to|trying to|tryna)"
_N1_YOU = r"(?:you|u|ya|yall|y'all)"
_N1_NARRATED = [
    _re.compile(rf"(?<![@\w'])(?!(?:{_N1_NOT_SUBJ})\b)([a-z][\w']*)(?:'s|'re| is| are| was| were| r)? {_N1_FUT}\b"
                rf"(?: [\w']+){{1,3}}? {_N1_YOU}\b"),
    _re.compile(rf"(?<![@\w'])(?!(?:{_N1_NOT_SUBJ})\b)([a-z][\w']*) (?:wants?|wanted) {_N1_YOU} dead\b"),
    _re.compile(r"\b(?:you|u) (?:said|say|says|told|tell|claimed)\b.*\b(?:gonna|going to|will|would|were)\b"),
    _re.compile(r"\b(?:isnt|isn't|is not|not|aint|ain't|wont|won't|never|nobody|no one|noone|nothing|no way)"
                r"(?: [\w']+){0,2}? (?:gonna|going to|gon|will|be able to)\b")]


def threat_narrated(norm):
    """N1: the normalised text narrates a threat about someone other than the speaker."""
    return (not any(rx.search(norm) for rx in _N1_KEEP)) and any(rx.search(norm) for rx in _N1_NARRATED)


def message_toxicity(f):
    """Returns {"score": 0..1, "level": str, "kinds": [...], "target": str}."""
    score, kinds = 0.0, {}
    sure = set()           # severe kinds from keywords and coded vocabulary: not the model, not AutoMod

    def hit(kind, s, certain=False):
        nonlocal score
        kinds[kind] = max(kinds.get(kind, 0.0), s)
        score = max(score, s)
        if certain:
            sure.add(kind)

    for cue in f["cues"]:
        if cue in CUE_KIND:
            hit(*CUE_KIND[cue], certain=True)
    tox = f.get("ml_toxicity")
    if tox is not None:
        # the model alone flags praise ("you killed it"); trust it outright only
        # when very sure, otherwise as support for other evidence
        # ...and swearing with nobody insulted ("holy shit", "she fucking hates
        # combos") is at most "low": profanity isn't toxicity by itself
        hostile_cue = any(f["families"].get(k) for k in ("friction", "creep", "spam", "injection"))
        aimed = bool(f.get("aimed_at_chatter") or f["addresses_streamer"] or f.get("second_person"))
        insult = (f.get("ml_insult") or 0) >= 0.5 and (aimed or not INSULT_NEEDS_AIM)
        if tox >= 0.9 and (hostile_cue or insult):
            hit("toxic language", 0.6 * tox)
        elif tox >= 0.9:
            hit("profanity", PROFANITY_SCORE)
        if (f.get("ml_identity_attack") or 0) >= 0.4:
            hit("hate", 0.8 * f["ml_identity_attack"] + 0.2)
        # games are full of "shoot her", "kill him", "a single bullet": the
        # model's threat score counts only when it speaks to a real person
        aimed_you = bool(f.get("aimed_at_chatter") or (f["addresses_streamer"] and f.get("second_person")))
        if (f.get("ml_threat") or 0) >= 0.5 and aimed_you and not threat_narrated(f.get("norm") or ""):
            hit("threat", 0.8 * f["ml_threat"])
        # and it calls compliments ("she is so cute") sexual: only when very sure
        if (f.get("ml_sexual") or 0) >= 0.85:
            hit("sexual", 0.55 * f["ml_sexual"])
    am = f.get("automod") or {}
    if am.get("A", 0) >= 5:
        hit("hostile", 0.12 * am["A"])
    if am.get("I", 0) >= 5:
        hit("hate", 0.12 * am["I"])
    if am.get("S", 0) >= 6:
        hit("sexual", 0.1 * am["S"])
    # coded vocabulary: T1/U are hateful on their own; directed is worse;
    # first-person suicide vocabulary is distress, never toxicity
    for h in f.get("vocab") or ():
        if h["distress"] or h["tier"] in ("T2", "C", "T3"):
            continue
        directed = bool(f["addresses_streamer"] or f.get("aimed_at_chatter"))
        if h["tier"] == "T1-violent":
            hit("threat", 0.9, certain=True)
        else:
            hit("hate", 0.85 if directed else 0.65, certain=True)
    if f.get("evaded_cues"):
        hit("evasion", 0.6)
        score = min(1.0, score + 0.15)       # disguising it makes it worse, not better
    # the distilled head reads meaning the keyword layer misses ("everyone
    # hates you"); it may lift a message to low/medium, never to high alone
    hp = head_probability(f, score)
    t = HEAD_T if HEAD_T is not None else max((_head() or {}).get("t", 0.9), HEAD_T_MIN)
    if hp is not None and hp >= t and score < 0.7:
        hit("toxic (learned)", min(0.65, 0.3 + 0.35 * (hp - t) / max(1 - t, 1e-6)))
    # chants are not discounted: a chanted slur is still a slur (fixed standard,
    # docs/DESIGN.md "Absolute standard")

    if f["addresses_streamer"]:
        target = "streamer"
    elif f.get("aimed_at_chatter"):
        target = f"@{f['target_chatter']}"
    elif f.get("about_third_party"):
        target = "third party"
    else:
        target = "general"
    level = "high" if score >= 0.75 else "medium" if score >= 0.5 else "low" if score >= 0.3 else "none"
    out = {"score": round(min(score, 1.0), 3), "level": level,
            "kinds": sorted(kinds, key=lambda k: -kinds[k]), "target": target if score >= 0.3 else "",
            # the model's identity-attack score also fires on identity *discussion* (political chats) and
            # AutoMod's I/S flags on words like "pussy's" or "baby back bitch": both count as toxicity,
            # neither as a sure slur (FINDINGS "heads-up panel")
            "sure": sorted(sure & {"hate", "threat", "sexual", "harassment"})}
    if hp is not None:
        out["head"] = round(float(hp), 3)
    return out
