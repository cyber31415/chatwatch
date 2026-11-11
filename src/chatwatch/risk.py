"""User-level feature aggregates and a learned "mods will act" risk score.

The archetype layer explains; this layer ranks. It is a logistic regression
over interpretable per-user aggregates, fitted by fit_risk.py on collected
chat with *future* mod actions as the label, cross-validated across channels
so it can't just memorise one channel's rules. Coefficients are stored in
data/models/risk_lr.json and each live score comes with its top
contributions.

Label caveat (see docs/FINDINGS.md): a mod action means "broke this channel's
rules", which overlaps bad faith but isn't the same thing.
"""
import json
import math
import os

from . import archetypes as A

AGG_KEYS = ("friction", "injection", "creep", "spam", "benign", "evasion", "url",
            "caps", "template", "copypasta", "ascii_art", "moderated_topic",
            "targeted_negative", "self_repeat", "questions", "emote_only", "hype",
            "slur", "odd", "first_msg")


BEHAVIOUR_KEYS = ("b_tox", "b_tox_strong", "b_hostile", "b_reply", "b_at_chatter", "b_at_streamer",
                  "b_command", "b_emotes", "b_chant", "b_heated", "b_pile", "b_non_latin", "b_positive")
ROOM_WINDOW_MS = 20 * 60 * 1000


def new_agg():
    a = dict.fromkeys(AGG_KEYS, 0)
    a.update(max_tox=0.0, max_insult=0.0, max_identity=0.0, max_bf_ev=0.0, sum_bf_ev=0.0,
             tw_A=0, tw_I=0, tw_S=0, tw_P=0, max_sexual=0.0, max_threat=0.0, max_proto_bad=0.0,
             crowd_echo=0, aimed_negative=0)
    # behaviour counts from the chat behaviour study (docs/STUDY.md): how this
    # person tends to chat, which the study found to be personal and stable
    a.update(dict.fromkeys(BEHAVIOUR_KEYS, 0), b_tox_score=0.0, b_words=0.0)
    return a


def update(a, f, bf_ev, tox=None):
    fam = f["families"]
    if tox is not None:
        lvl = tox["level"]
        a["b_tox"] += lvl != "none"
        a["b_tox_strong"] += lvl in ("medium", "high")
        a["b_tox_score"] += float(tox.get("head", f.get("ml_toxicity") or 0.0) or 0.0)
        a["b_hostile"] += bf_ev >= 1.0
        a["b_words"] += min(math.log1p(f["n_words"]) / 3, 1.5)
        a["b_reply"] += bool(f["is_reply"])
        a["b_at_chatter"] += bool(f["aimed_at_chatter"])
        a["b_at_streamer"] += bool(f["addresses_streamer"])
        a["b_command"] += bool(f["command"])
        a["b_emotes"] += f["n_emotes"] > 0
        a["b_chant"] += (f["copypasta_others"] > 0) or (f.get("crowd_echo", 0) > 0)
        a["b_heated"] += bool(f["room_heated"])
        a["b_pile"] += bool(f["pile_on"])
        a["b_non_latin"] += (f.get("non_latin") or 0) > 0.5
        a["b_positive"] += f["positive"] > 0
    a["friction"] += fam.get("friction", 0) > 0
    a["injection"] += fam.get("injection", 0) > 0
    a["creep"] += fam.get("creep", 0) > 0
    a["spam"] += fam.get("spam", 0) > 0
    a["benign"] += fam.get("benign", 0) > 0
    a["evasion"] += bool(f["evasion"])
    a["url"] += bool(f["has_url"])
    a["caps"] += bool(f["caps"])
    a["template"] += f["template_new_accounts"] >= 2
    a["copypasta"] += f["copypasta_others"] >= 2
    a["ascii_art"] += bool(f["ascii_art"])
    a["moderated_topic"] += bool(f["moderated_topics"])
    a["targeted_negative"] += bool(f["targeted_negative"])
    a["self_repeat"] += bool(f["self_repeat"])
    a["questions"] += bool(f["question"])
    a["emote_only"] += bool(f["emote_only"])
    a["hype"] += "hype" in f["cues"]
    a["slur"] += "slur" in f["cues"]
    a["odd"] += (f.get("oddness") or 0) >= 0.97
    a["first_msg"] = a["first_msg"] or f["first_msg"]
    a["max_tox"] = max(a["max_tox"], f.get("ml_toxicity") or 0.0)
    a["max_insult"] = max(a["max_insult"], f.get("ml_insult") or 0.0)
    a["max_identity"] = max(a["max_identity"], f.get("ml_identity_attack") or 0.0)
    a["max_bf_ev"] = max(a["max_bf_ev"], bf_ev)
    a["max_sexual"] = max(a["max_sexual"], f.get("ml_sexual") or 0.0)
    a["max_threat"] = max(a["max_threat"], f.get("ml_threat") or 0.0)
    pr = f.get("proto")
    if pr:
        a["max_proto_bad"] = max(a["max_proto_bad"],
                                 max(pr.get(k, 0) for k in ("provocateur", "concern_troll", "injector", "harasser", "spammer"))
                                 - pr.get("_benign", 0))
    a["crowd_echo"] += f.get("crowd_echo", 0) >= 3
    a["aimed_negative"] += bool((f.get("ml_toxicity") or 0) >= 0.5 and (f["addresses_streamer"] or f["target_chatter"]))
    for c, lvl in (f.get("automod") or {}).items():
        if "tw_" + c in a:
            a["tw_" + c] = max(a["tw_" + c], lvl)
    a["sum_bf_ev"] += bf_ev


def vector(u, room_rate_actions, room_ctx=None):
    """Flat numeric features for one user right now. Counts become shares
    or any-indicators so long and short histories are comparable.
    room_ctx: the room's last 20 minutes (Engine.room_context)."""
    a, n = u.agg, max(u.n, 1)
    f = u.last_f or {}
    post = u.post()
    v = {
        "log_msgs": math.log1p(u.n),
        "first_msg": float(a["first_msg"]),
        "sub": float(u.sub_months > 0),
        "sub12": float(u.sub_months >= 12),
        "badges": float(bool(u.badges)),
        "default_color": float(not u.color),
        "name_random": f.get("name_randomness", 0.0),
        "acct_new99": float((f.get("account_newness") or 0) >= 0.99),
        "acct_new": f.get("account_newness") if f.get("account_newness") is not None else 0.5,
        "acct_lt7d": float((f.get("account_age_days") if f.get("account_age_days") is not None else 9e9) < 7),
        "acct_lt30d": float((f.get("account_age_days") if f.get("account_age_days") is not None else 9e9) < 30),
        "acct_log_age": math.log1p(f.get("account_age_days") or 0.0),
        "twitch_A": a["tw_A"], "twitch_I": a["tw_I"], "twitch_S": a["tw_S"], "twitch_P": a["tw_P"],
        "callouts": math.log1p(sum(1 for r in u.reactions if r[1] == "callout")),
        "room_action_rate": room_rate_actions,
        "badfaith_post": sum(post[k] for k in A.BAD),
        "max_bf_ev": a["max_bf_ev"],
        "mean_bf_ev": a["sum_bf_ev"] / n,
        "max_tox": a["max_tox"], "max_insult": a["max_insult"], "max_identity": a["max_identity"],
        "max_sexual": a["max_sexual"], "max_threat": a["max_threat"], "max_proto_bad": a["max_proto_bad"],
        "share_crowd_echo": a["crowd_echo"] / n, "any_aimed_negative": float(a["aimed_negative"] > 0),
    }
    for k in AGG_KEYS:
        if k != "first_msg":
            v["any_" + k] = float(a[k] > 0)
            v["share_" + k] = a[k] / n
    for k in BEHAVIOUR_KEYS:
        v["share_" + k] = a.get(k, 0) / n
    v["mean_b_tox_score"] = a.get("b_tox_score", 0.0) / n
    v["mean_b_words"] = a.get("b_words", 0.0) / n
    rc = room_ctx or {}
    v["room_toxic_share"] = rc.get("toxic_share", 0.0)
    v["room_log_msgs_per_min"] = math.log1p(rc.get("msgs_per_min", 0.0))
    return v


class RiskModel:
    def __init__(self, path="data/models/risk_lr.json"):
        self.ok = os.path.exists(path)
        if self.ok:
            m = json.load(open(path))
            self.coef, self.bias = m["coef"], m["bias"]
            self.mean, self.scale = m["mean"], m["scale"]
            self.meta = m.get("meta", {})

    def score(self, v):
        """(probability, top contributions)"""
        if not self.ok:
            return None, []
        contrib = {}
        z = self.bias
        for k, w in self.coef.items():
            x = (v.get(k, 0.0) - self.mean[k]) / (self.scale[k] or 1.0)
            contrib[k] = w * x
            z += w * x
        top = sorted(contrib.items(), key=lambda kv: -kv[1])[:4]
        return 1 / (1 + math.exp(-z)), [(k, round(c, 2)) for k, c in top if c > 0.05]
