"""Scenario suite: the question this project is about.

"Someone just chatted for the first time (or a few times, or oddly). What
might they be?" Each scenario feeds a small, realistic room through the real
engine and checks the first-impression read. These are behavioural specs, so a
failing scenario means the model's judgement changed, not just its code.

Run:  bin/cw python -m pytest -q tests/test_scenarios.py
"""
import itertools
import zlib

from chatwatch.engine import Engine
from chatwatch.serve import ReadQueue

_ids = itertools.count(1_000_000)
T0 = 1_792_525_000_000
OLD_ID = "40000000"            # 2013 account
NEW_ID = "1567400000"          # created ~hours before T0


def msg(login, text, t, *, first=False, uid=None, sub=0, color="#8A2BE2", reply_to="", emotes=()):
    return {"kind": "msg", "ts": T0 + t * 1000, "channel": "cozy", "room_id": "", "id": f"m{next(_ids)}",
            "user_id": uid or str(50_000_000 + zlib.crc32(login.encode()) % 10_000_000), "login": login, "display": login,
            "text": text, "action": False, "badges": {"subscriber": "1"} if sub else {},
            "badge_info": {"subscriber": str(sub)} if sub else {}, "color": color, "emotes": list(emotes),
            "first_msg": first, "returning": False, "mod": False, "broadcaster": False, "vip": False,
            "sub": bool(sub), "bits": 0, "reply_to_login": reply_to, "reply_to_id": "", "reply_to_msg": "",
            "source_room": "", "automod": {}, "tags": {}}


CALM_ROOM = ["this boss fight is so tense", "you got this!", "nice dodge", "LUL that was close",
             "the music in this area is great", "what level are you", "gg", "love this game",
             "the lighting here is gorgeous", "i think the key is upstairs", "ok that was cool",
             "haha the cat walked across the keyboard", "hydrate chat", "how long have you been streaming today"]


def room(eng, n=60):
    """Warm up a calm, friendly room of regulars."""
    for i in range(n):
        eng.process(msg(f"regular{i % 12}", CALM_ROOM[i % len(CALM_ROOM)], i * 3, sub=6))
    return n * 3 + 5


def read(eng, login):
    u = eng.users[("cozy", next(uid for (c, uid), x in eng.users.items() if x.login == login))]
    post = u.post()
    top = max(post, key=post.get)
    return top, u.badfaith(post), post


def first(text, **kw):
    eng = Engine(legacy=False)
    t = room(eng)
    eng.process(msg("newbie", text, t, first=True, **kw))
    return read(eng, "newbie")


# ------------------------------------------------------------ friendly firsts
def test_friendly_newcomer():
    top, bf, _ = first("hi everyone! first time here, found you through a clip")
    assert top == "newcomer" and bf < 0.1


def test_honest_question():
    top, bf, _ = first("what game is this? looks really fun")
    assert top == "asker" and bf < 0.1


def test_plain_hype():
    top, bf, _ = first("LETS GOOO")
    assert top == "reactor" and bf < 0.1


def test_backseater_is_not_bad_faith():
    top, bf, _ = first("you should go upstairs and use the shotgun")
    assert top == "backseater" and bf < 0.2


# --------------------------------------------------------------- blatant
def test_blatant_dismissal_first_message():
    top, bf, _ = first("who even watches this guy, so boring")
    assert top == "provocateur" and bf > 0.6


def test_spam_first_message():
    top, bf, _ = first("Best viewers on streamboo .com check it out")
    assert top == "spammer" and bf > 0.6


def test_creep_first_message():
    top, bf, _ = first("you look so hot today show us your feet")
    assert top == "harasser" and bf > 0.6


# ---------------------------------------------------------------- subtle
def test_concern_troll():
    top, bf, _ = first("no offense but genuine question, why is he always this bad at games?")
    assert top == "concern_troll" and bf > 0.5


def test_topic_injection_into_calm_room():
    top, bf, post = first("what do you think about trump and the immigrants situation")
    assert post["injector"] == max(post[k] for k in ("injector", "provocateur", "harasser", "spammer"))
    assert bf > 0.35


def test_disguised_insult_is_worse_than_plain():
    _, bf_plain, _ = first("you are an idiot")
    _, bf_disg, _ = first("you are an id1ot")
    assert bf_disg > bf_plain


# ---------------------------------------------------------------- context
def test_new_account_raises_suspicion_of_same_text():
    _, bf_old, _ = first("this stream is boring", uid=OLD_ID)
    _, bf_new, _ = first("this stream is boring", uid=NEW_ID)
    assert bf_new > bf_old


def test_longtime_sub_gets_benefit_of_doubt():
    _, bf_sub, _ = first("this stream is boring", sub=24)
    _, bf_anon, _ = first("this stream is boring")
    assert bf_sub < bf_anon


def test_evidence_accumulates_and_plain_messages_dont_move_it():
    eng = Engine(legacy=False)
    t = room(eng)
    eng.process(msg("drifter", "lol", t, first=True))
    _, bf1, _ = read(eng, "drifter")
    for i, s in enumerate(["nice", "gg", "what level is this"]):
        eng.process(msg("drifter", s, t + 5 + i))
    _, bf2, _ = read(eng, "drifter")
    assert bf2 <= bf1 + 0.01            # ordinary chat doesn't build suspicion
    for i, s in enumerate(["this streamer is trash", "who even watches this", "this channel fell off hard"]):
        eng.process(msg("drifter", s, t + 20 + i * 5))
    _, bf3, _ = read(eng, "drifter")
    assert bf3 > 0.8


def test_instigator_vs_joiner():
    """Same line: starting friction in a calm room vs joining a heated one."""
    eng = Engine(legacy=False)
    t = room(eng)
    eng.process(msg("starter", "this streamer is so bad lmao", t, first=True))
    calm = eng.users[("cozy", next(k[1] for k, u in eng.users.items() if u.login == "starter"))]

    eng2 = Engine(legacy=False)
    t2 = room(eng2)
    for i in range(14):   # the room turns on the streamer
        eng2.process(msg(f"angry{i}", ["this streamer is trash", "so boring", "unwatchable", "fell off"][i % 4],
                         t2 + i))
    eng2.process(msg("joiner", "this streamer is so bad lmao", t2 + 16, first=True))
    joined = eng2.users[("cozy", next(k[1] for k, u in eng2.users.items() if u.login == "joiner"))]
    assert calm.instigated == 1 and joined.joined == 1
    assert calm.badfaith() > joined.badfaith()


def test_callout_is_not_evidence():
    # chat's and mods' reactions are community enforcement, not evidence about the
    # person (engine.REACTIONS_AS_EVIDENCE = False): a callout leaves the read as it was
    eng = Engine(legacy=False)
    t = room(eng)
    eng.process(msg("troll", "this stream is boring", t, first=True))
    before = eng.users[("cozy", next(k[1] for k, u in eng.users.items() if u.login == "troll"))].badfaith()
    eng.process(msg("regular1", "@troll who asked", t + 3, sub=6))
    eng.process(msg("regular2", "mods", t + 4, sub=6))
    after = eng.users[("cozy", next(k[1] for k, u in eng.users.items() if u.login == "troll"))].badfaith()
    assert after == before


def test_coordinated_template_from_new_accounts():
    eng = Engine(legacy=False)
    t = room(eng)
    alerts = []
    for i in range(5):
        rec = eng.process(msg(f"x{i}qz{i}", "this channel is a joke and so are you", t + i,
                              first=True, uid=str(1567400000 + i)))
        alerts += rec.get("alerts", [])
    assert any(a["kind"] == "coordinated" for a in alerts)
    top, _, _ = read(eng, "x4qz4")
    assert top in ("spammer", "provocateur")


def test_crowd_copypasta_is_not_a_template():
    eng = Engine(legacy=False)
    t = room(eng)
    for i in range(8):
        rec = eng.process(msg(f"regular{i}", "Choo Chooo the hype train is here", t + i, sub=6))
        assert not any(a["kind"] == "coordinated" for a in rec.get("alerts", []))


def test_ban_evasion_lookalike():
    eng = Engine(legacy=False)
    t = room(eng)
    eng.process(msg("trollking", "you are an idiot", t))
    eng.process({"kind": "clearchat", "ts": T0 + (t + 2) * 1000, "channel": "cozy", "room_id": "",
                 "target_login": "trollking", "target_user_id": "", "duration": -1})
    rec = eng.process(msg("trollking2", "hi", t + 30, first=True, uid=NEW_ID))
    assert any(a["kind"] == "lookalike" for a in rec.get("alerts", []))


def test_moderated_topic_learned_from_mod_actions():
    eng = Engine(legacy=False)
    t = room(eng)
    for i in range(4):
        eng.process(msg(f"asker{i}", "when are you playing elden ring", t + i * 4))
        eng.process({"kind": "clearchat", "ts": T0 + (t + i * 4 + 1) * 1000, "channel": "cozy",
                     "room_id": "", "target_login": f"asker{i}", "target_user_id": "", "duration": 30})
    rec = eng.process(msg("next", "elden ring when", t + 30, first=True))
    assert "moderated-topic" in rec["flags"]


def test_privileged_same_standard():
    # mods, VIPs and the broadcaster are held to the same standard; the alert
    # shows their role (docs/DASHBOARD.md)
    reads = {}
    for who in ("modguy", "viewer"):
        eng = Engine(legacy=False)
        t = room(eng)
        ev = msg(who, "who even watches this, so boring, fell off", t)
        if who == "modguy":
            ev["mod"], ev["badges"] = True, {"moderator": "1"}
        reads[who] = eng.process(ev)
    assert reads["modguy"]["badfaith"] == reads["viewer"]["badfaith"]
    assert [a.get("role") for a in reads["modguy"].get("alerts", [])] == ["mod"]


def test_forget_removes_user():
    eng = Engine()
    t = room(eng, 20)
    eng.process(msg("someone", "hello there", t, uid="777"))
    assert eng.forget("777") == 1
    assert not any(k[1] == "777" for k in eng.users)
    assert eng.legacy["cozy"].profile("777") is None


def test_bad_moment_fades_over_weeks():
    eng = Engine(legacy=False)
    t = room(eng)
    eng.process(msg("grumpy", "this streamer is trash, who even watches this", t, uid="555"))
    hot = eng.users[("cozy", "555")].badfaith()
    later = t + 21 * 86400          # three weeks later, an ordinary message
    eng.process(msg("grumpy", "hi chat, good to be back", later, uid="555"))
    assert eng.users[("cozy", "555")].badfaith() < hot / 2


def test_politics_in_a_political_room_is_not_injection():
    eng = Engine(legacy=False)
    t = room(eng)
    for i, s in enumerate(["the election results are wild", "trump is leading in ohio",
                           "democrats need a better candidate", "biden polling is rough"]):
        eng.process(msg(f"pol{i}", s, t + i * 5, sub=6))
    eng.process(msg("newbie", "what do you think about trump honestly", t + 30, first=True))
    _, bf_in_room, post = read(eng, "newbie")
    _, bf_cold, _ = first("what do you think about trump honestly")
    assert bf_in_room < bf_cold


def test_wave_of_brand_new_accounts():
    eng = Engine(legacy=False)
    t = room(eng)
    alerts = []
    for i in range(7):
        rec = eng.process(msg(f"q{i}x8k2m{i}", ["hello", "lol", "this stream sucks", "hi", "yo", "LUL", "ok"][i],
                              t + i * 2, first=True, uid=str(1567500000 + i)))
        alerts += rec.get("alerts", [])
    waves = [a for a in alerts if a["kind"] == "wave"]
    assert len(waves) == 1 and waves[0]["new_accounts"] >= 6


def test_ordinary_first_timers_are_not_a_wave():
    eng = Engine(legacy=False)
    t = room(eng)
    for i in range(10):
        rec = eng.process(msg(f"viewer{i}", "hi chat", t + i * 2, first=True))   # old accounts
        assert not any(a["kind"] == "wave" for a in rec.get("alerts", []))


def test_state_saved_by_older_code_still_loads():
    """Restoring data/state/engine.pkl from an older version must not crash
    message processing or user cards (it once caused a reconnect loop)."""
    import pickle
    eng = Engine()
    t = room(eng, 20)
    eng.process(msg("olduser", "hello there", t, uid="888"))
    eng.risk = eng.ml = eng.emotes = None            # as save_engine() stores it
    old = pickle.loads(pickle.dumps(eng))
    r = old.rooms["cozy"]
    del r.chants, r.pile                             # attributes added in later versions
    u = old.users[("cozy", "888")]
    del u.held                                       # a slot added later
    del u.agg["max_sexual"]                          # an aggregate key added later
    old.upgrade()
    rec = old.process(msg("olduser", "still here", t + 5, uid="888"))
    assert rec and rec["type"] == "msg"
    assert old.profile("cozy", "888")["messages"] == 2


def test_at_mention_that_answers_is_not_an_attack():
    """User-reported misread: '@X that's sweat mode, those guys never touch
    grass' answers X and mocks third parties; it is not aimed at X."""
    eng = Engine(legacy=False)
    t = room(eng)
    eng.process(msg("asker7", "why is everyone running the same build?", t))
    eng.process(msg("helper", "@asker7 That's super sweat mode. Those guys never touch grass.", t + 3, first=True))
    _, bf_answer, _ = read(eng, "helper")
    u = next(x for x in eng.users.values() if x.login == "helper")
    assert not u.last_f["aimed_at_chatter"] and u.last_f["about_third_party"]

    eng2 = Engine(legacy=False)
    t2 = room(eng2)
    eng2.process(msg("asker7", "why is everyone running the same build?", t2))
    eng2.process(msg("jerk", "@asker7 you never touch grass, loser", t2 + 3, first=True))
    _, bf_attack, _ = read(eng2, "jerk")
    assert bf_attack > bf_answer + 0.3


def test_toxicity_is_flagged_whoever_it_targets():
    """User requirement: 'those guys never touch grass' is toxic even though it
    targets nobody in chat; bad faith (intent toward the room) stays lower."""
    eng = Engine(legacy=False)
    t = room(eng)
    rec = eng.process(msg("helper", "@asker7 That's super sweat mode. Those guys never touch grass.", t, first=True))
    assert rec["toxicity"]["level"] != "none" and rec["toxicity"]["target"] == "third party"
    rec2 = eng.process(msg("fan", "what a clean play, love this stream", t + 2))
    assert rec2["toxicity"]["level"] == "none"


def test_coded_vocabulary_marker_needs_breadth_and_skips_counter_speech():
    from chatwatch import vocab
    assert vocab.LEX.match("arent all women foids")[0]["id"] == "foid"
    assert vocab.LEX.match("what does foid mean") == []            # asking about it
    assert vocab.LEX.match("stop calling women foids") == []       # counter-speech
    assert vocab.LEX.match("we roasted him so hard") == []          # roast != roastie
    assert vocab.LEX.match("i should just rope")[0]["distress"]     # first person -> distress

    eng = Engine(legacy=False)
    t = room(eng)
    rec = eng.process(msg("vocabguy", "the blackpill is real, hypergamy explains it", t))
    assert rec["toxicity"]["level"] == "none"                       # jargon alone isn't toxic
    eng.process(msg("vocabguy", "smv is everything, awalt", t + 5))
    p = eng.profile("cozy", next(k[1] for k, u in eng.users.items() if u.login == "vocabguy"))
    assert p["vocabulary"] and p["vocabulary"][0]["family"] == "incel"

    rec2 = eng.process(msg("hater", "you're a foid lol", t + 9))
    assert rec2["toxicity"]["level"] in ("medium", "high") and "coded-vocab" in rec2["flags"]


def test_badfaith_alert_needs_a_toxic_message(monkeypatch):
    # no toxic message yet, no bad-faith alert (FINDINGS 2026-10-25); the old rule fired on the link
    import chatwatch.engine as E
    kinds = {}
    for flag in (False, True):
        monkeypatch.setattr(E, "ALERT_NEEDS_TOXIC", flag)
        eng = Engine()
        rec = eng.process(msg("newbie", "check out my channel twitch.tv/xx_free_followers", room(eng),
                              first=True, uid=NEW_ID))
        kinds[flag] = [a["kind"] for a in rec.get("alerts", [])]
    assert kinds == {False: ["badfaith"], True: []}


def test_watch_comes_from_the_llm_read_of_the_person():
    # FINDINGS 2026-10-26 "D7": with the local LLM reading people, toxic counts and alerts ask for
    # a read; "bad_faith" puts someone on watch. DB (2026-11-02): "mild" no longer does, with 2+ toxic
    # in 10 min or an alert alike; slurs stay 'act' regardless
    eng = Engine()
    eng.person_review = True
    t = room(eng)
    recs = [eng.process(msg("heckler", text, t + i * 20)) for i, text in
            enumerate(["you are trash", "worst streamer ever, you are trash", "you are trash lol"])]
    one = eng.process(msg("oneoff", "you are trash", t + 30))
    assert eng.wants_person_review(recs[0])
    ch, uid = recs[0]["channel"], recs[0]["user_id"]
    level = lambda: {d["display"]: d["level"] for d in eng.notices(ch, T0 + (t + 70) * 1000)[0]["items"]}
    assert "heckler" not in level()                       # 3 toxic in 10 min: not a card without the read
    eng.apply_person_review(ch, uid, {"label": "good_faith", "rationale": "banter"}, T0)
    assert "heckler" not in level()
    eng.apply_person_review(ch, uid, {"label": "mild", "rationale": "keeps jabbing"}, T0)
    assert "heckler" not in level()                       # DB: mild + the pattern makes no card
    eng.apply_person_review(ch, one["user_id"], {"label": "mild", "rationale": "one-off"}, T0)
    assert "oneoff" not in level()                        # mild, one toxic message: no card
    eng.apply_person_review(ch, uid, {"label": "bad_faith", "rationale": "keeps insulting the streamer"}, T0)
    assert level()["heckler"] == "watch"
    rec = eng.process(msg("slurguy", "you are a nigger", t + 80))
    assert eng.notices(ch, T0 + (t + 81) * 1000)[0]["items"][0]["display"] == "slurguy"   # act, no read needed
    assert eng.wants_person_review(rec)


def test_severe_without_a_slur_is_act_only_when_the_llm_read_is_at_least_mild():
    # FINDINGS 2026-10-27 "A4h" (bf_design/RULE_ACT_TIER.md): with people read by the local LLM, a
    # threat/sexual/harassment message keeps the chatter listed, but at 'act' only when the read is
    # mild or bad faith; a pending read keeps 'act', and slurs stay 'act' whatever the read
    eng = Engine()
    eng.person_review = True
    t = room(eng)
    rec = eng.process(msg("threatguy", "i will kill you", t))
    slur = eng.process(msg("slurguy", "you are a nigger", t + 1))
    ch = rec["channel"]
    level = lambda: {d["display"]: d["level"] for d in eng.notices(ch, T0 + (t + 5) * 1000)[0]["items"]}
    assert level() == {"threatguy": "act", "slurguy": "act"}          # reads pending
    for label, want in (("good_faith", "watch"), ("unclear", "watch"), ("mild", "act"), ("bad_faith", "act")):
        eng.apply_person_review(ch, rec["user_id"], {"label": label, "rationale": "x"}, T0)
        assert level()["threatguy"] == want, label
    eng.apply_person_review(ch, slur["user_id"], {"label": "good_faith", "rationale": "x"}, T0)
    assert level()["slurguy"] == "act"
    eng.person_review = False                                          # without reads: unchanged
    eng.apply_person_review(ch, rec["user_id"], {"label": "good_faith", "rationale": "x"}, T0)
    assert level()["threatguy"] == "act"


def test_wished_or_threatened_sexual_violence_is_severe_but_discussion_is_not():
    # FINDINGS 2026-10-27, data/study/cue_eval/RULE_CUEFIX.md: the sexual_threat cue
    from chatwatch.engine import _tox_class
    eng = Engine()
    t = room(eng)
    for i, text in enumerate(["i hope you get raped", "you deserve to be raped", "go get raped", "im gonna rape you"]):
        tox = eng.process(msg(f"bad{i}", text, t + i))["toxicity"]
        assert tox["level"] == "high" and _tox_class(tox) == "severe", text
    for i, text in enumerate(["he doesnt deserve to be raped", "we got raped by the boss", "rape culture is real",
                              "she was raped and nobody believed her"]):
        tox = eng.process(msg(f"ok{i}", text, t + 10 + i))["toxicity"]
        assert _tox_class(tox) is None, text


def test_automod_on_a_schizo_emote_is_not_counted_but_other_emote_spans_are():
    # FINDINGS 2026-10-27, RULE_CUEFIX.md: AutoMod tags the SCHIZO reaction emote as aggression/identity;
    # judges called 19 of 19 such flags not toxic. Insulting emote names ("hesRetarded") still count.
    from chatwatch import signals
    ev = {"text": "no no no SCHIZO", "automod_spans": [(9, 14, {"A": 6, "I": 6})]}
    assert signals.automod_levels(ev, frozenset({"SCHIZO"})) == {}
    assert signals.automod_levels(ev, frozenset()) == {"A": 6, "I": 6}          # typed as a word: counts
    ev2 = {"text": "lol hesRetarded", "automod_spans": [(4, 14, {"I": 6})]}
    assert signals.automod_levels(ev2, frozenset({"hesRetarded"})) == {"I": 6}


# ------------------------------------------------------------ review phase 2 (tests_docs.md 2c #1-#3)
def _panel(eng, at, **kw):
    """{display: level} of the live heads-up panel, and of the earlier list, at time `at` (seconds after T0)."""
    live, earlier = eng.notices("cozy", T0 + at * 1000, **kw)
    return {d["display"]: d["level"] for d in live["items"]}, {d["display"]: d["level"] for d in earlier}


def test_an_llm_reread_keeps_a_keyword_slur_s1():
    """S1 (docs/review/SUMMARY.md B; was the G1 pin of today's behaviour). The LLM's toxicity re-read
    replaces the keyword layer's read for display, but cannot clear a keyword slur: when the layer's
    `sure` list has "hate", the message keeps the class hate on the panel and counts in toxic_n at
    least at the layer's level."""
    from chatwatch.engine import _tox_class

    def run():
        eng = Engine()
        eng.person_review = True
        t = room(eng)
        rec = eng.process(msg("slurfriend", "thats my nigga fr", t))
        ch, uid, mid = rec["channel"], rec["user_id"], rec["id"]
        u = eng.users[(ch, uid)]
        assert rec["toxicity"]["level"] == "high" and "hate" in rec["toxicity"]["sure"]
        assert _panel(eng, t + 5)[0] == {"slurfriend": "act"} and u.toxic_n == 1
        upd = eng.apply_llm_toxicity(ch, uid, mid, {"toxicity": "none", "target": "", "kind": ""})
        assert upd["toxicity"]["source"] == "llm" and upd["toxicity"]["layer"]["level"] == "high"
        assert upd["toxicity"]["level"] == "none" and upd["toxicity"]["kinds"] == []   # the LLM's own read, for display
        return eng, u, ch, uid, mid, t

    eng, u, ch, uid, mid, t = run()
    assert _tox_class(u.msgs[-1]["toxicity"]) == "hate"
    assert _panel(eng, t + 5)[0] == {"slurfriend": "act"} and u.toxic_n == 1   # S1: the slur stays, counted once
    eng.apply_person_review(ch, uid, {"label": "good_faith", "rationale": "friendly"}, T0)
    assert _panel(eng, t + 5)[0] == {"slurfriend": "act"}                       # keyword slurs: act whatever the read
    for lvl, kind in (("high", "hate_speech"), ("high", "insult"), ("medium", "insult"), ("low", "joke"), ("none", "")):
        upd = eng.apply_llm_toxicity(ch, uid, mid, {"toxicity": lvl, "target": "chatter", "kind": kind})
        assert upd["toxicity"]["layer"]["sure"] == ["hate"] and upd["toxicity"]["level"] == lvl   # layer kept, not nested
        assert _tox_class(u.msgs[-1]["toxicity"]) == "hate" and u.toxic_n == 1, (lvl, kind)
    assert eng.apply_llm_toxicity(ch, uid, mid, {"toxicity": "extreme"}) is None   # an unknown level is ignored
    assert eng.apply_llm_toxicity(ch, uid, "no-such-id", {"toxicity": "none"}) is None
    # a message with no keyword slur is still the LLM's to clear
    rec = eng.process(msg("jabber", "you are trash", t + 10))
    v = eng.users[(ch, rec["user_id"])]
    assert v.toxic_n == 1
    eng.apply_llm_toxicity(ch, rec["user_id"], rec["id"], {"toxicity": "none", "target": "", "kind": ""})
    assert v.toxic_n == 0 and _tox_class(v.msgs[-1]["toxicity"]) is None


def test_llm_only_hate_is_likely_only_with_a_person_read_of_at_least_mild_s11():
    """S11 (docs/review/SUMMARY.md C). Hate that comes only from the LLM's toxicity re-read (no keyword or
    coded-vocabulary slur in the layer) is treated like severe content under A4h: a person read below
    mild (good faith, unclear) lists the chatter at 'watch'; with no read, or a pending one, they keep
    'act' (RULE_PHASE3). Keyword slurs stay 'act' whatever the read. Examples: "this level is aids" (LLM high, slur) and
    "Gamer of the Year, GOY" (hate_code medium, read good faith)."""
    from chatwatch.engine import _tox_class

    def setup():
        eng = Engine()
        eng.person_review = True
        t = room(eng)
        a = eng.process(msg("zoner", "this level is aids", t))
        g = eng.process(msg("goyer", "Gamer of the Year, GOY", t + 1))
        s = eng.process(msg("slurguy", "you are a nigger", t + 2))
        assert a["toxicity"]["sure"] == [] and g["toxicity"]["sure"] == [] and s["toxicity"]["sure"] == ["hate"]
        eng.apply_llm_toxicity("cozy", a["user_id"], a["id"], {"toxicity": "high", "target": "group", "kind": "slur"})
        eng.apply_llm_toxicity("cozy", g["user_id"], g["id"], {"toxicity": "medium", "target": "group", "kind": "hate_code"})
        assert _tox_class(eng.users[("cozy", a["user_id"])].msgs[-1]["toxicity"]) == "hate"
        assert _tox_class(eng.users[("cozy", g["user_id"])].msgs[-1]["toxicity"]) == "hate"
        return eng, a, g, s, t

    eng, a, g, s, t = setup()
    # RULE_PHASE3 S11: a chatter with no read, or a pending one, keeps 'act' (as A4h's code does for severe)
    assert _panel(eng, t + 5)[0] == {"zoner": "act", "goyer": "act", "slurguy": "act"}       # no read at all
    eng.mark_review_pending("cozy", a["user_id"], T0 + (t + 3) * 1000)
    assert _panel(eng, t + 5)[0]["zoner"] == "act"                                          # a pending read keeps act
    assert _panel(eng, t + 3 + 11 * 60)[0]["zoner"] == "act"                                # a lost read is no read
    for label, want in (("good_faith", "watch"), ("unclear", "watch"), ("mild", "act"), ("bad_faith", "act")):
        eng.apply_person_review("cozy", a["user_id"], {"label": label, "rationale": "x"}, T0)
        eng.apply_person_review("cozy", g["user_id"], {"label": label, "rationale": "x"}, T0)
        eng.apply_person_review("cozy", s["user_id"], {"label": label, "rationale": "x"}, T0)
        lv = _panel(eng, t + 5)[0]
        assert (lv["zoner"], lv["goyer"], lv["slurguy"]) == (want, want, "act"), label
    why = {d["display"]: d["why"] for d in eng.notices("cozy", T0 + (t + 5) * 1000)[0]["items"]}
    assert why["zoner"][0] == "slur or dehumanising language"
    # severe content without a slur: A4h unchanged
    r = eng.process(msg("threatguy", "i will kill you", t + 6))
    assert _panel(eng, t + 7)[0]["threatguy"] == "act"                                      # no read: A4h keeps act
    eng.apply_person_review("cozy", r["user_id"], {"label": "good_faith", "rationale": "x"}, T0)
    assert _panel(eng, t + 7)[0]["threatguy"] == "watch"
    eng.person_review = False                                                               # without person reads: unchanged
    assert _panel(eng, t + 7)[0] == {"zoner": "act", "goyer": "act", "slurguy": "act", "threatguy": "act"}


def test_an_llm_only_hate_reread_queues_a_person_read_s11(monkeypatch):
    """S11 (RULE_PHASE3). A re-read (the tox batcher's msg_update) that gives a chatter LLM-only hate
    queues a person read, as publish() does for a toxic message: at most REVIEW_MAX reads, none while one
    is pending; the lossless queue is never full, so a long queue still takes it. Not for a keyword slur (its 'act' needs no
    read), not for other re-reads, and not without person reads."""
    import asyncio
    from chatwatch import llm
    from chatwatch.engine import REVIEW_MAX
    from chatwatch.serve import Hub

    def run(text, says, reads=0, review=True, full=False):
        monkeypatch.setattr(llm, "rate_toxicity_window", lambda room, lines, emotes=frozenset(): [says])
        eng = Engine()
        eng.person_review = review
        t = room(eng)
        hub = Hub(eng)
        hub.log_sessions = False
        hub.review_q = ReadQueue()
        if full:                                  # a long queue (the old maxsize was 200)
            for i in range(500):
                hub.review_q.put_nowait((0, i, "other", {}))
        rec = eng.process(msg("quiet", text, t))
        eng.clock = rec["ts"]
        u = eng.users[("cozy", rec["user_id"])]
        if reads:
            u.review = {"label": "good_faith", "reads": reads, "pending": False}
        item = {"channel": "cozy", "user_id": rec["user_id"], "id": rec["id"], "line": f"quiet: {text}", "room": []}
        asyncio.run(hub._review_one("toxwin", {"channel": "cozy", "items": [item], "user_id": rec["user_id"]}))
        q = []
        while not hub.review_q.empty():
            q.append(hub.review_q.get_nowait()[2:])
        return eng, u, q

    hate = {"toxicity": "high", "target": "group", "kind": "slur"}
    eng, u, q = run("this level is aids", hate)
    assert q == [("person", {"channel": "cozy", "user_id": u.uid})] and u.review["pending"]
    assert u.review["queued"] == eng.clock
    assert run("this level is aids", hate, review=False)[2] == []                           # no person reads
    assert run("this level is aids", {"toxicity": "high", "target": "chatter", "kind": "insult"})[2] == []   # not hate
    assert run("this level is aids", hate, reads=REVIEW_MAX)[2] == []                       # within REVIEW_MAX
    assert len(run("this level is aids", hate, reads=REVIEW_MAX - 1)[2]) == 1
    assert run("you are a nigger", hate)[2] == []                                          # a keyword slur
    eng, u, q = run("this level is aids", hate, full=True)                                  # lossless: still queued
    assert ("person", {"channel": "cozy", "user_id": u.uid}) in q and len(q) == 501 and u.review["pending"]
    assert eng.stats["llm_person_errors"] == 0 and eng.stats["llm_person_dropped"] == 0
    e = Engine()
    e.person_review = True
    t = room(e)
    r = e.process(msg("pend", "this level is aids", t))
    e.mark_review_pending("cozy", r["user_id"], r["ts"])
    assert not e.wants_person_review_llm("cozy", r["user_id"], {**hate, "level": "high", "kinds": ["slur"],
                                                               "source": "llm", "layer": r["toxicity"]}, r["ts"])
    assert e.wants_person_review_llm("cozy", r["user_id"], {"level": "high", "kinds": ["slur"], "source": "llm",
                                                           "layer": r["toxicity"]}, r["ts"] + 11 * 60 * 1000)   # lost


def test_resub_milestone_and_announcement_text_is_scored_s5():
    """S5 (docs/review/SUMMARY.md B; live review P2). The text a chatter types into a resub, a viewer
    milestone or an announcement (a USERNOTICE) is scored as a chat message; Twitch's own wording
    (system-msg) is not part of it. Notices without text (subgift, raid) and bots' are not scored.
    The keyword layer (KeywordLayer) stays in step."""
    from chatwatch import irc
    from chatwatch.engine import KeywordLayer

    def notice(msg_id, login, text, t, uid="777"):
        tags = (f"@badge-info=subscriber/12;badges=subscriber/12;color=#FF0000;display-name={login.title()};emotes=;"
                f"id=n-{msg_id}-{t};login={login};mod=0;msg-id={msg_id};room-id=1;subscriber=1;"
                rf"system-msg={login}\ssubscribed\sat\sTier\s1.\sThey've\ssubscribed\sfor\s12\smonths!;"
                f"tmi-sent-ts={T0 + t * 1000};user-id={uid}")
        return irc.to_event(f"{tags} :tmi.twitch.tv USERNOTICE #cozy" + (f" :{text}" if text else ""))

    eng, kl = Engine(legacy=False), KeywordLayer()
    t = room(eng)
    evs = [notice("resub", "resubber", "you are a stupid idiot streamer", t),
           notice("viewermilestone", "milestoner", "gg", t + 1, uid="778"),
           notice("subgift", "gifter", "", t + 2, uid="779"),
           notice("announcement", "nightbot", "you are a stupid idiot streamer", t + 3, uid="780")]
    recs = [eng.process(ev) for ev in evs]
    assert [kl.process(ev) for ev in evs] == [r and r["toxicity"]["score"] for r in recs]
    r = recs[0]
    assert r["type"] == "msg" and r["notice"] == "resub" and r["text"] == "you are a stupid idiot streamer"
    assert r["toxicity"]["level"] != "none" and r["display"] == "Resubber" and r["id"] == f"n-resub-{t}"
    assert "subscribed" not in r["text"]
    assert recs[1]["notice"] == "viewermilestone" and recs[1]["toxicity"]["level"] == "none"
    assert recs[2] is None and recs[3] is None                     # no text; a bot
    u = eng.users[("cozy", "777")]
    assert u.toxic_n == 1 and u.msgs[-1]["text"] == r["text"]
    assert irc.notice_as_msg(notice("resub", "x", "  ", 0)) is None


def test_chatters_idle_over_30_days_are_pruned_at_load_and_autosave_s7(monkeypatch, tmp_path):
    """S7 (live review H1/P4). Engine.prune drops chatters idle for more than 30 days: profile, name
    index, the channel's sender names, legacy profile, room-vocabulary membership; the account-age index
    is rebuilt from the chatters kept. serve.load_engine prunes a restored state."""
    from chatwatch import serve
    day = 86_400
    eng = Engine()
    t = room(eng)
    for i in range(30):
        eng.process(msg(f"old{i}", "this boss fight is so tense", t + i, uid=str(10_000_000 + i)))
    eng.process(msg("oldfoid", "foids are all the same", t + 40, uid="10000099"))
    later = t + 40 * day
    eng.process(msg("fresh", "hello chat", later, uid="20000000"))
    eng.process(msg("regular1", "back again", later + 1, sub=6))
    n0, ids0 = len(eng.users), len(eng.ids.ids)
    assert ("cozy", "oldfoid") in eng.by_login and "oldfoid" in eng.rooms["cozy"].vocab.chatters
    assert eng.prune(T0 + (t + 41 + 29 * day) * 1000) == 0             # 29 days idle: kept
    gone = eng.prune(T0 + (later + 2) * 1000)
    kept = {u.login for u in eng.users.values()}
    assert gone == n0 - 2 and kept == {"fresh", "regular1"}
    assert set(eng.names["cozy"]) == kept and set(l for (_, l) in eng.by_login) == kept
    assert eng.rooms["cozy"].vocab.chatters <= kept and not any(eng.rooms["cozy"].vocab.users.values())
    assert set(eng.legacy["cozy"].profiles) == {u.uid for u in eng.users.values()}
    assert eng.ids.ids == sorted(int(u.uid) for u in eng.users.values() if u.uid.isdigit()) and len(eng.ids.ids) < ids0
    assert len(eng.channel_users("cozy")) == 2 and eng.stats["pruned_chatters"] == gone
    r = eng.process(msg("old3", "hi again", later + 5, uid="10000003"))          # a returning chatter starts over
    assert r["n"] == 1
    # serve.load_engine prunes the restored state
    st = tmp_path / "engine.pkl"
    old = Engine()
    tt = room(old)
    for i in range(3):
        old.process(msg(f"gone{i}", "gg", tt + i))
    monkeypatch.setattr(serve, "STATE", str(st))
    serve.save_engine(old)
    monkeypatch.setattr(serve.time, "time", lambda: (T0 + (tt + 31 * day) * 1000) / 1000)
    assert len(serve.load_engine(None, None).users) == 0


def test_the_panel_without_llm_person_reads(monkeypatch):
    """PINS TODAY'S BEHAVIOUR (G2): the panel rule with person reads off (--no-llm, or the LLM down).
    The identity check never calls notices(), so only this test guards it."""
    import chatwatch.engine as E
    monkeypatch.setattr(E, "ALERT_BADFAITH", 2.0)          # alerts off: only the toxic-count rule
    eng = Engine()
    eng.person_review = False
    t = room(eng)
    for i in range(2):
        eng.process(msg("twice", "you are trash", t + i * 60))
    for i in range(3):
        eng.process(msg("thrice", "ur bad", t + 500 + i * 60))
    for i in range(3):
        eng.process(msg("spread", "chat is so cringe", t + i * 330))      # 0, 5.5 and 11 minutes
    now = t + 660
    assert _panel(eng, now) == ({"thrice": "watch"}, {})                   # 3 toxic in 10 min; 2 is not enough
    assert _panel(eng, t + 620 + 601) == ({}, {})                          # 10 min later the pattern has gone,
    #                                                                        and the card with it (not "earlier")
    monkeypatch.undo()
    eng = Engine()
    eng.person_review = False
    t = room(eng)
    eng.process(msg("slurguy", "you are a nigger", t))
    a = eng.process(msg("alerted", "worst streamer ever, you are trash", t))
    assert [x["kind"] for x in a["alerts"]] == ["badfaith"]
    assert _panel(eng, t + 1) == ({"slurguy": "act", "alerted": "watch"}, {})   # one toxic message plus an alert
    why = {d["display"]: d["why"] for d in eng.notices("cozy", T0 + (t + 1) * 1000)[0]["items"]}
    assert why["alerted"] == ["bad-faith pattern (provocateur)"]
    # watch cards show for 15 minutes after the last message, act cards for 30; then "earlier"
    assert _panel(eng, t + 15 * 60 + 1) == ({"slurguy": "act"}, {"alerted": "watch"})
    assert _panel(eng, t + 30 * 60 + 1) == ({}, {"slurguy": "act", "alerted": "watch"})
    assert _panel(eng, t + 30 * 60 + 1, since=T0 + (t + 1) * 1000) == ({}, {})   # "earlier": this run only

    # an alert with no toxic message: 'watch' without person reads, nothing with them
    monkeypatch.setattr(E, "ALERT_NEEDS_TOXIC", False)
    eng = Engine()
    t = room(eng)
    r = eng.process(msg("spammer", "check out my channel twitch.tv/xx_free_followers", t, first=True, uid=NEW_ID))
    assert [x["kind"] for x in r["alerts"]] == ["badfaith"] and r["toxicity"]["level"] == "none"
    eng.person_review = False
    assert _panel(eng, t + 1)[0] == {"spammer": "watch"}
    eng.person_review = True
    assert _panel(eng, t + 1)[0] == {}


def test_same_text_is_one_card_and_the_panel_shows_forty():
    """PINS TODAY'S BEHAVIOUR (G2): identical lines (a copypasta with a slur) are one card that names
    everyone; each person still counts. At most 40 cards; the rest are counted in "more"."""
    eng = Engine()
    t = room(eng)
    eng.process(msg("paster1", "you are a nigger", t))
    eng.process(msg("paster2", "You are a nigger!", t + 1))
    live, _ = eng.notices("cozy", T0 + (t + 2) * 1000)
    assert live["counts"] == {"act": 2, "watch": 0} and len(live["items"]) == 1
    assert sorted(g["display"] for g in live["items"][0]["group"]) == ["paster1", "paster2"]
    for i in range(45):
        eng.process(msg(f"slur{i}", f"@someone{i} you are a nigger", t + 10 + i))
    live, _ = eng.notices("cozy", T0 + (t + 60) * 1000)
    assert len(live["items"]) == 40 and live["more"] == 6 and live["counts"]["act"] == 47


def test_the_alert_budget_holds_normal_alerts_and_lets_urgent_ones_through():
    """PINS TODAY'S BEHAVIOUR (G3): engine.py's per-channel hourly alert budget. Normal-priority alerts
    over the budget are held (counted once per chatter); a high-priority chatter at bad faith >= 0.85
    gets through marked over_budget; after an hour the budget is free again; a chatter whose bad-faith
    score falls below 0.25 can alert again."""
    eng = Engine(alert_budget=2)
    t = room(eng)
    hostile = "worst streamer ever, you are trash"         # bad faith 0.92 on the first line; normal priority
    alerts = lambda r: [(a["kind"], a["priority"], a["over_budget"]) for a in r.get("alerts", ())]
    assert alerts(eng.process(msg("a", hostile, t))) == [("badfaith", "normal", False)]
    assert alerts(eng.process(msg("b", hostile, t + 10))) == [("badfaith", "normal", False)]
    c = eng.process(msg("c", hostile, t + 20))
    assert alerts(c) == [] and eng.stats["alerts_held_by_budget"] == 1
    assert eng.users[("cozy", c["user_id"])].held
    assert alerts(eng.process(msg("c", hostile, t + 30))) == [] and eng.stats["alerts_held_by_budget"] == 1
    eng.process(msg("d", "you are a nigger", t + 40))                       # a slur cue: high priority
    d = eng.process(msg("d", "you are a nigger", t + 50))
    assert d["badfaith"] >= 0.85 and alerts(d) == [("badfaith", "high", True)]
    # each alert counts for one hour: at t+3605 a has gone but b and d still fill the budget; at t+3615 b has gone
    assert alerts(eng.process(msg("c", hostile, t + 3605))) == []
    assert alerts(eng.process(msg("c", hostile, t + 3615))) == [("badfaith", "normal", False)]
    # re-arm: four weeks later one ordinary line takes "a" below 0.25, and a new pattern alerts again
    r = eng.process(msg("a", "gg", t + 28 * 86400))
    assert r["badfaith"] < 0.25 and not eng.users[("cozy", r["user_id"])].alerted
    again = [alerts(eng.process(msg("a", hostile, t + 28 * 86400 + k))) for k in (10, 20)]
    assert again == [[("badfaith", "normal", False)], []]


def test_a_ban_on_one_name_is_no_evidence_about_a_lookalike():
    # audit LB-7 (BASELINE 8.1): the lookalike is shown (flag, room warning) but a mod action on account A
    # adds nothing to the read of account B; live_backend test gap 13 (the evidence's size: zero)
    reads = {}
    for banned in (False, True):
        eng = Engine(legacy=False)
        t = room(eng)
        eng.process(msg("trollking", "you are an idiot", t))
        if banned:
            eng.process({"kind": "clearchat", "ts": T0 + (t + 2) * 1000, "channel": "cozy", "room_id": "",
                         "target_login": "trollking", "target_user_id": "", "duration": -1})
        rec = eng.process(msg("trollking2", "you are trash lol", t + 30, first=True, uid=NEW_ID))
        reads[banned] = rec
    assert reads[True]["badfaith"] == reads[False]["badfaith"]
    assert "lookalike" in reads[True]["flags"] and "lookalike" not in reads[False]["flags"]
    assert any(a["kind"] == "lookalike" for a in reads[True].get("alerts", []))
    assert [a["kind"] for a in reads[True].get("alerts", []) if a["kind"] == "badfaith"] == \
           [a["kind"] for a in reads[False].get("alerts", []) if a["kind"] == "badfaith"]


def test_s12_and_s11_together_on_llm_only_hate_from_an_earlier_session():
    # live_backend test gap 13: one chatter whose only 'likely' content is LLM-only hate from before this run.
    # S12 makes it at most 'possible' (earlier session) whatever the read; without a run start, S11 decides.
    day = 86_400_000
    for label, now_level, plain_level in (("mild", "watch", "act"), ("good_faith", "watch", "watch")):
        eng = Engine()
        eng.person_review = True
        t = room(eng)
        rec = eng.process(msg("subtle", "those people are all the same, you know what they are", t))
        upd = eng.apply_llm_toxicity("cozy", rec["user_id"], rec["id"], {"toxicity": "high", "kind": "hate", "target": "group"})
        assert upd and "hate" in upd["toxicity"]["kinds"]
        eng.apply_person_review("cozy", rec["user_id"], {"label": label, "rationale": "x"}, rec["ts"])
        start = rec["ts"] + day
        eng.process({**msg("subtle", "hello again", 0), "ts": start + 60_000, "user_id": rec["user_id"]})
        later = start + 2 * 60_000
        items = {d["display"]: d for d in eng.notices("cozy", later, since=start)[0]["items"]}
        assert items["subtle"]["level"] == now_level and "(earlier session)" in " ".join(items["subtle"]["why"]), label
        plain = {d["display"]: d for d in eng.notices("cozy", later)[0]["items"]}
        assert plain["subtle"]["level"] == plain_level, label


def test_a_reread_slur_still_counts_as_severe_for_the_alert_shape_s1():
    # audit LB-12, live_backend test gap 9: after an LLM "none" re-read of a keyword slur (S1: the slur stands),
    # the alert rule must not treat the chatter as a one-off insult (_mild_only)
    eng = Engine(legacy=False)
    t = room(eng)
    eng.process(msg("mixed", "what level are you", t))
    eng.process(msg("mixed", "gg nice dodge", t + 2))
    rec = eng.process(msg("mixed", "shut up you stupid nigger", t + 4))
    u = eng.users[("cozy", rec["user_id"])]
    top = sorted(u.post().items(), key=lambda kv: -kv[1])[:3]
    assert not eng._mild_only(u, top)
    eng.apply_llm_toxicity("cozy", rec["user_id"], rec["id"], {"toxicity": "none", "target": "", "kind": ""})
    assert not eng._mild_only(u, top)


def test_a_pile_on_screen_raises_an_alert_only_with_a_toxic_message():
    # audit LB-13, live_backend test gap 9: the LLM screen's verdict is evidence, but the alert it may raise
    # follows the adopted rule "an alert needs a toxic message" like every other bad-faith alert
    eng = Engine(legacy=False)
    t = room(eng)
    bad = {"label": "bad_faith", "rules": ["R1", "R4"], "confidence": "high", "rationale": "x"}
    calm = eng.process(msg("quietnew", "hello what game is this", t, first=True, uid=NEW_ID))
    assert eng.apply_review("cozy", calm["user_id"], bad, calm["ts"]) is None          # no toxic message: no alert
    assert eng.users[("cozy", calm["user_id"])].badfaith() >= 0.35                      # the evidence still counts
    rude = eng.process(msg("rudenew", "you are trash streamer", t + 1, first=True, uid=str(int(NEW_ID) + 1)))
    eng.users[("cozy", rude["user_id"])].alerted = False
    a = eng.apply_review("cozy", rude["user_id"], bad, rude["ts"])
    assert a and a["via"] == "llm-screen"


def test_the_panel_when_every_llm_person_read_fails():
    # audit LB-3 (DASHBOARD.md "without LLM reads (... or the LLM down) the older rule applies"), live_backend
    # test gap 3: person reads on, the LLM down. A chatter whose read failed, or has been pending past 10 min,
    # is judged by the older rule; a read still within its 10 minutes waits; a read they already have counts.
    eng = Engine()
    eng.person_review = True
    t = room(eng)
    recs = [eng.process(msg("thrice", "ur bad", t + i * 60)) for i in range(3)]
    ch, uid = recs[0]["channel"], recs[0]["user_id"]
    eng.mark_review_pending(ch, uid, recs[0]["ts"])
    assert _panel(eng, t + 130)[0] == {}                                   # pending: D7 waits for the read
    eng.apply_person_review(ch, uid, {"error": "connection refused"}, recs[-1]["ts"])
    assert _panel(eng, t + 130)[0] == {"thrice": "watch"}                  # failed: the older rule (3 in 10 min)
    eng.mark_review_pending(ch, uid, recs[-1]["ts"])                       # a retry is queued: still the older rule
    assert _panel(eng, t + 130)[0] == {"thrice": "watch"}
    eng.apply_person_review(ch, uid, {"label": "good_faith", "rationale": "banter"}, recs[-1]["ts"])
    assert _panel(eng, t + 130)[0] == {}                                   # a read came back: D7 again
    # a read lost in the queue (pending for more than 10 minutes) also falls back
    eng2 = Engine()
    eng2.person_review = True
    t2 = room(eng2)
    r2 = [eng2.process(msg("lost", "ur bad", t2 + i * 60)) for i in range(3)]
    eng2.mark_review_pending("cozy", r2[0]["user_id"], r2[0]["ts"] - 11 * 60_000)
    assert _panel(eng2, t2 + 130)[0] == {"lost": "watch"}
    # slurs are 'likely' either way
    eng.process(msg("slurguy", "you are a nigger", t + 140))
    eng.apply_person_review(ch, eng.users and next(k[1] for k, u in eng.users.items() if u.login == "slurguy"),
                            {"error": "timeout"}, T0)
    assert _panel(eng, t + 141)[0]["slurguy"] == "act"
