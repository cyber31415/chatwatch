"""The two owner features of 2026-11-01 (docs/AGENDA.md B.5, docs/DASHBOARD.md "Why someone is listed" and "Roles"):
the messages behind a watchlist listing, kept past the 40 recent messages so they can still be corrected, and the
owner's hand-set "bot" role. No network, no LLM; the roles file and the register are in tmp_path."""
import asyncio
import json
import os
import pickle
import re
from pathlib import Path

import pytest

from chatwatch import engine as E
from chatwatch import roles as R
from chatwatch import serve
from chatwatch.engine import Engine

from test_dashboard import _http, _restart
from test_scenarios import CALM_ROOM, T0, msg, room

PAGE = {"X-Chatwatch": "1"}
INSULT = "you are a stupid idiot streamer"


def _user(eng, login):
    return next(u for u in eng.users.values() if u.login == login)


def _aged_out(eng, t, login="heckler", calm=45):
    """A chatter with one insult, followed by `calm` friendly lines (more than the 40 recent messages kept)."""
    rec = eng.process(msg(login, INSULT, t))
    for i in range(calm):
        eng.process(msg(login, CALM_ROOM[i % len(CALM_ROOM)] + f" {i}", t + 3 + i))
    return rec, _user(eng, login)


# ------------------------------------------------------------------ the evidence store
def test_a_toxic_message_is_kept_past_the_recent_messages_and_the_rule_still_reads_only_them():
    eng = Engine()
    t = room(eng)
    rec = eng.process(msg("heckler", INSULT, t))
    u = _user(eng, "heckler")
    assert u.evidence[0] is u.msgs[-1]                        # one dict: a re-read or a correction updates both
    for i in range(45):
        eng.process(msg("heckler", CALM_ROOM[i % len(CALM_ROOM)] + f" {i}", t + 3 + i))
    assert all(m["id"] != rec["id"] for m in u.msgs) and u.evidence[0]["id"] == rec["id"]
    items = eng.evidence("cozy", u.uid)
    # FLAGS_PERSIST (owner, 2026-11-07): the rule reads the kept line too ("counted"); a plain insult still makes no
    # listing by itself (only slur or severe content, a bad-faith read, an alert or a toxic run do)
    assert [(i["id"], i["counted"], i["older"]) for i in items] == [(rec["id"], True, True)]
    assert E._notice_reasons(u, None, u.last_seen, False) is None


def test_most_chatters_get_no_evidence_container():
    eng = Engine()
    room(eng)
    assert all(u.evidence is None for u in eng.users.values())


def test_the_evidence_is_capped_oldest_first():
    eng = Engine()
    t = room(eng)
    ids = [eng.process(msg("heckler", f"{INSULT} {i}", t + i))["id"] for i in range(E.EVIDENCE_KEPT + 5)]
    u = _user(eng, "heckler")
    assert len(u.evidence) == E.EVIDENCE_KEPT and [m["id"] for m in u.evidence] == ids[5:]


def test_an_llm_re_read_that_finds_toxicity_adds_the_message_and_one_that_clears_it_keeps_it():
    eng = Engine()
    t = room(eng)
    plain = eng.process(msg("quiet", "that boss is brutal", t))
    u = _user(eng, "quiet")
    assert u.evidence is None
    eng.apply_llm_toxicity("cozy", u.uid, plain["id"], {"toxicity": "medium", "kind": "insult", "target": "streamer"})
    assert u.evidence[0] is u.msgs[-1]
    eng.apply_llm_toxicity("cozy", u.uid, plain["id"], {"toxicity": "none"})
    item, = eng.evidence("cozy", u.uid)
    assert item["toxicity"]["level"] == "none" and not item["counted"] and not item["older"]   # "not toxic now"


def test_an_older_message_is_corrected_with_todays_effects_and_listed_to_undo():
    """apply_owner_override finds a message in the evidence too: toxic_n, toxic_kinds and the ALERT_NEEDS_TOXIC take-back
    work as for a recent one; "Not toxic" stays listed so it can be undone."""
    eng = Engine()
    t = room(eng)
    rec, u = _aged_out(eng, t)
    assert u.toxic_n == 1
    u.alerted = True                                          # an alert this insult allowed (ALERT_NEEDS_TOXIC)
    upd = eng.apply_owner_override("cozy", u.uid, rec["id"], "none")
    assert upd and upd["toxicity"]["source"] == "owner"
    assert u.toxic_n == 0 and not u.alerted and sum(u.toxic_kinds.values()) == 0
    item, = eng.evidence("cozy", u.uid)
    assert item["toxicity"]["source"] == "owner" and item["toxicity"]["level"] == "none"
    eng.apply_owner_override("cozy", u.uid, rec["id"], None)  # back to automatic: the alert comes back with it
    assert u.toxic_n == 1 and u.alerted and eng.evidence("cozy", u.uid)[0]["toxicity"].get("source") != "owner"


def test_a_message_corrected_up_from_not_toxic_is_listed():
    eng = Engine()
    t = room(eng)
    plain = eng.process(msg("quiet", "go back to where you came from", t))
    u = _user(eng, "quiet")
    eng.apply_owner_override("cozy", u.uid, plain["id"], "high", "hate")
    item, = eng.evidence("cozy", u.uid)
    assert item["counted"] and E._notice_reasons(u, None, u.last_seen, False)[0] == "act"


def test_the_evidence_survives_a_save_and_load_still_shared_with_the_recent_messages():
    eng = Engine()
    t = room(eng)
    rec = eng.process(msg("heckler", INSULT, t))
    _, data = serve.dump_engine(eng)
    eng2 = pickle.loads(data).upgrade()
    u = _user(eng2, "heckler")
    assert u.evidence[0] is u.msgs[-1]
    eng2.apply_owner_override("cozy", u.uid, rec["id"], "low")
    assert u.msgs[-1]["toxicity"]["source"] == "owner"


def test_a_state_saved_before_the_evidence_existed_still_loads_and_lists_its_recent_toxic_messages():
    eng = Engine()
    t = room(eng)
    rec = eng.process(msg("heckler", INSULT, t))
    for u in eng.users.values():
        del u.evidence                                        # as pickled by the code before this feature
    eng = _restart(eng)
    u = _user(eng, "heckler")
    assert u.evidence is None
    assert [i["id"] for i in eng.evidence("cozy", u.uid)] == [rec["id"]]
    assert eng.apply_owner_override("cozy", u.uid, rec["id"], "none") and u.evidence[0] is u.msgs[-1]


def test_forget_drops_the_evidence_with_the_chatter():
    eng = Engine()
    t = room(eng)
    _, u = _aged_out(eng, t)
    eng.forget(u.uid)
    assert eng.evidence("cozy", u.uid) is None and eng.held_message("cozy", u.uid, u.evidence[0]["id"]) is None


def test_the_endpoints_list_and_correct_an_older_message(monkeypatch, tmp_path):
    from chatwatch import session as S
    eng = Engine()
    t = room(eng)
    rec, u = _aged_out(eng, t)
    eng.clock = u.last_seen
    hub = serve.Hub(eng)
    hub.status.update(mode="live", running=True)
    hub.persist = True
    monkeypatch.setattr(S, "OVERRIDES", str(tmp_path / "owner_overrides.jsonl"))
    monkeypatch.setattr(hub, "_log", lambda rec: None)

    async def calls(c):
        r1 = await c.get(f"/api/evidence?channel=cozy&uid={u.uid}")
        ev = await r1.json()
        r2 = await c.post("/api/override", json={"channel": "cozy", "uid": u.uid, "id": rec["id"], "level": "none"}, headers=PAGE)
        r3 = await c.get(f"/api/profile?channel=cozy&uid={u.uid}")
        r4 = await c.get("/api/evidence?channel=cozy&uid=nobody")
        return ev, r2.status, await r2.json(), await r3.json(), r4.status
    ev, st, j, prof, missing = _http(hub, calls)
    assert ev["items"][0]["id"] == rec["id"] and ev["items"][0]["older"] and ev["items"][0]["counted"]   # FLAGS_PERSIST
    assert st == 200 and j["toxicity"]["source"] == "owner" and j["recorded"] and u.toxic_n == 0
    assert prof["evidence"][0]["toxicity"]["source"] == "owner" and prof["owner_role"] is None
    assert missing == 404


# ------------------------------------------------------------------ the owner's "bot" role
def _role_hub(monkeypatch, tmp_path, mode="live", persist=True):
    """A chatter 'gamebot' on the watchlist ("likely", from an owner correction), a Hub, the roles file in tmp_path."""
    from chatwatch import session as S
    eng = Engine()
    t = room(eng)
    rec = eng.process(msg("gamebot", "go back to where you came from", t))
    u = _user(eng, "gamebot")
    eng.apply_owner_override("cozy", u.uid, rec["id"], "high", "hate")
    eng.clock = u.last_seen
    hub = serve.Hub(eng)
    hub.status.update(mode=mode, running=mode == "live", channels=["cozy"])
    hub.persist = persist
    path = tmp_path / "owner_roles.json"
    monkeypatch.setattr(R, "ROLES", str(path))
    monkeypatch.setattr(S, "OVERRIDES", str(tmp_path / "owner_overrides.jsonl"))
    hub.logged = []
    monkeypatch.setattr(hub, "_log", lambda rec: hub.logged.append(rec))
    return hub, u, t, path


def _role(hub, *bodies, extra=None):
    async def calls(c):
        out = []
        for body in bodies:
            r = await c.post("/api/role", json=body, headers=PAGE)
            out.append((r.status, await r.json()))
        if extra:
            out.append(await extra(c))
        return out
    return _http(hub, calls)


def _listed(eng, uid):
    items = eng.room_snapshot("cozy", eng.clock)["notices"]["items"]
    return any(d["user_id"] == uid for d in items)


def test_the_bot_role_stops_scoring_and_listing_at_once_and_undo_restores_it(monkeypatch, tmp_path):
    hub, u, t, path = _role_hub(monkeypatch, tmp_path)
    eng = hub.eng
    assert _listed(eng, u.uid)
    (st, j), = _role(hub, {"channel": "cozy", "uid": u.uid, "role": "bot"})
    assert st == 200 and j["role"] == "bot" and j["recorded"] and j["entry"]["login"] == "gamebot"
    assert eng.owner_bots == {u.uid} and not _listed(eng, u.uid)
    assert json.loads(path.read_text())["roles"][u.uid]["role"] == "bot"
    assert eng.profile("cozy", u.uid)["notice"] is None
    n0 = u.n
    rec = eng.process(msg("gamebot", "!points 500 for everyone", t + 30))
    assert rec["type"] == "bot_msg" and "toxicity" not in rec and u.n == n0           # not scored, not profiled
    hub.publish(rec)
    assert hub.queue[-1] == {k: rec[k] for k in serve.UI_BOT}
    assert any(r.get("type") == "owner_role" and r["role"] == "bot" for r in hub.logged)
    # the page's list, then undo: new messages are scored again
    (st, j), lst = _role(hub, {"channel": "cozy", "uid": u.uid, "role": None},
                         extra=lambda c: c.get("/api/roles"))
    assert st == 200 and j["role"] is None and not eng.owner_bots
    assert json.loads(path.read_text())["roles"] == {}
    rec = eng.process(msg("gamebot", "ok that was cool", t + 40))
    assert rec["type"] == "msg" and u.n == n0 + 1 and _listed(eng, u.uid)


def test_the_role_list_names_every_role_and_a_restart_of_the_live_run_reads_them(monkeypatch, tmp_path):
    hub, u, t, path = _role_hub(monkeypatch, tmp_path)

    async def roles(c):
        r = await c.get("/api/roles")
        return await r.json()
    (st, _), lst = _role(hub, {"channel": "cozy", "uid": u.uid, "role": "bot"}, extra=roles)
    assert st == 200 and [(e["user_id"], e["role"], e["login"]) for e in lst["roles"]] == [(u.uid, "bot", "gamebot")]
    assert lst["saves"] is True and hub._status()["owner_roles"] == 1
    assert serve.live_roles(False) == R.load(str(path)) and R.bots(serve.live_roles(False)) == {u.uid}


def test_bad_role_requests_are_refused_and_there_is_no_mod_role(monkeypatch, tmp_path):
    hub, u, t, path = _role_hub(monkeypatch, tmp_path)
    out = _role(hub, {"channel": "cozy", "uid": u.uid, "role": "mod"},
                {"channel": "cozy", "uid": u.uid, "role": ["bot"]},
                {"channel": "cozy", "uid": "nobody", "role": "bot"},
                {"channel": "cozy", "uid": u.uid, "role": None})
    assert [s for s, _ in out] == [400, 400, 404, 404]
    assert R.KINDS == ("bot",) and not path.exists() and not hub.eng.owner_bots


def test_a_role_that_cannot_be_saved_is_not_applied(monkeypatch, tmp_path):
    hub, u, t, path = _role_hub(monkeypatch, tmp_path)

    def fail(*a, **k):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(R, "save", fail)
    (st, j), = _role(hub, {"channel": "cozy", "uid": u.uid, "role": "bot"})
    assert st == 500 and j["not_recorded"] and not hub.eng.owner_bots and not hub.roles
    assert hub.eng.stats["role_save_errors"] == 1 and _listed(hub.eng, u.uid)


def test_a_role_on_a_replay_or_an_unsaved_run_steers_the_view_only(monkeypatch, tmp_path):
    for mode, persist in (("replay", False), ("live", False)):
        hub, u, t, path = _role_hub(monkeypatch, tmp_path, mode=mode, persist=persist)
        (st, j), = _role(hub, {"channel": "cozy", "uid": u.uid, "role": "bot"})
        assert st == 200 and not j["recorded"] and not path.exists() and not _listed(hub.eng, u.uid)


def test_a_bot_that_never_spoke_in_a_channel_still_gets_a_card_to_undo_the_role(monkeypatch, tmp_path):
    hub, u, t, path = _role_hub(monkeypatch, tmp_path)
    _role(hub, {"channel": "cozy", "uid": u.uid, "role": "bot"})

    async def card(c):
        r = await c.get(f"/api/profile?channel=other&uid={u.uid}")
        return r.status, await r.json()
    (st, p), = _role(hub, extra=card)
    assert st == 200 and p["stub"] and p["owner_role"]["role"] == "bot" and p["display"] == "gamebot"


# ------------------------------------------------------------------ live only: nothing else reads the roles
def test_roles_are_live_only(monkeypatch, tmp_path):
    """The owner's roles steer the live dashboard only: a replay reads no roles, a saved state never carries them
    (replays and set builders load saved states), and an unreadable file is moved aside, not saved over."""
    path = tmp_path / "owner_roles.json"
    path.write_text(json.dumps({"roles": {"7": {"role": "bot", "login": "x", "display": "x", "set": "t"}}}))
    monkeypatch.setattr(R, "load", lambda *a, **k: pytest.fail("a replay read the roles"))
    assert serve.live_roles(True, str(path)) == {}
    monkeypatch.undo()
    assert R.bots(serve.live_roles(False, str(path))) == {"7"}
    eng = Engine()
    eng.owner_bots = frozenset({"7"})
    _, data = serve.dump_engine(eng)
    assert "owner_bots" not in vars(pickle.loads(data)) and eng.owner_bots == {"7"}
    stale = _restart(eng)                                     # a state pickled some other way: upgrade drops them
    assert not stale.owner_bots and not Engine().owner_bots
    path.write_text("{not json")
    assert serve.live_roles(False, str(path)) == {} and not path.exists()
    assert any(p.name.startswith("owner_roles.json.unreadable-") for p in tmp_path.iterdir())


def test_no_training_evaluation_or_replay_path_reads_the_roles():
    """Only the dashboard (serve.py, and aictl.py, whose hybrid review skips a bot), the opt-out tool (forget.py), roles.py itself and the engine's live-only set
    (Engine.owner_bots, empty unless serve.py fills it) name the roles; and, to keep roles OUT of measurement, the
    session log (session.py writes the ids in effect at the start, marked_bots reads them back), compare_session (counts
    their alerts apart, never scores with them) and export_session_sets (takes their alerts from its role-free replay,
    so a role keeps nobody out of a stratum). Audits LB3-4 and LB3-7."""
    root = Path(__file__).resolve().parents[1]
    pat = re.compile(r"owner_roles|owner_bots|marked_bots|chatwatch\.roles|from \.roles|from \. import[^\n]*\broles\b|import roles")
    hits = set()
    for d in ("src", "bin"):
        for p in (root / d).rglob("*"):
            if p.is_file() and p.suffix in (".py", ".sh", "") and pat.search(p.read_text(errors="ignore")):
                hits.add(str(p.relative_to(root)))
    assert hits == {"src/chatwatch/serve.py", "src/chatwatch/aictl.py", "src/chatwatch/forget.py", "src/chatwatch/roles.py",
                    "src/chatwatch/engine.py", "src/chatwatch/session.py"}, hits


def test_forget_removes_a_chatters_role(tmp_path):
    from chatwatch import forget
    path = tmp_path / "data" / "sessions" / "owner_roles.json"
    R.save({"1": R.entry("bot", "a", "A"), "2": R.entry("bot", "b", "B")}, str(path))
    R.save(R.load(str(path)), str(path))                     # a .prev that names them too
    assert forget.main(["1"], root=str(tmp_path)) == 0 and set(R.load(str(path))) == {"1", "2"}     # dry run
    assert forget.main(["1", "--go"], root=str(tmp_path)) == 0
    assert set(R.load(str(path))) == {"2"}
    assert "\"1\"" not in open(str(path) + ".prev").read()            # the .prev no longer names them (LB3-6)


# ------------------------------------------------------------------ standing flags (owner, 2026-11-07; FLAGS_PERSIST)
def _calm_after(eng, login, t, n=45):
    for i in range(n):
        eng.process(msg(login, CALM_ROOM[i % len(CALM_ROOM)] + f" {i}", t + 3 + i))
    return _user(eng, login)


def _level(eng, t, login):
    items = eng.notices("cozy", T0 + (t + 50) * 1000)
    return {d["display"]: d["level"] for d in items[0]["items"] + items[1]}.get(login)




def test_a_standing_slur_or_severe_flag_keeps_the_card_after_40_more_messages():
    """'if a user says even just one thing that is flagged and review maintains that it should be flagged then they
    will always be on the watchlist': the flag left the 40 recent messages, the card stays (with and without reads)."""
    for reviewed in (False, True):
        eng = Engine()
        eng.person_review = reviewed
        t = room(eng)
        slur = eng.process(msg("slurguy", "you are a nigger", t))
        threat = eng.process(msg("threatguy", "i will kill you", t + 1))
        for who in ("slurguy", "threatguy"):
            u = _calm_after(eng, who, t + 2)
            assert all(m["id"] not in (slur["id"], threat["id"]) for m in u.msgs)
        assert _level(eng, t + 2, "slurguy") == "act", reviewed
        assert _level(eng, t + 2, "threatguy") == "act", reviewed            # no read yet: A4h keeps 'act'
        why = E._notice_reasons(_user(eng, "slurguy"), None, _user(eng, "slurguy").last_seen, reviewed)
        assert why[0] == "act" and why[2]["id"] == slur["id"]                  # the card quotes the standing flag


def test_a_review_still_clears_a_flag_after_it_left_the_recent_messages_and_the_card_goes():
    eng = Engine()
    eng.person_review = True
    t = room(eng)
    rec = eng.process(msg("threatguy", "i will kill you", t))
    u = _calm_after(eng, "threatguy", t + 1)
    assert _level(eng, t + 1, "threatguy") == "act"
    upd = eng.apply_llm_toxicity("cozy", u.uid, rec["id"], {"toxicity": "none"})    # the re-read comes back late
    assert upd is not None and u.toxic_n == 0
    assert E._notice_reasons(u, None, u.last_seen, True) is None                   # nothing else flagged: off the list
    item, = eng.evidence("cozy", u.uid)
    assert not item["counted"]                                                     # "not toxic now", kept to undo
    eng.process(msg("threatguy", "i will kill you", t + 60))                       # a new flag: back on the list
    assert E._notice_reasons(u, None, u.last_seen, True)[0] == "act"


def test_a_keyword_slur_is_never_cleared_by_a_late_re_read_s1():
    eng = Engine()
    t = room(eng)
    rec = eng.process(msg("slurguy", "you are a nigger", t))
    u = _calm_after(eng, "slurguy", t + 1)
    eng.apply_llm_toxicity("cozy", u.uid, rec["id"], {"toxicity": "none"})
    assert E._notice_reasons(u, None, u.last_seen, False)[0] == "act"


def test_a_full_evidence_drops_plain_lines_before_standing_flags():
    eng = Engine()
    t = room(eng)
    slur = eng.process(msg("heckler", "you are a nigger", t))
    for i in range(E.EVIDENCE_KEPT + 5):
        eng.process(msg("heckler", f"{INSULT} {i}", t + 1 + i))
    u = _user(eng, "heckler")
    assert len(u.evidence) == E.EVIDENCE_KEPT and u.evidence[0]["id"] == slur["id"]
    assert E._notice_reasons(u, None, u.last_seen, False)[0] == "act"


def test_flags_persist_off_is_the_old_rule():
    eng = Engine()
    t = room(eng)
    eng.process(msg("slurguy", "you are a nigger", t))
    u = _calm_after(eng, "slurguy", t + 1)
    try:
        E.FLAGS_PERSIST = False
        assert E._notice_reasons(u, None, u.last_seen, False) is None
    finally:
        E.FLAGS_PERSIST = True
