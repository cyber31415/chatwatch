"""The live dashboard after the audit of 2026-11-01 (docs/review/audit_20261101/live_backend.md, LB3-n, and
live_frontend.md, LF3-n, server side). No network, no LLM: the LLM, the register, the roles file and the sessions are
fakes or in tmp_path; the chat is synthetic."""
import asyncio
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path

import pytest

from chatwatch import engine as E
from chatwatch.serve import ReadQueue
from chatwatch import roles as R
from chatwatch import serve
from chatwatch import session as S
from chatwatch.engine import Engine

from test_dashboard import _http, _restart
from test_scenarios import CALM_ROOM, NEW_ID, T0, msg, room

ROOT = Path(__file__).resolve().parents[1]
PAGE = {"X-Chatwatch": "1"}
INSULT = "you are a stupid idiot streamer"


def _user(eng, login):
    return next(u for u in eng.users.values() if u.login == login)


# ------------------------------------------------------------------ LB3-1: the alert take-back is order-free
def _two_toxic_and_alert():
    eng = Engine()
    t = room(eng)
    a = eng.process(msg("newacct", INSULT + " kys", t, first=True, uid=NEW_ID))
    b = eng.process(msg("newacct", "you are trash, worst streamer ever", t + 5, uid=NEW_ID))
    u = _user(eng, "newacct")
    assert u.alerted and u.toxic_n == 2
    return eng, u, a["id"], b["id"]


@pytest.mark.parametrize("first,second,back", [("a", "b", "a"), ("a", "b", "b"), ("b", "a", "a"), ("b", "a", "b")])
def test_the_alert_comes_back_when_any_correction_makes_the_chatter_toxic_again(first, second, back):
    """P1: "Not toxic" on A, then on B, took the alert back; "Back to automatic" on A left toxic_n 1 and no alert,
    because only B's own undo restored it. The toxic set decides now, not the order of the corrections."""
    eng, u, a, b = _two_toxic_and_alert()
    ids = {"a": a, "b": b}
    eng.apply_owner_override("cozy", u.uid, ids[first], "none")
    assert u.alerted and u.toxic_n == 1                       # another toxic message is left: the alert stands
    eng.apply_owner_override("cozy", u.uid, ids[second], "none")
    assert not u.alerted and u.toxic_n == 0 and u.owner_cleared
    eng.apply_owner_override("cozy", u.uid, ids[back], None)
    assert u.alerted and u.toxic_n == 1 and not u.owner_cleared
    assert E._notice_reasons(u, None, u.last_seen, False) is not None


def test_a_new_toxic_message_after_the_take_back_hands_the_alert_to_the_usual_path():
    eng, u, a, b = _two_toxic_and_alert()
    for m in (a, b):
        eng.apply_owner_override("cozy", u.uid, m, "none")
    assert not u.alerted and u.owner_cleared
    r = eng.process(msg("newacct", "you are trash", (u.last_seen - T0) // 1000 + 30, uid=NEW_ID))
    assert r["toxicity"]["level"] != "none"
    assert not u.owner_cleared                                # the usual path decides from here (re-alert or not)


def test_a_take_back_saved_on_the_message_by_the_previous_version_moves_to_the_chatter():
    eng, u, a, b = _two_toxic_and_alert()
    for m in (a, b):
        eng.apply_owner_override("cozy", u.uid, m, "none")
    u.owner_cleared = False
    next(m for m in u.msgs if m["id"] == b)["owner_cleared_alert"] = True      # how LB2-2 kept it
    old = _restart(eng)
    v = old.users[("cozy", NEW_ID)]
    del v.owner_cleared                                        # a User pickled before the slot existed
    v = old.upgrade().users[("cozy", NEW_ID)]
    assert v.owner_cleared
    old.apply_owner_override("cozy", NEW_ID, a, None)
    assert v.alerted


# ------------------------------------------------------------------ LB3-2: kinds after LLM re-reads
def test_llm_re_reads_count_each_message_once_by_its_current_read():
    """P2: a layer insult re-read as harassment showed insult 1 + harassment 1, and a second re-read harassment 2."""
    eng = Engine()
    t = room(eng)
    rec = eng.process(msg("heckler", INSULT, t))
    u = _user(eng, "heckler")
    kinds = lambda: {k: v for k, v in u.toxic_kinds.items() if v > 0}
    assert kinds() == {k: 1 for k in rec["toxicity"]["kinds"]}
    for _ in range(2):
        eng.apply_llm_toxicity("cozy", u.uid, rec["id"], {"toxicity": "medium", "kind": "harassment", "target": "streamer"})
        assert kinds() == {"harassment": 1}
    eng.apply_owner_override("cozy", u.uid, rec["id"], "none")
    assert kinds() == {} and u.toxic_n == 0
    eng.apply_owner_override("cozy", u.uid, rec["id"], None)          # back to the LLM's read
    assert kinds() == {"harassment": 1}
    eng.apply_llm_toxicity("cozy", u.uid, rec["id"], {"toxicity": "none"})
    assert kinds() == {} and eng.profile("cozy", u.uid)["toxic"]["kinds"] == []


def test_a_keyword_slur_re_read_as_none_keeps_its_kind_on_the_card():
    eng = Engine()
    t = room(eng)
    rec = eng.process(msg("slurguy", "you are a nigger", t))
    u = _user(eng, "slurguy")
    eng.apply_llm_toxicity("cozy", u.uid, rec["id"], {"toxicity": "none"})
    assert u.toxic_n == 1 and u.toxic_kinds["hate"] == 1                 # S1: it still counts, as hate


# ------------------------------------------------------------------ LB3-3: Start while Stop is still closing
def test_start_is_refused_until_a_slow_stop_has_saved(monkeypatch, tmp_path):
    """P6: a second Start during Stop opened a session that copied the state before Stop saved it."""
    import threading
    release = threading.Event()
    monkeypatch.setattr(S.Session, "_prepare", lambda self: release.wait(5))
    real_init = S.Session.__init__
    monkeypatch.setattr(S.Session, "__init__",
                        lambda self, ch, argv, eng, base=None, ai=None: real_init(self, ch, argv, eng, base=str(tmp_path / "s"), ai=ai))

    async def idle(*a):
        await asyncio.sleep(3600)
    monkeypatch.setattr(serve, "live_source", idle)
    hub = serve.Hub(Engine())
    hub.status["mode"] = "live"

    async def go():
        hub.start_live(["cozy"])
        stop = asyncio.create_task(hub.stop_live())
        await asyncio.sleep(0.1)
        with pytest.raises(RuntimeError, match="still stopping"):
            hub.start_live(["cozy"])
        during = dict(hub._status())
        release.set()
        await stop
        hub.start_live(["cozy"])                               # once Stop has returned, Start works again
        ok = hub.session is not None
        hub.live_task.cancel()
        return during, ok
    during, ok = asyncio.run(go())
    assert during["stopping"] and ok and not hub.stopping


def test_the_start_endpoint_answers_409_while_stopping():
    hub = serve.Hub(Engine())
    hub.status["mode"] = "live"
    hub.stopping = True

    async def call(c):
        r = await c.post("/api/start", json={"channels": "cozy"}, headers=PAGE)
        return r.status, await r.json()
    st, j = _http(hub, call)
    assert st == 409 and "stopping" in j["error"] and hub.session is None


# ------------------------------------------------------------------ LB3-4: compare_session and roles
def test_a_session_records_the_bots_marked_at_its_start(monkeypatch, tmp_path):
    monkeypatch.setattr(S.Session, "_prepare", lambda self: None)
    eng = Engine()
    eng.owner_bots = frozenset({"9", "7"})
    sess = S.Session(["cozy"], ["x"], eng, base=str(tmp_path))
    sess.close(dict(eng.stats))
    meta = json.load(open(os.path.join(sess.dir, "meta.json")))
    assert meta["owner_bots_at_start"] == ["7", "9"]
    with open(os.path.join(sess.dir, "outputs.jsonl"), "a") as f:
        f.write(json.dumps({"type": "owner_role", "user_id": "8", "role": "bot"}) + "\n")
        f.write(json.dumps({"type": "owner_role", "user_id": "6", "role": None}) + "\n")
    assert S.marked_bots(sess.dir) == {"7", "8", "9"}


# ------------------------------------------------------------------ LB3-5: a marked bot is not profiled
def test_a_marked_bot_gets_no_profile_changes_from_mod_actions_or_sub_notices(monkeypatch, tmp_path):
    from chatwatch import irc
    eng = Engine()
    t = room(eng)
    eng.process(msg("gamebot", "!points giveaway", t))
    u = _user(eng, "gamebot")
    eng.owner_bots = frozenset({u.uid})
    before = (list(u.mod_actions), list(u.reactions), dict(u.why), dict(u.react_ev), dict(u.msg_ev))
    rec = eng.process({"kind": "clearchat", "ts": T0 + (t + 2) * 1000, "channel": "cozy", "room_id": "",
                       "target_login": "gamebot", "target_user_id": u.uid, "duration": 600})
    tags = (f"@badge-info=;badges=;color=;display-name=gamebot;emotes=;id=n1;login=gamebot;mod=0;msg-id=sub;room-id=1;subscriber=0;"
            rf"system-msg=gamebot\ssubscribed.;tmi-sent-ts={T0 + (t + 3) * 1000};user-id={u.uid}")
    eng.process(irc.to_event(f"{tags} :tmi.twitch.tv USERNOTICE #cozy"))
    assert rec["type"] == "modaction" and not rec["known"]       # the room sees the action; their profile does not
    assert (list(u.mod_actions), list(u.reactions), dict(u.why), dict(u.react_ev), dict(u.msg_ev)) == before
    assert eng.stats["mod_timeout"] == 1                        # the action is still counted (room context)


def test_a_correction_on_a_marked_bots_message_starts_no_llm_read(monkeypatch, tmp_path):
    eng = Engine()
    eng.person_review = True
    t = room(eng)
    rec = eng.process(msg("gamebot", "nice run", t))
    u = _user(eng, "gamebot")
    eng.owner_bots = frozenset({u.uid})
    eng.clock = u.last_seen
    hub = serve.Hub(eng)
    hub.status.update(mode="live", running=True)
    hub.persist = True
    hub.review_q = ReadQueue()
    monkeypatch.setattr(S, "OVERRIDES", str(tmp_path / "o.jsonl"))
    monkeypatch.setattr(hub, "_log", lambda r: None)

    async def call(c):
        r = await c.post("/api/override", json={"channel": "cozy", "uid": u.uid, "id": rec["id"], "level": "high",
                                                "kind": "hate"}, headers=PAGE)
        return r.status
    assert _http(hub, call) == 200
    assert hub.review_q.qsize() == 0 and not hub.person_queued


# ------------------------------------------------------------------ LB3-6: forget and the roles' copies
def test_forget_finds_a_role_left_only_in_the_prev_copy(tmp_path):
    from chatwatch import forget
    path = tmp_path / "data" / "sessions" / "owner_roles.json"
    R.save({"1": R.entry("bot", "a", "A")}, str(path))
    R.save({}, str(path))                                      # removed on the page: only the .prev names them
    assert forget.main(["1", "--go"], root=str(tmp_path)) == 0
    assert '"1"' not in open(str(path) + ".prev").read() and R.load(str(path)) == {}


def test_forget_reports_unreadable_copies_and_never_stops_half_way(tmp_path, capsys):
    from chatwatch import forget
    path = tmp_path / "data" / "sessions" / "owner_roles.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    (tmp_path / "data" / "sessions" / "owner_roles.json.unreadable-20261101-020000").write_text('{"roles": {"1": {"rol')
    assert forget.main(["1", "--go"], root=str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "owner_roles.json: could not be read" in out and "unreadable-20261101-020000: an unreadable copy" in out


# ------------------------------------------------------------------ LB3-7: roles never pick session-sourced silver
def _chat():
    evs = [msg(f"regular{i % 9}", CALM_ROOM[i % len(CALM_ROOM)], i * 2, sub=6) for i in range(40)]
    for k in range(30):
        evs.append(msg("gamebot", f"!points giveaway round {k}", 90 + k * 3))
        evs.append(msg(f"regular{k % 9}", CALM_ROOM[k % len(CALM_ROOM)] + f" {k}", 91 + k * 3, sub=6))
        if k % 7 == 3:
            evs.append(msg("heckler", INSULT + f" {k}", 92 + k * 3))
    return evs


def _session(tmp_path, tag, role_at, evs):
    """The same synthetic chat through a Hub that logs a session; the owner marks 'gamebot' a bot at event `role_at`
    (None: never). Returns the session folder, renamed to one fixed name under <tmp>/<tag>/data/sessions."""
    eng = Engine()
    hub = serve.Hub(eng)
    sess = S.Session.__new__(S.Session)                         # no worker thread, no code snapshot
    base = tmp_path / tag / "data" / "sessions" / "20261101-010000_cozy"
    (base / "raw").mkdir(parents=True)
    sess.dir, sess.n, sess.meta, sess.stats_at_start = str(base), 0, {}, {}
    sess.out = open(base / "outputs.jsonl", "w")
    hub.session = sess
    for i, ev in enumerate(json.loads(json.dumps(evs))):
        if i == role_at:
            eng.owner_bots = frozenset({msg("gamebot", "", 0)["user_id"]})
        serve.handle(hub, ev)
    sess.out.close()
    assert not eng.stats.get("event_errors")
    types = [json.loads(x)["type"] for x in open(base / "outputs.jsonl")]
    return base, types


# ------------------------------------------------------------------ contract: a repeated click records nothing
def test_a_repeated_click_answers_not_recorded(monkeypatch, tmp_path):
    eng = Engine()
    t = room(eng)
    rec = eng.process(msg("corr", INSULT, t))
    u = _user(eng, "corr")
    eng.clock = u.last_seen
    hub = serve.Hub(eng)
    hub.status.update(mode="live", running=True)
    hub.persist = True
    monkeypatch.setattr(S, "OVERRIDES", str(tmp_path / "o.jsonl"))
    monkeypatch.setattr(R, "ROLES", str(tmp_path / "r.json"))
    monkeypatch.setattr(hub, "_log", lambda r: None)
    body = {"channel": "cozy", "uid": u.uid, "id": rec["id"], "level": "none"}
    role = {"channel": "cozy", "uid": u.uid, "role": "bot"}

    async def calls(c):
        out = []
        for url, b in (("/api/override", body), ("/api/override", body), ("/api/role", role), ("/api/role", role)):
            r = await c.post(url, json=b, headers=PAGE)
            out.append(await r.json())
        return out
    o1, o2, r1, r2 = _http(hub, calls)
    assert o1["recorded"] and not o1.get("unchanged") and o2["unchanged"] and o2["recorded"] is False
    assert r1["recorded"] and r2["unchanged"] and r2["recorded"] is False and r1["rev"] == r2["rev"] == 1
    assert len(open(tmp_path / "o.jsonl").read().splitlines()) == 1


# ------------------------------------------------------------------ LF3-2: the page gets the level that counts
def test_the_page_gets_the_level_a_slur_counts_at_under_s1(monkeypatch, tmp_path):
    """A keyword slur re-read as "none" by the LLM still counts at the layer's level (S1); the page showed the raw
    level. The server now adds counted_level wherever a read reaches the page; no decision changes."""
    eng = Engine()
    t = room(eng)
    rec = eng.process(msg("slurguy", "you are a nigger", t))
    u = _user(eng, "slurguy")
    eng.clock = u.last_seen
    upd = eng.apply_llm_toxicity("cozy", u.uid, rec["id"], {"toxicity": "none"})
    raw = upd["toxicity"]
    assert raw["level"] == "none" and E._level(raw) == "high" and u.toxic_n == 1
    hub = serve.Hub(eng)
    shown = hub._ui(upd)["toxicity"]
    assert shown["counted_level"] == "high" and shown["level"] == "none" and "counted_level" not in raw
    assert serve.shown_tox({"level": "medium", "source": "layer"}) == {"level": "medium", "source": "layer"}

    async def calls(c):
        p = await (await c.get(f"/api/profile?channel=cozy&uid={u.uid}")).json()
        ev = await (await c.get(f"/api/evidence?channel=cozy&uid={u.uid}")).json()
        return p, ev
    p, ev = _http(hub, calls)
    pick = lambda ms: next(m for m in ms if m["id"] == rec["id"])["toxicity"]
    assert pick(p["recent_messages"])["counted_level"] == "high" and pick(p["evidence"])["counted_level"] == "high"
    assert pick(ev["items"])["counted_level"] == "high" and next(i for i in ev["items"] if i["id"] == rec["id"])["counted"]
    # the engine's own records are untouched (the session log and every decision read them)
    assert "counted_level" not in next(m for m in u.msgs if m["id"] == rec["id"])["toxicity"]
