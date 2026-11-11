"""The live dashboard after the audit of 2026-11-01b (docs/review/audit_20261101b/live_backend.md, LB4-n, and the
server side of live_frontend.md LF4-6, LF4-7). No network, no LLM; the chat is synthetic, every file in tmp_path."""
import asyncio
import json
import pickle
import time

import pytest

from chatwatch import archetypes as A
from chatwatch import cues, emotes, forget, llm, serve, vocab
from chatwatch import engine as E
from chatwatch import roles as R
from chatwatch.engine import Engine

from test_dashboard import _http, _restart
from test_emotes import BTTV, FFZ, SEVENTV, _fake
from test_scenarios import T0, msg, room

PAGE = {"X-Chatwatch": "1"}
INSULT = "you are a stupid idiot streamer"


def _user(eng, login):
    return next(u for u in eng.users.values() if u.login == login)


class _WS:
    def __init__(self):
        self.frames = []

    async def send_str(self, d):
        self.frames.append(json.loads(d))


def _flush(hub):
    ws = _WS()
    hub.clients.add(ws)
    hub._last_room = hub._last_log = 0
    asyncio.run(hub._flush())
    hub.clients.discard(ws)
    return [r for f in ws.frames for r in f]


# ------------------------------------------------------------------ LB4-1: one bad snapshot never drops the batch
def _with_a_coded_term(monkeypatch):
    """A chatter and a room that used a coded-lexicon term, saved and loaded back; then the term id leaves the
    lexicon (an edit after the state was saved)."""
    eng = Engine()
    t = room(eng)
    term = next(x for x in vocab.LEX.terms if x["tier"] in vocab.MARKER_TIERS)
    eng.process(msg("coded", "hello chat", t, uid="777"))
    h = {"id": term["id"], "family": term["family"], "tier": term["tier"], "weight": term["weight"], "text": "x",
         "distress": False}
    for _ in range(3):
        eng.users[("cozy", "777")].vocab.add([h], eng.rooms["cozy"].vocab)
        eng.rooms["cozy"].vocab.observe("coded", [h])
    eng.clock = T0 + (t + 1) * 1000
    assert term["id"] in json.dumps(eng.room_snapshot("cozy", eng.clock, full=True)["vocabulary"])
    eng = _restart(eng)
    monkeypatch.setattr(vocab.LEX, "terms", [x for x in vocab.LEX.terms if x["id"] != term["id"]])
    return eng, term["id"]


def test_a_term_id_gone_from_the_lexicon_is_skipped_in_a_saved_state(monkeypatch):
    eng, gone = _with_a_coded_term(monkeypatch)
    assert gone not in json.dumps(eng.room_snapshot("cozy", eng.clock, full=True)["vocabulary"])
    assert gone not in json.dumps(eng.profile("cozy", "777")["vocabulary"])


def test_one_channels_snapshot_error_skips_that_channel_only(monkeypatch):
    """P2: an exception in one room snapshot dropped the whole batch (chat lines too) and its status frame, so the
    "server errors" pill never showed."""
    eng = Engine()
    room(eng)
    hub = serve.Hub(eng)
    hub.status.update(mode="replay", channels=["cozy", "broken"])
    real = eng.room_snapshot

    def snap(ch, *a, **k):
        if ch == "broken":
            raise KeyError("ropefuel")
        return real(ch, *a, **k)
    monkeypatch.setattr(eng, "room_snapshot", snap)
    hub.queue = [{"type": "msg", "channel": "cozy", "id": f"q{i}", "text": "hi"} for i in range(5)]
    sent = _flush(hub)
    types = [r["type"] for r in sent]
    assert types.count("msg") == 5 and types.count("room") == 1 and types[-1] == "status"
    assert sent[-1]["stats"]["flusher_errors"] == 1 and hub.queue == []


def test_a_record_that_cannot_be_encoded_is_left_out_alone():
    eng = Engine()
    hub = serve.Hub(eng)
    hub.status.update(mode="replay", channels=[])
    hub.queue = [{"type": "msg", "id": "a"}, {"type": "msg", "id": "bad", "x": {("tuple", "key"): 1}},
                 {"type": "msg", "id": "b"}]
    sent = _flush(hub)
    assert [r.get("id") for r in sent if r["type"] == "msg"] == ["a", "b"]
    assert eng.stats["flusher_errors"] == 1


# ------------------------------------------------------------------ LB4-2: a second Stop waits for the first
def test_a_second_stop_waits_for_the_first_so_start_stays_refused():
    """P7: the second Stop returned at once and cleared `stopping` while the first was still closing the session."""
    hub = serve.Hub(Engine())
    hub.status["mode"] = "live"
    hub.persist = False
    hub.log_sessions = False
    order = []

    class SlowSess:
        dir = "/tmp/slow"

        def log(self, r):
            pass

        def close(self, stats):
            time.sleep(0.5)
            order.append("closed")

    async def go():
        hub.session = SlowSess()
        s1 = asyncio.ensure_future(hub.stop_live())
        await asyncio.sleep(0.05)
        await hub.stop_live()
        order.append("stop 2 returned")
        hub.start_live(["cozy"])                               # both Stops are done: Start works
        hub.live_task.cancel()
        await s1
    asyncio.run(go())
    assert order == ["closed", "stop 2 returned"] and not hub.stopping


# ------------------------------------------------------------------ LB4-3: forget past a torn copy
def test_forget_goes_on_past_a_torn_state_copy_and_names_the_set_aside_ones(tmp_path, capsys):
    root = tmp_path
    (root / "data" / "state").mkdir(parents=True)
    for d in ("20261101-000000_cozy", "20261102-000000_cozy"):
        (root / "data" / "sessions" / d).mkdir(parents=True)
    eng = Engine()
    t = room(eng)
    eng.process(msg("victim", "hello", t, uid="4242"))
    eng.ml = eng.emotes = eng.risk = eng.persist = None
    blob = pickle.dumps(eng)
    for p in ("data/state/engine.pkl", "data/state/engine.pkl.unreadable-20261101T000000Z",
              "data/sessions/20261102-000000_cozy/state_at_start.pkl"):
        (root / p).write_bytes(blob)
    (root / "data/sessions/20261101-000000_cozy/state_at_start.pkl").write_bytes(blob[:5000])   # torn
    (root / "data/sessions/owner_overrides.jsonl").write_text(json.dumps({"user_id": "4242", "msg_id": "m1"}) + "\n")
    R.save({"4242": R.entry("bot", "victim", "victim")}, str(root / "data/sessions/owner_roles.json"))
    assert forget.main(["4242", "--go"], root=str(root)) == 0
    out = capsys.readouterr().out
    newer = pickle.loads((root / "data/sessions/20261102-000000_cozy/state_at_start.pkl").read_bytes())
    assert not any(k[1] == "4242" for k in newer.users)
    assert "4242" not in (root / "data/sessions/owner_overrides.jsonl").read_text()
    assert "4242" not in (root / "data/sessions/owner_roles.json").read_text()
    assert "20261101-000000_cozy/state_at_start.pkl: could not be read" in out
    assert "engine.pkl.unreadable-20261101T000000Z: a set-aside copy that may name them" in out


# ------------------------------------------------------------------ LB4-4: a quiet chat still ages
def _quiet_hub(mode):
    eng = Engine()
    t = room(eng)
    for i in range(3):
        eng.process(msg("loud", INSULT + f" {i}", t + i, uid="99"))
    eng.clock = T0 + (t + 2) * 1000                           # the last event, as serve.handle sets it
    hub = serve.Hub(eng)
    hub.status.update(mode=mode, running=True, channels=["cozy"])
    return hub


def test_a_live_snapshot_uses_the_wall_clock_when_chat_goes_quiet():
    """P9: with no events the snapshot stayed at the last message's time: the rate and the live cards froze."""
    r = next(x for x in _flush(_quiet_hub("live")) if x["type"] == "room")
    assert r["msgs_per_min"] == 0 and not r["notices"]["items"]
    r = next(x for x in _flush(_quiet_hub("replay")) if x["type"] == "room")      # a replay keeps its own clock
    assert r["msgs_per_min"] > 0 and r["notices"]["items"]


# ------------------------------------------------------------------ LB4-5: malformed bodies answer 400
@pytest.mark.parametrize("path,body", [("/api/start", "not json"), ("/api/start", "[1,2]"),
                                       ("/api/llm_review", "[1]"), ("/api/draft_warning", '"x"')])
def test_a_body_that_is_not_a_json_object_answers_400(path, body):
    eng = Engine()
    room(eng)
    hub = serve.Hub(eng)
    hub.status["mode"] = "live"
    hub.log_sessions = False

    async def call(c):
        r = await c.post(path, data=body, headers=PAGE)
        return r.status, await r.json()
    st, j = _http(hub, call)
    assert st == 400 and "JSON object" in j["error"] and hub.live_task is None


# ------------------------------------------------------------------ LB4-6: "keeps it up" only where it was fitted
def test_keeps_it_up_is_shown_only_up_to_the_fitted_message_counts():
    assert E.PERSIST_UP_TO == 10                      # the cold-start fit's last checkpoint (1, 3, 5, 10)
    eng = Engine()
    if not eng.persist.ok:
        pytest.skip("no persist model in data/models")
    t = room(eng)
    for i in range(60):
        eng.process(msg("talker", INSULT if i % 10 == 0 else f"nice play number {i}", t + i, uid="5151"))
        p = eng.profile("cozy", "5151")
        assert (p["persist"] is None) == (i + 1 > E.PERSIST_UP_TO)
    item = next(x for x in eng.notices("cozy", T0 + (t + 60) * 1000)[0]["items"] if x["user_id"] == "5151")
    assert item["persist"] is None and not any("keep it up" in w for w in item["why"])


# ------------------------------------------------------------------ LB4-7, LF4-7: unread numbers are gone
def test_no_unread_will_act_score_or_profile_fields_are_computed_or_sent():
    eng = Engine()
    t = room(eng)
    rec = eng.process(msg("heckler", INSULT, t))
    assert "risk" not in rec and "risk_why" not in rec
    p = eng.profile("cozy", rec["user_id"])
    assert not {"risk", "state_vs_trait", "account_newness", "mod_actions", "elsewhere"} & set(p)
    assert llm._content(p, list(eng.rooms["cozy"].context))          # the person read's input is complete
    hub = serve.Hub(eng)

    async def call(c):
        return await (await c.get(f"/api/profile?channel=cozy&uid={rec['user_id']}")).json()
    j = _http(hub, call)
    assert not {"risk", "state_vs_trait", "friction", "badfaith", "first_seen", "mod_actions"} & set(j)
    assert j["register"] is None or set(j["register"]) == set(serve.CARD_REGISTER)


# ------------------------------------------------------------------ LF4-6: plain words on the folded read
def test_the_first_impression_read_says_signs_and_cue_words_not_keys():
    eng = Engine()
    t = room(eng)
    for i in range(7):
        eng.process(msg("provo", INSULT + f" {i}", t + i, uid="31"))
    p = eng.profile("cozy", "31")
    assert "%" not in p["verdict"] and "worth watching" not in p["verdict"]
    assert "bad-faith signs" in p["verdict"]
    text = " ".join(r["reason"] for r in p["reasons"])
    assert not any(f"{k}:" in text for k in cues.CUE_REASON)
    assert cues.CUE_REASON["insult_target"] in text
    assert E.verdict({a: (1.0 if a == "provocateur" else 0.0) for a in A.NAMES}, 7).endswith("strong bad-faith signs")


# ------------------------------------------------------------------ LB4-9: a long live run refreshes emote sets
def test_a_live_run_refetches_a_room_set_once_it_is_refresh_s_old(tmp_path, monkeypatch):
    _fake(monkeypatch, {"7tv.io/v3/users": SEVENTV, "betterttv.net/3/cached/users": BTTV, "frankerfacez.com/v1/room": FFZ})
    es = emotes.EmoteSets(str(tmp_path), background=True)
    es.codes("42")
    for _ in range(200):
        if "42" in es.sets:
            break
        time.sleep(0.01)
    assert "Okayge" in es.sets["42"] and es.refresh["42"] - time.time() > emotes.REFRESH_S - 60
    assert es.codes("42") is es.sets["42"] and "42" not in es.pending          # not due: no refetch
    new = {"emote_set": {"emotes": [{"name": "Okayge"}, {"name": "newOne"}]}}
    _fake(monkeypatch, {"7tv.io/v3/users": new, "betterttv.net/3/cached/users": BTTV, "frankerfacez.com/v1/room": FFZ})
    d = json.load(open(tmp_path / "42.json"))
    d["fetched"] -= emotes.REFRESH_S + 1                         # the set is now 6 hours old
    json.dump(d, open(tmp_path / "42.json", "w"))
    es.refresh["42"] = time.time() - 1
    es.codes("42")
    for _ in range(200):
        if "newOne" in es.sets["42"]:
            break
        time.sleep(0.01)
    assert "newOne" in es.sets["42"] and "catJAM" not in es.sets["42"] - es.global_codes


# ------------------------------------------------------------------ LF4-1: a boot id makes a restart exact
def test_the_status_carries_one_boot_id_per_process():
    a, b = serve.Hub(Engine())._status(), serve.Hub(Engine())._status()
    assert a["boot"] == b["boot"] == serve.BOOT and len(serve.BOOT) == 16

    async def call(c):
        return await (await c.get("/api/status")).json()
    assert _http(serve.Hub(Engine()), call)["boot"] == serve.BOOT


def test_first_timers_this_session_count_since_start():
    """Owner 2026-11-08: "first-timers this min (just make that first-timers this session)". The room snapshot counts
    chatters whose first message ever in the channel (Twitch's first-msg tag) came since the run started."""
    from chatwatch.engine import Engine
    from test_scenarios import T0, msg
    eng = Engine()
    eng.process(msg("oldnewbie", "hi all first time here", 0, first=True))       # before Start
    since = T0 + 100_000
    eng.process(msg("newbie", "hello, first time watching", 200, first=True))
    eng.process(msg("regular", "gg", 210))
    eng.process(msg("oldnewbie", "back again", 220))
    snap = eng.room_snapshot("cozy", T0 + 230_000, since=since, full=False)
    assert snap["first_timers_session"] == 1 and snap["chatters"] == 3
    assert eng.room_snapshot("cozy", T0 + 230_000, full=False)["first_timers_session"] == 2
