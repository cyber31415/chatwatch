"""The live dashboard after audit 2026-10-29 (docs/review/audit_20261029/live_backend.md, LB-n, and
live_frontend.md, LF-n). No network, no LLM: the LLM, the IRC socket and the browsers are fakes."""
import asyncio
import json
import os
import pickle
import threading
import time

import pytest

from chatwatch import llm, serve, session
from chatwatch.engine import Engine
from chatwatch.serve import Hub, ReadQueue

from test_dashboard import FakeIRC, _fake_connect, _http
from test_scenarios import T0, msg, room

PAGE = {"X-Chatwatch": "1"}


def _state(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "STATE", str(tmp_path / "state" / "engine.pkl"))
    monkeypatch.setattr(serve, "_saved", [0])
    os.makedirs(tmp_path / "state", exist_ok=True)
    return serve.STATE


def _saved_users():
    return len(pickle.load(open(serve.STATE, "rb")).users)


# ------------------------------------------------------------------ LB-1: nothing saves over a state it did not load
def test_stop_on_a_replay_dashboard_is_refused_and_saves_nothing(tmp_path, monkeypatch):
    _state(tmp_path, monkeypatch)
    live = Engine()
    room(live)
    serve.save_engine(live)
    n = _saved_users()
    hub = Hub(Engine())                                  # a replay's engine: always fresh
    hub.status["mode"] = "replay"

    async def calls(c):
        r = await c.post("/api/stop", headers=PAGE)
        return r.status, (await r.json())["error"]
    assert _http(hub, calls) == (400, "not a live run")
    asyncio.run(hub.stop_live())                         # and a Hub that did not load the state never saves
    assert _saved_users() == n > 0


def test_an_unreadable_state_is_moved_aside_never_overwritten(tmp_path, monkeypatch):
    st = _state(tmp_path, monkeypatch)
    open(st, "wb").write(b"not a pickle")
    eng = serve.load_engine(None, None)
    assert len(eng.users) == 0 and not os.path.exists(st)
    aside = [f for f in os.listdir(tmp_path / "state") if f.startswith("engine.pkl.unreadable-")]
    assert len(aside) == 1 and open(tmp_path / "state" / aside[0], "rb").read() == b"not a pickle"
    serve.save_engine(eng)                                # the next save writes a new file beside it
    assert open(tmp_path / "state" / aside[0], "rb").read() == b"not a pickle"


def test_a_state_that_cannot_be_moved_aside_refuses_the_start(tmp_path, monkeypatch):
    st = _state(tmp_path, monkeypatch)
    open(st, "wb").write(b"not a pickle")

    def no(*a):
        raise OSError(13, "Permission denied")
    monkeypatch.setattr(serve.os, "replace", no)
    with pytest.raises(SystemExit):
        serve.load_engine(None, None)
    assert open(st, "rb").read() == b"not a pickle"


def test_fresh_keeps_the_saved_state_unless_told_to_replace_it(tmp_path, monkeypatch):
    st = _state(tmp_path, monkeypatch)
    assert serve.saves_state(replaying=False, fresh=True, replace_state=False)          # nothing saved yet
    open(st, "wb").write(b"x")
    assert not serve.saves_state(replaying=False, fresh=True, replace_state=False)
    assert serve.saves_state(replaying=False, fresh=True, replace_state=True)
    assert serve.saves_state(replaying=False, fresh=False, replace_state=False)
    assert not serve.saves_state(replaying=True, fresh=False, replace_state=True)


def test_each_save_keeps_the_previous_state(tmp_path, monkeypatch):
    _state(tmp_path, monkeypatch)
    eng = Engine()
    serve.save_engine(eng)
    room(eng)
    serve.save_engine(eng)
    prev = pickle.load(open(serve.STATE + ".prev", "rb"))
    assert len(prev.users) == 0 and _saved_users() > 0
    assert not [f for f in os.listdir(tmp_path / "state") if ".tmp" in f]


# ------------------------------------------------------------------ LB-24: the autosave pickles in a child process
def test_the_autosave_pickles_off_the_loop_in_a_child(tmp_path, monkeypatch):
    _state(tmp_path, monkeypatch)
    eng = Engine()
    t = room(eng)
    for i in range(400):
        eng.process(msg(f"chatter{i}", "hello there chat", t + i, uid=str(80_000_000 + i)))
    on_loop = []
    real = pickle.dumps

    def spy(*a, **kw):
        on_loop.append(os.getpid())
        return real(*a, **kw)
    monkeypatch.setattr(serve.pickle, "dumps", spy)

    async def go():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0)
                ticks += 1
        tk = asyncio.create_task(ticker())
        await serve.save_engine_async(eng, fork=True)
        tk.cancel()
        return ticks
    assert asyncio.run(go()) > 0
    assert os.getpid() not in on_loop                     # the parent never pickled
    assert _saved_users() == len(eng.users) and eng.risk is not None


def test_a_failed_save_child_leaves_the_state_as_it_was(tmp_path, monkeypatch):
    _state(tmp_path, monkeypatch)
    eng = Engine()
    room(eng)
    serve.save_engine(eng)
    n = _saved_users()
    eng.users = {"unpicklable": threading.Lock()}         # the child's pickle fails
    with pytest.raises(OSError):
        asyncio.run(serve.save_engine_async(eng, fork=True))
    assert _saved_users() == n
    assert not [f for f in os.listdir(tmp_path / "state") if ".tmp" in f]


# ------------------------------------------------------------------ LB-16: the lock is taken by the server, atomically
def test_the_lock_is_atomic_and_only_its_owner_removes_it(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "LOCK", str(tmp_path / "dashboard.lock"))
    monkeypatch.setenv("CW_HOST", "desk")
    mine = serve.take_lock(8766)
    assert mine.split()[0] == "desk" and mine.split()[3] == "8766"
    with pytest.raises(SystemExit):
        serve.take_lock(8765)                             # a second dashboard
    monkeypatch.setenv("CW_HOST", "laptop")
    theirs = serve.take_lock(8765, force=True)            # CW_FORCE=1 on another machine
    serve.release_lock(mine)                              # the first one exits later: not its lock any more
    assert open(serve.LOCK).read() == theirs
    serve.release_lock(theirs)
    assert not os.path.exists(serve.LOCK)


# ------------------------------------------------------------------ LB-8, LB-15: session copies and per-session stats
def test_only_the_newest_sessions_keep_a_state_copy_and_stats_are_per_session(tmp_path, monkeypatch):
    st = tmp_path / "engine.pkl"
    monkeypatch.setattr(session, "STATE", str(st))
    eng, base = Engine(), str(tmp_path / "sessions")
    made = []
    for i in range(5):
        st.write_bytes(f"state {i}".encode())             # every Stop saves a new state
        s = session.Session([f"c{i}"], [], eng, base=base)
        eng.stats["msgs"] += 10
        s.close(eng.stats)
        made.append(s)
        time.sleep(1.05)                                   # session folder names are per second
    copies = sorted(os.path.basename(s.dir) for s in made if os.path.exists(os.path.join(s.dir, "state_at_start.pkl")))
    assert copies == sorted(os.path.basename(s.dir) for s in made[-session.STATE_COPIES_KEPT:])
    meta = json.load(open(os.path.join(made[-1].dir, "meta.json")))
    assert meta["stats"] == {"msgs": 10} and meta["stats_lifetime"]["msgs"] == 50
    assert meta["state_sha256"] and open(os.path.join(made[-1].dir, "state_at_start.pkl"), "rb").read() == b"state 4"
    assert "feedback.jsonl" not in os.listdir(made[-1].dir)


def test_starting_a_session_does_not_copy_the_state_on_the_loop(tmp_path, monkeypatch):
    st = tmp_path / "engine.pkl"
    st.write_bytes(b"state")
    monkeypatch.setattr(session, "STATE", str(st))
    where = []
    real = session.Session._state_at_start

    def spy(self):
        where.append(threading.current_thread() is threading.main_thread())
        return real(self)
    monkeypatch.setattr(session.Session, "_state_at_start", spy)
    s = session.Session(["c"], [], Engine(), base=str(tmp_path / "sessions"))
    s.close(Engine().stats)
    assert where == [False]


# ------------------------------------------------------------------ LB-5: a client that stops reading is closed
def test_a_browser_that_misses_a_send_is_closed_so_it_reconnects(monkeypatch):
    monkeypatch.setattr(serve, "SEND_TIMEOUT_S", 0.05)
    hub = Hub(Engine())
    closed, aborted = [], []

    class Stuck:
        async def send_str(self, data):
            await asyncio.Event().wait()

        async def close(self, code=None, message=b""):
            closed.append(code)

    class Hung(Stuck):
        async def close(self, code=None, message=b""):
            await asyncio.Event().wait()                  # its socket buffer is full: the close frame cannot go

    class Transport:
        def abort(self):
            aborted.append(1)
    stuck, hung = Stuck(), Hung()
    hub.clients = {stuck, hung}
    hub.transports = {hung: Transport()}
    hub.queue = [{"type": "msg", "id": "m1"}]
    hub._last_room = time.time()

    async def go():
        await hub._flush()
        await asyncio.sleep(0.3)
    asyncio.run(go())
    assert hub.clients == set() and closed == [1013] and aborted == [1]


def test_a_real_page_socket_is_closed_by_the_server_when_dropped(monkeypatch):
    hub = Hub(Engine())

    async def calls(c):
        ws = await c.ws_connect("/ws")
        first = await ws.receive_json()
        server_ws = next(iter(hub.clients))
        hub._drop_client(server_ws)
        msg = await asyncio.wait_for(ws.receive(), 5)
        return first[0]["type"], msg.type.name, ws.close_code
    kind, mtype, code = _http(hub, calls)
    assert kind == "status" and mtype in ("CLOSE", "CLOSED", "CLOSING") and code == 1013


# ------------------------------------------------------------------ LB-6: "Stop now" ends the LLM work it queued
def test_stop_now_drops_queued_llm_work_and_does_not_apply_a_late_read(monkeypatch):
    """A plain Stop finishes the queue (test_stop_finishes_the_queue_before_the_session_closes); "Stop now" drops
    the rest, counted (one per read, one per line), and a read already running is not applied (LB-6)."""
    eng = Engine()
    eng.person_review = True
    t = room(eng)
    hub = Hub(eng)
    hub.status["mode"] = "live"
    hub.review_q = ReadQueue()
    hub.tox_per_min = 1
    serve.handle(hub, msg("first", "you are trash", t))
    serve.handle(hub, msg("second", "ur bad lol", t + 1))
    u1, u2 = (next(u for u in eng.users.values() if u.login == n) for n in ("first", "second"))
    assert u1.review["pending"] and u2.review["pending"] and hub.tox_pending
    lines = len(hub.tox_pending)
    started, release = threading.Event(), threading.Event()

    def slow_review(p, ctx, emotes=None):
        started.set()
        release.wait(5)
        return {"label": "bad_faith", "rationale": "late"}
    monkeypatch.setattr(llm, "review", slow_review)

    async def go():
        _, _, kind, a = hub.review_q.get_nowait()        # the reviewer took the first read ...
        task = asyncio.create_task(hub._review_one(kind, a))
        while not started.is_set():
            await asyncio.sleep(0.01)
        busy = hub._status()["llm_busy"]
        await hub.stop_live(now=True)                     # ... and "Stop now" comes while it runs
        release.set()
        await task
        return busy
    assert asyncio.run(go())
    assert hub.review_q.empty() and not hub.tox_pending and not hub.person_queued
    assert u1.review.get("label") is None and not u1.review["pending"]     # the late read was not applied
    assert u2.review["reads"] == 0 and not u2.review["pending"]            # released, not spent
    assert eng.stats["llm_person_errors"] == 0 and eng.stats["llm_late_dropped"] == 1
    assert eng.stats["llm_dropped_at_stop"] == 1 + lines and not hub._status()["llm_busy"]


# ------------------------------------------------------------------ LB-10, test gap 11: LLM endpoints are POSTs
def test_llm_endpoints_need_the_page_and_answer_before_any_event(monkeypatch):
    eng = Engine()
    t = room(eng)
    rec = eng.process(msg("heckler", "you are trash", t))
    hub = Hub(eng)                                       # no handle() yet, as right after a restart: clock set here
    hub.status["mode"] = "live"
    logged = []
    hub._log = logged.append
    monkeypatch.setattr(llm, "review", lambda p, ctx, emotes=None: {"label": "mild", "rationale": "jabs",
                                                                     "confidence": "low", "rules": []})
    monkeypatch.setattr(llm, "draft_warning", lambda p, ctx, emotes=None: {"warranted": True, "warning": "@heckler pls",
                                                                           "why": "insult"})
    who = {"channel": "cozy", "uid": rec["user_id"]}

    async def calls(c):
        out = [(await c.get(f"/api/llm_review?channel=cozy&uid={rec['user_id']}")).status,
               (await c.post("/api/llm_review", json=who)).status]               # no X-Chatwatch: a cross-site form
        r = await c.post("/api/llm_review", json=who, headers=PAGE)
        out.append((r.status, (await r.json())["label"]))
        r = await c.post("/api/draft_warning", json=who, headers=PAGE)
        out.append((r.status, (await r.json())["warning"]))
        return out
    assert _http(hub, calls) == [405, 403, (200, "mild"), (200, "@heckler pls")]
    assert [x["type"] for x in logged] == ["draft_warning"] and logged[0]["ts"] == 0


# ------------------------------------------------------------------ LB-11: the backoff resets on chat only
def test_a_server_that_accepts_and_closes_is_retried_ever_more_slowly(monkeypatch):
    hub = Hub(Engine())
    pauses = []

    async def pause(s):
        pauses.append((s, hub.status["connected"]))
    monkeypatch.setattr(serve, "_pause", pause)
    welcome = ":tmi.twitch.tv 001 justinfan123 :Welcome, GLHF!\r\n:tmi.twitch.tv CAP * ACK :twitch.tv/tags"

    async def go():
        from test_dashboard import _closed_ok
        done = asyncio.Event()
        conns = [FakeIRC(welcome, _closed_ok()) for _ in range(5)]
        _fake_connect(monkeypatch, conns, done.set)
        task = asyncio.create_task(serve._live_loop(hub, ["cozy"], None, None))
        await asyncio.wait_for(done.wait(), 5)
        task.cancel()
    asyncio.run(go())
    assert [p for p, _ in pauses][:5] == [1, 2, 4, 8, 16]                  # the welcome lines reset nothing
    assert not any(c for _, c in pauses)


def test_connected_means_the_server_answered(monkeypatch):
    hub = Hub(Engine())
    conn = FakeIRC()

    async def recv(self):                                  # nothing until a line is scripted
        while not self.script:
            await asyncio.sleep(0.01)
        return self.script.pop(0)
    monkeypatch.setattr(FakeIRC, "recv", recv)
    _fake_connect(monkeypatch, [conn], lambda: None)

    async def go():
        task = asyncio.create_task(serve._live_loop(hub, ["cozy"], None, None))
        await asyncio.sleep(0.05)
        before = hub.status["connected"]
        conn.script.append(":tmi.twitch.tv 001 justinfan1 :Welcome")
        await asyncio.sleep(0.05)
        task.cancel()
        return before
    assert asyncio.run(go()) is False and hub.status["connected"] is True


# ------------------------------------------------------------------ LB-14: one person read per chatter in the queue
def test_a_chatter_waiting_in_the_queue_is_not_queued_twice():
    eng = Engine()
    eng.person_review = True
    t = room(eng)
    hub = Hub(eng)
    hub.review_q = ReadQueue()
    serve.handle(hub, msg("heckler", "you are trash", t))
    serve.handle(hub, msg("heckler", "worst streamer ever, you are trash", t + 11 * 60))   # pending > 10 min
    kinds = [hub.review_q.get_nowait()[2] for _ in range(hub.review_q.qsize())]
    assert kinds == ["person"] and eng.stats["llm_person_requeue_skipped"] == 1
    u = next(u for u in eng.users.values() if u.login == "heckler")
    assert u.review["pending"] and u.review["queued"] == T0 + (t + 11 * 60) * 1000          # renewed


# ------------------------------------------------------------------ the lossless queue's windows (was LB-18's batcher)
def _item(ch, ts, prio=0, age=60.0):
    return (prio, ts, {"channel": ch, "user_id": f"u{ts}", "id": f"m{ts}", "line": f"x: line {ts}", "room": [],
                       "prio": prio, "ts": ts, "q_at": time.time() - age})


def test_a_window_takes_the_best_oldest_lines_of_one_channel():
    """The head is the best line (priority, then oldest: the H4 queue order); the window takes up to 12 lines of
    its channel in that order, sent in chat order. The rest waits; nothing is dropped."""
    hub = Hub(Engine())
    hub.review_q = ReadQueue()
    hub._q_cost = {"solo": 3.2}                     # a backlog: 6 lines go alone at most (SOLO_CLEAR_S 20 s)
    hub.tox_pending = [_item("a", ts, 0) for ts in range(20)] + [_item("b", 100 + ts, 1) for ts in range(5)]
    job = hub._next_job(time.time())
    assert job["kind"] == "toxwin" and job["a"]["channel"] == "a"
    assert [it["id"] for it in job["a"]["items"]] == [f"m{ts}" for ts in range(12)]        # oldest 12, chat order
    assert sorted(x[1] for x in hub.tox_pending) == list(range(12, 20)) + list(range(100, 105))
    hub.tox_pending = [_item("a", ts, 0) for ts in range(1000)]                            # a long backlog
    served = []
    while hub.tox_pending:
        served += [it["ts"] for it in hub._next_job(time.time())["a"]["items"]]
    assert served == list(range(1000)) and not hub.eng.stats.get("llm_tox_skipped")       # every line, in order
    mixed = [_item("a", ts, ts % 4) for ts in range(40)]
    hub.tox_pending = list(mixed)
    win = hub._next_job(time.time())["a"]["items"]
    assert sorted(it["prio"] for it in win) == [0] * 10 + [1] * 2                         # best priority first


def test_reads_go_before_lines_and_lines_wait_briefly_for_company():
    hub = Hub(Engine())
    hub.review_q = ReadQueue()
    hub._q_cost = {"solo": 10.0}                    # 2 lines go alone at most: 3 are a backlog (windows)
    now = time.time()
    hub.tox_pending = [_item("a", ts, 0, age=0) for ts in range(3)]                         # fresh: they gather
    assert hub._next_job(now) is None
    hub._enqueue(0, "person", {"channel": "a", "user_id": "p"})
    assert hub._next_job(now)["kind"] == "person"                                           # a read: at once
    assert hub._next_job(now + serve.LINE_GATHER_S + 0.1)["kind"] == "toxwin"               # then the lines
    hub.tox_pending = [_item("a", ts, 0, age=0) for ts in range(serve.WINDOW)]
    assert len(hub._next_job(now)["a"]["items"]) == serve.WINDOW                            # a full window: at once
    hub.tox_pending = [_item("a", 1, 0, age=0)]
    hub.finishing = True                                                                     # Stop: no gather
    assert hub._next_job(now)["kind"] == "toxwin"
    hub._enqueue(1, "screen", {"channel": "a", "user_id": "s"})
    assert hub._next_job(time.time()) is None                                                # a screen waits 1 s
    assert hub._next_job(time.time() + serve.SCREEN_DELAY_S + 0.1)["kind"] == "screen"
    hub.llm_open = False
    hub._enqueue(0, "person", {"channel": "a", "user_id": "p"})
    assert hub._next_job(time.time()) is None                                                # live, before Start


def test_only_low_priority_lines_age_out_and_are_counted():
    """Decided 2026-11-06 from H4: priority 1-3 re-reads older than 15 min age out, counted (the page shows it).
    Flagged / coded lines (0) and every read never do."""
    hub = Hub(Engine())
    hub.review_q = ReadQueue()
    old, new = serve.LINE_AGE_OUT_S + 5, 30
    hub.tox_pending = [_item("a", 1, 0, old), _item("a", 2, 1, old), _item("a", 3, 3, old), _item("a", 4, 2, new)]
    hub.review_q.put(0, 1, "person", {"channel": "a", "user_id": "p"}, at=time.time() - 3 * serve.LINE_AGE_OUT_S)
    assert hub._age_out(time.time()) == 2
    assert [x[1] for x in hub.tox_pending] == [1, 4] and hub.review_q.qsize() == 1
    assert hub.eng.stats["llm_lines_aged_out"] == 2 and hub._llm_status()["aged_out"] == 2


# ------------------------------------------------------------------ LF-21: a fast replay leaves the server responsive
def test_a_replay_faster_than_the_server_lets_the_loop_run(monkeypatch):
    from chatwatch import replay
    hub = Hub(Engine())
    hub.eng.clock = 0
    evs = [msg(f"c{i % 50}", "hello there chat", i) for i in range(1000)]
    monkeypatch.setattr(replay, "iter_events", lambda paths, chans: iter(evs))

    async def go():
        ticks = []

        async def ticker():
            while True:
                await asyncio.sleep(0)
                ticks.append(hub.eng.stats["msgs"])
        tk = asyncio.create_task(ticker())
        await serve.replay_source(hub, ["x"], [], 0)
        tk.cancel()
        return ticks
    ticks = asyncio.run(go())
    assert any(0 < n < 1000 for n in ticks) and hub.status["mode"] == "replay finished"


# ------------------------------------------------------------------ contract: what the page gets
def test_the_page_gets_slim_records_and_each_alert_once():
    eng = Engine()
    t = room(eng)
    hub = Hub(eng)
    serve.handle(hub, msg("newbie", "check out my channel twitch.tv/xx_free_followers you are trash", t,
                          first=True, uid="1567400000"))
    serve.handle(hub, {"kind": "usernotice", "msg_id": "raid", "ts": T0 + (t + 1) * 1000, "channel": "cozy",
                       "login": "raider", "viewers": 5, "user_id": "9", "room_id": "", "tags": {}})
    m = next(r for r in hub.queue if r["type"] == "msg")
    assert set(m) <= set(serve.UI_MSG) and "alerts" not in m and "risk" not in m
    assert not [r for r in hub.queue if r["type"] == "room_event"]


def test_a_bot_wave_shows_one_coordinated_banner_per_text():
    eng = Engine(legacy=False)
    t = room(eng)
    hub = Hub(eng)
    logged = []
    hub._log = logged.append
    for i in range(8):
        serve.handle(hub, msg(f"x{i}qz{i}", "this channel is a joke and so are you", t + i, first=True,
                              uid=str(1567400000 + i)))
    shown = [r for r in hub.queue if r.get("kind") == "coordinated"]
    assert len(shown) == 1 and len([r for r in logged if r.get("kind") == "coordinated"]) >= 4


def test_a_stopped_live_run_sends_no_room_snapshots():
    eng = Engine()
    t = room(eng)
    hub = Hub(eng)
    hub.status.update(mode="live", channels=["cozy"], running=True)
    sent = []

    class Client:
        async def send_str(self, data):
            sent.extend(json.loads(data))
    hub.clients = {Client()}
    asyncio.run(hub._flush())
    snap = next(r for r in sent if r["type"] == "room")
    assert "vocabulary" not in snap and "friction_now" not in snap and "notices" in snap   # lean for the page
    asyncio.run(hub.stop_live())
    sent.clear()
    hub._last_room = 0
    asyncio.run(hub._flush())
    st = [r for r in sent if r["type"] == "status"][0]
    assert not [r for r in sent if r["type"] == "room"] and st["stopped_at"] and st["channels"] == []


def test_a_person_read_reaches_an_open_user_card(monkeypatch):
    eng = Engine()
    eng.person_review = True
    t = room(eng)
    rec = eng.process(msg("heckler", "you are trash", t))
    hub = Hub(eng)
    monkeypatch.setattr(llm, "review", lambda p, ctx, emotes=None: {"label": "bad_faith", "rationale": "insults"})
    eng.mark_review_pending("cozy", rec["user_id"])
    asyncio.run(hub._review_one("person", {"channel": "cozy", "user_id": rec["user_id"]}))
    pr = [r for r in hub.queue if r["type"] == "person_read"]
    assert pr == [{"type": "person_read", "channel": "cozy", "user_id": rec["user_id"], "ts": 0,
                   "label": "bad_faith", "rationale": "insults"}]


def test_llm_calls_run_in_daemon_threads():
    async def go():
        return await serve.in_thread(lambda: (threading.current_thread().daemon, threading.current_thread().name))
    assert asyncio.run(go()) == (True, "cw-llm")


def test_start_clears_the_pile_on_screen_memory(monkeypatch):
    hub = Hub(Engine())
    hub.log_sessions = False
    hub.screened.add(("cozy", "1"))

    async def nothing(*a):
        return None
    monkeypatch.setattr(serve, "live_source", nothing)

    async def go():
        hub.start_live(["cozy"])
    asyncio.run(go())
    assert not hub.screened


# ------------------------------------------------------------------ RESPONSIBLE_USE (mismatch 8): forget covers everything
def _forget_room():
    eng = Engine()
    t = room(eng)
    eng.process(msg("trollking", "you are a nigger", t, uid="777"))
    eng.process(msg("trollking", "kys idiot streamer", t + 2, uid="777"))
    eng.process({"kind": "clearchat", "ts": T0 + (t + 3) * 1000, "channel": "cozy", "room_id": "",
                 "target_login": "trollking", "target_user_id": "777", "duration": -1})
    eng.process(msg("trollking2", "hi", t + 30, first=True, uid="1567400000"))   # lookalike of a banned name
    eng.process({**msg("trollking", "goyim again", t + 40, uid="777"), "channel": "other"})
    return eng


def _mentions(eng, needle="trollking"):
    """Every place in the engine that still holds the forgotten chatter's id or name."""
    found = set()
    for (ch, uid), u in eng.users.items():
        if uid == "777":
            found.add("users")
    for k in eng.by_login:
        if k[1] == needle:
            found.add("by_login")
    for a in eng.alerts:
        if a.get("user_id") == "777" or a.get("login") == needle or a.get("resembles") == needle:
            found.add("alerts")
    for ch, r in eng.rooms.items():
        if needle in eng.names.get(ch, ()):
            found.add("names")
        if any(x[1].lower() == needle for x in r.context):
            found.add("context")
        if needle in r.vocab.chatters or any(needle in s for s in r.vocab.users.values()):
            found.add("vocab")
        if needle in r.modlearn.last_terms or any(needle in s for s in r.modlearn.users.values()):
            found.add("modlearn")
        if any(x[1] == needle for x in r.banned):
            found.add("banned")
        if any(x[1] == needle for dq in list(r.shapes.values()) + list(r.chants.values()) for x in dq):
            found.add("shapes/chants")
    if "777" in eng.ids.seen or 777 in eng.ids.ids:
        found.add("ids")
    if eng.legacy and any("777" in lg.profiles for lg in eng.legacy.values()):
        found.add("legacy")
    return found


def test_forget_leaves_nothing_about_the_chatter():
    eng = _forget_room()
    assert {"users", "by_login", "alerts", "names", "context", "banned", "ids", "legacy"} <= _mentions(eng)
    others = len(eng.users)
    assert eng.forget("777") == 2
    assert _mentions(eng) == set() and len(eng.users) == others - 2
    assert eng.room_snapshot("cozy", T0 + 10**6) is not None                  # the rooms still work


def test_the_forget_tool_covers_every_saved_copy(tmp_path, monkeypatch):
    from chatwatch import forget
    root = tmp_path
    eng = _forget_room()
    eng.risk = eng.persist = eng.ml = eng.emotes = None
    blob = pickle.dumps(eng)
    paths = [root / "data" / "state" / "engine.pkl", root / "data" / "state" / "engine.pkl.prev",
             root / "data" / "sessions" / "20261029-100000_cozy" / "state_at_start.pkl"]
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(blob)
    raw = root / "data" / "sessions" / "20261029-100000_cozy" / "raw" / "2026-10-29"
    raw.mkdir(parents=True)
    (raw / "cozy.log").write_text("1 :trollking!trollking@trollking.tmi.twitch.tv PRIVMSG #cozy :hi\n")
    forget.main(["777"], root=str(root))                                       # dry run
    assert all(p.read_bytes() == blob for p in paths)
    (root / "data" / "state" / "dashboard.lock").write_text("desk 1 x 8765\n")
    with pytest.raises(SystemExit):
        forget.main(["777", "--go"], root=str(root))
    (root / "data" / "state" / "dashboard.lock").unlink()
    forget.main(["777", "--go"], root=str(root))
    for p in paths:
        assert _mentions(pickle.loads(p.read_bytes()).upgrade()) == set()
    assert forget.log_mentions({"trollking"}, str(root)) == {"20261029-100000_cozy": 1}


# ------------------------------------------------------------------ LB-19, LB-22, LF-13
def test_card_lists_are_bounded_but_callouts_are_all_kept():
    from chatwatch import engine as E
    eng = Engine()
    t = room(eng)
    rec = eng.process(msg("heckler", "you are trash", t))
    u = eng.users[("cozy", rec["user_id"])]
    for i in range(3):
        u.reactions.append((i, "callout", "x"))
    for i in range(300):
        eng.process({"kind": "clearchat", "ts": T0 + (t + 1 + i) * 1000, "channel": "cozy", "room_id": "",
                     "target_login": "heckler", "target_user_id": rec["user_id"], "duration": 1})
    assert len(u.mod_actions) == E.MOD_ACTIONS_KEPT and len(u.reactions) == E.REACTIONS_KEPT
    assert sum(r[1] == "callout" for r in u.reactions) == 3


def test_the_card_has_no_instruction_field_and_group_members_keep_their_reason():
    eng = Engine()
    t = room(eng)
    a = eng.process(msg("one", "you are a nigger lol", t))
    eng.process(msg("two", "you are a nigger lol", t + 1))
    assert "suggestion" not in eng.profile("cozy", a["user_id"])
    card = eng.notices("cozy", T0 + (t + 2) * 1000)[0]["items"][0]
    assert len(card["group"]) == 2 and all(g["why"] for g in card["group"])


# ------------------------------------------------------------------ LB-2 (compliance, BASELINE 8.1), test gap 4
def test_the_llm_person_read_is_not_told_about_mod_actions():
    """Pins the prompt's signals: mod actions are never evidence, and this read decides 'possible'."""
    eng = Engine()
    t = room(eng)
    rec = eng.process(msg("heckler", "you are trash", t))
    eng.process({"kind": "clearchat", "ts": T0 + (t + 1) * 1000, "channel": "cozy", "room_id": "",
                 "target_login": "heckler", "target_user_id": rec["user_id"], "duration": 600})
    p = eng.profile("cozy", rec["user_id"])
    assert any(r[1] == "timeout" for r in p["reactions"])        # the card still shows them, as context
    text = llm._content(p, list(eng.rooms["cozy"].context))
    signals = text.split("### Signals\n")[1].splitlines()
    assert [s.split()[0] for s in signals] == ["messages", "first-ever", "account", "subscribed", "started"]
    assert "mod action" not in text.lower() and "timed out" not in text.lower()
