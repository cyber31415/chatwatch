"""The live dashboard (serve.py and the engine's live-only paths), review phase 2 batch 2
(docs/review/live.md). No network, no LLM: the IRC websocket and the LLM are fakes, and the
HTTP tests run the app on a local aiohttp test server."""
import asyncio
import gc
import json
import os
import pickle
import threading
import time

import pytest

from chatwatch import llm, serve
from chatwatch.engine import Engine
from chatwatch.serve import Hub, ReadQueue

from test_scenarios import T0, msg, room


def _restart(eng):
    """What save_engine + load_engine do to the state across a dashboard restart."""
    eng.risk = eng.ml = eng.emotes = eng.persist = None
    return pickle.loads(pickle.dumps(eng)).upgrade()


# ------------------------------------------------------------------ F6: stale pending person reads
def test_a_pending_person_read_does_not_survive_a_restart():
    eng = Engine()
    eng.person_review = True
    t = room(eng)
    rec = eng.process(msg("heckler", "you are trash", t))
    assert eng.wants_person_review(rec)
    eng.mark_review_pending(rec["channel"], rec["user_id"])        # queued, then the dashboard stops
    eng = _restart(eng)
    eng.person_review = True
    rec = eng.process(msg("heckler", "worst streamer ever, you are trash", t + 60))
    assert eng.wants_person_review(rec)


def test_a_pending_person_read_older_than_ten_minutes_is_not_pending():
    eng = Engine()
    eng.person_review = True
    t = room(eng)
    rec = eng.process(msg("heckler", "you are trash", t))
    eng.mark_review_pending(rec["channel"], rec["user_id"])
    rec = eng.process(msg("heckler", "you are trash lol", t + 60))
    assert not eng.wants_person_review(rec)                         # still queued: no second read
    rec = eng.process(msg("heckler", "worst streamer ever, you are trash", t + 11 * 60))
    assert eng.wants_person_review(rec)                             # the read was lost somewhere
    assert eng.users[(rec["channel"], rec["user_id"])].review["reads"] == 0


# ------------------------------------------------------------------ F7: errors in the background tasks
class FullDisk:
    """A session whose every write fails, as on the full CIFS share."""
    dir = "/nowhere/20261027-000000_cozy"

    def __init__(self):
        self.closed = False

    def log(self, rec):
        raise OSError(28, "No space left on device")

    flush = raw_line = log

    def close(self, stats):
        self.closed = True


def _http(hub, calls):
    """Run calls(client) against the dashboard's app on a local test server."""
    from aiohttp.test_utils import TestClient, TestServer, unused_port
    port = unused_port()

    async def go():
        async with TestClient(TestServer(serve.make_app(hub, port), host="127.0.0.1", port=port)) as client:
            return await calls(client)
    return asyncio.run(go())


def test_a_failing_session_log_ends_no_loop_and_the_page_still_gets_the_record():
    eng = Engine()
    t = room(eng)
    eng.clock = 0                                    # as serve.main sets it
    hub = Hub(eng)
    hub.session = FullDisk()
    hub.status["channels"] = ["cozy"]

    async def go():
        task = asyncio.create_task(hub.flusher())
        await asyncio.sleep(0.6)                     # the first flush logs a room snapshot
        alive = not task.done()
        serve.handle(hub, msg("viewer", "hello there", t))
        got = [r["text"] for r in hub.queue if r.get("type") == "msg"]
        task.cancel()
        return alive, got
    assert asyncio.run(go()) == (True, ["hello there"])
    assert eng.stats["session_log_errors"] >= 2 and not eng.stats["event_errors"]


def test_an_error_in_a_read_does_not_end_the_reviewer(monkeypatch):
    eng = Engine()
    eng.person_review = True
    t = room(eng)
    rec = eng.process(msg("heckler", "you are trash", t))
    monkeypatch.setattr(llm, "review", lambda p, ctx, emotes=None: {"label": "mild", "rationale": "jabs"})
    real, calls = eng.apply_person_review, []

    def flaky(*a):
        calls.append(a)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return real(*a)
    eng.apply_person_review = flaky
    eng.clock = 0                                    # as serve.main sets it
    hub = Hub(eng)

    async def go():
        hub.review_q = ReadQueue()
        task = asyncio.create_task(hub.reviewer())
        for _ in range(2):
            hub._enqueue(0, "person", {"channel": "cozy", "user_id": rec["user_id"]})
        for _ in range(50):
            await asyncio.sleep(0.02)
            if len(calls) == 2:
                break
        alive = not task.done()
        task.cancel()
        return alive
    assert asyncio.run(go())
    assert eng.stats["reviewer_errors"] == 1 and eng.stats["llm_person_reads"] == 1


def test_an_error_in_a_batch_does_not_end_the_tox_batcher():
    hub = Hub(Engine())
    hub.tox_pending = [(1, 1, None)]                 # malformed: its age cannot be read

    async def go():
        task = asyncio.create_task(hub.tox_batcher(every=0.01))
        await asyncio.sleep(0.1)
        alive = not task.done()
        task.cancel()
        return alive
    assert asyncio.run(go())
    assert hub.eng.stats["tox_batcher_errors"] == 1


def test_stop_closes_the_session_before_saving_and_survives_both_failing(monkeypatch):
    order = []

    class Session(FullDisk):
        def close(self, stats):
            order.append("close")
            raise OSError(5, "Input/output error")

    def save(seq, data):
        order.append("save")
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(serve, "write_state", save)
    hub = Hub(Engine())
    hub.persist = True                                # a live run that loaded the saved state
    hub.session = Session()
    asyncio.run(hub.stop_live())
    assert order == ["close", "save"] and hub.session is None
    assert hub.eng.stats["session_close_errors"] == 1 and hub.eng.stats["save_errors"] == 1


def test_a_session_that_cannot_be_written_gives_a_clean_error_on_start(monkeypatch):
    def no_disk(*a, **kw):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(serve, "Session", no_disk)
    hub = Hub(Engine())
    hub.status["mode"] = "live"

    async def calls(c):
        r = await c.post("/api/start", json={"channels": "somechannel"}, headers={"X-Chatwatch": "1"})
        return r.status, (await r.json())["error"]
    status, err = _http(hub, calls)
    assert status == 503 and err.startswith("could not open the session log")
    assert hub.live_task is None and not hub.status["running"]


# ------------------------------------------------------------------ F8: only the page may use the server
def test_foreign_origin_host_or_missing_header_is_refused(monkeypatch):
    started = []
    hub = Hub(Engine())
    hub.status["mode"] = "live"
    monkeypatch.setattr(hub, "start_live", lambda chans: started.append(chans))

    async def calls(c):
        own = f"http://127.0.0.1:{c.server.port}"
        page = {"X-Chatwatch": "1"}
        start = {"channels": "somechannel"}
        out = [
            (await c.post("/api/start", json=start, headers={**page, "Origin": "http://evil.example"})).status,
            (await c.post("/api/start", json=start)).status,                              # no custom header
            (await c.post("/api/start", data='{"channels": "x"}', headers={"Content-Type": "text/plain",
                                                                            "Origin": "http://evil.example"})).status,
            (await c.post("/api/stop", headers={"Origin": "http://evil.example"})).status,
            (await c.get("/api/status", headers={"Host": f"rebind.example:{c.server.port}"})).status,
            (await c.get("/api/status", headers={"Origin": "http://evil.example"})).status,
            (await c.post("/api/feedback", json={}, headers=page)).status,                # removed: rule 8.1
            (await c.post("/api/msg_feedback", json={}, headers=page)).status,
        ]
        refused_ws = False
        try:
            ws = await c.ws_connect("/ws", headers={"Origin": "http://evil.example"})
            await ws.close()
        except Exception:
            refused_ws = True
        out.append(refused_ws)
        assert not started
        out.append((await c.post("/api/start", json=start, headers={**page, "Origin": own})).status)
        ws = await c.ws_connect("/ws", headers={"Origin": own})
        out.append((await ws.receive_json())[0]["type"])
        await ws.close()
        out.append((await c.get("/api/status", headers={"Host": f"localhost:{c.server.port}"})).status)
        out.append((await c.get("/api/status", headers={"Host": f"cw-live:{c.server.port}"})).status)  # jobs.gpu_busy
        return out
    assert _http(hub, calls) == [403, 403, 403, 403, 403, 403, 404, 404, True, 200, "status", 200, 200]
    assert started == [["somechannel"]]


# ------------------------------------------------------------------ F9: long runs (GC, saving off the loop)
def test_the_restored_state_is_frozen_out_of_the_collector():
    th = gc.get_threshold()
    try:
        serve.tune_gc()
        assert gc.get_freeze_count() > 0 and gc.get_threshold()[2] >= 1000
    finally:
        gc.unfreeze()
        gc.set_threshold(*th)


def test_the_state_is_written_off_the_loop_through_a_temp_file(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "STATE", str(tmp_path / "state" / "engine.pkl"))
    eng = Engine()
    room(eng)
    on_main, real = [], serve.write_state

    def spy(seq, data):
        on_main.append(threading.current_thread() is threading.main_thread())
        return real(seq, data)
    monkeypatch.setattr(serve, "write_state", spy)
    asyncio.run(serve.save_engine_async(eng))
    assert on_main == [False]
    back = pickle.load(open(serve.STATE, "rb")).upgrade()
    assert list(back.users) == list(eng.users) and eng.risk is not None       # models re-attached
    assert not os.path.exists(serve.STATE + ".tmp")


def test_an_older_dump_never_replaces_a_newer_save(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "STATE", str(tmp_path / "engine.pkl"))
    eng = Engine()
    old = serve.dump_engine(eng)
    room(eng)
    new = serve.dump_engine(eng)
    assert serve.write_state(*new) and not serve.write_state(*old)
    assert len(pickle.load(open(serve.STATE, "rb")).users) == len(eng.users) > 0


def test_room_snapshot_reads_a_per_channel_index_with_the_same_items():
    eng = Engine()
    t = room(eng)
    chans = ("cozy", "other", "third")
    for i in range(90):                          # the same chatters in several channels, some toxic
        ch = chans[i % 3]
        text = ["you are trash", "nice play", "you are a nigger", "gg", "worst streamer ever, you are trash"][i % 5]
        eng.process({**msg(f"chatter{i % 17}", text, t + i * 7, uid=str(70_000_000 + i % 17)), "channel": ch})

    def same_as_a_full_scan():
        for ts in (T0 + (t + 700) * 1000, T0 + (t + 4000) * 1000):          # cards live, then earlier
            for ch in chans:
                us = [u for (c, _), u in eng.users.items() if c == ch]     # what room_snapshot scanned before
                snap = eng.room_snapshot(ch, ts)
                assert snap["chatters"] == len(us)
                assert (snap["notices"], snap["notices_earlier"]) == eng.notices(ch, ts, us)
        return True
    assert same_as_a_full_scan() and eng.room_snapshot("other", T0 + (t + 700) * 1000)["notices"]["items"]
    eng.forget("70000003")
    assert same_as_a_full_scan()
    del eng.users[("third", "70000005")]                                   # as coldstore prunes
    assert same_as_a_full_scan()
    eng.process({**msg("latecomer", "you are trash", t + 690, uid="70000099"), "channel": "other"})
    assert same_as_a_full_scan()
    eng = _restart(eng)
    assert same_as_a_full_scan()

    calls = []

    class Counting(dict):
        def items(self):
            calls.append(1)
            return super().items()
    eng.users = Counting(eng.users)
    eng.room_snapshot("cozy", T0 + (t + 700) * 1000)
    assert not calls                                                       # no scan of every chatter


# ------------------------------------------------------------------ F10: IRC liveness (dashboard and collector)
class FakeIRC:
    """A Twitch IRC websocket: replies with the scripted frames, then never again (a half-open
    connection), or raises the scripted exception (a close)."""
    def __init__(self, *script):
        self.script, self.sent = list(script), []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def send(self, m):
        self.sent.append(m)

    async def recv(self):
        if not self.script:
            await asyncio.Event().wait()
        x = self.script.pop(0)
        if isinstance(x, Exception):
            raise x
        return x

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await self.recv()
        except Exception as e:
            if type(e).__name__ == "ConnectionClosedOK":
                raise StopAsyncIteration
            raise


def _closed_ok():
    from websockets.exceptions import ConnectionClosedOK
    from websockets.frames import Close
    return ConnectionClosedOK(Close(1000, ""), Close(1000, ""), True)


def _fake_connect(monkeypatch, conns, then):
    import websockets
    opened = []

    def connect(url, **kw):
        opened.append(url)
        if not conns:
            then()
            return FakeIRC()
        return conns.pop(0)
    monkeypatch.setattr(websockets, "connect", connect)
    return opened


def test_the_dashboard_reconnects_after_silence_a_reconnect_line_or_a_normal_close(monkeypatch):
    hub = Hub(Engine())
    hub.eng.clock = 0
    pauses = []

    async def pause(s):
        pauses.append((s, hub.status["connected"]))
    monkeypatch.setattr(serve, "IRC_IDLE_S", 0.05)
    monkeypatch.setattr(serve, "_pause", pause)
    line = "@user-id=7;display-name=Viewer :viewer!viewer@viewer.tmi.twitch.tv PRIVMSG #cozy :hello there"

    async def go():
        done = asyncio.Event()
        conns = [FakeIRC(),                                          # silent from the start
                 FakeIRC(":tmi.twitch.tv RECONNECT"),
                 FakeIRC(_closed_ok()),                              # the close that follows RECONNECT
                 FakeIRC(line, _closed_ok())]
        opened = _fake_connect(monkeypatch, conns, done.set)
        task = asyncio.create_task(serve._live_loop(hub, ["cozy"], None, None))
        await asyncio.wait_for(done.wait(), 5)
        task.cancel()
        return len(opened)
    assert asyncio.run(go()) == 5
    assert pauses == [(1, False), (2, False), (4, False), (1, False)]   # backoff, reset by a chat line
    assert hub.eng.stats["msgs"] == 1


def test_a_failing_raw_log_is_not_a_disconnect(monkeypatch):
    hub = Hub(Engine())
    hub.eng.clock = 0
    monkeypatch.setattr(serve, "_pause", lambda s: asyncio.sleep(0))
    line = "@user-id=7;display-name=Viewer :viewer!viewer@viewer.tmi.twitch.tv PRIVMSG #cozy :hello there"

    async def go():
        done = asyncio.Event()
        opened = _fake_connect(monkeypatch, [FakeIRC(line, line)], done.set)
        task = asyncio.create_task(serve._live_loop(hub, ["cozy"], None, FullDisk()))
        for _ in range(50):
            await asyncio.sleep(0.01)
            if hub.eng.stats["msgs"] == 2:
                break
        task.cancel()
        return len(opened)
    assert asyncio.run(go()) == 1
    assert hub.eng.stats["msgs"] == 2 and hub.eng.stats["raw_log_errors"] == 2


def _collector(tmp_path, monkeypatch, conns, logs):
    from chatwatch import collect
    col = collect.Collector(["cozy"], str(tmp_path / "raw"), run="r1", log=logs.append)
    col.recv_timeout, col.idle_s = 0.01, 0.1
    pauses = []

    async def pause(s):
        pauses.append(s)
    monkeypatch.setattr(collect, "_pause", pause)
    opened = _fake_connect(monkeypatch, conns, col.stop)
    return col, opened, pauses


def test_the_collector_reconnects_after_six_minutes_of_silence(tmp_path, monkeypatch):
    line = "@id=1 :viewer!viewer@viewer.tmi.twitch.tv PRIVMSG #cozy :hello there"
    logs = []
    col, opened, pauses = _collector(tmp_path, monkeypatch, [FakeIRC(), FakeIRC(line)], logs)

    async def go():
        task = asyncio.create_task(col.run())
        await asyncio.wait_for(task, 5)
    asyncio.run(go())
    assert len(opened) == 3 and pauses == [1.0, 1.0]
    assert "nothing received" in col.status.last_error
    days = os.listdir(tmp_path / "raw")
    assert line in open(tmp_path / "raw" / days[0] / "cozy.r1.log").read()


def test_a_collector_disk_error_is_reported_and_is_not_a_disconnect(tmp_path, monkeypatch):
    from chatwatch import collect
    line = "@id=1 :viewer!viewer@viewer.tmi.twitch.tv PRIVMSG #cozy :hello there"
    logs, writes = [], []
    real = collect.RawSink.write

    def flaky(self, chan, ln, ms):
        writes.append(ln)
        if len(writes) <= 2:
            raise OSError(28, "No space left on device")
        return real(self, chan, ln, ms)
    monkeypatch.setattr(collect.RawSink, "write", flaky)
    col, opened, pauses = _collector(tmp_path, monkeypatch, [FakeIRC(line, line, line)], logs)
    col.idle_s, col.stats_every = 60, 0                 # flush (and so notice the recovery) every round

    async def go():
        task = asyncio.create_task(col.run())
        for _ in range(100):
            await asyncio.sleep(0.01)
            if len(writes) == 3:
                break
        col.stop()
        await asyncio.wait_for(task, 5)
    asyncio.run(go())
    snap = col.status.snapshot()
    assert len(opened) == 1 and col.status.reconnects == 0 and not pauses
    assert snap["disk_errors"] == 2 and snap["disk_error"] == ""   # reported, then written again
    assert sum("disk" in m.lower() for m in logs) == 2              # once when it fails, once when it recovers


def test_a_cancelled_collector_closes_its_files(tmp_path, monkeypatch):
    from chatwatch import collect
    closed = []
    monkeypatch.setattr(collect.RawSink, "close", lambda self: closed.append(1))
    col, opened, pauses = _collector(tmp_path, monkeypatch, [FakeIRC()], [])
    col.idle_s = 60

    async def go():
        task = asyncio.create_task(col.run())
        await asyncio.sleep(0.1)
        task.cancel()                                   # what stop_collector does after 15 s
        try:
            await task
        except asyncio.CancelledError:
            pass
    asyncio.run(go())
    assert closed


# ------------------------------------------------------------------ F15: small live fixes (server side)
def test_unknown_chatters_get_json_404s():
    hub = Hub(Engine())
    hub.eng.clock = 0

    async def calls(c):
        out = []
        r = await c.get("/api/profile?channel=nowhere&uid=1")
        out.append((r.status, "error" in await r.json()))
        for path in ("/api/llm_review", "/api/draft_warning"):          # POST since LB-10
            r = await c.post(path, json={"channel": "nowhere", "uid": "1"}, headers={"X-Chatwatch": "1"})
            out.append((r.status, "error" in await r.json()))
        return out
    assert _http(hub, calls) == [(404, True)] * 3


def test_the_user_card_gets_the_llm_person_read():
    hub = Hub(Engine())
    hub.eng.clock = 0
    serve.handle(hub, msg("heckler", "you are a stupid idiot streamer", T0))
    uid = [r for r in hub.queue if r.get("type") == "msg"][0]["user_id"]

    async def card(c):
        r = await c.get(f"/api/profile?channel=cozy&uid={uid}")
        return (await r.json())["person_read"]
    assert _http(hub, card) is None
    hub.eng.apply_person_review("cozy", uid, {"label": "bad_faith", "rationale": "insults the streamer"}, T0)
    assert _http(hub, card) == {"label": "bad_faith", "rationale": "insults the streamer", "n": 1, "ts": T0, "reads": 1}


def test_trimming_keeps_alerts_reviews_and_updates():
    hub = Hub(Engine())
    hub.eng.clock = 0
    sent = []

    class Client:
        async def send_str(self, data):
            sent.extend(json.loads(data))
    hub.clients = {Client()}
    keep = [{"type": "alert", "kind": "badfaith"}, {"type": "review", "alert_id": "a"}, {"type": "msg_update", "id": "m1"}]
    hub.queue = keep[:1] + [{"type": "msg", "id": f"m{i}"} for i in range(450)] + keep[1:]
    hub._last_room = time.time()                       # no room or status records this round
    asyncio.run(hub._flush())
    assert [r for r in sent if r["type"] != "msg"] == keep
    assert [r["id"] for r in sent if r["type"] == "msg"] == [f"m{i}" for i in range(50, 450)]
    assert hub.eng.stats["ui_records_dropped"] == 50


def test_a_stuck_browser_does_not_block_the_others(monkeypatch):
    monkeypatch.setattr(serve, "SEND_TIMEOUT_S", 0.05)
    hub = Hub(Engine())
    hub.eng.clock = 0
    got = []

    class Stuck:
        async def send_str(self, data):
            await asyncio.Event().wait()

    class Fine:
        async def send_str(self, data):
            got.append(data)
    stuck, fine = Stuck(), Fine()
    hub.clients = {stuck, fine}
    hub.queue = [{"type": "msg", "id": "m1"}]
    hub._last_room = time.time()
    asyncio.run(asyncio.wait_for(hub._flush(), 2))
    assert got and hub.clients == {fine}


def test_the_page_queue_is_capped_without_a_flusher():
    eng = Engine()
    t = room(eng)
    eng.clock = 0
    hub = Hub(eng)
    hub.queue = [{"type": "alert", "kind": "wave"}]
    for i in range(serve.QUEUE_MAX + 10):
        hub.publish({"type": "msg", "channel": "cozy", "user_id": "1", "ts": t, "id": f"m{i}", "text": "hi",
                     "flags": [], "n": 9})
    assert len(hub.queue) <= serve.QUEUE_MAX and hub.queue[0] == {"type": "alert", "kind": "wave"}
    assert hub.queue[-1]["id"] == f"m{serve.QUEUE_MAX + 9}"


def test_a_malformed_subscriber_tag_does_not_drop_the_message():
    eng = Engine()
    t = room(eng)
    rec = eng.process({**msg("odd", "hello there", t, sub=1), "badge_info": {"subscriber": "abc"}})
    assert rec and rec["type"] == "msg" and eng.users[("cozy", rec["user_id"])].sub_months == 0


def test_the_shown_once_error_set_is_bounded(monkeypatch, capsys):
    monkeypatch.setattr(serve, "_errors_seen", set())
    eng = Engine()
    for i in range(500):
        try:
            raise ValueError(f"bad value for user {i}")
        except ValueError as e:
            serve._log_error(eng, "event_errors", e, "on a msg event")
    assert eng.stats["event_errors"] == 500 and len(serve._errors_seen) <= serve.ERRORS_SHOWN + 1
    assert capsys.readouterr().out.count("[serve] error") == serve.ERRORS_SHOWN


# ------------------------------------------------------------------ F15: wording (rule 8.9) and the session's span
def test_a_strong_first_impression_is_worded_as_likely():
    from chatwatch import archetypes as A
    from chatwatch.engine import verdict
    post = {a: 0.0 for a in A.NAMES}
    post.update(harasser=0.72, provocateur=0.2, ordinary=0.08)
    v = verdict(post, 5)
    assert v.startswith("Likely harasser") and "(5 messages)" in v
    assert verdict({**post, "harasser": 0.45, "provocateur": 0.44, "ordinary": 0.11}, 5).startswith("Harasser")  # "A or B"


def test_earlier_and_chatters_seen_cover_this_session_only(monkeypatch):
    day = 86_400_000
    eng = Engine()
    t = room(eng)                                                  # regulars, three days before the session
    eng.process(msg("oldie", "you are a nigger", t + 10))
    start = T0 + 3 * day
    eng.process({**msg("newbie", "you are a nigger", 0), "ts": start + 20 * 60_000})
    now = start + 60 * 60_000                                      # both cards have expired (30 min)
    whole = eng.room_snapshot("cozy", now)
    assert {d["display"] for d in whole["notices_earlier"]} == {"oldie", "newbie"}
    snap = eng.room_snapshot("cozy", now, since=start)
    assert [d["display"] for d in snap["notices_earlier"]] == ["newbie"] and snap["chatters"] == 1
    assert snap["notices"] == whole["notices"]                     # the live panel is unchanged

    hub = Hub(eng)
    hub.log_sessions = False
    sent = []

    class Client:
        async def send_str(self, data):
            sent.extend(json.loads(data))
    hub.clients = {Client()}

    async def nothing(*a):
        return None
    monkeypatch.setattr(serve, "live_source", nothing)

    async def go():
        hub.start_live(["cozy"])
        assert abs(hub.since - time.time() * 1000) < 5000
        hub.since, eng.clock = start, now
        await hub._flush()
    asyncio.run(go())
    ui = [r for r in sent if r["type"] == "room"][0]
    assert [d["display"] for d in ui["notices_earlier"]] == ["newbie"] and ui["chatters"] == 1


# ------------------------------------------------------------------ F16: the LLM's output contract (tests_docs.md 2c #4, G4)
def _llm_says(monkeypatch, content):
    monkeypatch.setattr(llm, "_chat", lambda body, timeout: {"message": {"content": content}})


def test_llm_output_outside_the_contract_is_an_error_not_a_read(monkeypatch):
    """A person read needs one of the policy's four labels and a toxicity read one of the four levels;
    anything else is an error (the caller then spends no read). The JSON schema sent to Ollama already
    asks for these, so valid answers are unchanged."""
    monkeypatch.setattr(llm, "_content", lambda p, ctx, emotes=frozenset(): "x")
    for bad in ("not json", "{}", '{"label": "Bad Faith"}', '["bad_faith"]'):
        _llm_says(monkeypatch, bad)
        with pytest.raises(ValueError):
            llm.review({}, [])
    _llm_says(monkeypatch, '{"label": "mild", "rules": [], "confidence": "low", "rationale": "jabs"}')
    assert llm.review({}, [])["label"] == "mild"
    for bad in ("not json", "{}", '{"toxicity": "extreme", "target": "", "kind": ""}'):
        _llm_says(monkeypatch, bad)
        with pytest.raises(ValueError):
            llm.rate_toxicity("you suck")
    _llm_says(monkeypatch, '{"toxicity": "low", "target": "chatter", "kind": "insult"}')
    assert llm.rate_toxicity("you suck")["toxicity"] == "low"


def test_a_window_read_keeps_lines_numbered_as_text_and_drops_unknown_levels(monkeypatch):
    """rate_toxicity_window (live re-reads and the silver teacher): a line number sent as "1" still
    counts; a rating with a level outside none/low/medium/high is dropped (None, as a skipped line);
    a reply that is not JSON raises (the dashboard counts llm_tox_errors, a silver round logs it)."""
    _llm_says(monkeypatch, json.dumps({"ratings": [
        {"n": "1", "toxicity": "low", "target": "chatter", "kind": "insult"},
        {"n": 2, "toxicity": "extreme", "target": "", "kind": ""},
        {"n": 3, "toxicity": "none", "target": "", "kind": ""},
        {"n": "x", "toxicity": "high", "target": "", "kind": ""}]}))
    got = llm.rate_toxicity_window([], ["a: you suck", "b: hmm", "c: gg", "d: ok"])
    assert [g and g["toxicity"] for g in got] == ["low", None, "none", None]
    _llm_says(monkeypatch, "not json")
    with pytest.raises(ValueError):
        llm.rate_toxicity_window([], ["a: you suck"])


def test_a_person_read_without_a_valid_label_spends_no_read(monkeypatch):
    eng = Engine()
    eng.person_review = True
    t = room(eng)
    rec = eng.process(msg("heckler", "you are trash", t))
    eng.clock = rec["ts"]
    hub = Hub(eng)
    hub.log_sessions = False
    u = eng.users[("cozy", rec["user_id"])]
    for content, reads in (("{}", 0), ('{"label": "Bad Faith"}', 0), ('{"label": "bad_faith", "rationale": "x"}', 1)):
        _llm_says(monkeypatch, content)
        eng.mark_review_pending("cozy", rec["user_id"])
        asyncio.run(hub._review_one("person", {"channel": "cozy", "user_id": rec["user_id"]}))
        assert u.review["reads"] == reads and not u.review["pending"], content
    assert eng.stats["llm_person_errors"] == 2 and u.review["label"] == "bad_faith"
    assert "llm_person_None" not in eng.stats and "llm_person_Bad Faith" not in eng.stats


# ------------------------------------------------------------------ F16: the publish glue (tests_docs.md 2c #5, G5)
def test_every_person_read_is_queued_however_long_the_queue():
    """Lossless (docs/AGENDA.md): publish() marks a candidate's read pending and queues it. The queue is never
    full, so no read is released as "queue full" (the old maxsize of 200 made it an error), and both wait."""
    eng = Engine()
    eng.person_review = True
    t = room(eng)
    hub = Hub(eng)
    hub.review_q = ReadQueue()
    for i in range(300):                                   # more than the old maxsize
        hub.review_q.put_nowait((0, i, "other", {}))
    serve.handle(hub, msg("first", "you are trash", t))
    serve.handle(hub, msg("second", "ur bad", t + 1))
    u1, u2 = (next(u for u in eng.users.values() if u.login == n) for n in ("first", "second"))
    assert u1.review["pending"] and u2.review["pending"] and u1.review["reads"] == u2.review["reads"] == 0
    assert eng.stats["llm_person_errors"] == 0 and hub.review_q.qsize() == 302
    kinds = [hub.review_q.get_nowait()[2:] for _ in range(302)]
    assert [a for k, a in kinds if k == "person"] == [{"channel": "cozy", "user_id": u1.uid},
                                                      {"channel": "cozy", "user_id": u2.uid}]


def test_flagged_messages_are_read_by_the_llm_first():
    """PINS TODAY'S BEHAVIOUR: Hub._tox_priority, the order of LLM toxicity re-reads (lower = sooner)."""
    hub = Hub(Engine())
    P = lambda text, score=0.0, **kw: hub._tox_priority(
        {"text": text, "toxicity": {"score": score}, "n": 9, "flags": [], "badfaith": 0.0, **kw})
    assert P("!points", 0.9) is None                          # commands: never
    assert P("shit aim", 0.3) == 0                            # the layer flags it: confirm or clear first
    assert P("you suck", tox=0.9) is None                     # under 3 words and unflagged: skipped
    assert P("is this the final boss", 0.15) == 1             # the layer is unsure
    assert P("is this the final boss", tox=0.5) == 1          # ... or the ML score is
    assert P("hello there everyone", n=2) == 1                # newcomers
    assert P("hello there everyone", flags=["pile-on"]) == 1  # brigades
    assert P("@someone what do you mean") == 2                # aimed at someone
    assert P("hello there everyone", badfaith=0.3) == 2       # already watched
    assert P("hello there everyone") == 3
    eng = hub.eng
    t = room(eng)
    hub.review_q = ReadQueue()
    hub.tox_per_min = 30
    serve.handle(hub, msg("heckler", "you are a stupid idiot streamer", t))
    prio, ts, item = hub.tox_pending[-1]
    assert prio == 0 and item["line"] == "heckler: you are a stupid idiot streamer" and len(item["room"]) == 6


def test_coded_language_is_read_first_short_lines_included():
    """C1 (branch coded1dev, data/scratch/coded1dev/DEV.md): a line with a coded-glossary trigger (llm.glossary_lookup,
    the teacher's own trigger list) is re-read at priority 0 even when the fast layer does not flag it and it is
    under 3 words. Commands are still never read, a login cannot trigger it, and lines without a trigger keep
    today's priority (test_flagged_messages_are_read_by_the_llm_first)."""
    hub = Hub(Engine())
    P = lambda text, score=0.0, display="someone", **kw: hub._tox_priority(
        {"text": text, "display": display, "toxicity": {"score": score}, "n": 9, "flags": [], "badfaith": 0.0, **kw})
    assert P("goyslop") == 0                                   # one word, unflagged: read first
    assert P("check his early life") == 0                     # would be 3 today
    assert P("every single time") == 0                        # an innocent use is read too: the LLM decides
    assert P("!goyslop") is None                               # commands: never
    assert P("hi", display="goyslop_enjoyer") is None          # the login is not the line
    assert P("hello there everyone") == 3                      # no trigger: unchanged
    assert P("you suck", tox=0.9) is None
    hub.coded_first = False                                    # master's rule
    assert P("goyslop") is None and P("check his early life") == 3
    hub.coded_first = True
    eng = hub.eng
    t = room(eng)
    hub.review_q = ReadQueue()
    hub.tox_per_min = 30
    serve.handle(hub, msg("someone", "goyslop", t))
    prio, ts, item = hub.tox_pending[-1]
    assert prio == 0 and item["line"] == "someone: goyslop"


def test_the_status_reports_the_head_this_run_loaded(tmp_path, monkeypatch):
    """PF-11: the pipeline page compares this with the live head to say whether a restart is still needed."""
    import hashlib
    from chatwatch import toxicity
    p = tmp_path / "head.json"
    p.write_text('{"t": 0.5}')
    monkeypatch.setenv("CW_TOX_HEAD", str(p))
    monkeypatch.setattr(toxicity, "_head", lambda: None)
    assert serve.loaded_head_sha() == hashlib.sha256(p.read_bytes()).hexdigest()
    monkeypatch.setenv("CW_TOX_HEAD", str(tmp_path / "missing.json"))
    assert serve.loaded_head_sha() is None


# ------------------------------------------------------------------ reclassifying a message (the owner's correction)
def _two_msgs():
    """A chatter with a plain message and an insult in cozy; returns (engine, uid, plain id, insult id)."""
    eng = Engine()
    room(eng)
    a = msg("corr", "nice dodge, that boss is brutal", 400)
    b = msg("corr", "you are a stupid idiot streamer", 410)
    eng.process(a)
    eng.process(b)
    return eng, a["user_id"], a["id"], b["id"]


def test_the_owner_correction_decides_the_watchlist_and_stands_over_a_re_read():
    from chatwatch import engine as E
    eng, uid, plain, insult = _two_msgs()
    u = eng.users[("cozy", uid)]
    n0 = u.toxic_n
    # severe hate on the plain line: "likely" at once, and it stands like a keyword slur
    upd = eng.apply_owner_override("cozy", uid, plain, "high", "hate")
    m = next(x for x in u.msgs if x["id"] == plain)
    assert upd["toxicity"]["source"] == "owner" and E._tox_class(m["toxicity"]) == "hate" and E._sure_hate(m["toxicity"])
    assert u.toxic_n == n0 + 1
    nr = E._notice_reasons(u, None, u.last_seen, False)
    assert nr and nr[0] == "act"
    # an LLM re-read afterwards does not replace the correction; it only updates the automatic read under it (and the
    # correction's tooltip, LB2-8)
    upd = eng.apply_llm_toxicity("cozy", uid, plain, {"toxicity": "none"})
    assert upd["toxicity"]["source"] == "owner" and upd["toxicity"]["auto"]["source"] == "llm"
    assert m["toxicity"]["source"] == "owner" and m["toxicity_auto"]["source"] == "llm"
    # back to automatic: the re-read the owner overrode now counts
    eng.apply_owner_override("cozy", uid, plain, None)
    assert m["toxicity"]["source"] == "llm" and m["toxicity"]["level"] == "none" and "toxicity_auto" not in m
    assert u.toxic_n == n0


def test_the_owner_can_clear_a_false_alarm_including_a_keyword_slur():
    from chatwatch import engine as E
    eng = Engine()
    room(eng)
    r = msg("catlover", "my maine coon knocked the cup over again lol", 400)
    eng.process(r)
    u = eng.users[("cozy", r["user_id"])]
    m = u.msgs[-1]
    m["toxicity"] = {"level": "high", "score": 0.85, "kinds": ["hate"], "sure": ["hate"], "target": ""}  # a keyword hit
    u.toxic_n = 1
    assert E._notice_reasons(u, None, u.last_seen, False)[0] == "act"
    eng.apply_owner_override("cozy", r["user_id"], r["id"], "none")
    assert E._level(m["toxicity"]) == "none" and u.toxic_n == 0 and E._notice_reasons(u, None, u.last_seen, False) is None
    assert m["toxicity"]["auto"]["level"] == "high"


def test_bad_override_requests_are_refused():
    eng, uid, plain, _ = _two_msgs()
    assert eng.apply_owner_override("cozy", uid, plain, "extreme") is None
    assert eng.apply_owner_override("cozy", uid, plain, "high", "spam") is None
    assert eng.apply_owner_override("cozy", uid, "no-such-id", "low") is None
    assert eng.apply_owner_override("cozy", uid, plain, None) is None          # nothing to undo
    assert eng.apply_owner_override("nowhere", uid, plain, "low") is None


def test_the_override_endpoint_logs_queues_and_asks_for_a_person_read(tmp_path, monkeypatch):
    from chatwatch import session as S
    eng, uid, plain, _ = _two_msgs()
    eng.clock = 0
    eng.person_review = True
    hub = Hub(eng)
    hub.status.update(mode="live", running=True)                 # a live session that saves its state
    hub.persist = True
    hub.review_q = ReadQueue()
    reg = tmp_path / "owner_overrides.jsonl"
    monkeypatch.setattr(S, "OVERRIDES", str(reg))
    logged = []
    monkeypatch.setattr(hub, "_log", lambda rec: logged.append(rec))
    eng.users[("cozy", uid)].review = None

    async def calls(c):
        h = {"X-Chatwatch": "1"}
        out = []
        for body in ({"channel": "cozy", "uid": uid, "id": plain, "level": "medium"},
                     {"channel": "cozy", "uid": uid, "id": plain, "level": "bogus"},
                     {"channel": "cozy", "uid": uid, "id": "gone", "level": "low"}):
            r = await c.post("/api/override", json=body, headers=h)
            out.append((r.status, await r.json()))
        r = await c.post("/api/override", json={"channel": "cozy", "uid": uid, "id": plain, "level": "low"})
        out.append((r.status, None))                               # no X-Chatwatch header: refused
        return out
    out = _http(hub, calls)
    assert out[0][0] == 200 and out[0][1]["toxicity"]["source"] == "owner" and out[0][1]["toxicity"]["level"] == "medium"
    assert out[1][0] == 400 and out[2][0] == 404 and out[3][0] == 403
    lines = [json.loads(x) for x in reg.read_text().splitlines()]
    assert len(lines) == 1 and lines[0]["msg_id"] == plain and lines[0]["level"] == "medium" and "text" not in lines[0]
    assert any(r.get("type") == "owner_override" for r in logged) and any(r.get("type") == "msg_update" for r in hub.queue)
    assert hub.review_q.qsize() == 1                              # a correction to toxic asks for the person read
    assert S.owner_override_ids(str(reg)) == {plain}


def test_forget_removes_a_chatters_corrections(tmp_path):
    from chatwatch import session as S
    reg = tmp_path / "o.jsonl"
    reg.write_text('{"user_id": "1", "msg_id": "a"}\n{"user_id": "2", "msg_id": "b"}\nnot json\n')
    assert S.forget_overrides({"1"}, False, str(reg)) == 1 and "\"1\"" in reg.read_text()
    assert S.forget_overrides({"1"}, True, str(reg)) == 1 and S.owner_override_ids(str(reg)) == {"b"}
