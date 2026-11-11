"""The lossless LLM work queue (docs/AGENDA.md "Core architecture"; owner 2026-11-05): nothing is dropped
automatically, a failed call is retried at the head, Stop finishes the queue, a shutdown keeps it in the saved
state, and the page sees how far behind it is."""
import asyncio
import pickle
import time
import urllib.error

from chatwatch import llm, serve
from chatwatch.engine import Engine
from chatwatch.serve import Hub, ReadQueue

from test_dashboard import _http
from test_scenarios import msg, room

LOW = {"toxicity": "low", "target": "chatter", "kind": "insult"}
PAGE = {"X-Chatwatch": "1"}


def _live_hub(person=True, lines=True):
    eng = Engine()
    eng.person_review = person
    t = room(eng)
    hub = Hub(eng)
    hub.log_sessions = False
    hub.status["mode"] = "live"
    hub.review_q = ReadQueue()
    hub.tox_per_min = 1 if lines else 0
    return eng, hub, t


async def _until(cond, s=5.0):
    end = time.time() + s
    while not cond() and time.time() < end:
        await asyncio.sleep(0.01)
    return cond()


def _fast_retries(monkeypatch, up=lambda: True):
    monkeypatch.setattr(serve, "RETRY_WAIT_S", (0,))
    monkeypatch.setattr(llm, "reachable", lambda timeout=5: up())


def test_stop_finishes_the_queue_before_the_session_closes(monkeypatch):
    """Chat stops at once; every queued read and line is still read and logged to this session, then it closes.
    Nothing is counted as dropped."""
    eng, hub, t = _live_hub()
    logged = []
    hub._log = logged.append
    monkeypatch.setattr(llm, "review", lambda p, ctx, emotes=None: {"label": "mild", "rationale": "jabs",
                                                                     "confidence": "low", "rules": []})
    monkeypatch.setattr(llm, "rate_toxicity_window", lambda room_, lines, emotes=frozenset(): [LOW] * len(lines))
    order = []
    monkeypatch.setattr(serve, "save_engine_async", lambda e, fork=False: order.append("save") or asyncio.sleep(0))
    hub.persist = True
    for i, (who, text) in enumerate([("first", "you are trash"), ("second", "ur bad lol"),
                                     ("third", "worst streamer ever honestly"), ("fourth", "you suck at this")]):
        serve.handle(hub, msg(who, text, t + i))
    reads, lines = hub.review_q.qsize(), len(hub.tox_pending)
    assert reads >= 2 and lines >= 2

    async def go():
        task = asyncio.create_task(hub.reviewer())
        await hub.stop_live()
        order.append("stopped")
        task.cancel()
    asyncio.run(go())
    users = [u for u in eng.users.values() if u.login in ("first", "second", "third", "fourth")]
    assert hub.review_q.empty() and not hub.tox_pending and hub.llm_job is None
    assert sum(1 for r in logged if r.get("type") == "review" and r.get("kind") == "person") == reads
    assert all(not (u.review or {}).get("pending") for u in users)
    assert eng.stats["llm_tox_reads"] == lines and eng.stats["llm_dropped_at_stop"] == 0
    assert order == ["save", "stopped"] and not hub.llm_open


def test_a_failed_call_stays_at_the_head_and_is_retried(monkeypatch):
    """An HTTP 500 (a runner that aborted on CUDA out of memory) or an empty reply is retried; the read lands
    once, with no error. After RETRY_MAX calls that fail while the server answers, the error is recorded as
    before. A reply outside the schema is final at once."""
    _fast_retries(monkeypatch)
    calls = []

    def run(review):
        monkeypatch.setattr(llm, "review", review)
        calls.clear()
        eng, hub, t = _live_hub(lines=False)
        serve.handle(hub, msg("first", "you are trash", t))
        u = next(u for u in eng.users.values() if u.login == "first")

        async def go():
            task = asyncio.create_task(hub.reviewer())
            await _until(lambda: not (u.review or {}).get("pending"))
            task.cancel()
        asyncio.run(go())
        return eng, u

    # the JSONDecodeError case: an empty reply is a JSONDecodeError (a ValueError), worth a retry
    def empty_then_ok(p, ctx, emotes=None):
        calls.append(1)
        if len(calls) < 3:
            raise (urllib.error.HTTPError("http://x", 500, "boom", {}, None) if len(calls) == 1
                   else __import__("json").JSONDecodeError("Expecting value", "", 0))
        return {"label": "mild", "rationale": "jabs", "confidence": "low", "rules": []}
    eng, u = run(empty_then_ok)
    assert len(calls) == 3 and u.review["reads"] == 1 and u.review["label"] == "mild"
    assert eng.stats["llm_person_errors"] == 0 and eng.stats["llm_retries"] == 2

    def always_500(p, ctx, emotes=None):
        calls.append(1)
        raise urllib.error.HTTPError("http://x", 500, "boom", {}, None)
    eng, u = run(always_500)
    assert len(calls) == serve.RETRY_MAX and eng.stats["llm_person_errors"] == 1 and u.review.get("failed")

    def off_schema(p, ctx, emotes=None):
        calls.append(1)
        raise ValueError("the LLM's person read is not one of good_faith/mild/bad_faith/unclear")
    eng, u = run(off_schema)
    assert len(calls) == 1 and eng.stats["llm_person_errors"] == 1 and eng.stats["llm_retries"] == 0


def test_time_the_server_does_not_answer_spends_no_retry(monkeypatch):
    """A restarting runner or container: the worker waits it out (the page says it is retrying) and the read lands."""
    answering = [False]
    _fast_retries(monkeypatch, up=lambda: answering[0])
    calls = []

    def down_then_up(p, ctx, emotes=None):
        calls.append(1)
        if len(calls) <= 6:
            if len(calls) == 6:
                answering[0] = True
            raise urllib.error.URLError("connection refused")
        return {"label": "good_faith", "rationale": "fine", "confidence": "low", "rules": []}
    monkeypatch.setattr(llm, "review", down_then_up)
    eng, hub, t = _live_hub(lines=False)
    serve.handle(hub, msg("first", "you are trash", t))
    u = next(u for u in eng.users.values() if u.login == "first")
    seen = []

    async def go():
        task = asyncio.create_task(hub.reviewer())
        await _until(lambda: hub.llm_retry is not None)
        seen.append(hub._llm_status()["retrying"])
        await _until(lambda: not (u.review or {}).get("pending"))
        task.cancel()
    asyncio.run(go())
    assert len(calls) == 7 and u.review["reads"] == 1 and eng.stats["llm_person_errors"] == 0
    assert seen[0].startswith("the local LLM is not answering") and hub.llm_retry is None


def test_a_shutdown_keeps_the_queue_in_the_saved_state_and_the_next_start_resumes_it(monkeypatch):
    eng, hub, t = _live_hub()
    hub.persist = True
    saved = []

    async def fake_save(e, fork=False):
        saved.append(pickle.loads(pickle.dumps(e.llm_queue)))
    monkeypatch.setattr(serve, "save_engine_async", fake_save)
    serve.handle(hub, msg("first", "you are trash", t))
    serve.handle(hub, msg("second", "ur bad lol", t + 1))
    reads, lines = hub.review_q.qsize(), len(hub.tox_pending)
    hub.llm_job = {"kind": "person", "a": {"channel": "cozy", "user_id": "999"}, "prio": 0, "at": time.time() - 30}
    asyncio.run(hub.stop_live(finish=False))
    snap = saved[-1]
    assert len(snap["reads"]) == reads + 1 and len(snap["lines"]) == lines
    assert snap["reads"][0][1:3] == ["person", {"channel": "cozy", "user_id": "999"}]        # the running job first
    assert eng.stats["llm_dropped_at_stop"] == 0
    # the next run: back in line, waiting for Start
    hub2 = Hub(eng)
    hub2.review_q = ReadQueue()
    hub2.llm_open = False
    assert hub2.restore_queue(snap) == reads + 1 + lines and ("cozy", "999") in hub2.person_queued
    assert hub2._next_job(time.time()) is None
    assert hub2._llm_status()["behind_s"] >= 30                                               # ages carried over
    hub2.llm_open = True
    job = hub2._next_job(time.time())
    assert job["kind"] == "person" and job["a"]["user_id"] == "999"
    assert eng.stats["llm_carried_over"] == reads + 1 + lines


def test_a_shutdown_during_a_stops_finishing_keeps_the_rest(monkeypatch):
    """SIGTERM while Stop waits for the queue: the rest is kept (saved), not dropped."""
    eng, hub, t = _live_hub()
    hub.persist = True
    saved = []

    async def fake_save(e, fork=False):
        saved.append(e.llm_queue)
    monkeypatch.setattr(serve, "save_engine_async", fake_save)
    serve.handle(hub, msg("first", "you are trash", t))
    n = hub.review_q.qsize() + len(hub.tox_pending)

    async def go():
        stopper = asyncio.create_task(hub.stop_live())          # no worker: it would wait forever
        await _until(lambda: hub.finishing)
        await hub.stop_live(finish=False)
        await stopper
    asyncio.run(go())
    assert saved and len(saved[-1]["reads"]) + len(saved[-1]["lines"]) == n
    assert eng.stats["llm_dropped_at_stop"] == 0


def test_stop_now_skips_the_rest_and_counts_it():
    eng, hub, t = _live_hub()
    hub.status["running"] = True                               # a live session runs
    serve.handle(hub, msg("first", "you are trash", t))
    serve.handle(hub, msg("second", "ur bad lol", t + 1))
    n = hub.review_q.qsize() + len(hub.tox_pending)

    async def calls(c):
        stopper = asyncio.create_task(c.post("/api/stop", headers=PAGE))       # no worker: it waits
        await _until(lambda: hub.finishing)
        finishing = hub._status()["llm_queue"]["finishing"]
        r = await c.post("/api/stop", json={"now": True}, headers=PAGE)
        first = await stopper
        return finishing, r.status, first.status
    finishing, s2, s1 = _http(hub, calls)
    assert finishing and s1 == s2 == 200
    assert eng.stats["llm_dropped_at_stop"] == n and hub.review_q.empty() and not hub.tox_pending
    users = [u for u in eng.users.values() if u.login in ("first", "second")]
    assert all(u.review["lost"] and not u.review["pending"] for u in users)


def test_the_status_shows_the_backlog_lag_catch_up_and_chat_rate():
    eng, hub, t = _live_hub()
    now = time.time()
    for i in range(30):
        serve.handle(hub, msg(f"c{i}", "you are trash honestly", t + i))
    hub._q_cost = {"read": 2.0, "line": 0.5}
    hub.status["mode"] = "replay"                             # the chat clock (test times are not today)
    for x in hub.review_q.items:
        x[4] = now - 120
    for x in hub.tox_pending:
        x[2]["q_at"] = now - 60
    hub.tox_pending.append((2, 1, {"channel": "cozy", "user_id": "u", "id": "m", "line": "x: y", "room": [], "prio": 2,
                               "ts": 1, "q_at": now - 300}))
    s = hub._status()
    q = s["llm_queue"]
    assert q["reads"] == hub.review_q.qsize() and q["lines"] == len(hub.tox_pending)
    assert 299 <= q["behind_s"] <= 301 and 119 <= q["behind_hi_s"] <= 121
    assert q["catchup_s"] is not None and q["catchup_s"] >= q["reads"] * 2 + q["lines"] * 0.5
    assert s["llm_busy"] and s["chat_per_min"]["cozy"] >= 30
    hub.llm_open = False                                       # live, stopped: queued work does not use the GPU
    assert not hub._status()["llm_busy"]
    hub.llm_open = True
    hub._q_in = [(now, "read")] * 1000                         # arrivals beyond what it can serve
    assert hub._llm_status()["catchup_s"] is None              # falling behind: no estimate


def test_a_hold_pauses_after_the_call_in_flight_and_chat_keeps_queueing():
    """The swap protocol's first steps (hybrid reviewer, later): hold, wait for the call in flight, swap, release."""
    eng, hub, t = _live_hub()
    serve.handle(hub, msg("first", "you are trash", t))
    hub.hold_llm("swap")
    assert hub._next_job(time.time() + 60) is None
    serve.handle(hub, msg("second", "ur bad lol", t + 1))
    assert hub.review_q.qsize() >= 2
    asyncio.run(asyncio.wait_for(hub.llm_idle(), 1))
    hub.release_llm("swap")
    assert hub._next_job(time.time())["kind"] == "person"


# ------------------------------------------------------------------ review fixes (2026-11-07)
def test_stop_now_during_a_call_that_then_fails_drops_the_job_cleanly(monkeypatch):
    """Stop now while the worker's call hangs; the call then fails (a timeout): the job is dropped and counted, the
    person read released and no longer marked queued, nothing is left running or "retrying", and the next run's Stop
    does not wait on it."""
    import threading
    _fast_retries(monkeypatch)
    started, release = threading.Event(), threading.Event()

    def hung(p, ctx, emotes=None):
        started.set()
        release.wait(5)
        raise urllib.error.URLError("timed out")
    monkeypatch.setattr(llm, "review", hung)
    eng, hub, t = _live_hub(lines=False)
    serve.handle(hub, msg("first", "you are trash", t))
    u = next(u for u in eng.users.values() if u.login == "first")

    async def go():
        task = asyncio.create_task(hub.reviewer())
        await _until(started.is_set)
        stopper = asyncio.create_task(hub.stop_live(now=True))
        await asyncio.sleep(0.1)
        release.set()
        await stopper
        await _until(lambda: hub.llm_job is None)
        task.cancel()
    asyncio.run(go())
    assert hub.llm_job is None and hub.llm_retry is None and not hub.person_queued
    assert eng.stats["llm_dropped_at_stop"] == 1 and u.review["lost"] and not u.review["pending"]
    hub.llm_open = True                                      # the next run: its Stop has nothing to wait for
    asyncio.run(asyncio.wait_for(hub.stop_live(), 2))


def test_stop_now_drops_a_job_waiting_to_retry_even_while_a_cards_read_runs(monkeypatch):
    eng, hub, t = _live_hub(lines=False)
    serve.handle(hub, msg("first", "you are trash", t))
    e = hub.review_q.take(time.time())
    hub.llm_job = {"kind": "person", "a": e[3], "prio": e[0], "at": e[4]}      # the worker waits to retry
    hub.llm_inflight = 1                                                      # a card's manual read is running
    asyncio.run(hub.stop_live(now=True))
    assert hub.llm_job is None and eng.stats["llm_dropped_at_stop"] == 1


def test_a_hold_puts_a_retrying_job_back_at_the_head(monkeypatch):
    _fast_retries(monkeypatch, up=lambda: False)
    calls = []

    def down(p, ctx, emotes=None):
        calls.append(1)
        raise urllib.error.URLError("refused")
    monkeypatch.setattr(llm, "review", down)
    eng, hub, t = _live_hub(lines=False)
    serve.handle(hub, msg("first", "you are trash", t))

    async def go():
        task = asyncio.create_task(hub.reviewer())
        await _until(lambda: hub.llm_retry is not None)
        hub.hold_llm("swap")
        await asyncio.wait_for(hub.llm_idle(), 2)                 # the swap does not wait on its retries
        task.cancel()
    asyncio.run(go())
    assert hub.llm_job is None and hub.review_q.qsize() == 1 and hub.review_q.items[0][2] == "person"
    assert hub.llm_retry is None and eng.stats["llm_dropped_at_stop"] == 0


def test_under_backlog_a_fresh_line_waits_for_company_even_when_another_channel_has_old_lines():
    hub = Hub(Engine())
    hub.review_q = ReadQueue()
    hub._q_cost = {"solo": 15.0}                    # one line goes alone at most: two are a backlog
    now = time.time()
    fresh = (0, 10, {"channel": "a", "user_id": "u", "id": "m1", "line": "x: y", "room": [], "prio": 0, "ts": 10,
                     "q_at": now})
    old = (3, 1, {"channel": "b", "user_id": "v", "id": "m2", "line": "x: z", "room": [], "prio": 3, "ts": 1,
                  "q_at": now - 60})
    hub.tox_pending = [fresh, old]
    assert hub._next_job(now) is None                              # the head (a) gathers its 4 s
    assert hub._next_job(now + serve.LINE_GATHER_S + 0.1)["a"]["channel"] == "a"


def test_a_stop_from_a_stale_page_keeps_a_restored_queue():
    eng, hub, t = _live_hub()
    hub.llm_open = False
    snap = {"reads": [[0, "person", {"channel": "cozy", "user_id": "9"}, time.time() - 5]], "lines": []}
    hub.restore_queue(snap)

    async def calls(c):
        return (await c.post("/api/stop", json={"now": True}, headers=PAGE)).status
    assert _http(hub, calls) == 200 and hub.review_q.qsize() == 1 and eng.stats["llm_dropped_at_stop"] == 0


def test_a_restored_person_read_is_pending_again():
    eng, hub, t = _live_hub(lines=False)
    serve.handle(hub, msg("first", "you are trash", t))
    u = next(u for u in eng.users.values() if u.login == "first")
    snap = {"reads": [[e[0], e[2], e[3], e[4]] for e in hub.review_q.items], "lines": []}
    eng.release_person_review("cozy", u.uid)                      # as upgrade() leaves it: lost
    hub2 = Hub(eng)
    hub2.review_q = ReadQueue()
    hub2.restore_queue(snap)
    assert u.review["pending"]


def test_a_job_stop_dropped_while_its_call_runs_is_not_saved(monkeypatch):
    eng, hub, t = _live_hub(lines=False)
    hub.persist = True
    saved = []

    async def fake_save(e, fork=False):
        saved.append(e.llm_queue)
    monkeypatch.setattr(serve, "save_engine_async", fake_save)
    hub.llm_job = {"kind": "person", "a": {"channel": "cozy", "user_id": "9"}, "prio": 0, "at": time.time()}
    hub._calling = True                                            # its call is still running
    asyncio.run(hub.stop_live(now=True))
    assert saved == [None]


def test_a_job_stop_dropped_is_not_saved_by_a_later_autosave_or_shutdown(monkeypatch):
    eng, hub, t = _live_hub(lines=False)
    hub.persist = True
    hub.llm_job = {"kind": "person", "a": {"channel": "cozy", "user_id": "9"}, "prio": 0, "at": time.time()}
    hub._calling = True                                            # its call still runs after Stop now
    hub._drain_llm()
    hub.stash_queue()                                              # an autosave meanwhile
    assert eng.llm_queue is None
    hub.stash_queue(include_job=True)                              # or a shutdown
    assert eng.llm_queue is None


# ------------------------------------------------ call shape: one line per call while the queue is short (Open 43)
def _line(ch, ts, prio=0, age=0.0):
    return (prio, ts, {"channel": ch, "user_id": f"u{ts}", "id": f"m{ts}", "line": f"x: line {ts}",
                       "room": [f"r: before {ts}"], "prio": prio, "ts": ts, "q_at": time.time() - age})


def _shape_hub(solo=3.2):
    hub = Hub(Engine())
    hub.review_q = ReadQueue()
    if solo is not None:
        hub._q_cost = {"solo": solo}
    return hub


def test_a_short_queue_sends_the_best_line_alone_at_once():
    """window_shape RESULT.md: lines at window positions 2+ lose mild reads; a lone line is the measured shape.
    While the queue is short the head (best priority, then oldest) goes alone with its own room lines, no gather."""
    hub = _shape_hub()
    hub.tox_pending = [_line("a", 1, prio=3), _line("a", 2, prio=0), _line("a", 3, prio=0)]
    job = hub._next_job(time.time())
    assert job["kind"] == "toxwin" and [it["id"] for it in job["a"]["items"]] == ["m2"]
    assert job["a"]["items"][0]["room"] == ["r: before 2"]                       # its own context
    assert [x[1] for x in hub.tox_pending] == [1, 3]
    assert [it["id"] for it in hub._next_job(time.time())["a"]["items"]] == ["m3"]
    assert [it["id"] for it in hub._next_job(time.time())["a"]["items"]] == ["m1"]
    assert hub._next_job(time.time()) is None


def test_the_switch_to_windows_follows_the_measured_one_line_cost():
    """SOLO_CLEAR_S = 20 s: at 3.2 s per one-line call 6 waiting lines go alone, 7 are a backlog (a window that
    gathers as before); a slower model keeps fewer alone, a faster one more; an unmeasured cost sends the first line
    alone (it measures it)."""
    now = time.time()
    hub = _shape_hub(3.2)
    assert (hub._solo_n(3.2), hub._solo_n(5.0), hub._solo_n(25.0), hub._solo_n(0.5)) == (6, 4, 0, 40)
    hub.tox_pending = [_line("a", ts) for ts in range(6)]
    assert len(hub._next_job(now)["a"]["items"]) == 1
    hub.tox_pending = [_line("a", ts) for ts in range(7)]
    assert hub._next_job(now) is None                                          # a backlog: it gathers
    assert len(hub._next_job(now + serve.LINE_GATHER_S + 0.1)["a"]["items"]) == 7
    hub.tox_pending = [_line("a", ts) for ts in range(serve.WINDOW + 3)]
    assert len(hub._next_job(now)["a"]["items"]) == serve.WINDOW                # a full window: at once
    hub._q_cost["solo"] = 5.0
    hub.tox_pending = [_line("a", ts) for ts in range(5)]
    assert hub._next_job(now) is None                                          # 5 x 5 s > 20 s: a backlog
    hub._q_cost["solo"] = 2.0
    hub.tox_pending = [_line("a", ts) for ts in range(7)]
    assert len(hub._next_job(now)["a"]["items"]) == 1                          # faster: 10 lines go alone
    hub = _shape_hub(None)
    hub.tox_pending = [_line("a", ts) for ts in range(30)]
    assert len(hub._next_job(now)["a"]["items"]) == 1                          # unmeasured: alone, and measured


def test_solo_clear_zero_is_the_windows_only_batcher(monkeypatch):
    monkeypatch.setattr(serve, "SOLO_CLEAR_S", 0)
    now = time.time()
    hub = _shape_hub(None)
    hub.tox_pending = [_line("a", 1)]
    assert hub._solo_n(None) == 0 and hub._next_job(now) is None               # a lone line gathers, as before
    assert len(hub._next_job(now + serve.LINE_GATHER_S + 0.1)["a"]["items"]) == 1


def test_stop_finishing_keeps_windows():
    """Stop drains the queue in windows, at once (the fastest drain), also when it is short."""
    hub = _shape_hub(3.2)
    hub.tox_pending = [_line("a", ts) for ts in range(3)] + [_line("b", 10)]
    hub.finishing = True
    job = hub._next_job(time.time())
    assert job["a"]["channel"] == "a" and [it["ts"] for it in job["a"]["items"]] == [0, 1, 2]


def test_each_call_shape_is_measured_and_counted():
    hub = _shape_hub(None)
    hub._served("line", 1, 3.0)
    hub._served("line", 12, 24.0)
    hub._served("read", 1, 6.0)
    assert hub._q_cost == {"solo": 3.0, "line": 2.0, "read": 6.0}
    s = hub.eng.stats
    assert (s["llm_tox_solo_calls"], s["llm_tox_solo_lines"], s["llm_tox_solo_ms"]) == (1, 1, 3000)
    assert (s["llm_tox_window_calls"], s["llm_tox_window_lines"], s["llm_tox_window_ms"]) == (1, 12, 24000)
    hub._served("line", 1, 4.0)
    assert abs(hub._q_cost["solo"] - 3.2) < 1e-9 and hub._q_cost["line"] == 2.0


def test_the_catch_up_estimate_costs_each_shape():
    """The backlog drains in windows down to the lines that go alone; new lines are costed alone while the queue is
    short and one per call keeps up with them, else in windows."""
    now = time.time()
    hub = _shape_hub(None)
    hub._q_cost = {"read": 5.0, "solo": 3.0, "line": 1.5}                    # 6 lines go alone
    hub.tox_pending = [_line("a", ts) for ts in range(30)]
    assert hub._llm_status(now)["catchup_s"] == 6 * 3 + 24 * 1.5              # 54 s, no arrivals
    hub.tox_pending = [_line("a", ts) for ts in range(4)]
    assert hub._llm_status(now)["catchup_s"] == 12
    hub._q_in = [(now - i, "line") for i in range(30)]                         # 0.1 line/s: alone keeps up
    assert hub._llm_status(now)["catchup_s"] == round(12 / (1 - 0.3))
    hub.tox_pending = [_line("a", ts) for ts in range(30)]                     # a backlog: arrivals join windows
    assert hub._llm_status(now)["catchup_s"] == round(54 / (1 - 0.15))
    hub._q_in = [(now - i / 4, "line") for i in range(120)]                    # 0.4 line/s: alone would not keep up
    hub.tox_pending = [_line("a", ts) for ts in range(4)]
    assert hub._llm_status(now)["catchup_s"] == round(12 / (1 - 0.6))
    hub._q_in = []
    hub._q_cost = {"solo": 3.0}                                                # one shape measured: it for both
    hub.tox_pending = [_line("a", ts) for ts in range(10)]
    assert hub._llm_status(now)["catchup_s"] == 30
    hub._q_cost = {"line": 1.5}
    assert hub._llm_status(now)["catchup_s"] == 15
    hub._q_cost = {}
    assert hub._llm_status(now)["catchup_s"] is None


def test_the_worker_reads_a_quiet_queue_one_line_per_call(monkeypatch):
    eng, hub, t = _live_hub(person=False)
    calls = []
    monkeypatch.setattr(llm, "rate_toxicity_window",
                        lambda room_, lines, emotes=frozenset(): calls.append((room_, lines)) or [LOW] * len(lines))
    serve.handle(hub, msg("first", "you are trash", t))
    serve.handle(hub, msg("second", "ur bad lol", t + 1))
    n = len(hub.tox_pending)
    assert n >= 2

    async def go():
        task = asyncio.create_task(hub.reviewer())
        await _until(lambda: eng.stats["llm_tox_reads"] >= n, 2.0)            # well inside the 4 s gather
        task.cancel()
    asyncio.run(go())
    assert [len(lines) for _, lines in calls] == [1] * n and eng.stats["llm_tox_solo_calls"] == n
    assert not eng.stats["llm_tox_window_calls"]
