"""Audit 2026-11-10 (TRACKER Open 42), the back half: one test per finding of docs/review/audit_20261110/FIXLIST.md fixed
in serve.py, llm.py, aictl.py, picker.py, engine.py, assets/models.json and start.sh. Each began as the auditor's probe
(data/scratch/audit1110/ in the audit worktree), which passed while the bug was there; here it asserts the fix."""
import asyncio
import json
import os
import subprocess
import threading
import time

from chatwatch import aictl, llm, picker, serve

from test_lossless_queue import LOW, _fast_retries, _live_hub, _until
from test_picker import (AMD16, NV12, FakeOllama, _hub, _hybrid, _machines)  # noqa: F401  (_machines: autouse)
from test_scenarios import msg

M31, M12 = "gemma4mtp:31b-n3-text", "gemma4mtp:12b-n2"
ROOT = os.path.join(os.path.dirname(__file__), "..")


def _check(hub, tag):
    async def go():
        hub.start_speed({"model": tag})
        await hub.ai_task_done()
    asyncio.run(go())
    return hub.ai_last["speed"]


def _line(i, prio=0, q_at=None, ch="cozy"):
    return (prio, i, {"channel": ch, "user_id": f"u{i}", "id": f"m{i}", "line": "x: you suck", "room": [],
                      "prio": prio, "ts": i, "q_at": time.time() if q_at is None else q_at})


# ------------------------------------------------------------------------------------------------ QL1
def test_ql1_a_failed_call_is_not_measured_and_not_counted_as_a_solo_call(monkeypatch):
    _fast_retries(monkeypatch)
    monkeypatch.setattr(serve, "RETRY_MAX", 1)                       # the first try is the last

    def slow_fail(room_, lines, emotes=frozenset()):
        time.sleep(0.3)
        raise TimeoutError("timed out")
    monkeypatch.setattr(llm, "rate_toxicity_window", slow_fail)
    eng, hub, t = _live_hub(person=False)
    hub._q_cost["solo"] = 0.1
    serve.handle(hub, msg("first", "you are trash", t))

    async def go():
        task = asyncio.create_task(hub.reviewer())
        await _until(lambda: eng.stats["llm_tox_lines_failed"] >= 1, 3.0)
        await asyncio.sleep(0.05)
        task.cancel()
    asyncio.run(go())
    s = eng.stats
    assert s["llm_tox_lines_failed"] == 1 and s["llm_tox_errors"] == 1
    assert s["llm_tox_solo_calls"] == 0 and s["llm_tox_solo_lines"] == 0 and s["llm_tox_solo_ms"] == 0
    assert hub._q_cost == {"solo": 0.1}                              # its 0.3 s did not enter the cost


def test_ql1_a_late_result_and_a_gone_chatter_are_not_measured(monkeypatch):
    eng, hub, t = _live_hub()
    hub.run_gen = 0

    def stop_meanwhile(room_, lines, emotes=frozenset()):
        hub.run_gen += 1                                             # Stop came while it ran
        return [LOW] * len(lines)
    monkeypatch.setattr(llm, "rate_toxicity_window", stop_meanwhile)
    a = {"channel": "cozy", "items": [_line(1)[2]], "user_id": "u1"}
    assert asyncio.run(hub._review_one("toxwin", a)) is None
    assert asyncio.run(hub._review_one("person", {"channel": "cozy", "user_id": "nobody"})) is None
    assert asyncio.run(hub._review_one("screen", {"channel": "cozy", "user_id": "nobody"})) is None


def test_ql1_the_models_load_time_is_left_out_of_the_call_cost(monkeypatch):
    # Ollama's load_duration (ns) in the reply
    monkeypatch.setattr(llm, "_post", lambda body, timeout: {"message": {"content": "{}"}, "load_duration": 2.5e9})
    llm._chat({"model": M31}, 5)
    assert abs(llm.last_load_s() - 2.5) < 1e-9
    # a call that found the model unloaded: the failed try and the reload are load time too
    tries = []

    def post(body, timeout):
        tries.append(1)
        if len(tries) == 1:
            time.sleep(0.05)
            raise TimeoutError("timed out")
        return {"message": {"content": "{}"}}
    monkeypatch.setattr(llm, "_post", post)
    monkeypatch.setattr(llm, "loaded", lambda m: False)
    monkeypatch.setattr(llm, "warm", lambda m, timeout=600: time.sleep(0.05))
    llm._chat({"model": M31}, 5)
    assert llm.last_load_s() >= 0.1
    # the queue's measured seconds: the call's wall time less its load
    eng, hub, t = _live_hub()

    def call():
        time.sleep(0.25)
        llm._call_info.load_s = 0.2
        return "ok"
    res, secs = asyncio.run(hub._timed(call))
    assert res == "ok" and 0.0 <= secs < 0.15


def test_ql1_the_measured_costs_start_again_when_the_model_changes():
    eng, hub, t = _live_hub(person=False)
    hub.apply_choice({"mode": "single", "live": "gemma4mtp:12b-n2"}, caps={})
    hub._q_cost.update(solo=0.5, line=0.3, read=1.0)                 # the 12B measured these
    hub.apply_choice({"mode": "single", "live": "gemma4mtp:12b-n2"}, caps={})
    assert hub._q_cost == {"solo": 0.5, "line": 0.3, "read": 1.0}    # the same model: kept
    hub.apply_choice({"mode": "single", "live": M31}, caps={})
    assert hub._q_cost == {}                                         # the 31B: measured afresh (its first call alone)
    hub.tox_pending = [_line(i) for i in range(10)]
    assert hub._llm_status()["catchup_state"] == "measuring"


# ------------------------------------------------------------------------------------------------ QL2
def test_ql2_stop_now_during_a_window_counts_its_lines_as_dropped_at_stop(monkeypatch):
    started, release = threading.Event(), threading.Event()

    def hung(room_, lines, emotes=frozenset()):
        started.set()
        release.wait(5)
        return [LOW] * len(lines)
    monkeypatch.setattr(llm, "rate_toxicity_window", hung)
    eng, hub, t = _live_hub(person=False)
    hub.status["running"] = True
    hub.tox_pending = [_line(i, q_at=time.time() - 10) for i in range(12)]
    hub._q_cost["solo"] = 100.0                                      # a backlog: one window of 12

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
    s = eng.stats
    assert s["llm_dropped_at_stop"] == 12 and s["llm_late_dropped"] == 0 and s["llm_tox_reads"] == 0


# ------------------------------------------------------------------------------------------------ QL3
def test_ql3_a_line_the_model_leaves_out_is_asked_again_alone_then_kept(monkeypatch):
    calls = []

    def skip_second(room_, lines, emotes=frozenset()):
        calls.append(list(lines))
        return [LOW if "keep" in ln else None for ln in lines]          # the model leaves out every "skip" line
    monkeypatch.setattr(llm, "rate_toxicity_window", skip_second)
    eng, hub, t = _live_hub(person=False)
    for who, text in (("alpha", "you are trash keep"), ("bravo", "you are trash skip"), ("charlie", "u suck keep")):
        serve.handle(hub, msg(who, text, t))
    hub.finishing = True                                             # Stop's drain: one window for all three
    job = hub._window(time.time())
    hub.finishing = False
    assert len(job["a"]["items"]) == 3
    asyncio.run(hub._review_one("toxwin", job["a"]))
    s = eng.stats
    assert s["llm_tox_reads"] == 2 and s["llm_tox_missing"] == 1 and s["llm_tox_missing_kept"] == 0
    assert len(hub.tox_pending) == 1 and hub.tox_pending[0][2]["missed"] == 1
    hub.tox_pending.extend(_line(10 ** 13 + i, prio=0) for i in range(20))       # a newer backlog behind it
    hub._q_cost["solo"] = 100.0
    again = hub._window(time.time())
    assert [it["line"] for it in again["a"]["items"]] == ["bravo: you are trash skip"]   # alone, at once
    asyncio.run(hub._review_one("toxwin", again["a"]))
    assert s["llm_tox_missing"] == 2 and s["llm_tox_missing_kept"] == 1                # a second miss: kept
    assert not any(x[2].get("missed") for x in hub.tox_pending)
    w = hub._window(time.time())                                     # a window never carries a missed line
    assert w is not None and len(w["a"]["items"]) == 12


# ------------------------------------------------------------------------------------------------ QL4
def test_ql4_kept_lines_that_age_out_before_start_show_under_aged_out(monkeypatch):
    eng, hub, t = _live_hub(person=False)
    hub.llm_open = False                                             # a live server before Start
    old = time.time() - 20 * 60
    saved = {"reads": [], "lines": [[2, t + i, {**_line(i, prio=2, q_at=old)[2]}] for i in range(5)]
                                   + [[0, t + 9, {**_line(9, prio=0, q_at=old)[2]}]]}
    assert hub.restore_queue(saved) == 6 and hub._restored_low == 5
    assert hub._age_out(time.time()) == 5                            # the sweeper, before anyone presses Start

    async def idle(*a, **k):
        await asyncio.sleep(3600)
    monkeypatch.setattr(serve, "live_source", idle)

    async def go():
        hub.start_live(["cozy"])
        q = hub._llm_status()
        hub.live_task.cancel()
        return q
    q = asyncio.run(go())
    assert q["aged_out"] == 5 and q["lines"] == 1
    hub.tox_pending = [_line(20, prio=3, q_at=old)]                  # one aged during this run: counted once
    hub._age_out(time.time())
    assert hub._llm_status()["aged_out"] == 6


# ------------------------------------------------------------------------------------------------ QL5
def test_ql5_the_status_tells_measuring_from_falling_behind():
    now = time.time()
    eng, hub, t = _live_hub(person=False)
    q = hub._llm_status(now)
    assert q["catchup_s"] is None and q["catchup_state"] == "idle"
    hub.tox_pending = [_line(i) for i in range(4)]
    q = hub._llm_status(now)
    assert q["catchup_s"] is None and q["catchup_state"] == "measuring"       # nothing timed yet
    hub._q_cost = {"solo": 3.0, "line": 1.5}
    q = hub._llm_status(now)
    assert q["catchup_s"] == 12 and q["catchup_state"] == "estimate"
    hub._q_in = [(now - i / 10, "line") for i in range(600)]         # 10 lines/s: far past what it serves
    q = hub._llm_status(now)
    assert q["catchup_s"] is None and q["catchup_state"] == "behind"
    hub._q_in = []
    hub.review_q.put(0, 1, "person", {"channel": "cozy", "user_id": "x"})
    assert hub._llm_status(now)["catchup_state"] == "measuring"     # a read, and no read timed yet


# ------------------------------------------------------------------------------------------------ PH1
def test_ph1_a_check_that_measured_nothing_saves_nothing_and_says_why(monkeypatch):
    fo = FakeOllama(monkeypatch)
    fo.loaded = [M31]
    monkeypatch.setattr(aictl, "CHECK_BUDGET_S", 0)
    real = fo.window

    def window(ctx, lines, **kw):
        if len(lines) == 1:                                          # a reply outside the schema every time
            raise ValueError("the LLM's toxicity is not one of none/low/medium/high")
        return real(ctx, lines, **kw)
    monkeypatch.setattr(llm, "rate_toxicity_window", window)
    eng, hub, t = _hub()
    r = _check(hub, M12)
    assert "too few calls succeeded" in r["error"] and "1-line windows" in r["error"] and "nothing was saved" in r["error"]
    assert M12 not in (hub.machine.get("speed") or {})
    assert M12 not in ((picker.load_machine("box", NV12) or {}).get("speed") or {})
    asyncio.run(hub.ai_info())                                       # the page's GET /api/ai still answers
    assert fo.loaded == [M31]                                        # and the live model is back


def test_ph1_a_bad_speed_entry_breaks_neither_the_picker_nor_the_status(monkeypatch, tmp_path):
    fo, eng, hub, t = _hybrid(monkeypatch)
    bad = {"at": "2026-11-10T00:00:00Z", "n_person": 5, "num_gpu": 30, "aborts": 0, "calls": 9}
    hub.machine.setdefault("speed", {})[M31] = bad                  # the hybrid's reviewer
    hub.machine["speed"][M12] = dict(bad)
    s = hub._status()                                                # every status message
    assert s["ai"]["review"]["basis"].startswith("tested on")
    rows = picker.rows(picker.classes(NV12), "nvidia", {M12, M31}, "0.40.11", hub.machine)
    assert not next(r for r in rows if r["id"] == M12)["speed"]["measured"]
    asyncio.run(hub.ai_info())
    good = {**picker.summarize([6.3] * 9, [2.7, 2.7], [18.4, 18.4]), "at": "2026-11-09T22:00:00Z"}
    picker.save_machine("box", NV12, {"speed": {M31: bad, M12: good}})
    assert list(picker.load_machine("box", NV12)["speed"]) == [M12]             # dropped on load


# ------------------------------------------------------------------------------------------------ PH2
def test_ph2_a_retried_timeout_is_a_retry_not_a_runner_abort(monkeypatch):
    fo = FakeOllama(monkeypatch)
    fo.loaded = [M31]
    monkeypatch.setattr(aictl, "CHECK_BUDGET_S", 0)
    real, state = fo.window, {"n": 0}

    def window(ctx, lines, **kw):
        state["n"] += 1
        if state["n"] == 1:
            raise TimeoutError("timed out")                          # one slow reply, retried and then fine
        return real(ctx, lines, **kw)
    monkeypatch.setattr(llm, "rate_toxicity_window", window)
    eng, hub, t = _hub()                                             # 12 GB Nvidia: the study's cap 30, no log
    assert hub._cap(M31)["value"] == 30
    r = _check(hub, M31)
    assert not r.get("error"), r
    assert r["aborts"] == 0 and r["retries"] == 1 and r["num_gpu"] == 30
    assert any("retried" in n and "not runner aborts" in n for n in r["notes"])
    assert hub.machine["caps"][M31]["value"] == 30 and hub.machine["speed"][M31]["retries"] == 1


def test_ph2_with_no_log_an_http_500_is_still_a_runner_abort(monkeypatch):
    fo = FakeOllama(monkeypatch)
    fo.loaded = [M31]
    fo.abort_until[M31] = 28                                         # HTTP 500 above 28 layers, no log to read
    monkeypatch.setattr(aictl, "CHECK_BUDGET_S", 0)
    eng, hub, t = _hub()
    r = _check(hub, M31)
    assert not r.get("error"), r
    assert r["num_gpu"] == 28 and r["aborts"] == 0 and any("stepping down to 28" in n for n in r["notes"])


# ------------------------------------------------------------------------------------------------ PH3
def test_ph3_the_reviewer_rereads_a_held_flagged_line_beyond_the_recent_40(monkeypatch):
    fo, eng, hub, t = _hybrid(monkeypatch)
    serve.handle(hub, msg("zulu", "you are trash and nobody likes you", t))
    u = next(u for u in eng.users.values() if u.login == "zulu")
    first = u.msgs[-1]["id"]
    assert any(mid == first for mid, n, seq in hub._queued[("cozy", u.uid)])
    for i in range(45):
        serve.handle(hub, msg("zulu", f"what game is this {i}", t + 1 + i))
    assert not any(str(m.get("id")) == str(first) for m in u.msgs) and eng.held_message("cozy", u.uid, first)
    u.review = {"label": "bad_faith", "rationale": "x", "reads": 1, "n": 1, "ts": 0, "pending": False}
    c = next(c for c in hub.review_candidates("b") if c["uid"] == u.uid)
    assert first in [m for m, _ in c["lines"]]

    async def go():
        hub.start_review({"scope": "b"})
        await hub.ai_task_done()
    asyncio.run(go())
    assert hub.ai_last["review"]["reviewed"] >= 1, hub.ai_last["review"]
    m = eng.held_message("cozy", u.uid, first)
    assert m["toxicity"]["source"] == "llm" and m["toxicity"].get("reviewer") == M31
    assert not any(str(x[2]["id"]) == str(first) for x in hub.tox_pending)    # the live queue's copy is covered


def test_ph3_scope_b_sees_a_toxic_line_kept_only_as_evidence(monkeypatch):
    fo, eng, hub, t = _hybrid(monkeypatch)
    serve.handle(hub, msg("yankee", "you are trash and nobody likes you", t))
    for i in range(45):
        serve.handle(hub, msg("yankee", f"what game is this {i}", t + 1 + i))
    u = next(u for u in eng.users.values() if u.login == "yankee")
    assert not any(m.get("toxicity", {}).get("level", "none") != "none" for m in u.msgs)
    assert any(c["uid"] == u.uid for c in hub.review_candidates("b"))


# ------------------------------------------------------------------------------------------------ PH4 / PH5 / PH6
def _reviewed_one(monkeypatch):
    from test_picker import _chatters
    fo, eng, hub, t = _hybrid(monkeypatch)
    _chatters(hub, eng, t, listed=("alpha",), flagged=())

    async def go():
        hub.start_review({"scope": "a"})
        await hub.ai_task_done()
    asyncio.run(go())
    assert hub.ai_last["review"]["reviewed"] == 1, hub.ai_last["review"]
    return fo, eng, hub, t, next(u for u in eng.users.values() if u.login == "alpha")


def test_ph4_forget_drops_the_reviewers_mark(monkeypatch):
    fo, eng, hub, t, u = _reviewed_one(monkeypatch)
    assert ("cozy", u.uid) in eng.__dict__["hybrid_reviewed"]
    eng.forget(u.uid)
    assert not any(k[1] == u.uid for k in eng.__dict__["hybrid_reviewed"])


def test_ph5_a_pruned_chatter_who_returns_is_reviewed_on_new_content(monkeypatch):
    fo, eng, hub, t, u = _reviewed_one(monkeypatch)
    u.last_seen = 0                                                  # idle for longer than PRUNE_IDLE_DAYS
    assert eng.prune(int(time.time() * 1000)) >= 1
    assert ("cozy", u.uid) not in eng.__dict__["hybrid_reviewed"]
    serve.handle(hub, msg("alpha", "you are trash and nobody likes you", t + 100))
    u2 = eng.users[("cozy", u.uid)]
    u2.review = {"label": "bad_faith", "rationale": "x", "reads": 1, "n": u2.n, "ts": 0, "pending": False}
    assert u2.n == 1 and [c for c in hub.review_candidates("a") if c["uid"] == u.uid]


def test_ph6_the_batch_counts_the_lines_it_reread(monkeypatch):
    fo, eng, hub, t, u = _reviewed_one(monkeypatch)
    r = hub.ai_last["review"]
    assert r["lines"] == eng.stats["hybrid_lines_reread"] == 2


# ------------------------------------------------------------------------------------------------ PH7
def test_ph7_a_check_on_a_machine_with_no_gpu_reads_no_fit_and_saves_no_cap(monkeypatch, tmp_path):
    log = str(tmp_path / "ollama.log")
    open(log, "w").close()
    cpu = {"host": "laptop", "gpu_vendor": "none", "gpu_name": None, "vram_mb": 0, "ram_mb": 31800, "cores": 8,
           "threads": 16, "ram_avail_mb": 26000}
    fo = FakeOllama(monkeypatch, log=log, fit={M12: (0, 49)})        # were the runner to log 0/49 on a CPU
    fo.loaded = []
    monkeypatch.setattr(aictl, "CHECK_BUDGET_S", 0)
    eng, hub, t = _hub(specs=cpu, log=log)
    assert hub._tier()["key"] == "F32" and hub._cap(M12)["source"] == "no GPU"
    r = _check(hub, M12)
    assert not r.get("error"), r
    assert r.get("fit") is None and r["num_gpu"] is None
    assert M12 not in (hub.machine.get("caps") or {}) and hub._cap(M12)["source"] == "no GPU"
    assert M12 in hub.machine["speed"]
    assert all(c[2] is None for c in fo.calls if c[0] == "warm" and c[1] == M12)


# a laptop with a 6 GB Nvidia card: under 7000 MB, so the CPU tier F32, but its GPU still takes layers
SMALL_GPU = {"host": "laptop", "gpu_vendor": "nvidia", "gpu_name": "RTX 3060 Laptop", "vram_mb": 6144, "ram_mb": 64000,
             "cores": 8, "threads": 16, "vram_avail_mb": 5600, "ram_avail_mb": 52000}


def test_ph_r2_1_a_small_gpu_on_a_cpu_tier_reads_the_fit_and_stores_its_cap(monkeypatch, tmp_path):
    """PH-R2-1 (round 2): "no GPU" comes from the hardware, not the tier. A 6 GB card on F32 is checked as before the
    sweep: Ollama's fit read (20/49), cap 17 stored "found here", the speed measured at 17, and the live cap 17 after."""
    # no free-VRAM read (the cap shown is "no GPU"), then read (the free-VRAM estimate 22); each its own machine file
    for specs in ({**{k: v for k, v in SMALL_GPU.items() if k != "vram_avail_mb"}, "host": "laptop-a"}, SMALL_GPU):
        llm.NUM_GPU_BY_MODEL.pop(M12, None)
        log = str(tmp_path / f"ollama{len(specs)}.log")
        open(log, "w").close()
        fo = FakeOllama(monkeypatch, log=log, fit={M12: (20, 49)})
        fo.abort_until[M12] = 17                                      # the runner aborts above 17 layers on this card
        fo.loaded = []
        monkeypatch.setattr(aictl, "CHECK_BUDGET_S", 0)
        eng, hub, t = _hub(specs=specs, log=log)
        assert hub._tier()["key"] == "F32" and not hub._no_gpu()
        before = hub._cap(M12)
        assert (before["source"], before["value"]) == (("no GPU", None) if "vram_avail_mb" not in specs else
                                                       ("estimated from 5.4 GB available", 22)), before
        r = _check(hub, M12)
        assert not r.get("error"), r
        assert r["fit"] == [20, 49] and r["num_gpu"] == 17 and not r["aborts"]
        warms = [c[2] for c in fo.calls if c[0] == "warm" and c[1] == M12]
        assert warms[0] is None and warms[-1] == 17                  # uncapped only to read the fit, then capped
        e = hub.machine["caps"][M12]
        assert e["value"] == 17 and e["tier"] == "F32" and e["how"] == "found"
        assert hub._cap(M12)["value"] == 17 and hub._cap(M12)["how"] == "found"
        assert llm.NUM_GPU_BY_MODEL.get(M12) == 17


def test_ph_r2_1_no_gpu_is_read_from_the_hardware(monkeypatch):
    FakeOllama(monkeypatch)
    cases = [({"gpu_vendor": "none", "vram_mb": 0}, True), ({"gpu_vendor": "cpu", "vram_mb": None}, True),
             ({"gpu_vendor": "nvidia", "vram_mb": 0}, True), ({"gpu_vendor": "nvidia", "vram_mb": 6144}, False),
             ({"gpu_vendor": "amd", "vram_mb": 4096}, False), ({"gpu_vendor": "nvidia", "vram_mb": None}, False)]
    for hw, want in cases:
        eng, hub, t = _hub(specs={"host": "laptop", "ram_mb": 31800, **hw})
        assert hub._no_gpu() is want, hw


# ------------------------------------------------------------------------------------------------ PH8
def test_ph8_a_checks_keep_up_is_the_tier_figure_for_the_same_timings():
    # what the page can list (picker.shown_rows: no plain twin of an MTP build, nothing in NOT_LISTED)
    listed = {s["id"] for s in picker.catalogue()["setups"]
              if s["id"] not in picker._plain_twins() and not s["id"].startswith(picker.NOT_LISTED)}
    assert listed and all(picker.is_mtp(x) for x in listed)
    same = total = rr_same = rr_total = 0
    above = []
    for s in picker.catalogue()["setups"]:
        for k, t in s["tiers"].items():
            n = picker.check_numbers(t["person_s"], t["a"], t["b"])
            total += 1
            same += n["keepup"] == t["keepup"]
            if n["keepup"] > t["keepup"]:
                above.append((s["id"], k))
            if s["id"] in listed:
                # PH-R2-2: never above the tier figure (never overstate keep-up), and in fact the same
                assert n["keepup"] <= t["keepup"], (s["id"], k, n)
                assert n["keepup"] == t["keepup"] and n["keepup_plus"] == bool(t["keepup_plus"]), (s["id"], k, n)
            else:
                steps = list(picker.KEEPUP_STEPS)
                assert abs(steps.index(n["keepup"]) - steps.index(t["keepup"])) <= 1, (s["id"], k, n)
            for key in ("all", "most"):
                if t.get(f"reread_{key}") is not None:
                    rr_total += 1
                    rr_same += n[f"reread_{key}"] == t[f"reread_{key}"]
                    steps = list(picker.REREAD_STEPS)
                    assert abs(steps.index(n[f"reread_{key}"]) - steps.index(t[f"reread_{key}"])) <= 1, (s["id"], k, key)
    # the one cell that reads a step above (PH-R2-2, named in picker.py's comment): the plain 12B on 8 GB, never listed
    assert above == [("gemma4:12b-it-qat", "E")], above
    assert same >= 19 and rr_same >= 35, (same, total, rr_same, rr_total)
    # the e4b + MTP on CPU 16 GB, the auditor's example: 20, as tested (it read 10)
    e4 = picker.setup("gemma4mtp:e4b-n4")["tiers"]["F16"]
    assert picker.check_numbers(e4["person_s"], e4["a"], e4["b"])["keepup"] == 20 == e4["keepup"]


# ------------------------------------------------------------------------------------------------ PH9
def test_ph9_two_gpus_under_one_host_name_keep_their_own_files():
    picker.save_machine("box", NV12, {"choice": {"mode": "single", "live": M31}, "caps": {M31: 30}})
    picker.save_machine("box", AMD16, {"choice": {"mode": "single", "live": M12}})       # same host, another GPU
    assert picker.load_machine("box", NV12)["choice"]["live"] == M31        # kept aside, not overwritten
    assert picker.load_machine("box", AMD16)["choice"]["live"] == M12
    d = picker.machines_dir()
    assert os.path.exists(os.path.join(d, f"box.{picker.gpu_sig(NV12)}.json"))
    assert os.path.exists(os.path.join(d, f"box.{picker.gpu_sig(NV12)}.env"))
    picker.save_machine("box", NV12, {**picker.load_machine("box", NV12), "override": {}})    # the 12 GB card again
    assert picker.load_machine("box", AMD16)["choice"]["live"] == M12
    assert picker.load_machine("box", NV12)["caps"][M31] == 30
    assert not os.path.exists(os.path.join(d, f"box.{picker.gpu_sig(NV12)}.json"))   # its own copy: the main file


def test_ph9_the_start_script_reads_this_gpus_file_set_aside(tmp_path):
    sh = open(os.path.join(ROOT, "start.sh")).read()
    part = sh[sh.index('MENV="$MDIR/$HOSTKEY.env"'):sh.index("AI_MODE=single")]
    d = tmp_path / "m"
    d.mkdir()
    (d / "box.env").write_text("CW_MACHINE_GPU=amd-16368\nCW_LIVE_MODEL=gemma4mtp:12b-n2\n")
    (d / "box.nvidia-12282.env").write_text("CW_MACHINE_GPU=nvidia-12282\nCW_LIVE_MODEL=gemma4mtp:31b-n3-text\n")
    stub = tmp_path / "t.sh"
    stub.write_text("set -euo pipefail\nMDIR=%s HOSTKEY=box\n" % d
                    + "CW_SPEC_GPU_VENDOR=nvidia CW_SPEC_VRAM_MB=12282\n" + part + 'echo "A $MENV"\n'
                    + "CW_SPEC_GPU_VENDOR=amd CW_SPEC_VRAM_MB=16368\n" + part + 'echo "B $MENV"\n')
    out = subprocess.run(["bash", str(stub)], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert f"A {d}/box.nvidia-12282.env" in out.stdout and f"B {d}/box.env" in out.stdout


# ------------------------------------------------------------------------------------------------ PH10 / FE-9b
def test_ph10_the_read_ahead_keeps_no_unread_fields(monkeypatch, tmp_path):
    root = tmp_path / "ollama"
    blobs = root / "models" / "blobs"
    blobs.mkdir(parents=True)
    d = "a" * 64
    man = root / "models" / "manifests-v2" / "ollama.com" / "library" / "gemma4mtp" / "12b-n2"
    man.parent.mkdir(parents=True)
    (blobs / f"sha256-{d}").write_bytes(b"x" * 2_000_000)
    man.write_text(json.dumps({"layers": [{"digest": f"sha256:{d}", "size": 2_000_000}]}))
    monkeypatch.setenv("CW_OLLAMA_MODELS", str(root))
    fo, eng, hub, t = _hybrid(monkeypatch)
    st = asyncio.run(hub._prefetch(M12))
    assert st["state"] == "ready" and not {"was_cached", "gb_per_s"} & set(st)


# ------------------------------------------------------------------------------------------------ FE-6
def test_fe6_no_log_copier_means_no_log_and_the_check_refuses(monkeypatch, tmp_path):
    sh = open(os.path.join(ROOT, "start.sh")).read()
    assert 'CW_OLLAMA_LOG="${TAIL_PGID:+/work/data/state/machines/$HOSTKEY.ollama.log}"' in sh
    fns = sh[sh.index('OLOG="$MDIR/$HOSTKEY.ollama.log"'):sh.index("start_tail() {")]
    after = sh[sh.index('[[ -z "$TAIL_PGID" ]] && rm -f "$OLOG"'):].splitlines()[0]
    olog = tmp_path / "box.ollama.log"
    olog.write_text("load_tensors: offloaded 29/61 layers to GPU\n")        # the last run's copy
    stub = tmp_path / "t.sh"
    stub.write_text(f"set -euo pipefail\nMDIR={tmp_path} HOSTKEY=box\n" + fns + after + "\n"
                    + 'test -e "$OLOG" && echo STALE || echo GONE\n'
                    + ': > "$OLOG"; TAIL_PGID=999999999; stop_tail; test -e "$OLOG" && echo KEPT || echo REMOVED\n')
    out = subprocess.run(["bash", str(stub)], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert "GONE" in out.stdout and "REMOVED" in out.stdout
    # the server with no log: an MTP build whose cap must be found here is refused, as without a log
    fo = FakeOllama(monkeypatch)
    fo.loaded = [M31]
    monkeypatch.setattr(aictl, "CHECK_BUDGET_S", 0)
    monkeypatch.setitem(picker.setup(M31)["tiers"]["A"], "num_gpu_known", False)
    eng, hub, t = _hub(log=None)
    hub.ollama_log = picker.OllamaLog(None)
    assert not hub._cap(M31)["known"]                                # a cap to find here: Ollama's fit is needed
    r = _check(hub, M31)
    assert "log is not readable" in r["error"]
    # PH-R2-3: only a --no-llm run gets here now, so the advice names it (not "start with start.sh")
    assert "without --no-llm" in r["error"] and "start the dashboard with start.sh" not in r["error"]
