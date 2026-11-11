"""The live dashboard after round 2 of the audit of 2026-11-01b (docs/review/audit_20261101b/live_backend_r2.md,
LB5-n). No network, no LLM; the chat is synthetic, every file in tmp_path."""
import copy
import os

import pytest

from chatwatch import fileio, serve
from chatwatch import roles as R
from chatwatch.engine import Engine

from test_dashboard_audit4 import _flush, _user
from test_scenarios import T0, msg, room

ANGRY = ["this streamer is trash", "so boring", "unwatchable", "fell off"]


# ------------------------------------------------------------------ LB5-1: the wall-clock snapshot changes no read
def _burst():
    """A calm room, then a burst of friction 1 s apart; eng.clock = the last event, as serve.handle sets it."""
    eng = Engine(legacy=False)
    t = room(eng)
    for i in range(10):
        eng.process(msg(f"angry{i}", ANGRY[i % 4], t + i))
    eng.clock = T0 + (t + 9) * 1000
    return eng, t


def _newcomer(eng, T):
    """A first-timer's friction line at chat time T (ms): what its record and their read come to."""
    rec = eng.process(msg("newbie", "this streamer is so bad lmao", (T - T0) / 1000, first=True))
    u = _user(eng, "newbie")
    return (eng.rooms["cozy"].short.n, u.instigated, u.joined, u.agg["b_heated"], round(u.badfaith(), 9),
            rec["toxicity"]["level"])


@pytest.mark.parametrize("ahead_ms", [100, 300, 1000])
def test_a_live_flush_ahead_of_a_message_in_flight_leaves_its_read_unchanged(monkeypatch, ahead_ms):
    """R1/R1b: a snapshot at wall time trimmed the room's windows; a message whose chat time was still behind the
    wall then read the room as calm (joined -> instigated, bad faith 0.46 -> 0.52, b_heated)."""
    eng0, t = _burst()
    end = T0 + (t + 10) * 1000
    bites = 0
    for k in range(0, 22):                       # arrival times across the edge where the burst leaves the 90 s
        T = end + 81000 + k * 50
        plain, flushed, early = copy.deepcopy(eng0), copy.deepcopy(eng0), copy.deepcopy(eng0)
        hub = serve.Hub(flushed)
        hub.status.update(mode="live", running=True, channels=["cozy"])
        hub.log_sessions = False
        with monkeypatch.context() as m:
            m.setattr(serve.time, "time", lambda: (T + ahead_ms) / 1000)
            assert any(r["type"] == "room" for r in _flush(hub))
        early.rooms["cozy"].friction_now(T + ahead_ms)        # what the LB4-4 snapshot did to the windows
        want = _newcomer(plain, T)
        assert _newcomer(flushed, T) == want
        bites += _newcomer(early, T) != want
    assert bites                                   # the band holds arrivals the old trim changed


def test_the_snapshot_reads_the_windows_as_a_trim_would_without_trimming():
    eng, t = _burst()
    r = eng.rooms["cozy"]
    for dt in (0, 30_000, 85_000, 95_000, 3_600_000):
        ts = eng.clock + dt
        before = (r.short.n, r.short.fr, r.long.n, r.long.fr, len(r.short.q), len(r.long.q))
        snap = r.snapshot(ts, full=True)
        assert (r.short.n, r.short.fr, r.long.n, r.long.fr, len(r.short.q), len(r.long.q)) == before
        twin = copy.deepcopy(r)
        cur, base, n = twin.friction_now(ts)
        assert snap["msgs_per_min"] == round(n * 60 / 90)
        assert (snap["friction_now"], snap["friction_usual"]) == (round(cur, 3), round(base, 3))
        assert snap["heated"] == twin.heated(ts)
        w = twin.short                            # friction_now trimmed the twin at ts
        assert snap["mood"] == round((w.pos - w.fr) / w.n if w.n else 0.0, 3)


# ------------------------------------------------------------------ LB5-2: forget names both set-aside roles copies
def test_forget_names_the_set_aside_prev_copy_of_the_roles_too(tmp_path):
    p = str(tmp_path / "owner_roles.json")
    R.save({"4242": R.entry("bot", "victim", "victim")}, p)
    R.save({"4242": R.entry("bot", "victim", "victim"), "1": R.entry("bot", "x", "x")}, p)     # -> .prev
    for q in (p, p + fileio.PREV):                # both torn, as a crash on the share would leave them
        s = open(q).read()
        open(q, "w").write(s[:-3])
    assert serve.live_roles(False, p) == {}
    aside = sorted(os.listdir(tmp_path))
    assert len(aside) == 2 and all(".unreadable-" in f for f in aside)
    out = []
    R.forget({"4242"}, False, p, log=out.append)
    assert sorted(line.split(":")[0] for line in out) == aside


# ------------------------------------------------------------------ LB5-3: first-timers this minute, on a quiet chat
def test_first_timers_this_minute_follow_the_display_time():
    eng = Engine(legacy=False)
    t = room(eng)
    eng.process(msg("fresh1", "hi first time here", t, first=True))
    eng.process(msg("fresh2", "hello!", t + 1, first=True))
    r = eng.rooms["cozy"]
    now = T0 + (t + 1) * 1000
    assert r.snapshot(now, full=False)["first_timers_this_min"] == 2
    later = r.snapshot(now + 3_600_000, full=True)
    assert later["first_timers_this_min"] == 0 and later["first_timer_share"] == 0 and later["msgs_per_min"] == 0
    assert r.snapshot(now, full=False)["first_timers_this_min"] == 2         # read-only: nothing was dropped
