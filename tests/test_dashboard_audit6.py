"""The live dashboard after round 3 of the audit of 2026-11-01b (docs/review/audit_20261101b/live_backend_r3.md,
LB6-n). No network, no LLM; the chat is synthetic, every file in tmp_path."""
import os

import pytest

from chatwatch import fileio, serve
from chatwatch import roles as R
from chatwatch.engine import Engine

from test_dashboard import _http
from test_dashboard_audit4 import _flush
from test_scenarios import T0, msg, room

HOT = ["you are a stupid idiot streamer", "shut up you moron", "this chat is full of idiots"]


# ------------------------------------------------------------------ LB6-1: the user card at the watchlist's time
@pytest.mark.parametrize("later_min", [0, 11, 20])
def test_on_a_quiet_live_chat_the_user_card_and_the_watchlist_agree(monkeypatch, later_min):
    """S3b: 4 toxic lines, then no chat. At +11 min the watchlist had dropped '3 toxic messages in 10 min' but the
    card still gave it; at +20 min the watchlist had moved them to Earlier with the card unchanged."""
    eng = Engine()
    t = room(eng)
    for i in range(4):
        eng.process(msg("hot", HOT[i % 3] + f" {i}", t + i, uid="6161"))
    eng.clock = T0 + (t + 3) * 1000
    hub = serve.Hub(eng)
    hub.status.update(mode="live", running=True, channels=["cozy"])
    wall = (eng.clock + later_min * 60_000) / 1000
    monkeypatch.setattr(serve.time, "time", lambda: wall)
    r = next(x for x in _flush(hub) if x["type"] == "room")
    live = [x for x in r["notices"]["items"] if x["user_id"] == "6161"]
    early = [x for x in r["notices_earlier"] if x["user_id"] == "6161"]
    shown = (live or early)[0]

    async def call(c):
        return await (await c.get("/api/profile?channel=cozy&uid=6161")).json()
    card = _http(hub, call)["notice"]
    assert (card["level"], card["why"]) == (shown["level"], shown["why"])
    assert bool(live) == (later_min < 20)
    assert ("3 toxic messages in 10 min" in card["why"]) == (later_min == 0)


def test_the_display_time_is_the_chat_clock_in_replay_and_never_behind_it_live(monkeypatch):
    eng = Engine()
    eng.clock = 5_000_000
    hub = serve.Hub(eng)
    monkeypatch.setattr(serve.time, "time", lambda: 9_000.0)
    hub.status["mode"] = "replay"
    assert hub._display_ts() == 5_000_000
    hub.status["mode"] = "live"
    assert hub._display_ts() == 9_000_000 and hub._display_ts(1_000.0) == 5_000_000


# ------------------------------------------------------------------ LB7-1: after a live Stop the card stops aging
@pytest.mark.parametrize("after_min", [11, 20, 60])
def test_after_a_live_stop_the_user_card_matches_the_last_list_sent(monkeypatch, after_min):
    """Stop 1 s after the last toxic line: the page keeps the last list sent ('3 toxic messages in 10 min'); the
    card, asked 11, 20 or 60 min later, must still read the same, not age on the wall clock."""
    eng = Engine()
    t = room(eng)
    for i in range(4):
        eng.process(msg("hot", HOT[i % 3] + f" {i}", t + i, uid="6161"))
    eng.clock = T0 + (t + 3) * 1000
    hub = serve.Hub(eng)
    hub.status.update(mode="live", running=True, channels=["cozy"])
    stop = (eng.clock + 1000) / 1000
    monkeypatch.setattr(serve.time, "time", lambda: stop)
    r = next(x for x in _flush(hub) if x["type"] == "room")
    shown = next(x for x in r["notices"]["items"] if x["user_id"] == "6161")
    hub.status.update(running=False, stopped_at=int(stop * 1000))
    monkeypatch.setattr(serve.time, "time", lambda: stop + after_min * 60)

    async def call(c):
        return await (await c.get("/api/profile?channel=cozy&uid=6161")).json()
    card = _http(hub, call)["notice"]
    assert (card["level"], card["why"]) == (shown["level"], shown["why"])
    assert "3 toxic messages in 10 min" in card["why"]
    assert hub._display_ts() == int(stop * 1000)


# ------------------------------------------------------------------ LB6-4: the console names both set-aside copies
def test_the_console_names_both_set_aside_roles_copies(tmp_path, capsys):
    p = str(tmp_path / "owner_roles.json")
    R.save({"4242": R.entry("bot", "victim", "victim")}, p)
    R.save({"4242": R.entry("bot", "victim", "victim"), "1": R.entry("bot", "x", "x")}, p)     # -> .prev
    for q in (p, p + fileio.PREV):
        s = open(q).read()
        open(q, "w").write(s[:-3])
    assert serve.live_roles(False, p) == {}
    out = capsys.readouterr().out
    aside = sorted(os.listdir(tmp_path))
    assert len(aside) == 2 and all(f in out for f in aside)
