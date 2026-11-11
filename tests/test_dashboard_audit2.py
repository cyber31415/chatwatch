"""The live dashboard after the second final-audit pass (docs/review/audit_20261029b/live_backend.md, LB2-n, and
live_frontend.md, N-n). No network, no LLM: the LLM, the register and the browsers are fakes."""
import asyncio
import json

from chatwatch import engine as E
from chatwatch.serve import ReadQueue
from chatwatch.engine import Engine

from test_scenarios import NEW_ID, msg, room


def _user(eng, login):
    return next(u for u in eng.users.values() if u.login == login)


# ------------------------------------------------------------------ compliance LB2-2: an alert needs a toxic message
def test_not_toxic_on_the_alerting_message_takes_the_alert_back_and_undo_restores_it():
    """ALERT_NEEDS_TOXIC (FINDINGS 2026-10-25) and DASHBOARD.md "Not toxic removes the message; if it was the only
    reason, the chatter leaves the watchlist": the alert could not have been raised without that message."""
    eng = Engine()
    t = room(eng)
    rec = eng.process(msg("newacct", "you are a stupid idiot streamer kys", t, first=True, uid=NEW_ID))
    assert any(a["kind"] == "badfaith" for a in rec.get("alerts", ()))
    u = _user(eng, "newacct")
    assert E._notice_reasons(u, None, u.last_seen, False) is not None
    eng.apply_owner_override("cozy", u.uid, rec["id"], "none")
    assert u.toxic_n == 0 and not u.alerted
    assert E._notice_reasons(u, None, u.last_seen, False) is None                  # the older rule
    u.review = {"label": "mild", "rationale": "r", "reads": 1, "pending": False}
    assert E._notice_reasons(u, None, u.last_seen, True) is None                   # D7: mild + an alert no more
    # "Back to automatic": the toxic message counts again, and so does the alert it raised
    eng.apply_owner_override("cozy", u.uid, rec["id"], None)
    assert u.toxic_n == 1 and u.alerted
    assert "bad-faith pattern" in " ".join(E._notice_reasons(u, None, u.last_seen, False)[1])   # the older rule
    assert E._notice_reasons(u, None, u.last_seen, True) is None    # DB (2026-11-02): a mild read makes no card


def test_a_correction_that_leaves_another_toxic_message_keeps_the_alert():
    eng = Engine()
    t = room(eng)
    rec = eng.process(msg("newacct", "you are a stupid idiot streamer kys", t, first=True, uid=NEW_ID))
    u = _user(eng, "newacct")
    r2 = eng.process(msg("newacct", "you are trash, worst streamer ever", t + 5, uid=NEW_ID))
    assert u.alerted and u.toxic_n == 2, r2["toxicity"]
    eng.apply_owner_override("cozy", u.uid, rec["id"], "none")
    assert u.toxic_n == 1 and u.alerted


# ------------------------------------------------------------------ compliance LB2-3: a read dropped at Stop is lost
def _three_toxic_and_alert():
    """A chatter with 3 toxic messages in 10 min and a bad-faith alert, person reads on (probe P7)."""
    eng = Engine()
    eng.person_review = True
    t = room(eng)
    recs = [eng.process(msg("newacct", "you are a stupid idiot streamer kys", t, first=True, uid=NEW_ID))]
    recs += [eng.process(msg("newacct", "you are trash, worst streamer ever", t + 20 * i, uid=NEW_ID)) for i in (1, 2)]
    u = _user(eng, "newacct")
    assert u.alerted and u.toxic_n == 3
    return eng, u, recs


def test_a_person_read_dropped_at_stop_or_lost_at_a_restart_falls_back_to_the_older_rule():
    """LB-3 (DASHBOARD.md): "anyone whose last person read failed ... or is lost ... is judged by [the older rule]
    until a read comes back". Before LB2-3 a read released at Stop, or pending when the state was saved, left
    {reads: 0, pending: False}: neither a read nor the fallback, so the chatter dropped off "possible"."""
    from chatwatch import serve
    eng, u, recs = _three_toxic_and_alert()
    eng.clock = recs[-1]["ts"]
    hub = serve.Hub(eng)
    hub.review_q = ReadQueue()
    hub._queue_person("cozy", u.uid, recs[-1]["ts"])
    assert E._notice_reasons(u, None, u.last_seen, True) is None          # pending: D7 waits for the read
    hub._drain_llm()                                                       # Stop
    assert u.review["lost"] and not u.review["pending"] and u.review.get("reads", 0) == 0
    nr = E._notice_reasons(u, None, u.last_seen, True)
    assert nr and nr[0] == "watch"                                         # the older rule: 3 toxic in 10 min
    # a read that comes back starts D7 again
    eng.apply_person_review("cozy", u.uid, {"label": "good_faith", "rationale": "banter"}, recs[-1]["ts"])
    assert "lost" not in u.review and E._notice_reasons(u, None, u.last_seen, True) is None
    # a read pending when the state was saved: upgrade() at the next start marks it lost
    eng2, u2, recs2 = _three_toxic_and_alert()
    eng2.mark_review_pending("cozy", u2.uid, recs2[-1]["ts"])
    eng2.upgrade()
    assert u2.review["lost"] and E._notice_reasons(u2, None, u2.last_seen, True)[0] == "watch"


def test_a_late_read_after_stop_is_not_applied_and_leaves_the_read_lost(monkeypatch):
    from chatwatch import serve
    eng, u, recs = _three_toxic_and_alert()
    eng.clock = recs[-1]["ts"]
    hub = serve.Hub(eng)
    hub.review_q = ReadQueue()
    eng.mark_review_pending("cozy", u.uid, recs[-1]["ts"])

    async def go():
        async def slow(fn):
            hub.run_gen += 1                     # Stop comes while the read runs
            return {"label": "bad_faith", "rationale": "x"}
        monkeypatch.setattr(hub, "_llm", slow)
        await hub._review_one("person", {"channel": "cozy", "user_id": u.uid})
    asyncio.run(go())
    assert not u.review.get("label") and u.review["lost"] and eng.stats["llm_late_dropped"] == 1
    assert E._notice_reasons(u, None, u.last_seen, True)[0] == "watch"


# ------------------------------------------------------------------ the override endpoint (LB2-1, 4-9, 12, 13; N-3, N-5, N-11)
PAGE = {"X-Chatwatch": "1"}


def _hub(monkeypatch, tmp_path, mode="live", running=True, persist=True, person_review=True):
    """A dashboard Hub with a plain and an insult message from 'corr', the register in tmp_path, and a review queue."""
    from chatwatch import serve, session as S
    eng = Engine()
    t = room(eng)
    a, b = msg("corr", "nice dodge, that boss is brutal", t), msg("corr", "you are a stupid idiot streamer", t + 10)
    eng.process(a)
    eng.process(b)
    eng.clock = b["ts"]
    eng.person_review = person_review
    hub = serve.Hub(eng)
    hub.status.update(mode=mode, running=running)
    hub.persist = persist
    hub.review_q = ReadQueue()
    reg = tmp_path / "owner_overrides.jsonl"
    monkeypatch.setattr(S, "OVERRIDES", str(reg))
    hub.logged = []
    monkeypatch.setattr(hub, "_log", lambda rec: hub.logged.append(rec))
    return hub, eng.users[("cozy", a["user_id"])], a["id"], b["id"], reg


def _post(hub, *bodies, raw=False):
    from test_dashboard import _http

    async def calls(c):
        out = []
        for body in bodies:
            r = await (c.post("/api/override", data=body, headers={**PAGE, "Content-Type": "application/json"}) if raw
                       else c.post("/api/override", json=body, headers=PAGE))
            out.append((r.status, await r.json()))
        return out
    return _http(hub, calls)


def _lines(reg):
    return [json.loads(x) for x in reg.read_text().splitlines()] if reg.exists() else []


def test_a_correction_after_stop_is_recorded_but_starts_no_llm_work(monkeypatch, tmp_path):
    """LB2-1 (reopened LB-6): no LLM work while no live session runs."""
    hub, u, plain, _, reg = _hub(monkeypatch, tmp_path, running=False)
    u.review = None
    (st, j), = _post(hub, {"channel": "cozy", "uid": u.uid, "id": plain, "level": "medium"})
    assert st == 200 and j["toxicity"]["source"] == "owner" and j["recorded"]
    assert hub.review_q.qsize() == 0 and not (u.review or {}).get("pending")
    assert len(_lines(reg)) == 1                                  # the saved state holds it: recorded


def test_a_correction_on_a_replay_or_an_unsaved_run_steers_the_view_only(monkeypatch, tmp_path):
    """LB2-12, N-11: a replay, or a --fresh run that is not saved, writes nothing to the register (the set
    builders would leave those messages out for good) and, on a replay, asks for no person read."""
    hub, u, plain, insult, reg = _hub(monkeypatch, tmp_path, mode="replay", running=False, persist=False)
    u.review = None
    (st, j), = _post(hub, {"channel": "cozy", "uid": u.uid, "id": plain, "level": "high", "kind": "hate"})
    assert st == 200 and not j["recorded"] and E._sure_hate(next(m for m in u.msgs if m["id"] == plain)["toxicity"])
    assert not reg.exists() and hub.review_q.qsize() == 0
    hub2, u2, plain2, _, reg2 = _hub(monkeypatch, tmp_path / "f", mode="live", running=True, persist=False)
    (tmp_path / "f").mkdir()
    u2.review = None
    (st, j), = _post(hub2, {"channel": "cozy", "uid": u2.uid, "id": plain2, "level": "low"})
    assert st == 200 and not j["recorded"] and not reg2.exists()
    assert hub2.review_q.qsize() == 1                             # a live session runs: its LLM may read them
    assert any(r.get("type") == "owner_override" and r["recorded"] is False for r in hub2.logged)


def test_a_register_write_that_fails_is_reported_and_the_correction_not_applied(monkeypatch, tmp_path):
    """N-3: the set builders leave corrected messages out only through the register."""
    from chatwatch import serve
    hub, u, plain, _, reg = _hub(monkeypatch, tmp_path)

    def ro(rec, path=None):
        raise OSError(30, "Read-only file system")
    monkeypatch.setattr(serve, "log_override", ro)
    before = dict(next(m for m in u.msgs if m["id"] == plain)["toxicity"])
    (st, j), = _post(hub, {"channel": "cozy", "uid": u.uid, "id": plain, "level": "high", "kind": "hate"})
    assert st == 500 and j["not_recorded"] and "not applied" in j["error"]
    assert next(m for m in u.msgs if m["id"] == plain)["toxicity"] == before
    assert hub.eng.stats["override_log_errors"] == 1 and not hub.queue and hub.review_q.qsize() == 0


def test_malformed_bodies_and_pairs_the_menu_never_sends_are_refused(monkeypatch, tmp_path):
    """LB2-5 (500s with a traceback) and LB2-6 (low + hate made a 'likely' slur card)."""
    hub, u, plain, _, reg = _hub(monkeypatch, tmp_path)
    raw = _post(hub, "[1, 2]", '"text"', "not json", raw=True)
    assert [s for s, _ in raw] == [400, 400, 400]
    base = {"channel": "cozy", "uid": u.uid, "id": plain}
    bad = [{"level": []}, {"level": {}}, {"level": "low", "kind": 5}, {"level": "low", "kind": "hate"},
           {"level": "none", "kind": "threat"}, {"level": "high"}, {"level": "medium", "kind": "harassment"},
           {"level": None, "kind": "hate"}, {"level": "extreme"}]
    out = _post(hub, *({**base, **b} for b in bad))
    assert [s for s, _ in out] == [400] * len(bad), out
    assert all(not E._owner(m["toxicity"]) for m in u.msgs) and not reg.exists()
    assert hub.eng.apply_owner_override("cozy", u.uid, plain, "low", "hate") is None


def test_repeated_corrections_count_kinds_once_and_undo_takes_them_back(monkeypatch, tmp_path):
    """LB2-7, N-5: one message corrected hate, hate (a repeated click), threat, hate, back to automatic."""
    hub, u, plain, _, reg = _hub(monkeypatch, tmp_path)
    auto = +hub.eng.users[("cozy", u.uid)].toxic_kinds
    one = lambda lv, k=None: {"channel": "cozy", "uid": u.uid, "id": plain, "level": lv, "kind": k}  # noqa: E731
    out = _post(hub, one("high", "hate"), one("high", "hate"))
    assert out[1][1]["unchanged"] and len(_lines(reg)) == 1          # the repeat records nothing
    assert +u.toxic_kinds == auto + __import__("collections").Counter({"hate": 1})
    _post(hub, one("high", "threat"), one("high", "hate"), one(None))
    assert +u.toxic_kinds == auto and u.toxic_n == 1                   # only the insult's own kinds are left
    prof = hub.eng.profile("cozy", u.uid)
    assert all(v > 0 for _, v in prof["toxic"]["kinds"])
    assert len(_lines(reg)) == 4


def test_only_a_correction_that_makes_the_chatter_toxic_spends_a_person_read(monkeypatch, tmp_path):
    """LB2-4: three corrections of one message used to spend all REVIEW_MAX reads."""
    hub, u, plain, insult, reg = _hub(monkeypatch, tmp_path)
    _post(hub, {"channel": "cozy", "uid": u.uid, "id": insult, "level": "none"})
    assert u.toxic_n == 0
    u.review = {"label": "good_faith", "rationale": "x", "reads": 1, "pending": False}
    one = lambda lv, k=None: {"channel": "cozy", "uid": u.uid, "id": plain, "level": lv, "kind": k}  # noqa: E731
    _post(hub, one("low"))
    assert hub.review_q.qsize() == 1                                  # no toxic message before: read them
    hub.review_q.get_nowait()
    hub.person_queued.clear()
    u.review = {"label": "mild", "rationale": "x", "reads": 2, "pending": False}
    _post(hub, one("medium"), one("high", "threat"))
    assert hub.review_q.qsize() == 0 and u.review["reads"] == 2        # already toxic, already read: no new read


def test_back_to_automatic_on_llm_only_hate_asks_for_the_s11_read(monkeypatch, tmp_path):
    """LB2-9 (probe_undo_s11) and LB2-8: a re-read under the correction refreshes its tooltip and is what the
    undo restores; restoring LLM-only hate asks for the person read S11 needs."""
    hub, u, plain, _, reg = _hub(monkeypatch, tmp_path)
    u.review = {"label": "good_faith", "rationale": "x", "reads": 1, "pending": False}
    _post(hub, {"channel": "cozy", "uid": u.uid, "id": plain, "level": "none"})
    hub.review_q = ReadQueue()
    upd = hub.eng.apply_llm_toxicity("cozy", u.uid, plain, {"toxicity": "high", "kind": "hate", "target": "third_party"})
    assert upd and upd["toxicity"]["source"] == "owner" and upd["toxicity"]["auto"] == {
        "level": "high", "kinds": ["hate"], "source": "llm"}                     # the tooltip follows the re-read
    (st, j), = _post(hub, {"channel": "cozy", "uid": u.uid, "id": plain, "level": None})
    assert st == 200 and j["toxicity"]["source"] == "llm" and E._tox_class(j["toxicity"]) == "hate"
    assert hub.review_q.qsize() == 1                                            # S11's read


def test_the_register_records_the_level_the_message_counted_at(monkeypatch, tmp_path):
    """LB2-13: an S1 re-read of a keyword slur counts at the layer's level, not the re-read's 'none'."""
    hub, u, plain, _, reg = _hub(monkeypatch, tmp_path)
    m = next(x for x in u.msgs if x["id"] == plain)
    m["toxicity"] = {"level": "none", "score": 0, "kinds": [], "source": "llm",
                     "layer": {"level": "high", "kinds": ["hate"], "sure": ["hate"]}}
    _post(hub, {"channel": "cozy", "uid": u.uid, "id": plain, "level": "none"})
    assert _lines(reg)[0]["before"] == {"level": "high", "class": "hate", "source": "llm"}


def test_the_watchlist_card_says_when_the_owner_set_its_reason():
    """N-4: a card made by an owner correction reads "set by you", not as the tool's own slur finding."""
    eng = Engine()
    t = room(eng)
    r = eng.process(msg("lefty", "lets keep politics out of this chat please", t))
    eng.apply_owner_override("cozy", r["user_id"], r["id"], "high", "hate")
    items = eng.notices("cozy", r["ts"])[0]["items"]
    card = next(d for d in items if d["user_id"] == r["user_id"])
    assert card["level"] == "act" and card["owner"] and card["quote_owner"]
    assert eng.profile("cozy", r["user_id"])["notice"]["owner"]
    r2 = eng.process(msg("slurguy", "you are a nigger", t + 5))
    card2 = next(d for d in eng.notices("cozy", r2["ts"])[0]["items"] if d["user_id"] == r2["user_id"])
    assert not card2["owner"] and not card2["quote_owner"]


# ------------------------------------------------------------------ LB2-10: Stop soon after Start
def test_stop_soon_after_start_does_not_freeze_the_loop(monkeypatch, tmp_path):
    """Session.close waits for the state copy; it used to run on the event loop (probe P8: a 4 s stall)."""
    import threading
    import time
    from chatwatch import serve, session
    release = threading.Event()
    monkeypatch.setattr(session.Session, "_prepare", lambda self: release.wait(5))
    monkeypatch.setattr(session, "ROOT", str(tmp_path))
    real_init = session.Session.__init__
    monkeypatch.setattr(session.Session, "__init__",
                        lambda self, ch, argv, eng, base=None, ai=None: real_init(self, ch, argv, eng, base=str(tmp_path / "s"), ai=ai))

    async def idle(*a):
        await asyncio.sleep(3600)
    monkeypatch.setattr(serve, "live_source", idle)
    hub = serve.Hub(Engine())
    hub.status["mode"] = "live"

    async def go():
        hub.start_live(["cozy"])
        ticks = []

        async def ticker():
            while True:
                ticks.append(time.monotonic())
                await asyncio.sleep(0.02)
        tk = asyncio.create_task(ticker())
        threading.Timer(0.6, release.set).start()             # the copy takes 0.6 s
        t0 = time.monotonic()
        await hub.stop_live()
        took = time.monotonic() - t0
        await asyncio.sleep(0.1)                               # the ticker runs again: a stall shows as a gap
        tk.cancel()
        gaps = [b - a for a, b in zip(ticks, ticks[1:])]
        return took, max(gaps or [0])
    took, worst = asyncio.run(go())
    assert took >= 0.5 and worst < 0.3, (took, worst)
    assert hub.session is None and not hub.status["running"]


# ------------------------------------------------------------------ small residues
def test_forget_skips_register_lines_that_are_not_objects(tmp_path):
    """LB2-16: a JSON line that is not an object ended `python -m chatwatch.forget --go` half-way."""
    from chatwatch import session as S
    reg = tmp_path / "o.jsonl"
    reg.write_text('[1, 2]\n"x"\n{"user_id": "1", "msg_id": "a"}\n{"user_id": "2", "msg_id": "b"}\n')
    assert S.forget_overrides({"1"}, True, str(reg)) == 1 and S.owner_override_ids(str(reg)) == {"b"}


def test_the_status_names_a_head_only_when_the_run_scores_with_it(monkeypatch):
    """LB2-18: --ml off, the pipeline page's "restart needed" compared a head the dashboard does not use."""
    from chatwatch import serve
    monkeypatch.setattr(serve, "loaded_head_sha", lambda: "abc")
    assert serve.head_for(None) is None and serve.head_for(object()) == "abc"


def test_env_cw_names_this_machine_in_the_lock():
    """LB2-19: a dashboard started with bin/cw directly wrote the container's hostname into the lock."""
    assert '-e CW_HOST="${CW_HOST:-${HOSTNAME:-$(uname -n)}}"' in open("bin/cw").read()


