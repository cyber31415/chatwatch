"""Local dashboard server: live anonymous chat or replay -> engine -> browser.

    # live (read-only, anonymous): start the server, then pick ONE channel and
    # press Start on the page (owner 2026-11-07: one channel per session; another
    # channel means Stop, then Start). With --log, each Start..Stop is one logged session in
    # data/sessions/<time>_<channel>/ (see session.py). --channels starts at once.
    python -m chatwatch.serve [--channels somechannel] [--ml] [--llm-review [--person-review]]
    # a replay may hold several channels (a dev tool for tests and studies; it never saves state)
    # replay a capture at 5x (nothing is saved):
    python -m chatwatch.serve --replay data/raw/2026-10-20 --channels somechannel --speed 5

Then open http://127.0.0.1:8765 . The server only binds inside the container;
bin/cw publishes it on 127.0.0.1 of the host, nowhere else.

State (per-user running sums, room models) is saved to data/state/engine.pkl
on exit and every 5 minutes, and reloaded on start, so profiles carry across
streams (the handoff's "persist by user_id" recommendation). Only a live run
that loaded that file (or found none) saves over it (audit LB-1): a replay never
saves, a state that cannot be read is moved aside first, and --fresh keeps the
saved state unless --replace-state is given. The file before each save is kept
as engine.pkl.prev. Live mode holds data/state/dashboard.lock (one dashboard at
a time on any machine, LB-16).
"""
import argparse
import asyncio
import calendar
import gc
import itertools
import json
import os
import pickle
import random
import re
import shutil
import socket
import sys
import threading
import time
import warnings

# The fast layer's tokenizers (models.py) run batched encodes; the server later forks small helpers (the model read,
# nvidia-smi, the Ollama log). On each fork after a batched encode, huggingface/tokenizers printed a 4-line warning and
# turned its own parallelism off for the rest of the process anyway. Off from the start: the same tokens, no warning
# (owner 2026-11-10: "no reason to make the backend more chatty"). An explicit setting in the environment still wins.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from aiohttp import web, WSMsgType

from . import aictl
from . import engine as eng_mod
from . import fileio, irc
from . import roles as roles_mod
from .emotes import EmoteSets
from .engine import Engine
from .session import Session, log_override

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEB = os.path.join(ROOT, "web")
STATE = os.path.join(ROOT, "data", "state", "engine.pkl")
LOCK = os.path.join(ROOT, "data", "state", "dashboard.lock")
UI_BATCH = 400            # chat lines per 250 ms batch to the page; alerts, reviews and updates always go
QUEUE_MAX = 5000          # records waiting for the page; more only if the flusher is stuck
SEND_TIMEOUT_S = 2.0      # a browser that does not take a batch in this long is closed (its page reconnects)
UI_KEEP = ("alert", "review", "msg_update", "person_read")
# what the page reads of a message record; the session log keeps the whole record
UI_MSG = ("type", "channel", "ts", "id", "user_id", "display", "color", "text", "flags", "toxicity", "top",
          "badfaith", "notice")
UI_MOD = ("type", "channel", "ts", "login", "kind", "detail", "user_id", "known")
UI_BOT = ("type", "channel", "ts", "id", "user_id", "display", "color", "text")   # a line from the owner's bot role
CARD_REGISTER = ("style", "runner_up", "margin", "summary", "words", "tags", "confidence")   # what the card reads
COORDINATED_UI_GAP_MS = 10 * 60 * 1000   # one "coordinated template" banner per text per 10 min (all are logged)
REPLAY_YIELD = 200        # a replay faster than the server gives the loop a turn this often (LF-21)
BOOT = os.urandom(8).hex()    # this process: a page that sees another one knows the server restarted (LF4-1)
# The lossless LLM work queue (docs/AGENDA.md "Core architecture"; owner 2026-11-05). Nothing is dropped automatically:
# - order: person, alert and pile-on reads first, then re-read lines by priority (0 = flagged or coded), oldest first;
# - only low-priority re-reads (priority 1-3) age out, after LINE_AGE_OUT_S, counted and shown (H4: under no drops their
#   tail waited hours in a channel's busiest hours). Person reads and flagged lines never age out;
# - a failed call stays at the head and is retried (RETRY_MAX calls while the server answers; time it does not answer
#   is waited out without spending one);
# - Stop finishes the queue before the session closes ("Stop now" skips the rest, counted); a shutdown keeps it in the
#   saved state, and the next Start resumes it.
# - call shape (TRACKER Open 43, 2026-11-10): one line per call while the queue is short, windows only under backlog.
#   data/study/window_shape/RESULT.md: lines at window positions 2+ lose mild toxic reads (about -0.045 F1 on the mild
#   band; medium+ unchanged) for about 1.9x the throughput (31B: 1.7 s of GPU per line in a window, 3.2 s alone). A lone
#   line is the shape every published quality figure was measured in. SOLO_CLEAR_S: a line goes alone, at once, while
#   the waiting lines would clear one per call within it at the measured one-line cost (31B: 6 lines); 20 s is about
#   one full 12-line window of the 31B (18.8-20.6 s), and in data/study/solo/sim.txt it is the largest value that
#   re-reads as many lines as windows do at every replay speed (30 s already loses lines at 1x); the picker's own
#   simulators move re-read coverage by at most 0.1 point for any catalogue model (cov_check.txt). Stop's finishing
#   keeps windows (the fastest drain). SOLO_CLEAR_S = 0: windows only (the batcher before 2026-11-10).
WINDOW = 12               # re-read lines per call under backlog (window_shape: ~1.9x the throughput of one per call)
LINE_GATHER_S = 4.0       # under backlog a line waits this long for its window to fill, unless WINDOW lines of its channel wait
SOLO_CLEAR_S = 20.0       # one line per call while waiting lines x measured one-line seconds <= this (data/study/solo)
LINE_AGE_OUT_S = 15 * 60  # priority 1-3 re-reads older than this age out (decided 2026-11-06 from H4; tunable)
SCREEN_DELAY_S = 1.0      # a pile-on screen waits for the chatter's first few messages
RETRY_MAX = 3             # calls per job while the server answers; then the job's error is recorded, as before
RETRY_WAIT_S = (2, 5, 10, 20, 30)
RATE_SPAN_S = 300         # arrivals and service times behind the catch-up estimate


def shown_tox(t):
    """A toxicity read as the page gets it: with "counted_level", the level it counts at (engine._level), when that
    differs from its own level. Under S1 a keyword slur the LLM re-read as "none" still counts at the layer's level;
    the page showed the raw level, so the slur looked clean (audit LF3-2). Display only: no decision reads this."""
    if not t:
        return t
    lv = eng_mod._level(t)
    return t if lv == t.get("level", "none") else {**t, "counted_level": lv}


def _shown_msg(m):
    """A held message (the card's recent messages, the evidence) with its read as the page gets it (shown_tox)."""
    t = m.get("toxicity")
    s = shown_tox(t)
    return m if s is t else {**m, "toxicity": s}


def _trim(recs, n):
    """Every alert, review and update, plus the newest n other records, in their order.
    Returns (kept, how many were dropped)."""
    drop = sum(r.get("type") not in UI_KEEP for r in recs) - n
    if drop <= 0:
        return recs, 0
    kept, left = [], drop
    for r in recs:
        if left and r.get("type") not in UI_KEEP:
            left -= 1
            continue
        kept.append(r)
    return kept, drop


def _coded(rec):
    """C1 (data/scratch/coded1dev/DEV.md): the line holds a coded-glossary trigger, the same test the teacher uses
    to add its glossary (llm.glossary_lookup on the "display: text" line the re-read sends, so a login can't
    trigger one). The fast layer catches about half the coded hate the LLM does; these lines are read first."""
    from .llm import glossary_lookup
    return bool(glossary_lookup([f"{rec.get('display') or rec.get('login') or '-'}: {rec['text']}"]))


def _settle(fut, res, exc):
    if not fut.done():
        if exc is not None:
            fut.set_exception(exc)
        else:
            fut.set_result(res)


def in_thread(fn):
    """Run a blocking call (an LLM request) in a daemon thread and await it. Unlike the default
    executor, shutdown never waits for these (a window read can take 180 s, a model load 600 s;
    live review L7, audit LB-17)."""
    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    def run():
        try:
            res, exc = fn(), None
        except BaseException as e:          # noqa: BLE001 - handed to the awaiting coroutine
            res, exc = None, e
        try:
            loop.call_soon_threadsafe(_settle, fut, res, exc)
        except RuntimeError:                 # the loop has closed: nobody is waiting any more
            pass
    threading.Thread(target=run, daemon=True, name="cw-llm").start()
    return fut


class _Retry(Exception):
    """A call failed in a way worth retrying (llm.transient): the job stays at the head of the queue."""


class ReadQueue:
    """Person, alert and pile-on reads waiting for the LLM: best priority first, then oldest. Never full (lossless).
    An entry is [prio, seq, kind, payload, queued at (wall s), not before (wall s)]."""

    def __init__(self):
        self.items = []

    def put(self, prio, seq, kind, payload, at=None, not_before=0.0):
        self.items.append([prio, seq, kind, payload, time.time() if at is None else at, not_before])

    def put_nowait(self, entry):
        """asyncio.Queue's call, for (prio, seq, kind, payload)."""
        self.put(*entry[:4])

    def qsize(self):
        return len(self.items)

    def empty(self):
        return not self.items

    def take(self, now=None):
        """The best entry that may run now, removed; None when none may."""
        now = time.time() if now is None else now
        ready = [e for e in self.items if e[5] <= now]
        if not ready:
            return None
        e = min(ready, key=lambda e: (e[0], e[1]))
        self.items.remove(e)
        return e

    def get_nowait(self):
        """asyncio.Queue's call: (prio, seq, kind, payload) of the best entry, eligible or not."""
        if not self.items:
            raise asyncio.QueueEmpty
        e = min(self.items, key=lambda e: (e[0], e[1]))
        self.items.remove(e)
        return tuple(e[:4])


class Hub(aictl.AIMixin):
    coded_first = True            # C1: a glossary trigger makes a line priority 0 for the LLM re-read

    def __init__(self, engine):
        self.eng = engine
        if not hasattr(engine, "clock"):
            engine.clock = 0                  # serve.main and handle() set it; endpoints may run before either
        self.clients = set()
        self.transports = {}                  # websocket -> its transport, to cut a client that stopped reading
        self.queue = []
        self.status = {"mode": "", "channels": [], "connected": False, "running": False, "session": None,
                       "stopped_at": None, "started_at": None, "stopping": False}
        self.review_q = None          # ReadQueue when --llm-review is on (lossless: never full)
        self.llm_open = True          # the worker may call the LLM: a live run (from Start), its Stop finishing, a replay
        self.llm_hold = set()         # reasons the worker holds after the call in flight (a model swap)
        self.llm_job = None           # (kind, payload) running or being retried: kept in a shutdown's saved queue
        self.llm_retry = None         # {"why", "since"} while a call is being retried
        self.finishing = False        # Stop is finishing the queue before the session closes
        self.stop_now = False         # "Stop now": skip what is left (counted)
        self.shutting = False         # the server is shutting down: the queue is kept in the saved state
        self._calling = False         # the worker's own call is running (a card's manual read is not the worker's)
        self._aged_base = 0           # llm_lines_aged_out at Start: the page shows this run's
        self._aged_closed = 0         # lines aged out while no run was open (a kept queue before Start): this run's too (QL4)
        self._restored_low = 0        # low-priority lines among the queue the last shutdown kept (they may age out)
        self._wake = None             # asyncio.Event: work arrived
        self._q_in = []               # (wall s, "read" | "line") arrivals, for the catch-up estimate
        self._q_cost = {}             # "read" / "solo" (a one-line call) / "line" (per line of a window) -> recent seconds
        self._chat = {}               # channel -> chat-clock ms of its recent messages (msgs/min on the page)
        self.session = None           # Session when logging (live runs)
        self.persist = False          # save data/state/engine.pkl (live runs that loaded it; LB-1)
        self.live_task = None
        self.screened = set()         # (channel, uid) already LLM-screened in a pile-on (cleared at Start)
        self.person_queued = set()    # (channel, uid) with a person read waiting in review_q (LB-14)
        self.tox_per_min = 0          # 0 = LLM toxicity reads off
        self.tox_pending = []         # (priority, ts, item) awaiting a window read
        self.llm_inflight = 0         # LLM calls running now (status "llm_busy")
        self.run_gen = 0              # +1 at every Stop: a read that started before it is not applied (LB-6)
        self._seq = 0
        self._coord_shown = {}        # (channel, text) -> ts of the last "coordinated" alert sent to the page
        self.log_sessions = True
        self.record = False
        self._last_room = self._last_log = 0     # when the flusher last sent / logged room snapshots
        self.since = None             # ms: start of the live run, for "this session" on the page
        self.roles = {}               # the owner's roles, user id -> entry (roles.py); a live run loads them
        self.roles_rev = 0            # +1 at every role change: another page offers a reload (audit LF3-8)
        self.stopping = False         # Stop is closing the session and saving: Start waits (audit LB3-3)
        self._stopper = None          # the running Stop: a second Stop waits for it (audit LB4-2)
        self.ai_init()                # the model picker, speed check and hybrid review (aictl.py)

    # ------------------------------------------------------- live control
    def start_live(self, channels):
        """Join channels (anonymously) and open a new logged session. Session() only creates its folder
        here; its code snapshot and state copy are written by a worker thread (LB-8)."""
        if self.live_task and not self.live_task.done():
            raise RuntimeError("already running; stop first")
        if self.stopping:             # the last Stop has not saved yet: a new session would copy the old state (LB3-3)
            raise RuntimeError("still stopping (saving the state): try again in a moment")
        if self._ai_job is not None and not self._ai_job.done():    # a model switch or speed check holds the GPU
            raise RuntimeError(f"wait: {(self.ai_task or {}).get('what', 'an AI task')} is running")
        if self.log_sessions:
            self.session = Session(channels, sys.argv + ["[ui-start]"] + channels, self.eng, ai=self.session_ai())
        self.since = int(time.time() * 1000)
        self.screened.clear()
        self._chat.clear()
        # QL4 (audit 2026-11-10): lines a shutdown kept that aged out while the queue waited for this Start count as this
        # run's "aged out" (the console said they wait for the next Start)
        self._aged_base = self.eng.stats.get("llm_lines_aged_out", 0) - self._aged_closed
        self._aged_closed = 0
        self.stop_now = False
        self.llm_open = True          # a queue kept by the last shutdown resumes now
        self._kick()
        self.status.update(channels=list(channels), running=True, connected=False, stopped_at=None, started_at=self.since,
                           session=os.path.basename(self.session.dir) if self.session else None)
        rec = os.path.join(ROOT, "data", "raw") if self.record else None
        self.live_task = asyncio.create_task(live_source(self, channels, rec, self.session))

    async def stop_live(self, finish=True, now=False):
        """Until this returns, Start is refused (409, `stopping` in the status): a session opened meanwhile would copy
        data/state/engine.pkl before this Stop saved it, so its state_at_start.pkl would be the wrong start (LB3-3).
        A second Stop meanwhile (a double POST, or Stop and SIGTERM) waits for the running one (LB4-2).
        - finish (a Stop): chat stops at once; the LLM queue is finished, its reads logged to this session, before
          the session closes. now ("Stop now", also on a second Stop's request) skips what is left, counted.
        - not finish (a shutdown): the queue is kept in the saved state and the next Start resumes it."""
        if self._ai_job is not None and not self._ai_job.done():
            # a review batch stops at its next safe point (between calls); _stop_live waits for it to swap back to the
            # live model, after chat has stopped (review fix 3). On a shutdown it skips the swap (the next start loads the
            # live model)
            self.cancel_ai_task()
        if now or not finish:
            self.stop_now = self.stop_now or now
            self._kick()
        if not finish:
            self.shutting = True           # a shutdown (also during a Stop's finishing): keep the queue, save it
            self.llm_open = False          # no new call starts
        if self._stopper is None or self._stopper.done():
            self._stopper = asyncio.ensure_future(self._stop_guarded(finish))
        await asyncio.shield(self._stopper)

    async def _stop_guarded(self, finish=True):
        self.stopping = True
        self.status["stopping"] = True
        try:
            await self._stop_live(finish)
        finally:
            self.stopping = False
            self.status["stopping"] = False

    async def _stop_live(self, finish=True):
        if self.live_task and not self.live_task.done():
            self.live_task.cancel()
            try:
                await self.live_task
            except asyncio.CancelledError:
                pass
        self.live_task = None
        self.status.update(running=False, connected=False, session=None, channels=[],
                           stopped_at=int(time.time() * 1000))
        if finish and not self.shutting and self._ai_job is not None and not self._ai_job.done():
            await self.ai_task_done()          # a review batch swaps back to the live model before the queue finishes
        if finish and self.llm_open and not self.shutting:
            await self._finish_llm()
        if finish and not self.shutting and not self.ai_off:
            self._drain_llm()                  # empty unless "Stop now" skipped the rest
        elif finish and not self.shutting:
            # review fix 10: with the AI off nothing ran the queue (a shutdown's, kept for the next Start): keep it in the
            # saved state for a run with the AI on; nothing is dropped automatically
            self.run_gen += 1
        else:
            self.run_gen += 1                  # a call still running is not applied; its job is kept (llm_job)
        self.llm_open = False
        # the session first: a failed save (full disk) must not lose the session's tail. Closing it waits for its
        # state copy (seconds to tens of seconds on the share), so it runs off the loop (audit LB2-10); it is
        # detached first, so nothing logs into it meanwhile
        sess, self.session = self.session, None
        if sess:
            try:
                await asyncio.get_running_loop().run_in_executor(None, sess.close, dict(self.eng.stats))
            except Exception as e:
                _log_error(self.eng, "session_close_errors", e, "closing the session")
        if self.persist:
            try:
                self.stash_queue(include_job=not finish or self.shutting)    # a job Stop dropped is not kept
                await save_engine_async(self.eng)
            except Exception as e:
                _log_error(self.eng, "save_errors", e, "saving the state")

    async def _finish_llm(self):
        """Stop: let the worker finish every queued job (no new gather wait), or until "Stop now"."""
        self.finishing = True
        self._kick()
        try:
            # with the AI off nothing runs the queue: what a shutdown kept is dropped by _drain_llm, counted
            while not self.stop_now and not self.shutting and not self.ai_off and (
                    self.llm_job is not None or self.tox_pending
                    or (self.review_q is not None and not self.review_q.empty())):
                await asyncio.sleep(0.2)
        finally:
            self.finishing = False

    def _drain_llm(self):
        """After Stop finished the queue, or "Stop now" (LB-6): what is left is dropped, counted in
        llm_dropped_at_stop (one per read, one per line); queued person reads are released (no read spent, not an
        error), and a read already running is not applied when it returns (run_gen)."""
        self.run_gen += 1
        self.eng.stats["llm_dropped_at_stop"] += len(self.tox_pending)
        self.tox_pending.clear()
        job = self.llm_job
        if job is not None and not self._calling:     # waiting to retry (its call running: dropped when it returns)
            self._drop_job(job)
        elif job is not None:
            job["dropped"] = True                     # never saved, even by an autosave or shutdown before it returns
        q = self.review_q
        while q is not None:
            try:
                _, _, kind, a = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            if kind == "person":
                self.eng.release_person_review(a["channel"], a["user_id"])
            self.eng.stats["llm_dropped_at_stop"] += 1
        self.person_queued.clear()
        self.stop_now = False

    def stash_queue(self, include_job=True):
        """Put the waiting LLM work on the engine (eng.llm_queue) for the state save, so a shutdown loses none of it
        and a crash loses no more than the state does. The job running or being retried is kept too."""
        if self.review_q is None:
            self.eng.llm_queue = None
            return
        reads = [[e[0], e[2], e[3], e[4]] for e in self.review_q.items]
        lines = [list(x) for x in self.tox_pending]
        job = self.llm_job if include_job and not (self.llm_job or {}).get("dropped") else None
        if job is not None:
            if job["kind"] == "toxwin":
                lines += [[it.get("prio", 0), it.get("ts", 0), it] for it in job["a"]["items"]]
            else:
                reads.insert(0, [job.get("prio", 0), job["kind"], job["a"], job.get("at", time.time())])
        self.eng.llm_queue = {"saved": time.time(), "reads": reads, "lines": lines} if reads or lines else None

    def restore_queue(self, saved):
        """The queue a shutdown kept (stash_queue), back in line; it runs at the next Start. Ages are kept, so the
        low-priority lines among them age out as usual."""
        if not saved or self.review_q is None:
            return 0
        n = 0
        for prio, kind, a, at in saved.get("reads", ()):
            self._seq += 1
            self.review_q.put(prio, self._seq, kind, a, at=at)
            if kind == "person":
                self.person_queued.add((a["channel"], a["user_id"]))
                self.eng.mark_review_pending(a["channel"], a["user_id"])    # upgrade() marked it lost
            n += 1
        for prio, ts, item in saved.get("lines", ()):
            self.tox_pending.append((prio, ts, item))
            n += 1
            self._restored_low += prio != 0
        self.eng.stats["llm_carried_over"] += n
        return n

    def _log(self, rec):
        """Session logging. A failed write (the share full or flaky) is counted and shown once; it
        never keeps the record from the page or stops a loop (it used to end the server)."""
        if self.session:
            try:
                self.session.log(rec)
            except Exception as e:
                _log_error(self.eng, "session_log_errors", e, "writing the session log")

    def _ui(self, rec):
        """The page's copy of a record: only the fields it reads (the session log keeps everything)."""
        t = rec.get("type")
        if t == "msg":
            out = {k: rec[k] for k in UI_MSG if k in rec}
            if "toxicity" in out:
                out["toxicity"] = shown_tox(out["toxicity"])
            return out
        if t == "msg_update":
            return {**rec, "toxicity": shown_tox(rec.get("toxicity"))}
        if t == "modaction":
            return {k: rec[k] for k in UI_MOD if k in rec}
        if t == "bot_msg":
            return {k: rec[k] for k in UI_BOT if k in rec}
        return rec

    def publish(self, rec):
        self._log(rec)
        if rec.get("type") != "room_event":         # a raid shows from the room snapshot
            self.queue.append(self._ui(rec))
        if len(self.queue) > QUEUE_MAX:            # the flusher is stuck: keep the newest half
            self.queue, dropped = _trim(self.queue, QUEUE_MAX // 2)
            self.eng.stats["ui_records_dropped"] += dropped
        for a in rec.get("alerts", ()):
            a["alert_id"] = f"{a['channel']}:{a.get('user_id', '')}:{a['ts']}"
            self._log(a)
            if self._show_alert(a):
                self.queue.append(a)
            if (self.ai_queue() is not None and a["kind"] == "badfaith" and a.get("via") != "llm-screen"
                    and not self.eng.person_review):
                self._enqueue(0, "alert", a)
        # the local LLM reads candidates as people; its read decides 'watch' (FINDINGS "bad-faith read")
        if (self.ai_queue() is not None and self.eng.person_review and rec.get("type") == "msg"
                and self.eng.wants_person_review(rec)):
            self._queue_person(rec["channel"], rec["user_id"], rec["ts"])
        if rec.get("type") == "msg":
            self._chat.setdefault(rec["channel"], []).append(rec["ts"])
        # toxicity escalation: the LLM re-reads prioritised messages (lossless: they wait in tox_pending)
        prio = None
        if self.ai_queue() is not None and self.tox_per_min and rec.get("type") == "msg":
            prio = self._tox_priority(rec)
            if prio is not None:
                ctx = list(self.eng.rooms[rec["channel"]].context)[-7:-1]   # lines just before it
                self.tox_pending.append((prio, rec["ts"], {"channel": rec["channel"], "user_id": rec["user_id"],
                                         "id": rec["id"], "line": f"{rec['display']}: {rec['text']}",
                                         "room": [f"{w}: {t}" for _, w, t in ctx],
                                         "prio": prio, "ts": rec["ts"], "q_at": time.time()}))
                self._arrived("line")
        if rec.get("type") == "msg":
            self.track_message(rec, prio)          # the hybrid review: context ring, and the lines it re-reads
        # pile-on mode: screen first-time chatters' first messages with the LLM
        if (self.ai_queue() is not None and rec.get("type") == "msg" and "pile-on" in rec.get("flags", ())
                and rec.get("n", 9) <= 3):
            key = (rec["channel"], rec["user_id"])
            if key not in self.screened:
                self.screened.add(key)
                self._enqueue(1, "screen", {"channel": rec["channel"], "user_id": rec["user_id"]})

    def _show_alert(self, a):
        """A bot wave posting one template raises a 'coordinated' alert per message (engine.py); the page
        gets one per text per COORDINATED_UI_GAP_MS (LF-6). Every alert is still logged."""
        if a.get("kind") != "coordinated":
            return True
        key = (a["channel"], re.sub(r"\W+", " ", str(a.get("text", "")).lower()).strip()[:80])
        last = self._coord_shown.get(key)
        if last is not None and 0 <= a["ts"] - last < COORDINATED_UI_GAP_MS:
            return False
        if len(self._coord_shown) > 1000:
            self._coord_shown.clear()
        self._coord_shown[key] = a["ts"]
        return True

    def _queue_person(self, ch, uid, ts):
        """Queue one person read. A chatter already waiting in the queue is not queued twice (a read
        pending past REVIEW_PENDING_MS makes them a candidate again; LB-14): the pending mark is renewed."""
        self.eng.mark_review_pending(ch, uid, ts)
        if (ch, uid) in self.person_queued:
            self.eng.stats["llm_person_requeue_skipped"] += 1
            return
        self._enqueue(0, "person", {"channel": ch, "user_id": uid})
        self.person_queued.add((ch, uid))

    def _tox_priority(self, rec):
        """Priority for an LLM toxicity read (lower = sooner), or None to skip.
        The LLM is far more precise than the fast layer (one
        judged session: 0.97 vs 0.75), so what the fast layer FLAGS is read
        first, short messages included ("shit aim"): those are the flags a mod
        sees. Then what it is unsure about and newcomers; the rest only when
        the GPU has room (big chats)."""
        if rec["text"].startswith("!"):
            return None
        sc = (rec.get("toxicity") or {}).get("score", 0)
        if sc >= 0.3:
            return 0                                 # flagged: confirm or clear it
        if self.coded_first and _coded(rec):
            return 0                                 # C1: coded language, short lines included
        if len(rec["text"].split()) < 3 and sc < 0.15:
            return None
        if sc >= 0.15 or (rec.get("tox") or 0) >= 0.5:
            return 1                                 # the layer is unsure
        if rec.get("n", 9) <= 3 or "pile-on" in rec.get("flags", ()):
            return 1                                 # newcomers / brigades
        if rec.get("badfaith", 0) >= 0.25 or "@" in rec["text"]:
            return 2                                 # aimed at someone / already watched
        return 3

    async def tox_batcher(self, every=5.0):
        """The queue's sweeper: low-priority re-reads (1-3) older than LINE_AGE_OUT_S age out, counted in
        llm_lines_aged_out and shown on the page. Person reads, alerts, screens and flagged (0) lines never do."""
        while True:
            await asyncio.sleep(every)
            try:
                self._age_out(time.time())
            except Exception as e:
                _log_error(self.eng, "tox_batcher_errors", e, "ageing out LLM re-reads")

    def _age_out(self, now):
        bad = [x for x in self.tox_pending if not isinstance(x[2], dict)]
        if bad:                                   # not a line: removed, so it cannot fail every round
            self.tox_pending[:] = [x for x in self.tox_pending if isinstance(x[2], dict)]
        keep = [x for x in self.tox_pending if x[0] == 0 or now - x[2].get("q_at", now) <= LINE_AGE_OUT_S]
        gone = len(self.tox_pending) - len(keep)
        if gone:
            self.tox_pending[:] = keep
            self.eng.stats["llm_lines_aged_out"] += gone
            if not self.llm_open:             # waiting for a Start: counted in that run's "aged out" (QL4)
                self._aged_closed += gone
        if bad:
            raise ValueError(f"{len(bad)} malformed queued line(s) removed")
        return gone

    def _solo_n(self, solo_s):
        """How many waiting lines go one per call at this one-line cost (SOLO_CLEAR_S of it; an unmeasured cost: every
        one, so the first call goes alone and measures it)."""
        if SOLO_CLEAR_S <= 0:
            return 0
        if not solo_s or solo_s <= 0:
            return float("inf")
        return int(SOLO_CLEAR_S / solo_s + 1e-9)

    def _window(self, now):
        """The next window read, removed from tox_pending, or None. The head is the best line (priority, then oldest).
        While the queue is short (_solo_n) it goes alone, at once: the shape the quality figures were measured in.
        Under backlog the window takes up to WINDOW lines of its channel in the same order (one channel per call: its
        lines share one room context), sent in chat order with the lines before the first as context; its lines wait
        LINE_GATHER_S for company. Stop's finishing sends windows at once (the fastest drain)."""
        if not self.tox_pending:
            return None
        order = sorted(self.tox_pending, key=lambda x: (x[0], x[1]))
        ch = order[0][2]["channel"]
        if order[0][2].get("missed") or (not self.finishing
                                         and len(self.tox_pending) <= self._solo_n(self._q_cost.get("solo"))):
            take = order[:1]                      # (QL3: a line the model left out of its answer is asked again alone)
        else:
            take = [x for x in order if x[2]["channel"] == ch and not x[2].get("missed")][:WINDOW]
            ripe = (self.finishing or len(take) >= WINDOW
                    or min(x[2].get("q_at", 0) for x in take) <= now - LINE_GATHER_S)
            if not ripe:
                return None
        ids = {id(x) for x in take}
        self.tox_pending[:] = [x for x in self.tox_pending if id(x) not in ids]
        take.sort(key=lambda x: x[1])
        items = [item for _, _, item in take]
        return {"kind": "toxwin", "a": {"channel": ch, "items": items, "user_id": items[0]["user_id"]}}

    def _next_job(self, now):
        """Reads first (person, alert, pile-on screen; best priority, then oldest), then a window of lines."""
        if not self.llm_open or self.llm_hold or self.stop_now or self.review_q is None or self.ai_off:
            return None
        e = self.review_q.take(now)
        if e is not None:
            return {"kind": e[2], "a": e[3], "prio": e[0], "at": e[4]}
        return self._window(now)

    def _enqueue(self, prio, kind, payload):
        """Person and alert reviews (0) jump ahead of pile-on screens (1). Lossless: always queued (True)."""
        self._seq += 1
        nb = time.time() + SCREEN_DELAY_S if kind == "screen" else 0.0     # let their first messages arrive
        self.review_q.put(prio, self._seq, kind, payload, not_before=nb)
        self._arrived("read")
        return True

    def _arrived(self, what):
        now = time.time()
        self._q_in.append((now, what))
        if len(self._q_in) > 50_000 or (self._q_in and self._q_in[0][0] < now - 2 * RATE_SPAN_S):
            self._q_in = [x for x in self._q_in if x[0] >= now - RATE_SPAN_S]
        self._kick()

    def _kick(self):
        if self._wake is not None:
            self._wake.set()

    def _served(self, what, n, seconds):
        """Measured seconds per read / one-line call / line of a window (smoothed), for the catch-up estimate and the
        call shape (_solo_n). Line calls are also summed by shape in the stats (llm_tox_solo_* / llm_tox_window_*)."""
        if what == "line":
            what = "solo" if n == 1 else "line"
            shape = "solo" if n == 1 else "window"
            self.eng.stats[f"llm_tox_{shape}_calls"] += 1
            self.eng.stats[f"llm_tox_{shape}_lines"] += n
            self.eng.stats[f"llm_tox_{shape}_ms"] += int(seconds * 1000)
        per = seconds / max(n, 1)
        old = self._q_cost.get(what)
        self._q_cost[what] = per if old is None else 0.8 * old + 0.2 * per

    def hold_llm(self, why):
        """Hold the worker after the call in flight (a model swap); chat keeps queueing. release_llm resumes it."""
        self.llm_hold.add(why)

    def release_llm(self, why):
        self.llm_hold.discard(why)
        self._kick()

    async def llm_idle(self):
        """Until no call runs (after hold_llm: the swap may unload the model)."""
        while self.llm_job is not None or self.llm_inflight:
            await asyncio.sleep(0.1)

    async def _llm(self, fn):
        """One blocking LLM call off the loop, counted in llm_inflight (status "llm_busy")."""
        self.llm_inflight += 1
        try:
            return await in_thread(fn)
        finally:
            self.llm_inflight -= 1

    async def reviewer(self):
        """The one worker of the lossless LLM queue (T3): reads, then windows of lines, one call at a time. A failed
        call stays at the head and is retried (_run_job). An error is counted and shown once; it never ends the loop."""
        if self._wake is None:
            self._wake = asyncio.Event()
        while True:
            try:
                job = self._next_job(time.time())
                if job is None:
                    self._wake.clear()
                    try:
                        await asyncio.wait_for(self._wake.wait(), 1.0)
                    except asyncio.TimeoutError:
                        pass
                    continue
                await self._run_job(job)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                job, self.llm_job = self.llm_job, None
                if job is not None and job["kind"] == "person":     # a bug-level error: the read is not left pending
                    self.eng.release_person_review(job["a"]["channel"], job["a"]["user_id"])
                    self.person_queued.discard((job["a"]["channel"], job["a"]["user_id"]))
                _log_error(self.eng, "reviewer_errors", e, "the LLM work queue")
                await asyncio.sleep(0.2)

    async def _run_job(self, job):
        """One job, retried while it fails in a way worth retrying: RETRY_MAX calls while the server answers; while
        it does not answer (a restarting runner) the worker waits without spending one. A Stop's "Stop now" or a
        shutdown ends the wait (the shutdown keeps the job in the saved queue)."""
        from .llm import reachable
        self.llm_job = job
        calls = waits = 0
        while True:
            if calls or waits:                    # it failed and waited: Stop now, a shutdown or a hold since?
                if self._job_cut(job, self.run_gen):
                    return
            gen = self.run_gen
            self._calling = True
            try:
                secs = await self._review_one(job["kind"], job["a"], final=calls + 1 >= RETRY_MAX)
            except _Retry as r:
                self._calling = False
                if self._job_cut(job, gen):
                    return
                up = await in_thread(reachable)
                calls += up
                self.eng.stats["llm_retries"] += 1
                self.llm_retry = {"since": (self.llm_retry or {}).get("since") or int(time.time() * 1000),
                                  "why": ("the local LLM failed: " if up else "the local LLM is not answering: ")
                                  + str(r.__cause__ or r)[:160]}
                await asyncio.sleep(RETRY_WAIT_S[min(waits, len(RETRY_WAIT_S) - 1)])
                waits += 1
                continue
            finally:
                self._calling = False
            self.llm_retry = None
            if self.llm_job is job:
                self.llm_job = None
            # QL1 (audit 2026-11-10): only a clean call is measured (it came back, was applied and loaded no model)
            if secs is not None:
                n = len(job["a"]["items"]) if job["kind"] == "toxwin" else 1
                self._served("line" if job["kind"] == "toxwin" else "read", n, secs)
            return

    def _job_cut(self, job, gen):
        """A job that failed and has not run again: is it to stop now? True when it ended here:
        - a shutdown keeps it in llm_job, for the saved queue;
        - Stop (now) dropped it: counted in llm_dropped_at_stop, a person read released (unless _drain_llm already did);
        - a hold (a model swap) puts it back at the head of the queue, so the swap does not wait on its retries."""
        if self.shutting:
            return True
        mine = self.llm_job is job
        if gen != self.run_gen or self.stop_now or not self.llm_open:
            if mine:
                self._drop_job(job)
            self.llm_retry = None
            return True
        if self.llm_hold and mine:
            self.llm_job = None
            self.llm_retry = None
            if job["kind"] == "toxwin":
                self.tox_pending.extend((it.get("prio", 0), it.get("ts", 0), it) for it in job["a"]["items"])
            else:
                self.review_q.put(job.get("prio", 0), -1, job["kind"], job["a"], at=job.get("at"))
                if job["kind"] == "person":
                    self.person_queued.add((job["a"]["channel"], job["a"]["user_id"]))
            return True
        return not mine

    def _drop_job(self, job):
        """A job Stop dropped (never to be read): counted, and a person read released."""
        self.llm_job = None
        self.eng.stats["llm_dropped_at_stop"] += len(job["a"]["items"]) if job["kind"] == "toxwin" else 1
        if job["kind"] == "person":
            self.eng.release_person_review(job["a"]["channel"], job["a"]["user_id"])
            self.person_queued.discard((job["a"]["channel"], job["a"]["user_id"]))

    async def _timed(self, fn):
        """(result, service seconds) of one call: its wall time less the time the model server spent loading the model
        (llm.last_load_s: Ollama's load_duration, or a reload after a failed try). QL1 (audit 2026-11-10)."""
        from . import llm as L
        box = {}

        def call():
            try:
                return fn()
            finally:
                box["load"] = L.last_load_s()
        t0 = time.time()
        res = await self._llm(call)
        return res, max(0.0, time.time() - t0 - box.get("load", 0.0))

    def _late(self, a, n):
        """A result that came back after a Stop: not applied. When "Stop now" dropped its job it is counted with what
        that Stop skipped, one per read and one per line (llm_dropped_at_stop; QL2, audit 2026-11-10); otherwise (a
        direct call, a shutdown) as one llm_late_dropped."""
        job = self.llm_job
        if job is not None and job.get("dropped") and job.get("a") is a:
            self.eng.stats["llm_dropped_at_stop"] += n
        else:
            self.eng.stats["llm_late_dropped"] += 1

    async def _review_one(self, kind, a, final=True):
        """One read and its result. final=False (the worker, before its last try): a failure worth retrying raises
        _Retry instead of recording the error. Returns the call's service seconds when it was a clean call (it came
        back and was applied; its model load left out), else None: only clean calls set the measured costs (QL1)."""
        from .llm import review, transient
        gen = self.run_gen

        def retry(e):
            if not final and transient(e):
                raise _Retry(str(e)) from e
        if kind == "toxwin":
            from .llm import rate_toxicity_window
            ch, items = a["channel"], a["items"]
            try:
                em = self.eng.emote_names(ch)
                res, secs = await self._timed(lambda: rate_toxicity_window(
                    items[0]["room"], [it["line"] for it in items], emotes=em))
                self.eng.stats["llm_tox_windows"] += 1
            except Exception as e:
                retry(e)
                self.eng.stats["llm_tox_errors"] += 1
                self.eng.stats["llm_tox_lines_failed"] += len(items)
                return None
            if gen != self.run_gen:                     # Stop came while it ran
                self._late(a, len(items))
                return None
            for it, r in zip(items, res):
                if not r:
                    # QL3 (audit 2026-11-10): a line the model left out of its answer (no rating, or an unknown level)
                    # goes back once, asked alone; a second miss keeps the engine's level. Both are counted
                    self.eng.stats["llm_tox_missing"] += 1
                    if it.get("missed"):
                        self.eng.stats["llm_tox_missing_kept"] += 1
                    else:
                        self.tox_pending.append((it.get("prio", 0), it.get("ts", 0), {**it, "missed": 1}))
                    continue
                upd = self.eng.apply_llm_toxicity(ch, it["user_id"], it["id"], r)
                self.eng.stats["llm_tox_reads"] += 1
                if upd:
                    self._log({**upd, "ts": self.eng.clock})
                    self.queue.append(self._ui(upd))
                    # S11: a re-read that gives the chatter LLM-only hate queues a person read
                    if self.eng.wants_person_review_llm(ch, it["user_id"], upd["toxicity"], self.eng.clock):
                        self._queue_person(ch, it["user_id"], self.eng.clock)
            return secs
        if kind == "person":
            self.person_queued.discard((a["channel"], a["user_id"]))
            p = self.eng.profile(a["channel"], a["user_id"])
            if p is None:
                return None
            ctx = list(self.eng.rooms[a["channel"]].context)[-20:]
            secs = None
            try:
                em = self.eng.emote_names(a["channel"])
                r, secs = await self._timed(lambda: review(p, ctx, emotes=em))
            except Exception as e:
                if not final and transient(e):
                    if gen == self.run_gen:          # still waiting for its retry: no second copy meanwhile
                        self.person_queued.add((a["channel"], a["user_id"]))
                    raise _Retry(str(e)) from e
                r = {"error": str(e)}
            if gen != self.run_gen:
                self.eng.release_person_review(a["channel"], a["user_id"])
                self._late(a, 1)
                return None
            self.eng.apply_person_review(a["channel"], a["user_id"], r, self.eng.clock)
            self._log({"type": "review", "kind": "person", "channel": a["channel"],
                       "user_id": a["user_id"], "ts": self.eng.clock, **r})
            if r.get("label"):                          # an open user card shows it without a reload
                self.queue.append({"type": "person_read", "channel": a["channel"], "user_id": a["user_id"],
                                   "ts": self.eng.clock, "label": r["label"], "rationale": r.get("rationale", "")})
            return None if "error" in r else secs
        p = self.eng.profile(a["channel"], a["user_id"])
        if p is None:
            return None
        ctx = list(self.eng.rooms[a["channel"]].context)[-20:]
        secs = None
        try:
            em = self.eng.emote_names(a["channel"])
            r, secs = await self._timed(lambda: review(p, ctx, emotes=em))
            if kind == "alert":
                self.eng.stats["llm_reviews"] += 1
                self.eng.stats["llm_agrees" if r["label"] == "bad_faith" else "llm_disagrees"] += 1
            else:
                self.eng.stats["llm_screens"] += 1
                self.eng.stats[f"llm_screen_{r.get('label')}"] += 1
        except Exception as e:
            retry(e)
            r, secs = {"error": str(e)}, None
        if gen != self.run_gen:
            self._late(a, 1)
            return None
        rv = {"type": "review", "kind": kind, "alert_id": a.get("alert_id"), "channel": a["channel"],
              "user_id": a["user_id"], "ts": self.eng.clock, **r}
        # in a pile-on the verdict becomes evidence (and may raise an alert)
        new_alert = self.eng.apply_review(a["channel"], a["user_id"], r, self.eng.clock) \
            if kind == "screen" else None
        self._log(rv)
        if new_alert:
            self._log(new_alert)
        if kind == "alert":
            self.queue.append(rv)
        if new_alert:
            new_alert["alert_id"] = f"{new_alert['channel']}:{new_alert['user_id']}:{new_alert['ts']}"
            new_alert["review"] = r
            self.queue.append(new_alert)
        return secs

    async def flusher(self):
        """Batch outgoing records every 250 ms so big chats don't flood the socket. An error is
        counted and shown once; it never ends the loop (it used to end the server)."""
        while True:
            await asyncio.sleep(0.25)
            try:
                await self._flush()
            except Exception as e:
                _log_error(self.eng, "flusher_errors", e, "sending to the page")

    def _status(self):
        q = self._llm_status()
        return {"type": "status", **self.status, "boot": BOOT,
                # the pipeline's jobs wait while this is true (jobs.gpu_busy, pipeline.py; the page does not read it: kept
                # for them, audit 2026-11-10 FE-9b): a call runs, or the worker has work
                "llm_busy": self.llm_inflight > 0 or bool(self.llm_open and q and (q["reads"] or q["lines"]))
                            or (self._ai_job is not None and not self._ai_job.done()),     # a swap, check or review
                "llm_queue": q, "chat_per_min": self._chat_rates(), "ai": self.ai_status(),
                "stats": dict(self.eng.stats), "owner_roles": len(self.roles), "owner_roles_rev": self.roles_rev}

    def _chat_rates(self):
        """Messages in the last minute per channel (the chat clock; live, the wall clock once chat is quiet)."""
        now = self._display_ts()
        out = {}
        for ch, ts in self._chat.items():
            cut = 0
            while cut < len(ts) and ts[cut] <= now - 60_000:
                cut += 1
            del ts[:cut]
            out[ch] = len(ts)
        return out

    def _llm_status(self, now=None):
        """The work queue for the page: what waits, how far behind its oldest item is (any, and person reads with
        flagged lines), a catch-up estimate from measured service times and recent arrivals, the low-priority lines
        aged out this run, and whether a call is being retried. None when the LLM is off.
        catchup_state (QL5, audit 2026-11-10) says what catchup_s is: "estimate" (seconds), "behind" (None: arrivals load
        the model past 0.95, falling behind), "measuring" (None: a kind of call waiting has not been timed yet since the
        server started or the model changed) or "idle" (None: nothing waits)."""
        if self.review_q is None or self.ai_off:
            return None
        now = time.time() if now is None else now
        job = self.llm_job
        reads = self.review_q.qsize() + (job is not None and job["kind"] != "toxwin")
        jl = job["a"]["items"] if job is not None and job["kind"] == "toxwin" else []
        lines = len(self.tox_pending) + len(jl)
        ats = [e[4] for e in self.review_q.items] + [x[2].get("q_at", now) for x in self.tox_pending]
        hi = [e[4] for e in self.review_q.items] + [x[2].get("q_at", now) for x in self.tox_pending if x[0] == 0]
        if job is not None:
            js = [it.get("q_at", now) for it in jl] if jl else [job.get("at", now)]
            ats += js
            hi += js if not jl else [it.get("q_at", now) for it in jl if it.get("prio") == 0]
        catchup, state = None, "idle" if not (reads or lines) else "measuring"
        cr = self._q_cost.get("read")
        cs, cl = self._q_cost.get("solo"), self._q_cost.get("line")
        cs, cl = (cs if cs is not None else cl), (cl if cl is not None else cs)     # one shape measured: it for both
        if (reads or lines) and (cr is not None or not reads) and (cl is not None or not lines):
            # mixed call shapes: the backlog drains in windows down to the lines that go one per call; new lines are
            # costed in the shape they will get (alone while the queue is short and one per call keeps up with them)
            k = min(lines, self._solo_n(cs)) if lines else 0
            recent = [w for t, w in self._q_in if t >= now - RATE_SPAN_S]
            r_load = recent.count("read") * (cr or 0) / RATE_SPAN_S
            l_rate = recent.count("line") / RATE_SPAN_S
            alone = SOLO_CLEAR_S > 0 and lines <= k and r_load + l_rate * (cs or 0) < 0.95
            load = r_load + l_rate * ((cs if alone else cl) or 0)
            state = "behind"
            if load < 0.95:
                catchup = round((reads * (cr or 0) + k * (cs or 0) + (lines - k) * (cl or 0)) / (1 - load))
                state = "estimate"
        return {"reads": reads, "lines": lines,
                "behind_s": round(now - min(ats)) if ats else 0,
                "behind_hi_s": round(now - min(hi)) if hi else 0,
                "catchup_s": catchup, "catchup_state": state,
                "aged_out": self.eng.stats.get("llm_lines_aged_out", 0) - self._aged_base,
                "retrying": (self.llm_retry or {}).get("why"), "held": bool(self.llm_hold),
                "finishing": self.finishing, "open": self.llm_open}

    def _display_ts(self, now=None):
        """The time the page is shown at (ms): the chat clock, and in live mode the wall clock once chat is quiet, so a
        quiet chat still ages: its rate falls, its cards move on (LB4-4). Display only: the snapshot and the card only
        read at this time and trim nothing, so a message still in flight is scored as the chat clock left the room
        (LB5-1). The watchlist and the user card both use it, so they agree (LB6-1). After a live Stop the lists the page
        shows stay as they were then, so the card stops aging at the Stop too (LB7-1)."""
        now = time.time() if now is None else now
        ts = self.eng.clock or int(now * 1000)
        if self.status["mode"] != "live":
            return ts
        wall = int(now * 1000)
        if not self.status.get("running") and self.status.get("stopped_at"):
            wall = min(wall, self.status["stopped_at"])
        return max(ts, wall)

    async def _flush(self):
        batch, dropped = _trim(self.queue, UI_BATCH)
        self.queue = []
        self.eng.stats["ui_records_dropped"] += dropped
        now = time.time()
        if now - self._last_room > 2:
            self._last_room = now
            ts = self._display_ts(now)
            log_rooms = self.session and now - self._last_log > 30
            if log_rooms:
                self._last_log = now
            # a stopped live run sends no snapshots: its lists would stand still as if live (LF-8)
            chans = self.status["channels"] if (self.status["mode"] != "live" or self.status["running"]) else []
            for ch in chans:
                # the page reads the lean snapshot; the 30 s log keeps the room profile too. One channel's
                # error skips that channel only: the chat lines and the status still go out (LB4-1)
                try:
                    snap = self.eng.room_snapshot(ch, ts, since=self.since, full=bool(log_rooms))
                except Exception as e:
                    _log_error(self.eng, "flusher_errors", e, f"the room snapshot of {ch}")
                    continue
                if snap:
                    if log_rooms:
                        self._log({"type": "room", "ts": ts, **snap})
                        for k in ("friction_now", "friction_usual", "heated", "mood", "first_timer_share",
                                  "roomstate", "messages_seen", "moderated_topics", "vocabulary"):
                            snap.pop(k, None)
                    batch.append({"type": "room", **snap})
            if log_rooms:
                try:
                    self.session.flush()
                except Exception as e:
                    _log_error(self.eng, "session_log_errors", e, "writing the session log")
            batch.append(self._status())
        if batch and self.clients:
            data = self._encode(batch)
            for ws in list(self.clients):
                try:
                    await asyncio.wait_for(ws.send_str(data), SEND_TIMEOUT_S)
                except Exception:          # gone, or not reading: one browser must not stall the rest
                    self._drop_client(ws)

    def _encode(self, batch):
        """The batch as JSON; an item that cannot be encoded is left out alone, never the whole batch (LB4-1)."""
        try:
            return json.dumps(batch, default=_jsonable)
        except Exception:
            items = []
            for r in batch:
                try:
                    items.append(json.dumps(r, default=_jsonable))
                except Exception as e:
                    _log_error(self.eng, "flusher_errors", e, f"encoding a {r.get('type')} record for the page")
            return "[" + ",".join(items) + "]"

    def _drop_client(self, ws):
        """A browser that did not take a batch in time is closed, not just forgotten: its page then
        reconnects instead of showing "connected" with nothing arriving (LB-5)."""
        self.clients.discard(ws)
        self.eng.stats["ui_clients_dropped"] += 1
        asyncio.ensure_future(self._close_client(ws))

    async def _close_client(self, ws):
        try:
            await asyncio.wait_for(ws.close(code=1013, message=b"too slow: reconnect"), SEND_TIMEOUT_S)
        except Exception:
            tr = self.transports.pop(ws, None)
            if tr is not None:
                tr.abort()


def _jsonable(o):
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def _stamp():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def load_engine(ml, emotes, fresh=False):
    """The saved state, or a fresh engine. A file that cannot be read is moved aside first
    (engine.pkl.unreadable-<UTC time>), so no later save can replace it (LB-1); when even that fails
    the server refuses to start."""
    if not fresh and os.path.exists(STATE):
        try:
            eng = pickle.load(open(STATE, "rb")).upgrade()
        except Exception as e:
            aside = f"{STATE}.unreadable-{_stamp()}"
            try:
                os.replace(STATE, aside)
            except OSError as e2:
                raise SystemExit(f"[serve] could not restore {STATE} ({e}) nor move it aside ({e2}); "
                                 "refusing to start, so that nothing saves over it")
            print(f"[serve] could not restore state ({e}); kept it as {os.path.basename(aside)} and starting "
                  "fresh", flush=True)
            return Engine(ml=ml, emotes=emotes)
        eng.ml, eng.emotes = ml, emotes
        from .risk import RiskModel
        from .engine import PERSIST_PATH
        eng.risk = RiskModel()
        eng.persist = RiskModel(PERSIST_PATH)
        gone = eng.prune(int(time.time() * 1000))        # S7: chatters idle > 30 days
        print(f"[serve] restored state: {len(eng.users):,} chatters, {len(eng.rooms)} rooms"
              + (f" ({gone:,} idle chatters dropped)" if gone else ""))
        return eng
    return Engine(ml=ml, emotes=emotes)


def loaded_head_sha():
    """The sha256 of the toxicity head this run uses, loaded now so the hash is of the file in use. The
    pipeline page compares it with the live head to say whether a restart is still needed (PF-11)."""
    import hashlib
    from . import toxicity
    toxicity._head()
    path = os.environ.get("CW_TOX_HEAD", "data/models/tox_head.json")
    return hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None


def head_for(ml):
    """The head this run scores with, for /api/status: only with --ml (the pipeline page's restart reminder
    compares it with the live head; without --ml it would describe a head nobody uses: audit LB2-18)."""
    return loaded_head_sha() if ml is not None else None


def saves_state(replaying, fresh, replace_state):
    """LB-1: only a live run that loaded the saved state, or found none, saves over it. --fresh keeps
    an existing state unless --replace-state is given; a replay never saves."""
    return not replaying and (not fresh or replace_state or not os.path.exists(STATE))


def live_roles(replaying, path=None):
    """The owner's roles for this run (roles.py): a live run reads data/sessions/owner_roles.json, a replay nothing.
    A file that cannot be read (nor its .prev) is moved aside, as the state is (LB-1), so the next change cannot
    save over it; the run then starts with none, and the console says so."""
    if replaying:
        return {}
    path = path or roles_mod.ROLES
    try:
        return roles_mod.load(path)
    except fileio.Corrupt as e:
        stamp = _stamp()
        moved = []
        for p in (path, path + fileio.PREV):
            if os.path.exists(p):
                try:
                    os.replace(p, f"{p}.unreadable-{stamp}")
                except OSError as e2:
                    raise SystemExit(f"[serve] could not read {path} ({e}) nor move it aside ({e2}); refusing to start, "
                                     "so that no role change saves over it")
                moved.append(f"{os.path.basename(p)}.unreadable-{stamp}")
        # every copy moved aside is named, the .prev one too (LB6-4)
        print(f"[serve] could not read the owner's roles ({e}); kept as {' and '.join(moved) or 'nothing moved'}, "
              "starting with none", flush=True)
        return {}


def tune_gc():
    """Long runs (live H1): the restored state is permanent, so take it out of the collector's
    scans (gc.freeze) and make full collections rarer; at 50k chatters a full collection took
    1.2-1.8 s, in the middle of a message. autosave() collects and freezes again every 5 min."""
    gc.collect()
    gc.freeze()
    gc.set_threshold(700, 10, 1000)


_save_seq = itertools.count(1)
_save_lock = threading.Lock()
_saved = [0]                    # sequence number of the dump now on disk


def _detached(eng):
    """Context: the engine without its models and caches, as it is pickled."""
    class _Ctx:
        def __enter__(self):
            self.kept = eng.ml, eng.emotes, eng.risk, getattr(eng, "persist", None)
            eng.ml = eng.emotes = eng.risk = eng.persist = None
            eng.__dict__.pop("_notice_p", None)      # panel cache: recomputed with the models loaded next time
            eng.__dict__.pop("_by_ch", None)         # channel index: rebuilt on the next room snapshot
            # the owner's roles live in their own file (roles.py), never in the state that replays and set builders load
            self.bots = eng.__dict__.pop("owner_bots", None)

        def __exit__(self, *a):
            eng.ml, eng.emotes, eng.risk, eng.persist = self.kept
            if self.bots is not None:
                eng.owner_bots = self.bots
    return _Ctx()


def dump_engine(eng):
    """(sequence number, pickled state), pickled on the calling thread (the loop). save_engine_async
    with fork=True pickles in a child process instead (LB-24)."""
    with _detached(eng):
        return next(_save_seq), pickle.dumps(eng, protocol=pickle.HIGHEST_PROTOCOL)


def _keep_previous():
    """engine.pkl.prev: the state before this save (a hard link where the share allows one, else a copy),
    so one bad save never costs the only copy (LB-1)."""
    if not os.path.exists(STATE):
        return
    prev, tmp = STATE + ".prev", STATE + ".prev.tmp"
    try:
        os.remove(tmp)
    except OSError:
        pass
    try:
        os.link(STATE, tmp)
    except OSError:
        shutil.copyfile(STATE, tmp)
    os.replace(tmp, prev)


def _install(seq, tmp):
    """Move a written dump into place, unless a newer one is already there (an autosave finishing
    after Stop's save). True when installed."""
    with _save_lock:
        if seq < _saved[0]:
            os.remove(tmp)
            return False
        _keep_previous()
        os.replace(tmp, STATE)
        _saved[0] = seq
        return True


def write_state(seq, data):
    """Write a dump to a temporary file, then rename it over the state, so a crash or a full disk
    never leaves half a file. Thread-safe; a dump older than the one on disk is dropped. True when written."""
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    tmp = f"{STATE}.tmp{seq}"
    with _save_lock:
        if seq < _saved[0]:
            return False
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return _install(seq, tmp)


def save_engine(eng):
    write_state(*dump_engine(eng))


def _fork_dump(eng, seq):
    """Pickle and write the state in a forked child (copy-on-write), so the loop pauses only for the
    fork (LB-24: pickling 38k chatters took 2.6 s on the loop). Returns the child's pid."""
    tmp = f"{STATE}.tmp{seq}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)      # "multi-threaded fork": the child only pickles
        pid = os.fork()
    if pid:
        return pid
    code = 1
    try:                                                         # the child: no prints (stdio locks), no loop
        with _detached(eng):
            data = pickle.dumps(eng, protocol=pickle.HIGHEST_PROTOCOL)
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        code = 0
    except BaseException as e:                                   # noqa: BLE001
        try:
            os.write(2, f"[serve] save child failed: {type(e).__name__}: {e}\n".encode())
        except OSError:
            pass
    finally:
        os._exit(code)


async def save_engine_async(eng, fork=False):
    """save_engine with the file write off the event loop; fork=True (the autosave) also takes the
    pickling off it."""
    loop = asyncio.get_running_loop()
    if fork and hasattr(os, "fork"):
        os.makedirs(os.path.dirname(STATE), exist_ok=True)
        seq = next(_save_seq)
        pid = _fork_dump(eng, seq)
        _, status = await loop.run_in_executor(None, os.waitpid, pid, 0)
        if os.waitstatus_to_exitcode(status) != 0:
            try:
                os.remove(f"{STATE}.tmp{seq}")
            except OSError:
                pass
            raise OSError(f"the state was not written (save child exit {os.waitstatus_to_exitcode(status)})")
        await loop.run_in_executor(None, _install, seq, f"{STATE}.tmp{seq}")
        return
    seq, data = dump_engine(eng)
    await loop.run_in_executor(None, write_state, seq, data)


# ---------------------------------------------------------------- one dashboard at a time (LB-16)
def take_lock(port, force=False):
    """Create data/state/dashboard.lock atomically ("host pid since port"). Another dashboard's lock
    refuses the start unless force (CW_FORCE=1: that machine is off or crashed). Returns the text
    written, for release_lock."""
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    host = os.environ.get("CW_HOST") or socket.gethostname()
    token = f"{host} {os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {port}\n"
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        if not force:
            try:
                held = open(LOCK).read().strip()
            except OSError:
                held = "?"
            raise SystemExit(f"[serve] refusing to start: data/state/dashboard.lock says a dashboard already "
                             f"runs ({held}). Stop that one first; if its machine is off or crashed, start with "
                             "CW_FORCE=1.")
        with open(LOCK + ".tmp", "w") as f:
            f.write(token)
        os.replace(LOCK + ".tmp", LOCK)
        return token
    with os.fdopen(fd, "w") as f:
        f.write(token)
    return token


def release_lock(token):
    """Remove the lock only if it is still ours (CW_FORCE on another machine may have taken it over)."""
    try:
        if open(LOCK).read() == token:
            os.remove(LOCK)
    except OSError:
        pass


async def live_source(hub, channels, record_dir=None, session=None):
    from .collect import RawSink
    sink = RawSink(record_dir) if record_dir else None
    try:
        await _live_loop(hub, channels, sink, session)
    finally:                          # Stop / shutdown cancels us here
        if sink:
            sink.close()


IRC_IDLE_S = 360           # Twitch PINGs about every 5 min: 6 min of nothing is a dead connection


async def _pause(seconds):
    await asyncio.sleep(seconds)


async def _live_loop(hub, channels, sink, session):
    """Read chat until cancelled. A dead connection (nothing for IRC_IDLE_S), Twitch's RECONNECT
    and a normal close are disconnects like any error: connected=False, a backoff pause, then a new
    connection (live M3). "connected" means the server has answered; only a chat message resets the
    backoff, so a server that accepts and then closes is retried ever more slowly (LB-11)."""
    import websockets
    from .collect import _channel_of
    backoff = 1
    while True:
        why = "closed by the server"
        try:
            async with websockets.connect("wss://irc-ws.chat.twitch.tv:443", ping_interval=None) as ws:
                await ws.send("CAP REQ :twitch.tv/tags twitch.tv/commands")
                await ws.send("PASS SCHMOOPIIE")
                await ws.send(f"NICK justinfan{random.randint(10000, 99999)}")
                await ws.send("JOIN " + ",".join("#" + c for c in channels))
                while why == "closed by the server":
                    try:
                        data = await asyncio.wait_for(ws.recv(), IRC_IDLE_S)
                    except asyncio.TimeoutError:
                        why = f"nothing received for {IRC_IDLE_S} s"
                        break
                    except websockets.ConnectionClosedOK:
                        break
                    hub.status["connected"] = True
                    ms = int(time.time() * 1000)
                    for line in data.split("\r\n"):
                        if not line:
                            continue
                        if line.startswith("PING"):
                            await ws.send("PONG :tmi.twitch.tv")
                            continue
                        if " RECONNECT" in line[:80]:
                            why = "the server asked to reconnect"
                            break
                        _raw(hub, sink, session, _channel_of(line), line, ms)
                        ev = irc.to_event(line, ms)
                        if ev:
                            if ev.get("kind") == "msg":
                                backoff = 1
                            handle(hub, ev)
        except Exception as e:
            why = str(e) or type(e).__name__
        hub.status["connected"] = False
        print(f"[serve] disconnected: {why}; retry in {backoff}s", flush=True)
        await _pause(backoff)
        backoff = min(backoff * 2, 60)


def _raw(hub, sink, session, chan, line, ms):
    """Record a raw line (--record, and the session). A write error is counted and shown once;
    it is not a disconnect (reconnecting would not help a full disk)."""
    for write in (sink and sink.write, session and session.raw_line):
        if write:
            try:
                write(chan, line, ms)
            except Exception as e:
                _log_error(hub.eng, "raw_log_errors", e, "writing raw chat")


async def replay_source(hub, paths, channels, speed):
    from .replay import iter_events
    hub.status["connected"] = True
    t_first = t0 = None
    since_yield = 0
    for ev in iter_events(paths, set(channels) if channels else None):
        if t_first is None:
            t_first, t0 = ev["ts"], time.time()
        since_yield += 1
        if speed > 0:
            due = t0 + (ev["ts"] - t_first) / 1000 / speed
            delay = due - time.time()
            if delay > 0:
                await asyncio.sleep(delay)
                since_yield = 0
        if since_yield >= REPLAY_YIELD:          # faster than the server: let the page and the API in
            await asyncio.sleep(0)
            since_yield = 0
        handle(hub, ev)
        if ev.get("channel") and ev["channel"] not in hub.status["channels"]:
            hub.status["channels"].append(ev["channel"])
    hub.status["connected"] = False
    hub.status["mode"] = "replay finished"


_errors_seen = set()
ERRORS_SHOWN = 100       # distinct errors printed; after that they are only counted (texts can hold per-user data)


def _log_error(eng, counter, e, where):
    """Count an error in eng.stats[counter] and print it with its traceback, once per kind;
    call from inside the except block."""
    eng.stats[counter] += 1
    sig = f"{counter} {type(e).__name__}: {e}"
    if sig in _errors_seen or len(_errors_seen) > ERRORS_SHOWN:
        return
    _errors_seen.add(sig)
    if len(_errors_seen) > ERRORS_SHOWN:
        print(f"[serve] {ERRORS_SHOWN} different errors shown; further ones are only counted (stats)", flush=True)
        return
    import traceback
    print(f"[serve] error {where} (skipped; shown once): {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()


def handle(hub, ev):
    """One event. An error here is logged (once per kind) and skipped; it must
    never take down the connection (it used to: one bad state -> reconnect loop)."""
    hub.eng.clock = ev["ts"]
    try:
        rec = hub.eng.process(ev)
        if rec:
            hub.publish(rec)
    except Exception as e:
        _log_error(hub.eng, "event_errors", e, f"on a {ev.get('kind')} event")


async def autosave(hub):
    eng = hub.eng
    while True:
        await asyncio.sleep(300)
        if not hub.persist:
            continue
        try:
            eng.prune(getattr(eng, "clock", None) or int(time.time() * 1000))     # S7: never per message
            hub.stash_queue()               # a crash loses no more of the LLM queue than of the state
            await save_engine_async(eng, fork=True)
        except Exception as e:          # a full disk must not end the server; the next save retries
            _log_error(eng, "save_errors", e, "saving the state")
        gc.collect()                    # a pause here is expected; then keep the survivors out of later scans
        gc.freeze()


def make_app(hub, port=8765):
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    origins = {f"http://{h}" for h in hosts}
    # the pipeline page and its jobs ask /api/status across the podman network (gpu_busy)
    hosts |= {f"cw-live:{port}", f"host.containers.internal:{port}"}

    @web.middleware
    async def same_origin(request, handler):
        """Only the page itself may use this server (the reads are for the mods only). A foreign
        Host is a DNS-rebinding page; a foreign Origin is another site reading the websocket or
        driving Start/Stop; a POST must also carry X-Chatwatch, which another site's form, link,
        image or no-cors fetch cannot send. Everything that starts an LLM call is a POST (LB-10).
        bin/cw publishes the same port on the host."""
        if request.host not in hosts:
            raise web.HTTPForbidden(text=f"open the dashboard at http://127.0.0.1:{port}")
        origin = request.headers.get("Origin")
        if origin is not None and origin not in origins:
            raise web.HTTPForbidden(text="refused: not the dashboard's own page")
        if request.method not in ("GET", "HEAD") and request.headers.get("X-Chatwatch") != "1":
            raise web.HTTPForbidden(text="refused: not sent by the dashboard page")
        return await handler(request)

    async def index(_):
        return web.FileResponse(os.path.join(WEB, "index.html"))

    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        hub.clients.add(ws)
        hub.transports[ws] = request.transport
        try:
            await ws.send_str(json.dumps([hub._status()], default=_jsonable))
            async for msg in ws:
                if msg.type == WSMsgType.ERROR:
                    break
        finally:
            hub.clients.discard(ws)
            hub.transports.pop(ws, None)
        return ws

    async def profile(request):
        uid = request.query.get("uid", "")
        ch = request.query.get("channel", "")
        p = hub.eng.profile(ch, uid, since=hub.since)
        role = hub.roles.get(uid)
        if p is None and role:
            # a chatter marked a bot before they ever spoke here has no profile in this channel: the card shows the
            # role, so it can be undone from their chat line
            p = {"user_id": uid, "login": role.get("login"), "display": role.get("display") or role.get("login"),
                 "channel": ch, "stub": True}
        if p is None:
            return web.json_response({"error": "unknown chatter or channel"}, status=404)
        u = hub.eng.users.get((ch, uid))
        if u is not None and not p.get("stub") and uid not in hub.eng.owner_bots:
            # the card's notice at the watchlist's display time, so on a quiet live chat they agree (LB6-1); display
            # only (engine.py is in the layer fingerprint, so it is recomputed here)
            nr = eng_mod._notice_reasons(u, (p.get("persist") or {}).get("p") if u.toxic_n else None, hub._display_ts(),
                                         getattr(hub.eng, "person_review", False), since=hub.since)
            p["notice"] = ({"level": nr[0], "why": nr[1],
                            "owner": any(eng_mod._owner(m.get("toxicity")) and eng_mod._level(m["toxicity"]) != "none"
                                         for m in u.msgs)} if nr else None)
        # the local LLM's latest person read, for the user card (display only; engine.py is in the layer
        # fingerprint, so it is added here)
        rv = getattr(hub.eng.users.get((ch, uid)), "review", None) or {}
        p["person_read"] = {k: rv.get(k) for k in ("label", "rationale", "n", "ts", "reads")} if rv.get("label") else None
        p["evidence"] = [_shown_msg(m) for m in hub.eng.evidence(ch, uid) or []]   # the messages behind a listing
        if p.get("recent_messages"):
            p["recent_messages"] = [_shown_msg(m) for m in p["recent_messages"]]      # the level each counts at (LF3-2)
        p["owner_role"] = role
        for k in ("first_seen", "badfaith", "friction"):      # the LLM reads these; the card does not (audit LF4-7)
            p.pop(k, None)
        if p.get("register"):
            p["register"] = {k: p["register"].get(k) for k in CARD_REGISTER}
        return web.json_response(p, dumps=lambda o: json.dumps(o, default=_jsonable))

    async def evidence(request):
        """The messages behind a chatter's listing, for the watchlist card's dropdown (Engine.evidence): each with
        its read, whether the watchlist rule reads it now ("counted"), and whether it has left the recent messages
        ("older": still correctable with the ⋯)."""
        ch, uid = request.query.get("channel", ""), request.query.get("uid", "")
        items = hub.eng.evidence(ch, uid)
        if items is None:
            return web.json_response({"error": "unknown chatter or channel"}, status=404)
        return web.json_response({"channel": ch, "user_id": uid, "items": [_shown_msg(m) for m in items]},
                                 dumps=lambda o: json.dumps(o, default=_jsonable))

    async def roles_list(request):
        """Every role the owner set (roles.py), newest first, so each can be seen and undone."""
        items = sorted(({"user_id": k, **v} for k, v in hub.roles.items()), key=lambda e: e.get("set") or "", reverse=True)
        return web.json_response({"roles": items, "saves": hub.status["mode"] == "live" and hub.persist})

    async def role(request):
        """The owner sets or unsets a chatter's role: {channel, uid, role}, role "bot" or null (roles.py; never
        automated, and there is no "mod" role). A bot is not scored, profiled or listed from now on
        (Engine.owner_bots), and its watchlist entry goes at the next room snapshot.
        - Saved: in a live run that saves its state, data/sessions/owner_roles.json is written first; if that fails
          the change is not applied (500, and the page's "roles not saved" pill). A replay or a --fresh run that is
          not saved changes only its own view. Every change is logged to the open session.
        - Setting needs the chatter known in that channel; unsetting needs only the user id (the page's list)."""
        try:
            body = await request.json()
        except Exception:
            body = None
        if not isinstance(body, dict):
            return web.json_response({"error": "the body must be a JSON object"}, status=400)
        want = body.get("role")
        if want is not None and want not in roles_mod.KINDS:
            return web.json_response({"error": "role must be one of " + ", ".join(roles_mod.KINDS) + ", or null (remove)"},
                                     status=400)
        ch, uid = str(body.get("channel", "")), str(body.get("uid", ""))
        have = hub.roles.get(uid)
        u = hub.eng.users.get((ch, uid))
        if want is not None and u is None and not have:
            return web.json_response({"error": "unknown chatter or channel"}, status=404)
        if want is None and not have:
            return web.json_response({"error": "that chatter has no role to remove"}, status=404)
        recorded = hub.status["mode"] == "live" and hub.persist
        if (have or {}).get("role") == want:                       # a repeated click: nothing changed or recorded
            return web.json_response({"user_id": uid, "role": want, "entry": have, "recorded": False, "unchanged": True,
                                      "rev": hub.roles_rev})
        new = dict(hub.roles)
        if want is None:
            new.pop(uid)
        else:
            new[uid] = roles_mod.entry(want, u.login, u.display)
        if recorded:
            try:
                roles_mod.save(new)
            except Exception as e:
                _log_error(hub.eng, "role_save_errors", e, "writing data/sessions/owner_roles.json")
                return web.json_response({"error": "not applied: the role could not be saved in "
                                          "data/sessions/owner_roles.json. See the server's console.",
                                          "not_recorded": True}, status=500)
        hub.roles = new
        hub.roles_rev += 1
        hub.eng.owner_bots = roles_mod.bots(new)
        hub._last_room = 0                                         # the watchlist without them, at the next flush
        e = new.get(uid) or have
        hub._log({"type": "owner_role", "ts": hub.eng.clock or int(time.time() * 1000),
                  "wall": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "channel": ch, "user_id": uid,
                  "login": e.get("login"), "role": want, "recorded": recorded})
        return web.json_response({"user_id": uid, "role": want, "entry": new.get(uid), "recorded": recorded,
                                  "rev": hub.roles_rev})

    async def _object(request):
        """The body as a JSON object; anything else (not JSON, a list, a string) answers 400, never 500 (LB4-5)."""
        try:
            body = await request.json()
        except Exception:
            body = None
        if not isinstance(body, dict):
            raise web.HTTPBadRequest(text=json.dumps({"error": "the body must be a JSON object"}),
                                     content_type="application/json")
        return body

    async def _who(request):
        body = await _object(request)
        ch, uid = str(body.get("channel", "")), str(body.get("uid", ""))
        return ch, uid, hub.eng.profile(ch, uid)

    async def llm_review(request):
        """Ask the optional local LLM (ollama container) for a policy read. POST: it costs GPU time."""
        from .llm import review
        ch, uid, p = await _who(request)
        if p is None:
            return web.json_response({"error": "unknown chatter or channel"}, status=404)
        busy = hub.llm_tools_busy()
        if busy:
            return web.json_response({"error": busy}, status=409)
        ctx = list(hub.eng.rooms[ch].context)[-20:]
        try:
            em = hub.eng.emote_names(ch)
            res = await hub._llm(lambda: review(p, ctx, emotes=em))
        except Exception as e:
            return web.json_response({"error": str(e)}, status=503)
        return web.json_response(res)

    CHAN = re.compile(r"^[a-z0-9_]{3,25}$")

    async def draft_warning(request):
        """Ask the local LLM for a calm, ready-to-paste warning. Never posted. POST: it costs GPU time."""
        from .llm import draft_warning as draft
        ch, uid, p = await _who(request)
        if p is None:
            return web.json_response({"error": "unknown chatter or channel"}, status=404)
        busy = hub.llm_tools_busy()
        if busy:
            return web.json_response({"error": busy}, status=409)
        ctx = list(hub.eng.rooms[ch].context)[-20:]
        em = hub.eng.emote_names(ch)
        try:
            res = await hub._llm(lambda: draft(p, ctx, emotes=em))
        except Exception as e:
            return web.json_response({"error": str(e)}, status=503)
        hub._log({"type": "draft_warning", "ts": hub.eng.clock, "channel": ch, "user_id": uid, **res})
        return web.json_response(res)

    async def start(request):
        if hub.status["mode"] != "live":
            raise web.HTTPBadRequest(text="replay mode")
        body = await _object(request)
        raw = body.get("channels", "")
        chans = [c.strip().lower().lstrip("#").replace("twitch.tv/", "")
                 for c in (raw if isinstance(raw, list) else str(raw).replace(" ", ",").split(","))]
        chans = list(dict.fromkeys(c for c in chans if c))
        bad = [c for c in chans if not CHAN.match(c)]
        if not chans or bad:
            return web.json_response({"error": f"invalid channel name(s): {', '.join(bad) or '(none)'}"}, status=400)
        if len(chans) > 1:                # owner 2026-11-07: one channel per live session
            return web.json_response({"error": "one channel at a time: Stop, then Start the other"}, status=400)
        try:
            hub.start_live(chans)
        except RuntimeError as e:
            return web.json_response({"error": str(e)}, status=409)
        except OSError as e:            # the session folder could not be written (disk full, share gone)
            return web.json_response({"error": f"could not open the session log: {e}"}, status=503)
        return web.json_response({"ok": True, "channels": chans, "session": hub.status["session"]})

    async def stop(request):
        """Stop: chat stops at once, the LLM queue is finished, then the session closes. {"now": true} (the page's
        "Stop now", also while a Stop finishes) skips the queue's rest, counted in llm_dropped_at_stop."""
        if hub.status["mode"] != "live":            # a replay has no session, and its engine is not the saved one
            return web.json_response({"error": "not a live run"}, status=400)
        try:
            body = await request.json() if request.can_read_body else {}
        except Exception:
            body = {}
        now = isinstance(body, dict) and body.get("now") is True
        if not hub.status["running"] and not hub.stopping:
            return web.json_response({"ok": True})   # nothing to stop (a stale page): a kept queue waits for Start
        await hub.stop_live(now=now)
        return web.json_response({"ok": True})

    async def override(request):
        """The owner reclassifies one message: {channel, uid, id, level, kind}, one of the menu's pairs
        (engine.OWNER_CHOICES; level null goes back to the automatic read). The correction steers the live
        watchlist only (engine.apply_owner_override); it is never a label.
        - Recorded: a correction to the saved state (a live run that saves it) is first written to
          data/sessions/owner_overrides.jsonl, which the set builders leave out. If that write fails the
          correction is not applied: 500 with the error, and the page's "corrections not recorded" pill
          (N-3). A replay or a --fresh run that is not saved steers only its own view and is not recorded
          (LB2-12, N-11). Every correction is logged to the open session, if any.
        - Person reads: only while a live session runs (LB2-1), and only when the correction gives the
          chatter their first toxic message or they have no read yet (LB2-4); "Back to automatic" runs the
          same checks on the restored read, S11 included (LB2-9).
        - The same correction twice changes nothing and records nothing (N-5)."""
        try:
            body = await request.json()
        except Exception:
            body = None
        if not isinstance(body, dict):
            return web.json_response({"error": "the body must be a JSON object"}, status=400)
        level, kind = body.get("level"), body.get("kind") or None
        if (not isinstance(level, (str, type(None))) or not isinstance(kind, (str, type(None)))
                or (level, kind) not in eng_mod.Engine.OWNER_CHOICES):
            return web.json_response({"error": "level and kind must be one of the menu's choices: none, low, medium, "
                                      "high with a kind (" + ", ".join(eng_mod.Engine.OWNER_KINDS) + "), or null "
                                      "(back to automatic)"}, status=400)
        ch, uid, mid = str(body.get("channel", "")), str(body.get("uid", "")), str(body.get("id", ""))
        u = hub.eng.users.get((ch, uid))
        m = hub.eng.held_message(ch, uid, mid)               # recent messages, then the evidence (EVIDENCE_KEPT)
        before = (m or {}).get("toxicity")
        if m is None or (level is None and not eng_mod._owner(before)):
            return web.json_response({"error": "that message is not held any more (only recent messages and the "
                                      "messages behind a listing are kept), or it has no correction to undo"}, status=404)
        live = hub.status["mode"] == "live"
        recorded = live and hub.persist
        if hub.eng.owner_same(before, level, kind):              # a repeated click: nothing changed or recorded
            upd = hub.eng.apply_owner_override(ch, uid, mid, level, kind)
            return web.json_response({**hub._ui(upd), "recorded": False, "unchanged": True})
        ts = hub.eng.clock or int(time.time() * 1000)
        rec = {"type": "owner_override", "ts": ts, "wall": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "channel": ch, "user_id": uid, "msg_id": mid, "level": level, "kind": kind,
               # the level the message counted at (S1: a re-read keyword slur counts at the layer's), and its class
               "before": {"level": eng_mod._level(before), "class": eng_mod._tox_class(before) if before else None,
                          "source": (before or {}).get("source", "layer")}}
        if recorded:
            try:
                log_override(rec)
            except Exception as e:
                _log_error(hub.eng, "override_log_errors", e, "writing data/sessions/owner_overrides.jsonl")
                return web.json_response({"error": "not applied: the correction could not be recorded in "
                                          "data/sessions/owner_overrides.jsonl (the set builders read it). "
                                          "See the server's console.", "not_recorded": True}, status=500)
        toxic_before = u.toxic_n
        upd = hub.eng.apply_owner_override(ch, uid, mid, level, kind)
        hub._log({**rec, "recorded": recorded})
        hub._log({**upd, "ts": ts})
        hub.queue.append(hub._ui(upd))
        # no LLM work on a chatter the owner marked a bot (audit LB3-5)
        if (live and hub.status["running"] and hub.review_q is not None and hub.eng.person_review
                and uid not in hub.eng.owner_bots):
            t = upd["toxicity"]
            if level is None and hub.eng.wants_person_review_llm(ch, uid, t, ts):
                hub._queue_person(ch, uid, ts)                    # S11 on the restored LLM-only hate
            elif (eng_mod._level(t) != "none" and (toxic_before == 0 or not (u.review or {}).get("label"))
                  and hub.eng.wants_person_review({"channel": ch, "user_id": uid, "ts": ts, "toxicity": t,
                                                   "alerts": []})):
                hub._queue_person(ch, uid, ts)
        return web.json_response({**hub._ui(upd), "recorded": recorded})

    # ------------------------------------------------------------------ the AI: picker, speed check, hybrid review
    async def ai_info(request):
        """The machine's specs, the setups that fit with their numbers, the choice and the running task (aictl.py).
        ?vram=..&ram=.. previews the dropdowns' override without saving it."""
        ov = {k: request.query[k] for k in ("vram", "ram") if k in request.query} or None
        return web.json_response(await hub.ai_info(ov), dumps=lambda o: json.dumps(o, default=_jsonable))

    def _ai_answer(fn):
        async def h(request):
            body = await _object(request)
            try:
                task = fn(body)
            except aictl.Busy as e:
                return web.json_response({"error": str(e)}, status=409)
            except ValueError as e:
                return web.json_response({"error": str(e)}, status=400)
            return web.json_response({"ok": True, "task": task}, dumps=lambda o: json.dumps(o, default=_jsonable))
        return h

    async def ai_override(request):
        body = await _object(request)
        return web.json_response({"ok": True, "override": hub.save_override(body)})

    async def ai_cancel(request):
        return web.json_response({"ok": True, "cancelling": hub.cancel_ai_task()})

    async def ai_install_cancel(request):
        return web.json_response({"ok": True, "cancelling": hub.cancel_install()})

    async def review_estimate(request):
        if hub.ai.get("mode") != "hybrid":
            return web.json_response({"error": "not a hybrid setup"}, status=409)
        return web.json_response(hub.review_summary(force=True))

    async def status(request):
        s = hub._status()
        s.pop("type")
        return web.json_response(s)

    app = web.Application(middlewares=[same_origin])
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/api/profile", profile)
    app.router.add_post("/api/llm_review", llm_review)
    app.router.add_post("/api/start", start)
    app.router.add_post("/api/draft_warning", draft_warning)
    app.router.add_post("/api/stop", stop)
    app.router.add_post("/api/override", override)
    app.router.add_get("/api/evidence", evidence)
    app.router.add_get("/api/roles", roles_list)
    app.router.add_post("/api/role", role)
    app.router.add_get("/api/status", status)
    app.router.add_get("/api/ai", ai_info)
    app.router.add_post("/api/ai/select", _ai_answer(hub.start_select))
    app.router.add_post("/api/ai/speed", _ai_answer(hub.start_speed))
    app.router.add_post("/api/ai/chatrate", _ai_answer(hub.start_chatrate))
    app.router.add_post("/api/ai/override", ai_override)
    app.router.add_post("/api/ai/cancel", ai_cancel)
    app.router.add_post("/api/ai/install", _ai_answer(hub.start_install))      # a model download (install.py)
    app.router.add_post("/api/ai/install_cancel", ai_install_cancel)
    app.router.add_post("/api/review", _ai_answer(hub.start_review))
    app.router.add_get("/api/review", review_estimate)
    app.router.add_static("/static", WEB)
    return app


def prune_sessions(days=None, root=None):
    """Session logs hold raw chat with usernames: those older than CW_SESSION_KEEP_DAYS (default 14; 0 keeps them all)
    are deleted at start. Only the session folders (<date>-<time>_<channel>); the files beside them stay."""
    try:
        days = float(os.environ.get("CW_SESSION_KEEP_DAYS", 14)) if days is None else days
    except ValueError:
        days = 14.0
    if days <= 0:
        return 0
    base = os.path.join(root or ROOT, "data", "sessions")
    cut, n = time.time() - days * 86400, 0
    for name in (os.listdir(base) if os.path.isdir(base) else ()):
        p = os.path.join(base, name)
        m = re.match(r"^(\d{8}-\d{6})_", name)
        if not m or not os.path.isdir(p):
            continue
        try:
            started = calendar.timegm(time.strptime(m.group(1), "%Y%m%d-%H%M%S"))   # the run's start (UTC, its name)
        except ValueError:
            continue
        if started < cut:
            shutil.rmtree(p, ignore_errors=True)
            n += 1
    if n:
        print(f"[serve] deleted {n} session log(s) older than {days:g} days (CW_SESSION_KEEP_DAYS)", flush=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", default="")
    ap.add_argument("--replay", nargs="*")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed; 0 = as fast as possible")
    ap.add_argument("--ml", action="store_true", help="load the local toxicity model (T1)")
    ap.add_argument("--log", action="store_true",
                    help="keep a session log (data/sessions/<run>/: raw chat lines with usernames and the reads); off "
                         "by default; logs older than CW_SESSION_KEEP_DAYS (14) are deleted at start")
    ap.add_argument("--no-log", action="store_true", help="no session log (the default; kept for older scripts)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to listen on (127.0.0.1; the container passes 0.0.0.0 and publishes it on 127.0.0.1)")
    ap.add_argument("--fresh", action="store_true",
                    help="start from an empty state; the saved state is kept and this run is not saved")
    ap.add_argument("--replace-state", action="store_true",
                    help="with --fresh: save this run over data/state/engine.pkl (the old one becomes engine.pkl.prev)")
    ap.add_argument("--alert-budget", type=int, default=20,
                    help="alerts per channel per hour before only the strongest get through (0 = no limit)")
    ap.add_argument("--tox-llm-per-min", type=int, default=1,
                    help="with --llm-review: LLM toxicity reads in windows (1 = on, 0 = off)")
    ap.add_argument("--person-review", action="store_true",
                    help="with --llm-review: the AI's read of each candidate decides 'watch' (./start.sh turns it on)")
    ap.add_argument("--llm-review", action="store_true",
                    help="send every bad-faith alert to the local policy model (bin/ollama.sh)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CW_PORT", 8765)))
    ap.add_argument("--ai", choices=("none", "saved"), default="saved",
                    help="none: this run starts with the AI off (./start.sh --no-llm); saved: this machine's choice "
                         "(data/state/machines/<host>.json, set on the page), else CW_LLM_MODEL")
    a = ap.parse_args()
    chans = [c.strip().lower().lstrip("#") for c in a.channels.split(",") if c.strip()]
    replaying = a.replay is not None
    if not replaying and len(chans) > 1:     # owner 2026-11-07: a live session watches one channel (a replay may hold several)
        raise SystemExit("[serve] one channel at a time: --channels takes one channel in live mode")

    lock = None if replaying else take_lock(a.port, force=os.environ.get("CW_FORCE") == "1")
    try:
        _main(a, chans, replaying)
    finally:
        if lock:
            release_lock(lock)


def _main(a, chans, replaying):
    ml = None
    if a.ml:
        from .models import Combined
        ml = Combined()
    emotes = EmoteSets(os.path.join(ROOT, "data", "emotes"), background=not a.replay)
    eng = load_engine(ml, emotes, fresh=a.fresh or replaying)
    saved_queue = getattr(eng, "llm_queue", None)
    eng.llm_queue = None
    tune_gc()
    eng.clock = 0
    eng.alert_budget = a.alert_budget
    eng.person_review = bool(a.llm_review and a.person_review)
    if not hasattr(eng, "alert_times"):
        eng.alert_times = {}
    hub = Hub(eng)
    # LB-1: only a live run that loaded the saved state (or found none) saves over it
    hub.persist = saves_state(replaying, a.fresh, a.replace_state)
    hub.roles = live_roles(replaying)              # the owner's roles: live runs only (roles.py)
    eng.owner_bots = roles_mod.bots(hub.roles)
    if not replaying and not hub.persist:
        print("[serve] --fresh: data/state/engine.pkl is kept, and this run's state will NOT be saved "
              "(add --replace-state to save over it)", flush=True)
    hub.tox_per_min = a.tox_llm_per_min if a.llm_review else 0
    hub._person_flag, hub._tox_flag = bool(a.person_review), a.tox_llm_per_min
    hub.llm_enabled = bool(a.llm_review)        # ./start.sh always passes it; --no-llm there is --ai none
    hub.allow_ai_in_replay = replaying and os.environ.get("CW_AI_REPLAY") == "1"     # tests and the replay check only
    hub.status.update(mode="replay" if replaying else "live", channels=list(chans) if replaying else [],
                      # the head this run scores with: only with --ml (the pipeline page's restart reminder; LB2-18)
                      head_sha256=head_for(ml),
                      saves=hub.persist)        # false: a correction here steers this view only (not recorded)
    hub.log_sessions = bool(a.log) and not a.no_log
    prune_sessions()
    hub.record = False                         # the public build never adds to a raw corpus

    async def run():
        runner = web.AppRunner(make_app(hub, a.port))
        await runner.setup()
        await web.TCPSite(runner, a.host, a.port).start()
        what = (", ".join(chans) or "all channels in capture") if replaying else (", ".join(chans) or "idle")
        print(f"[serve] dashboard on http://127.0.0.1:{a.port}  ({hub.status['mode']}: {what})", flush=True)
        tasks = [asyncio.create_task(hub.flusher())]
        if a.llm_review:
            hub.review_q = ReadQueue()             # lossless: never full (docs/AGENDA.md)
            if hub.persist:                       # (a replay or --fresh run does not load the saved queue)
                n = hub.restore_queue(saved_queue)
                if n:
                    low = hub._restored_low
                    print(f"[serve] {n:,} LLM jobs kept by the last shutdown wait for the next Start"
                          + (f" ({low:,} low-priority lines among them age out {LINE_AGE_OUT_S // 60} min after they were "
                             "queued, as usual; the page counts them under \"aged out\" at Start)" if low else ""),
                          flush=True)
            hub.llm_open = replaying              # live: from Start (start_live)
            tasks.append(asyncio.create_task(hub.reviewer()))
            tasks.append(asyncio.create_task(hub.tox_batcher()))     # also when the run starts with the AI off
            tasks.append(asyncio.create_task(hub.prefetch_loop()))   # hybrid: both models kept in the page cache
        elif saved_queue and hub.persist:
            n = len(saved_queue.get("reads", ())) + len(saved_queue.get("lines", ()))
            print(f"[serve] {n:,} LLM jobs kept by the last shutdown are dropped: this run has no --llm-review",
                  flush=True)
        c = hub.startup_choice(a.ai)          # this machine's saved choice (aictl.py), or CW_LLM_MODEL
        print(f"[serve] AI: {c['mode']}" + (f", live {c.get('live')}" if c["mode"] != "none" else "")
              + (f", reviewer {c.get('reviewer')}" if c["mode"] == "hybrid" else ""), flush=True)
        if replaying:
            tasks.append(asyncio.create_task(replay_source(hub, a.replay, chans, a.speed)))
        else:
            tasks.append(asyncio.create_task(autosave(hub)))
            if chans:
                hub.start_live(chans)
            else:
                print("[serve] idle: choose channels on the page and press Start", flush=True)
        import signal
        loop = asyncio.get_running_loop()
        # SIGHUP too: a closed terminal (podman run -it) must save and close the session (LB-17)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            loop.add_signal_handler(sig, lambda: [t.cancel() for t in tasks])
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            print("[serve] stopping", flush=True)
        finally:
            if not replaying:
                await hub.stop_live(finish=False)      # a shutdown: the LLM queue is kept in the saved state

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
