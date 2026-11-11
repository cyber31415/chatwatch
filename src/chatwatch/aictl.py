"""The dashboard's AI controls (docs/DASHBOARD.md "Choosing the AI" and "Hybrid: the reviewer"): the model picker, the
on-machine speed check, the channel-rate probe and the hybrid review batch. Mixed into serve.Hub; picker.py holds the
parts with no server state.

- **Choice** (only while stopped): No AI, one model, or a hybrid (a fast live model and a better reviewer). Applying it
  unloads the old model, loads the new one with its load options (its GPU layer cap on this machine) and saves the
  choice per machine. "No AI" turns the LLM work off: the engine runs the keyword layer and head with the no-LLM rule.
- **Speed check** (only while stopped): the speed2 method on a fixed synthetic sample, 1-3 min; for an MTP build with no
  known cap it first reads Ollama's fit from the log ./start.sh copies, sets the cap to the fit minus 3, and accepts it
  only with 0 runner aborts over the check's calls (stepping down 2 layers on an abort). Cached per model and machine.
- **Hybrid review** (owner, 2026-11-07: "one button ... one full swap between models rather than 5", "only reviewing
  new content"): one press = one swap for the whole batch. Hold the live worker after its call in flight, unload the live
  model, load the reviewer, for each chatter re-read their lines queued since their last review and read them as a
  person, apply and log both, then unload the reviewer, load the live model and release the worker. Chat keeps queueing
  (the lossless queue) and the live worker catches up after. A chatter is marked reviewed (their message count, the
  model, the time) only when both reads came back; the next press covers only chatters with messages after that.
  Scope: REVIEW_SCOPE_DEFAULT (owner, 2026-11-07: "b", quality first)."""
import asyncio
import collections
import json
import math
import os
import random
import time
import urllib.error

from . import llm, picker

# Owner decision 2026-11-07: the review covers the watchlist chatters with new content AND every other chatter with a
# toxic line not yet reviewed (the study's mode b, 114.8% of the 20b; mode a, the watchlist alone: 109.1%). The page's
# checkbox makes one press use "a". A one-line setting: change it here (or CW_REVIEW_SCOPE=a) to flip the default.
REVIEW_SCOPE_DEFAULT = os.environ.get("CW_REVIEW_SCOPE", "b") if os.environ.get("CW_REVIEW_SCOPE") in ("a", "b") else "b"
RING_LINES = 20_000          # chat lines kept per channel for a reviewed line's context (the 6 lines before it)
QUEUED_KEPT = 48             # a chatter's queued lines remembered for their review
QUEUED_CHATTERS_MAX = 200_000
CALL_TRIES = 3               # a reviewer or speed-check call that fails in a way worth retrying
CALL_WAITS = (2, 5, 10)
UNLOAD_WAIT_S = 30           # until /api/ps no longer lists the unloaded model
CHECK_BUDGET_S = 120         # the speed check's calls after the warm-up (more only to reach the minimums)
CHECK_MIN = {"person": 4, "w12": 2, "w1": 2}
CHECK_MAX = {"person": 10, "w12": 4, "w1": 3}
CHATRATE_S = (20, 120)       # the channel-rate probe's length, min and max (default 60)
LOAD_STEPS = 4              # the speed check's loads at a cap: each abort or failed load steps down 2 layers
FIT_WAIT_S = 10              # how long the speed check waits for Ollama's offload line in the log copy (polled every 2 s)
OTHER_SAMPLE_DELAY_S = 0.3   # after a call, before sampling the other work (the model server's threads go idle)
# A batch stops (and swaps back) once the live queue's oldest person read or flagged line has waited this long, so the
# live model stays current (H4's keep-up: p95 under 10 min; the swap back and the catch-up need the rest). The chatters
# left are not marked reviewed: the next press covers them. hybrid_effective.py simulates the same rule.
REVIEW_LIVE_LAG_S = 8 * 60
PREFETCH_EVERY_S = 20 * 60    # the hybrid's models are read again this often (kept in the page cache)
PREFETCH_READ = True          # read the reviewer's files ahead (False: only hold both models' files open)
PREFETCH_MB_S = 20            # its read from the share stays under this (line rate slowed the 12B ~95x, BASELINE 34)
PREFETCH_CHUNK = 4 * 1024 * 1024
PREFETCH_MEM_S = 0.01         # a chunk read faster than this came from memory: not paced
SWAP_KEY = "review_swap"      # the machine file's last batch swap, with its two models (BASELINE 34)


class Busy(RuntimeError):
    """The request cannot run now (the page shows the message)."""


class AIMixin:
    def ai_init(self, specs=None, host=None, ollama_log=None):
        self.specs = specs or picker.specs_from_env()
        self.host = host or self.specs.get("host") or "unknown"
        # a choice saved with a replaced setup id means its replacement (picker.resolve; owner 2026-11-09: the text-only
        # 31B for gemma4mtp:31b-n3): the file is migrated here and saved once at start (startup_choice)
        self.machine, self._machine_migrated = picker.migrate_machine(picker.load_machine(self.host, self.specs))
        nv = str(llm.NUM_GPU or "").strip()
        live = picker.resolve(llm.MODEL)     # CW_LLM_MODEL (gpu.env) may name a replaced build; apply_choice sets llm.MODEL
        # gpu.env's (or the .env's) cap; one set for a replaced build is carried (a floor: picker.cap_for)
        self._env_cap = ((live, int(nv), "env" if live == llm.MODEL else "carried", llm.MODEL)
                         if nv.isdigit() else None)
        self.ai = {"mode": "single", "live": live, "reviewer": None}
        self.ai_task = None             # the running picker task's progress, for the page
        self.ai_last = {}               # kind -> the last result (or error) of each task kind
        self._ai_job = None             # its asyncio task
        self.rate_task = None           # the chat count (no GPU: not an AI job)
        self._rate_job = None
        self._ai_cancel = False
        self.ollama_log = picker.OllamaLog(ollama_log if ollama_log is not None else os.environ.get("CW_OLLAMA_LOG"))
        self._ring = {}                 # channel -> deque[(seq, "display: text")]
        self._ring_seq = {}             # channel -> its last line number (gapless per channel)
        self._queued = {}               # (channel, user id) -> deque[(msg id, msg n, seq)]: their re-read lines
        self.prefetch = {}              # model -> its page-cache read (hybrid mode)
        self._pf_fds = {}               # model -> its blob files held open (keeps the share's cache valid)
        self._rv_cache = (0.0, None, None)
        self.llm_enabled = None         # set by serve._main: can this run reach an LLM at all (--llm-review)

    # ------------------------------------------------------------------ the choice
    @property
    def ai_off(self):
        return getattr(self, "ai", {}).get("mode") == "none"

    def ai_queue(self):
        """The LLM work queue when the AI is on (None: no LLM work is queued; the engine runs without it)."""
        return None if self.ai_off else self.review_q

    def session_ai(self):
        """What this run's AI is, for the session log (T5, sweep 2026-11-09: meta.json recorded CW_LLM_MODEL from the
        environment, not the page's choice): the mode (single / hybrid / none), the live model the worker calls
        (llm.MODEL), the reviewer, each one's GPU layer cap, and why the AI is off when a saved choice could not run."""
        a = dict(getattr(self, "ai", None) or {})
        mode = a.get("mode") or "single"
        live = llm.MODEL if mode != "none" else None
        rev = a.get("reviewer") if mode == "hybrid" else None
        out = {"mode": mode, "live": live, "reviewer": rev,
               "num_gpu": {t: llm.NUM_GPU_BY_MODEL.get(t) for t in (live, rev) if t}}
        if getattr(self, "ai_blocked", None):
            out["blocked"] = self.ai_blocked
        if getattr(self, "review_q", None) is None:
            out["llm_queue"] = False                 # started without --llm-review: no LLM work this run
        return out

    def apply_choice(self, choice, caps=None):
        """Make a choice live in this process: the model the worker calls, each model's cap, the engine's rule."""
        caps = caps if caps is not None else (self.machine.get("caps") or {})
        mode = choice.get("mode", "single")
        self.ai = {"mode": mode, "live": choice.get("live") or llm.MODEL,
                   "reviewer": choice.get("reviewer") if mode == "hybrid" else None}
        if mode != "none":
            if llm.MODEL != self.ai["live"] and isinstance(getattr(self, "_q_cost", None), dict):
                # QL1 (audit 2026-11-10): the queue's measured call costs were the old model's: the call shape and the
                # catch-up figure start again from this model's first calls ("measuring" until then)
                self._q_cost.clear()
            llm.MODEL = self.ai["live"]
        for t in (self.ai["live"], self.ai["reviewer"]):
            if not t or not hasattr(self, "specs"):
                continue
            cap = self._cap(t)
            # this machine's cap (found, set, gpu.env; "no cap" too: review fix 7), else the estimate or the tier's:
            # every chosen model gets its own, so gpu.env's cap never leaks to another model
            if cap["known"] and (mode != "none" or cap["how"] in ("found", "set", "lowered", "carried")):
                llm.NUM_GPU_BY_MODEL[t] = cap["value"]
        self.eng.person_review = mode != "none" and self.review_q is not None and bool(getattr(self, "_person_flag", True))
        self.tox_per_min = 0 if mode == "none" or self.review_q is None else (getattr(self, "_tox_flag", 1) or 0)
        self._rv_cache = (0.0, None, None)
        if mode == "hybrid":
            self.prefetch_kick()               # read both models into the page cache (a quick first swap)

    def startup_choice(self, cli_mode=None):
        """At start: this machine's saved choice, else today's setup from gpu.env (CW_LLM_MODEL / CW_LLM_NUM_GPU); a
        run with no LLM (--no-llm) is "none" for this run only."""
        c = dict(self.machine.get("choice") or {})
        if not c:
            c = {"mode": "single", "live": picker.resolve(llm.MODEL)}
            if llm.NUM_GPU:
                llm.NUM_GPU_BY_MODEL.setdefault(c["live"], int(llm.NUM_GPU))
        if cli_mode == "none" or self.review_q is None:
            c = {**c, "mode": "none"}
        self.ai_blocked = None
        if c.get("mode") != "none" and hasattr(self, "specs"):
            for t in filter(None, (c.get("live"), c.get("reviewer") if c.get("mode") == "hybrid" else None)):
                cap = self._cap(t)
                if not cap["known"]:
                    # third review 2: an MTP build whose cap is not known for this card never runs uncapped: this run
                    # starts with the AI off and the page says why; the saved choice is kept for after the check
                    self.ai_blocked = f"{t}: {cap['source']}"
                    c = {**c, "mode": "none"}
                    break
            if self.ai_blocked is None:
                why = self.ram_block(c)
                if why:
                    # ramblock (2026-11-08): a saved setup whose RAM part no longer fits does not start silently: this run
                    # starts with the AI off and the page says why and what fits; the saved choice is kept
                    self.ai_blocked = why
                    c = {**c, "mode": "none"}
        self.apply_choice(c)
        if getattr(self, "_machine_migrated", False) and hasattr(self, "specs"):
            # a replaced setup id in the saved file: the new id (and the carried cap) is written back once, so the .env
            # ./start.sh reads at the next start names the new build too
            self._machine_migrated = not self._save_machine()
        return c

    def ram_block(self, choice):
        """Why a choice must not start on this machine's RAM, or None (ramblock, 2026-11-08): a model whose system-RAM part
        (picker.ram_need_gb at its cap) plus headroom does not fit the free RAM read at start, or with no read the RAM
        class's nominal free RAM. For a hybrid each model is checked (they swap: one in memory at a time). The reason
        names the option the picker recommends here instead."""
        if not hasattr(self, "specs") or (choice or {}).get("mode") in (None, "none"):
            return None
        cls = picker.classes(self.specs, self.machine.get("override"))
        mem = self._mem()
        mv = picker._ram_view(cls, mem)
        read = mem.get("ram") is not None and not mem.get("ram_assumed")
        for t in filter(None, (choice.get("live"), choice.get("reviewer") if choice.get("mode") == "hybrid" else None)):
            cap = self._cap(t)
            r = picker.ram_check(picker.ram_need_gb(t, cap, mv), mv)
            if not r or r["state"] != "short":
                continue
            lab = (picker.setup(t) or {}).get("label", t)
            size = f"{mem['ram_total']:g}" if mem.get("ram_assumed") else cls["ram"]   # under 16 GB: the machine's own
            free = f"{r['avail_gb']:g} available" if read else f"{r['avail_gb']:g} assumed free with {size} GB of RAM"
            return (f"{lab} needs ~{math.ceil(r['need_gb'] + r['headroom_gb'])} GB free RAM, {free}: pick another "
                    f"(recommended: {self._ram_pick(cls, mem)}) or free memory")
        return None

    def _ram_pick(self, cls, mem):
        """The label of the option the picker recommends on this machine (what fits), or "No AI" when nothing fits."""
        vendor = self.specs.get("gpu_vendor")
        ids = {s["id"] for s in picker.catalogue()["setups"]}
        singles = picker.rows(cls, vendor, ids, None, self._mview(), mem)
        hybrids = picker.hybrid_rows(cls, vendor, ids, self._mview(), singles, mem)
        s, h = picker.shown_rows(singles, hybrids)
        rec = next((x for x in s if x["recommended"] and not (x.get("ram") or {}).get("state") == "short"), None)
        return rec["label"] if rec else "No AI"

    # ------------------------------------------------------------------ what the page reads
    def ai_status(self):
        """Small, for every status message: the choice, the running task, the review button's numbers."""
        out = {"mode": self.ai["mode"], "live": self.ai["live"], "reviewer": self.ai["reviewer"],
               "task": self.ai_task, "rate_task": self.rate_task,
               "last": {k: v for k, v in self.ai_last.items() if k in ("review", "select")},
               "scope_default": REVIEW_SCOPE_DEFAULT, "blocked": getattr(self, "ai_blocked", None)}
        if self.ai["mode"] == "hybrid" and self.review_q is not None:
            out["review"] = self.review_summary()
            out["prefetch"] = self.prefetch_summary()
        return out

    def _installed(self):
        """The cw-ollama tags. While Ollama is slow to answer (it lists its models from the shared store, which a 20 GB
        model load can saturate: seen on a 16 GB AMD card 2026-11-07, every row turned "not
        installed" right after a switch), the last list it gave stands; None only before any answer."""
        try:
            got = set(llm.installed_models(timeout=30))
        except Exception:
            return getattr(self, "_installed_last", None)
        self._installed_last = got
        return got

    def _version(self):
        try:
            return llm.version()
        except Exception:
            return None

    async def ai_info(self, override=None):
        """GET /api/ai: specs (read, and the page's override), the rows that fit, the hybrids, the choice."""
        from .serve import in_thread
        installed, version = await in_thread(lambda: (self._installed(), self._version()))
        ov = override if override is not None else (self.machine.get("override") or {})
        cls = picker.classes(self.specs, ov)
        vendor = self.specs.get("gpu_vendor")
        tr = picker.tier(cls, vendor)
        mem = picker.memory(self.specs)
        singles = picker.rows(cls, vendor, installed, version, self._mview(), mem) if tr["proxy"] else []
        hybrids = picker.hybrid_rows(cls, vendor, installed, self._mview(), singles, mem) if tr["proxy"] else []
        singles, hybrids = picker.shown_rows(singles, hybrids)      # MTP builds only, nothing dominated (owner 11-07)
        fit_note = picker.ram_fit_note(cls, vendor, self._mview(), mem) if not singles else None   # why only No AI
        cat = picker.catalogue()
        return {"specs": self.specs, "memory": mem, "classes": cls, "tier": tr, "ollama": {"version": version, "reachable": installed is not None,
                "installed": sorted(installed) if installed is not None else None,
                "mtp_min": ".".join(map(str, picker.MTP_MIN_VERSION))},
                "no_ai": cat["no_ai"], "rows": singles, "hybrids": hybrids, "choice": self.ai, "fit_note": fit_note,
                "swap": self._swap_estimate(), "chat_rate": self.ai_last.get("chatrate"), "task": self.ai_task,
                "last": self.ai_last, "choices": {"vram": picker.VRAM_CHOICES, "ram": picker.RAM_CHOICES},
                "llm_enabled": self.review_q is not None or bool(self.llm_enabled),
                "scope_default": REVIEW_SCOPE_DEFAULT,
                "hybrid_effective": cat.get("hybrid_effective")}

    def save_override(self, ov):
        clean = {k: v for k, v in (ov or {}).items()
                 if (k == "vram" and v in picker.VRAM_CHOICES) or (k == "ram" and v in picker.RAM_CHOICES)}
        self.machine["override"] = clean
        self._save_machine()
        return clean

    def _save_machine(self):
        try:
            # the .env carries only caps valid for the current tier (third review 1)
            c, tr, env_caps = self.machine.get("choice") or {}, self._tier(), {}
            for t in filter(None, (c.get("live"), c.get("reviewer"))):
                cap = self._cap(t, tr)
                # an estimate is worked out by the server at each start: ./start.sh does not warm it (review 4)
                env_caps[t] = (picker.ESTIMATE_CAP if cap.get("how") == "estimate" else cap["value"] if cap["known"]
                               else picker.UNKNOWN_CAP)
            self.machine = picker.save_machine(self.host, self.specs, self.machine, env_caps)
            return True
        except Exception as e:
            from .serve import _log_error
            _log_error(self.eng, "machine_save_errors", e, "saving this machine's AI choice")
            return False

    # ------------------------------------------------------------------ tasks: one at a time
    def llm_tools_busy(self):
        """Why the user card's LLM tools (a person read, a warning draft) cannot ask the model now, or None (review fix
        4): with the AI off nothing is loaded for them; during a swap, a speed check or a model switch a call would
        evict the model in memory (one at a time) and answer from the wrong one."""
        if self.ai_off and self.status.get("mode") == "live":     # (a replay keeps master's card tools: second review 3)
            return "the AI is off (No AI): pick a model in the AI panel to use the LLM tools"
        if self._ai_job is not None and not self._ai_job.done():
            return f"the LLM is busy: {(self.ai_task or {}).get('what', 'an AI task')}; try again when it is done"
        if self.llm_hold:
            return "the LLM is paused for a model swap; try again in a moment"
        return None

    def _refuse_unless_stopped(self, what):
        if self._ai_job is not None and not self._ai_job.done():
            raise Busy(f"wait: {self.ai_task.get('what', 'another AI task')} is running")
        if self.status.get("mode") == "live" and (self.status.get("running") or self.stopping):
            raise Busy(f"{what} only while stopped: press Stop first")
        if self.status.get("mode") not in ("live",) and not getattr(self, "allow_ai_in_replay", False):
            raise Busy(f"{what} is for live mode")

    def _run_task(self, kind, what, coro):
        self._ai_cancel = False
        self.ai_task = {"kind": kind, "what": what, "state": "starting", "started": int(time.time() * 1000)}
        self._ai_job = asyncio.ensure_future(self._task_wrap(kind, coro))
        return self.ai_task

    async def _task_wrap(self, kind, coro):
        res = None
        try:
            res = await coro
        except asyncio.CancelledError:
            res = {"error": "cancelled (the server is stopping)"}
            raise
        except Busy as e:
            res = {"error": str(e)}
        except Exception as e:
            from .serve import _log_error
            _log_error(self.eng, "ai_task_errors", e, f"the {kind} task")
            res = {"error": f"{type(e).__name__}: {e}"[:300]}
        finally:
            res = res or {}
            res.setdefault("at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            self.ai_last[kind] = res
            self.ai_task = None
            self._ai_cancel = False
        return res

    def cancel_ai_task(self):
        """Cancel at the next safe point (between calls); a review then swaps back as usual."""
        if self._ai_job is not None and not self._ai_job.done():
            self._ai_cancel = True
            if self.ai_task:
                self.ai_task["state"] = "cancelling: finishing the call in progress, then swapping back"
            return True
        return False

    async def ai_task_done(self):
        if self._ai_job is not None and not self._ai_job.done():
            try:
                await asyncio.shield(self._ai_job)
            except Exception:
                pass

    # ------------------------------------------------------------------ GPU steps
    async def _gpu(self, fn):
        from .serve import in_thread
        return await in_thread(fn)

    async def _unload(self, tag, step=None):
        """Unload and wait until Ollama no longer lists it (the VRAM is free for the next model)."""
        t0 = time.time()
        if step:
            self._step(step)
        try:
            await self._gpu(lambda: llm.unload(tag))
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
        end = time.time() + UNLOAD_WAIT_S
        while time.time() < end:
            if tag not in await self._gpu(llm.running_models):
                break
            await asyncio.sleep(0.3)
        return time.time() - t0

    async def _unload_all(self):
        for t in await self._gpu(llm.running_models):
            await self._unload(t)

    async def _load(self, tag, step=None):
        t0 = time.time()
        load_before = _loadavg()
        if step:
            self._step(step)
        last = None
        for i in range(CALL_TRIES):
            try:
                await self._gpu(lambda: llm.warm(tag))
                dt = time.time() - t0
                # each model's last load here, for the swap estimate; kept only when the machine was quiet before and
                # after it (owner 2026-11-07: a timing taken under load is never stored)
                if load_before is not None and load_before < quiet_threshold() and _running_now() < quiet_threshold():
                    self.machine.setdefault("loads", {})[tag] = {"s": round(dt, 1), "at": _now(),
                                                                  "load_before": load_before}
                return dt
            except Exception as e:
                last = e
                if not llm.transient(e) or self.shutting:
                    break
                await asyncio.sleep(CALL_WAITS[min(i, len(CALL_WAITS) - 1)])
        raise RuntimeError(f"could not load {tag}: {last}")

    def _step(self, s):
        if self.ai_task is not None:
            self.ai_task["state"] = s

    async def _call(self, fn):
        """One reviewer or check call; a failure worth retrying is tried again (CALL_TRIES) unless cancelled. Returns
        (result, seconds, retried failures): the failures as {"5xx": HTTP 5xx answers (a runner that aborted answers
        500), "other": timeouts, connection resets, cut-off replies}; a call that fails for good carries them as e.fails.
        PH2 (audit 2026-11-10): only the first kind can be a runner abort."""
        fails = {"5xx": 0, "other": 0}
        while True:
            t0 = time.time()
            try:
                return await self._gpu(fn), time.time() - t0, fails
            except Exception as e:
                tr = llm.transient(e)
                if tr:
                    fails["5xx" if isinstance(e, urllib.error.HTTPError) else "other"] += 1
                if not tr or sum(fails.values()) >= CALL_TRIES or self._ai_cancel or self.shutting:
                    e.fails = dict(fails)
                    raise
                await asyncio.sleep(CALL_WAITS[min(sum(fails.values()) - 1, len(CALL_WAITS) - 1)])

    # ------------------------------------------------------------------ select
    def start_select(self, body):
        self._refuse_unless_stopped("Changing the AI")
        mode = body.get("mode")
        if mode not in ("none", "single", "hybrid"):
            raise ValueError("mode must be none, single or hybrid")
        live, rev = body.get("live"), body.get("reviewer")
        for t in ((live,) if mode == "single" else (live, rev) if mode == "hybrid" else ()):
            if not isinstance(t, str) or not picker.TAG.match(t):
                raise ValueError(f"not a model tag: {t!r}")
        if mode == "hybrid" and live == rev:
            raise ValueError("a hybrid needs two different models")
        if mode != "none" and self.review_q is None:
            raise Busy("this run has no LLM work queue (started without --llm-review): start it with ./start.sh")
        what = {"none": "turning the AI off", "single": f"switching to {live}",
                "hybrid": f"switching to {live} live with {rev} reviewing"}[mode]
        return self._run_task("select", what, self._select({"mode": mode, "live": live, "reviewer": rev}))

    async def _select(self, c):
        installed = await self._gpu(self._installed)
        if c["mode"] != "none":
            if installed is None:
                raise Busy("the local LLM (cw-ollama) does not answer: start it with bin/ollama.sh start")
            for t in filter(None, (c["live"], c["reviewer"])):
                if t not in installed:
                    raise Busy(f"{t} is not installed: {picker.install_cmd(t)}")
            ver = picker.parse_version(await self._gpu(self._version))
            if any(picker.is_mtp(t) for t in filter(None, (c["live"], c["reviewer"]))) and ver and ver < picker.MTP_MIN_VERSION:
                raise Busy("MTP builds need Ollama 0.40.11 or later; this cw-ollama runs " + ".".join(map(str, ver)))
        saved_caps = dict(llm.NUM_GPU_BY_MODEL)
        saved_machine_caps = dict(self.machine.get("caps") or {})
        found = {}
        t0 = time.time()
        try:
            for t in filter(None, (c["reviewer"], c["live"])):       # the live model last: it stays loaded
                tier_key = self._tier().get("key")
                cap = self._cap(t)
                if not cap["known"]:
                    self._step(f"{t}: finding this card's GPU layer cap and speed (a speed check, 2-4 min)")
                    res = await self._speed_check(t, restore=False)
                    if res.get("error"):
                        raise Busy(f"{t}: {res['error']}")
                    found[t] = res
                    cap = self._cap(t)
                elif cap["how"] == "carried":
                    # W2 (sweep 2026-11-09): a carried cap stays carried, a floor under the new build's own figure: "Use
                    # this" in a session that ran on it must not freeze it as "set" below a cap measured or estimated
                    # later (a set cap is never raised). Stored with today's VRAM, so it is still lowered as a set cap is.
                    old = (self.machine.get("caps") or {}).get(t)
                    e = picker.cap_entry(cap["value"], tier_key, "carried",
                                         (old.get("avail_gb") if isinstance(old, dict) else None) or self._mem()["vram"])
                    e["from"] = cap.get("from")
                    self.machine.setdefault("caps", {})[t] = e
                elif cap["how"] in ("estimate", "study", "none", "env") and not (cap["value"] is None and picker.is_mtp(t)):
                    # "no cap" is stored too, so a start never guesses (review fix 7), with its tier; an MTP build's
                    # "none" is never stored (second review 1: only a fit read here may say it fits whole)
                    self.machine.setdefault("caps", {})[t] = picker.cap_entry(cap["value"], tier_key, "set",
                                                                             self._mem()["vram"])
                llm.NUM_GPU_BY_MODEL[t] = cap["value"]
            if c["mode"] == "none":
                for t in await self._gpu(llm.running_models):
                    await self._unload(t, f"unloading {t}")
            else:
                running = await self._gpu(llm.running_models)
                for t in running:
                    if t != c["live"]:
                        await self._unload(t, f"unloading {t}")
                await self._load(c["live"], f"loading {c['live']} (about 10-60 s; minutes the first time from disk)")
        except BaseException:
            # review fix 9: a failed switch leaves the old choice in force: its caps and its live model, nothing else
            llm.NUM_GPU_BY_MODEL.clear()
            llm.NUM_GPU_BY_MODEL.update(saved_caps)
            self.machine["caps"] = saved_machine_caps
            if not self.shutting:
                try:
                    await self._restore_live()
                except Exception:
                    pass
            raise
        self.apply_choice(c)
        self.machine["choice"] = {k: c[k] for k in ("mode", "live", "reviewer") if c.get(k)}
        self.ai_blocked = None
        saved = self._save_machine()
        return {"ok": True, "choice": self.ai, "seconds": round(time.time() - t0, 1), "saved": saved,
                "checked": list(found)}

    def _mem(self):
        """What ./start.sh read at start: the VRAM and RAM available for models (GiB), and the RAM in all."""
        return picker.memory(self.specs)

    def _cap(self, tag, tr=None):
        return picker.cap_for(tag, tr or self._tier(), self.specs.get("gpu_vendor"), self._mview(), self._mem())

    def _mview(self):
        """The machine file as the caps see it: gpu.env's CW_LLM_NUM_GPU counts as this machine's cap for the first-run
        live model until a choice is saved (review 2026-11-08: the estimate never raises a cap this machine has)."""
        m, ec = self.machine, getattr(self, "_env_cap", None)
        if not ec or m.get("choice"):
            return m
        key = self._tier().get("key")
        if picker.stored_cap(m, ec[0], key)[0]:
            return m
        e = picker.cap_entry(ec[1], key, ec[2])
        if ec[2] == "carried":
            e["from"] = ec[3]
        return {**m, "caps": {**(m.get("caps") or {}), ec[0]: e}}

    def _tier(self):
        return picker.tier(picker.classes(self.specs, self.machine.get("override")), self.specs.get("gpu_vendor"))

    # ------------------------------------------------------------------ speed check
    def start_speed(self, body):
        self._refuse_unless_stopped("The speed check")
        tag = body.get("model")
        if not isinstance(tag, str) or not picker.TAG.match(tag):
            raise ValueError("model must be a model tag")
        return self._run_task("speed", f"checking the speed of {tag}", self._speed_check(tag))

    def _no_gpu(self):
        """True when this machine has no GPU, read from the hardware (PH-R2-1, audit 2026-11-10 round 2): a real card
        under 7 GB takes a CPU tier (F16/F32) but still loads layers on its GPU, so its MTP check must read Ollama's fit
        and keep a cap. Only when the start script read no GPU at all (an old start script) does the tier decide."""
        vendor, mb = self.specs.get("gpu_vendor"), self.specs.get("vram_mb")
        if vendor in ("none", "cpu") or mb == 0:
            return True
        if vendor is None and mb is None:
            return self._tier().get("key") in ("F16", "F32")
        return False

    async def _speed_check(self, tag, restore=True):
        """See the module doc. restore: load the chosen live model back afterwards (not when a select runs it; the select
        restores on its own failure)."""
        installed = await self._gpu(self._installed)
        if installed is None:
            raise Busy("the local LLM (cw-ollama) does not answer: start it with bin/ollama.sh start")
        if tag not in installed:
            raise Busy(f"{tag} is not installed: {picker.install_cmd(tag)}")
        self.hold_llm("check")
        version = await self._gpu(self._version)
        res = {"model": tag, "ollama": version}
        had_cap, saved_cap = tag in llm.NUM_GPU_BY_MODEL, llm.NUM_GPU_BY_MODEL.get(tag)
        load_before = _loadavg()        # before anything loads: a model's own load must not count as "busy"
        ok = False
        try:
            await self.llm_idle()
            self._step("unloading the models in memory")
            await self._unload_all()
            cap = self._cap(tag)
            mtp = picker.is_mtp(tag)
            notes = []
            # a bigger card than the tested one (e.g. the 16 GB AMD desktop on the 12 GB tier's cap 26: 31B + MTP ran at
            # ~10 msgs/min there, 2026-11-07): the tier's cap is only a safe default, so the check finds this card's own
            # fit, as for an unknown cap, and keeps the faster setting
            # (vramcap, 2026-11-07: any cap not found here, the free-VRAM estimate included, is checked against the fit)
            # PH7 (audit 2026-11-10): on a machine with no GPU there is no GPU fit to read and no GPU cap to keep
            no_gpu = self._no_gpu()
            rsv_used = 0.0
            own_fit = mtp and not no_gpu and cap.get("how") != "found"
            mem = self._mem()
            est = picker.estimate_cap(tag, mem["vram"], mem["vram_cold"]) if mtp else None
            if est:                                      # beside the fit, to calibrate the estimate
                res["estimate"] = {"num_gpu": est["value"], "avail_gb": mem["vram_shown"], "how": est["how"],
                                   "cold": mem["vram_cold"]}
            if own_fit and not self.ollama_log.available():
                own_fit = False                          # no log to read the fit: keep the safe tier cap
                notes.append(f"Ollama's log is not readable here: kept the cap {cap['value']} ({cap['source']})")
            if (not cap["known"] or own_fit) and not no_gpu:
                # review fix 2: an MTP build's cap comes only from Ollama's own fit, read from its log; without the log
                # the check refuses (an uncapped MTP build can abort the runner) and never stores "no cap"
                if not self.ollama_log.available():
                    raise Busy(f"{tag} needs a GPU layer cap on this card and Ollama's log is not readable here "
                               "(a run with --no-llm copies no log): start ./start.sh without --no-llm, then check again")
                off = self.ollama_log.size()             # before the load: its offload line and any abort follow it
                llm.NUM_GPU_BY_MODEL[tag] = None
                # Ollama's own fit leaves out the MTP drafter ("failed to measure the memory of the extra model, fitting
                # without it"), so on some cards the uncapped load runs out of GPU memory (a 16 GB AMD card, text 31B,
                # 2026-11-10: 47/61 layers, then 895 MiB more failed; HTTP 500). Its offload line is logged before the
                # failure: the fit is read either way, and the capped loads below step down until one holds.
                fit_load_err = None
                try:
                    await self._load(tag, f"loading {tag} without a cap to read Ollama's own fit")
                except RuntimeError as e:
                    if not mtp:
                        raise
                    fit_load_err = e
                fit = None
                for _ in range(FIT_WAIT_S * 2):           # the copier polls every 2 s
                    fit = picker.OllamaLog.fit(self.ollama_log.since(off))
                    if fit:
                        break
                    await asyncio.sleep(0.5)
                res["fit"] = list(fit) if fit else None
                if fit is None:
                    if fit_load_err is not None:
                        raise fit_load_err
                    raise Busy(f"no offload line for {tag} in Ollama's log within {FIT_WAIT_S} s: the cap is not known; "
                               "check again (the log copier may be stalled)")
                value = picker.cap_from_fit(fit)
                if fit_load_err is not None:
                    notes.append(f"Ollama's own fit ({fit[0]}/{fit[1]} layers) ran out of GPU memory: it leaves out the MTP "
                                 "drafter; started from the cap below it")
                    if value is None:                     # it "fit whole" yet failed: start one margin below every layer
                        value = max(1, fit[1] - picker.CAP_MARGIN)
                # a cap that fills the card (not a whole model) on a card that also drives a screen: the desktop keeps
                # picker.DESKTOP_RESERVE_GB of it (owner 2026-11-10), as layers of this model
                rsv = self._mem().get("reserve_gb") or 0.0
                if rsv and value is not None:
                    g, _ = picker.gb_per_layer(tag)
                    if not g:
                        sizes = [sz for _, sz in picker.blob_files(tag)]
                        g = max(sizes) / 2 ** 30 / fit[1] if sizes and fit[1] else None
                    if g:
                        k = math.ceil(rsv / g - 1e-9)
                        notes.append(f"{k} layers fewer than Ollama's fit allows, so {rsv:g} GB of the GPU stays free "
                                     "for the desktop")
                        value, rsv_used = max(1, value - k), rsv
                if res.get("estimate"):               # to calibrate the free-VRAM estimate against Ollama's own fit
                    e = res["estimate"]
                    notes.append(f"Ollama's fit {fit[0]}/{fit[1]} layers, cap {value if value is not None else fit[1]}; "
                                 f"the estimate from {e['avail_gb']} GB available: "
                                 f"{e['num_gpu'] if e['num_gpu'] is not None else 'fits whole'}")
                if value is None:                         # it fit whole: the cap is every layer (a number, not "none")
                    value = fit[1]
                await self._unload(tag)
            else:
                value = None if no_gpu else cap["value"]
            run = None
            for attempt in range(LOAD_STEPS):
                llm.NUM_GPU_BY_MODEL[tag] = value
                res["num_gpu"] = value
                off = self.ollama_log.size()             # before the load: an abort while loading counts too
                try:
                    await self._load(tag, f"loading {tag}" + (f" with {value} GPU layers" if value is not None else ""))
                except RuntimeError:
                    # an MTP build that does not load at this cap (out of GPU memory): two layers fewer, as for an abort
                    if not mtp or no_gpu or value is None or value <= 1 or attempt == LOAD_STEPS - 1:
                        raise
                    notes.append(f"could not load at {value} GPU layers (out of GPU memory): stepping down to {max(1, value - 2)}")
                    value = max(1, value - 2)
                    continue
                run = await self._sample(tag, load_before, off)
                aborts = run.pop("aborts") + (picker.OllamaLog.aborts(self.ollama_log.since(off))
                                              if self.ollama_log.available() else 0)
                res.pop("empty", None)
                res.update(run)
                res["aborts"] = aborts
                if not aborts or not mtp or no_gpu:
                    break
                await self._unload(tag)
                top = value if value is not None else ((picker.setup(tag) or {}).get("vram_fit") or {}).get("layers") or 40
                value = max(1, top - 2)
                notes.append(f"{aborts} runner abort(s) at {res['num_gpu'] if res['num_gpu'] is not None else 'no cap'}: "
                             f"stepping down to {value}")
            if res.get("aborts") and mtp:
                raise Busy(f"{tag} still aborted the runner " + ("on the CPU" if no_gpu else
                           f"at {res['num_gpu']} GPU layers") + ": not stable on this machine")
            # PH1 (audit 2026-11-10): a check that measured nothing saves nothing, and says why
            if "keepup" not in run or run.get("n_person", 0) < CHECK_MIN["person"]:
                names = {"person": "person reads", "w12": "12-line windows", "w1": "1-line windows"}
                empty = [names[k] for k in run.get("empty") or () if k in names]
                raise Busy("too few calls succeeded to measure the speed"
                           + (f" (no {', '.join(empty)} came back)" if empty else "") + "; nothing was saved")
            res["notes"] = notes
            res["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            # owner 2026-11-07: "the point of the check speed feature is to see what my machine is capable of doing right
            # now ... the estimate value is just a placeholder until the user wants to do the check speed on the actual
            # system". So the check always replaces the estimate with what it measured; when other work was running it
            # says so beside the number (the quiet-machine rule is for the study's own figures, not the user's check)
            res["busy"] = not res.get("quiet")
            self.machine.setdefault("speed", {})[tag] = {k: res[k] for k in (
                "at", "person_s", "person_median_s", "a", "b", "window12_s", "keepup", "keepup_raw", "keepup_plus",
                "reread_all", "reread_most", "n_person", "n_w1", "n_w12", "num_gpu", "aborts", "retries", "calls", "ollama",
                "load", "busy", "fit", "estimate") if k in res}
            if res["busy"]:
                ld = res.get("load") or {}
                notes.append(f"measured while other work was running (load {ld.get('before')} before the check, "
                             f"other running threads up to {ld.get('other_max')}): the number reflects that")
            if res.get("retries"):
                notes.append(f"{res['retries']} call(s) retried after a timeout or a cut-off reply (not runner aborts)")
            if mtp and value is not None and not no_gpu:
                self.machine.setdefault("caps", {})[tag] = picker.cap_entry(value, self._tier().get("key"), "found",
                                                                            self._mem()["vram"], rsv_used)
            self._save_machine()
            ok = True
            return res
        finally:
            try:
                if not ok:                                 # review fix 9: a failed check leaves the old cap in force
                    if had_cap:
                        llm.NUM_GPU_BY_MODEL[tag] = saved_cap
                    else:
                        llm.NUM_GPU_BY_MODEL.pop(tag, None)
                if (restore or not ok) and not self.shutting:
                    await self._restore_live()
            except Exception as e:
                res["restore_error"] = f"{type(e).__name__}: {e}"[:200]
            finally:
                self.release_llm("check")

    async def _restore_live(self):
        """After a check or switch: the chosen live model loaded (with its cap), nothing else; with No AI, nothing."""
        live = self.ai["live"] if self.ai["mode"] != "none" else None
        for t in await self._gpu(llm.running_models):
            if t != live:
                await self._unload(t, f"unloading {t}")
        if live and live not in await self._gpu(llm.running_models):
            await self._load(live, f"loading {live} back")

    async def _sample(self, tag, load_before=None, log_off=None):
        """The check's calls, in blocks as speed2's runs made them (person reads, then 12-line windows, then 1-line
        windows; consecutive calls of one kind share Ollama's prompt cache, as live runs of windows do): one warm-up
        answer (not counted), then each block until its share of the budget (its minimum always). PH2 (audit
        2026-11-10): a runner abort is what Ollama's log says (picker.ABORT lines after log_off, counted by the caller),
        or, with no log to read, a retried HTTP 5xx; any other retried failure (a timeout, a reset, a cut-off reply) is
        a retry, counted apart, never an abort. The machine's load average is recorded with the result: a busy CPU slows
        a model whose layers partly run there. Without a summary ("keepup") the result says which blocks are empty."""
        persons, windows = picker.speed_sample()
        t = {"person": [], "w12": [], "w1": []}
        aborts = retries = calls = errors = 0
        http_aborts = not self.ollama_log.available()
        ab = lambda f: (f or {}).get("5xx", 0) if http_aborts else 0          # noqa: E731
        log_ab = lambda: (picker.OllamaLog.aborts(self.ollama_log.since(log_off))     # noqa: E731
                          if log_off is not None and not http_aborts else 0)
        load0, others = (_loadavg() if load_before is None else load_before), []
        self._step(f"{tag}: first answer (not counted)")
        try:
            await self._call(lambda: llm.review(persons[0][0], persons[0][1], model=tag))
        except Exception as e:
            f = getattr(e, "fails", None) or {}
            if not (ab(f) or log_ab()):
                raise
            return {"aborts": ab(f), "retries": f.get("other", 0), "calls": 1, "n_person": 0,
                    "empty": list(t)}                                     # the runner aborted: a lower cap next
        t0 = time.time()
        share = {"person": 0.5, "w12": 0.35, "w1": 0.15}
        for kind in ("person", "w12", "w1"):
            tk = time.time()
            for j in range(1, CHECK_MAX[kind] + 1):
                if self._ai_cancel:
                    raise Busy("cancelled")
                if len(t[kind]) >= CHECK_MIN[kind] and time.time() - tk > CHECK_BUDGET_S * share[kind]:
                    break
                if kind == "person":
                    p, room = persons[j % len(persons)]
                    fn = lambda p=p, room=room: llm.review(p, room, model=tag)
                else:
                    ctx, lines = windows[j % len(windows)]
                    fn = lambda ctx=ctx, lines=lines, k=kind: llm.rate_toxicity_window(
                        ctx, lines if k == "w12" else lines[:1], model=tag)
                what = {"person": "person reads", "w12": "12-line windows", "w1": "1-line windows"}[kind]
                self._step(f"{tag}: measuring {what} ({len(t[kind]) + 1}; {round(time.time() - t0)} s)")
                calls += 1
                try:
                    _, s, f = await self._call(fn)
                    aborts += ab(f)
                    retries += f.get("other", 0)
                    t[kind].append(s)
                    await asyncio.sleep(OTHER_SAMPLE_DELAY_S)     # the model idle: what still runs is other work
                    others.append(_running_now())
                except Exception as e:
                    f = getattr(e, "fails", None) or {}
                    aborts += ab(f)
                    retries += f.get("other", 0)
                    errors += 1
                    if aborts or log_ab():         # a runner abort: no point measuring at this cap
                        return {"aborts": aborts, "retries": retries, "calls": calls, "n_person": len(t["person"]),
                                "empty": [k for k in t if not t[k]]}
                    if errors >= 4:
                        raise Busy(f"the model keeps failing ({e})")
        if not t["person"] or not t["w12"] or not t["w1"]:
            return {"aborts": aborts, "retries": retries, "calls": calls, "n_person": len(t["person"]),
                    "empty": [k for k in t if not t[k]]}
        thr = quiet_threshold()
        others.sort()
        load = {"before": load0, "after": _loadavg(), "threshold": thr, "other_median": others[len(others) // 2] if others else None,
                "other_max": others[-1] if others else None}
        quiet = (load0 is not None and load0 < thr and bool(others) and load["other_median"] < thr
                 and load["other_max"] < 2 * thr)
        return {**picker.summarize(t["person"], t["w1"], t["w12"]), "aborts": aborts, "retries": retries, "calls": calls,
                "check_s": round(time.time() - t0, 1), "load": load, "quiet": quiet}

    # ------------------------------------------------------------------ the channel's chat rate
    # Review fix 13: the count uses no GPU, so it is not an AI job: it does not block Start, a switch or a check, and it
    # does not make llm_busy true. One count at a time; the page reads its progress as `rate_task`.
    def start_chatrate(self, body):
        if self._rate_job is not None and not self._rate_job.done():
            raise Busy("a chat count is already running")
        ch = str(body.get("channel", "")).strip().lower().lstrip("#")
        import re
        if not re.match(r"^[a-z0-9_]{3,25}$", ch):
            raise ValueError("not a channel name")
        secs = body.get("seconds", 60)
        secs = int(min(max(secs if isinstance(secs, (int, float)) else 60, CHATRATE_S[0]), CHATRATE_S[1]))
        self.rate_task = {"kind": "chatrate", "what": f"counting #{ch}'s chat for {secs} s", "state": "starting",
                          "started": int(time.time() * 1000)}
        self._rate_job = asyncio.ensure_future(self._rate_wrap(ch, secs))
        return self.rate_task

    async def _rate_wrap(self, ch, secs):
        res = None
        try:
            res = await self._chat_rate(ch, secs)
        except asyncio.CancelledError:
            res = {"error": "cancelled (the server is stopping)"}
            raise
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}"[:300], "channel": ch}
        finally:
            res = res or {}
            res.setdefault("at", _now())
            self.ai_last["chatrate"] = res
            self.rate_task = None
        return res

    async def _chat_rate(self, ch, secs):
        """A short anonymous read-only join that only counts chat lines (nothing is kept or shown)."""
        import websockets
        n = 0
        t0 = time.time()
        async with websockets.connect("wss://irc-ws.chat.twitch.tv:443", ping_interval=None) as ws:
            await ws.send("PASS SCHMOOPIIE")
            await ws.send(f"NICK justinfan{random.randint(10000, 99999)}")
            await ws.send(f"JOIN #{ch}")
            t0 = time.time()
            needle = f" PRIVMSG #{ch} :"
            while time.time() - t0 < secs and not self.shutting:
                if self.rate_task is not None:
                    self.rate_task["state"] = f"#{ch}: {n} messages in {round(time.time() - t0)} of {secs} s"
                try:
                    data = await asyncio.wait_for(ws.recv(), 1.0)
                except asyncio.TimeoutError:
                    continue
                for line in data.split("\r\n"):
                    if line.startswith("PING"):
                        await ws.send("PONG :tmi.twitch.tv")
                    elif needle in line:
                        n += 1
        el = max(time.time() - t0, 1.0)
        return {"channel": ch, "msgs_per_min": round(n * 60 / el, 1), "messages": n, "seconds": round(el),
                "note": "" if n else "no messages: the channel may be offline"}

    # ------------------------------------------------------------------ the hybrid's models kept in the page cache
    # The store is on the project share (the owner's rule: everything in the project folder), mounted CIFS with
    # cache=strict and closetimeo=1. Measured here 2026-11-10 (BASELINE 34, data/study/readahead/RESULT.md):
    # - a blob's cached pages are dropped about a second after its LAST open handle closes: the 12B came back from an
    #   unload in 76-83 s when nothing else held its files and in about 3 s while they were held; the 31B loaded in 11 s
    #   right after a load that left it cached, about 160 s from the share;
    # - the old read-ahead (cat each file, then open them all) so kept nothing it read of the 31B (0 % cached after it,
    #   the load still about 160 s), and while it read the 31B at line rate every live 12B call took ~15 s (0.16 s
    #   otherwise): Ollama reads its small files over the same share connection.
    # So, in hybrid mode both models' blob files are opened and held FIRST (nothing read for the live model: its pages
    # come from Ollama's own load and now survive the swap); the reviewer's files are then read through the held
    # handles at most PREFETCH_MB_S from the share (a chunk at memory speed is not paced), paused while a swap, a check
    # or a switch loads a model; again every PREFETCH_EVERY_S. PREFETCH_READ = False keeps only the held files.
    # Read-only; nothing in cw-ollama.
    async def prefetch_loop(self):
        if getattr(self, "_prefetch_wake", None) is None:
            self._prefetch_wake = asyncio.Event()
        while True:
            try:
                self._prefetch_letgo()
                if self.ai.get("mode") == "hybrid" and not self.shutting:
                    for tag, read in ((self.ai.get("live"), False), (self.ai.get("reviewer"), PREFETCH_READ)):
                        if tag and self.ai.get("mode") == "hybrid":
                            await self._prefetch(tag, read=read)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                from .serve import _log_error
                _log_error(self.eng, "ai_task_errors", e, "keeping the hybrid's models in memory")
            self._prefetch_wake.clear()
            try:
                await asyncio.wait_for(self._prefetch_wake.wait(), PREFETCH_EVERY_S)
            except asyncio.TimeoutError:
                pass

    def prefetch_kick(self):
        ev = getattr(self, "_prefetch_wake", None)
        if ev is not None:
            ev.set()

    def _prefetch_letgo(self):
        """Close the held files of a model no longer in the hybrid (its pages are then the kernel's to drop)."""
        keep = {self.ai.get("reviewer"), self.ai.get("live")} if self.ai.get("mode") == "hybrid" else set()
        for t in [t for t in self._pf_fds if t not in keep]:
            for fd in self._pf_fds.pop(t):
                _close(fd)
            self.prefetch.pop(t, None)

    def _prefetch_paused(self):
        """A model is being loaded (a review swap, a speed check, a switch): the share is its."""
        return bool(getattr(self, "llm_hold", None)) or (self._ai_job is not None and not self._ai_job.done())

    async def _prefetch(self, tag, read=True):
        from .serve import in_thread
        files = picker.blob_files(tag)
        st = self.prefetch.setdefault(tag, {})
        if not files:
            st.update(state="unknown", why="its manifest was not found under models/ollama")
            return st
        if tag not in self._pf_fds:              # hold first: what is read (by us or by Ollama) then stays cached
            fds = []
            try:
                for p, _ in files:
                    fds.append(os.open(p, os.O_RDONLY))
            except OSError as e:
                for fd in fds:
                    _close(fd)
                st.update(state="error", why=f"could not open {os.path.basename(e.filename or '')}")
                return st
            self._pf_fds[tag] = fds
        if not read:      # the live model: Ollama's load fills its pages; a reviewer held but not read is not in memory
            st.update(state="ready" if tag == self.ai.get("live") else "held", total=0, done=0, at=_now())
            return st
        total = sum(s for _, s in files)
        # a re-read of a reviewer already in memory stays "ready" (no flicker on the page, no cold estimate) unless a
        # chunk turns out to come from the share (its pages were dropped): then it is "reading" again
        again = st.get("state") == "ready"
        st.update(total=total, done=0, started=_now(), **({} if again else {"state": "reading"}))
        t0 = time.time()
        share_b = share_s = 0.0
        for fd, (path, size) in zip(self._pf_fds[tag], files):
            off = 0
            while off < size:
                while self._prefetch_paused() and not self.shutting:
                    await asyncio.sleep(1.0)
                if self.shutting or self._pf_fds.get(tag) is None:
                    st.update(state="cold")
                    return st
                c0 = time.time()
                try:
                    n = await in_thread(lambda: len(os.pread(fd, PREFETCH_CHUNK, off)))
                except OSError:
                    st.update(state="error", why=f"could not read {os.path.basename(path)}")
                    return st
                if n <= 0:
                    break
                off += n
                st["done"] += n
                dt = time.time() - c0
                if dt > PREFETCH_MEM_S:          # from the share: keep under PREFETCH_MB_S
                    if st.get("state") == "ready":
                        st["state"] = "reading"
                    share_b, share_s = share_b + n, share_s + dt
                    if share_s >= 0.5:           # the share's own speed (pauses left out), for the cold swap estimate
                        st["share_mb_s"] = round(share_b / share_s / 1e6, 1)
                    await asyncio.sleep(max(0.0, n / (PREFETCH_MB_S * 1e6) - dt))
        st.update(state="ready", at=_now(), seconds=round(max(time.time() - t0, 0.001), 1))
        return st

    def prefetch_summary(self):
        """For the page: are both hybrid models in memory (a quick first swap)?"""
        tags = [t for t in (self.ai.get("reviewer"), self.ai.get("live")) if t]
        sts = [self.prefetch.get(t) or {} for t in tags]
        if sts and all(s.get("state") == "ready" for s in sts):
            return {"state": "ready", "at": min(s["at"] for s in sts)}
        reading = [s for s in sts if s.get("state") == "reading"]
        if reading:
            done = sum(s.get("done", 0) for s in sts)
            total = sum(s.get("total", 0) for s in sts) or 1
            return {"state": "reading", "done_gb": round(done / 1e9, 1), "total_gb": round(total / 1e9, 1)}
        return {"state": "cold" if sts else "none"}

    # ------------------------------------------------------------------ the hybrid: which chatters, what it costs
    def track_message(self, rec, queued_prio):
        """Every chat line goes into its channel's ring (a reviewed line's context), numbered per channel without gaps
        (review fix 1: one counter for all channels made a channel's seqs gappy and the context wrong); a line queued for
        a re-read is remembered for its chatter's review."""
        ch = rec["channel"]
        seq = self._ring_seq[ch] = self._ring_seq.get(ch, 0) + 1
        ring = self._ring.get(ch)
        if ring is None:
            ring = self._ring[ch] = collections.deque(maxlen=RING_LINES)
        ring.append((seq, f"{rec.get('display') or rec.get('login') or '-'}: {rec['text']}"))
        if queued_prio is None:
            return
        key = (ch, rec["user_id"])
        q = self._queued.get(key)
        if q is None:
            if len(self._queued) >= QUEUED_CHATTERS_MAX:
                for k in list(self._queued)[:QUEUED_CHATTERS_MAX // 10]:      # the oldest tenth
                    del self._queued[k]
            q = self._queued[key] = collections.deque(maxlen=QUEUED_KEPT)
        q.append((rec["id"], rec.get("n", 0), seq))

    def _context(self, ch, seq, k=6):
        """The k chat lines of the channel just before line `seq` (its own channel's numbering)."""
        ring = self._ring.get(ch)
        if not ring or seq <= ring[0][0]:
            return []
        i = seq - ring[0][0]                     # gapless per channel: index of line `seq` in the ring
        return [ring[j][1] for j in range(max(0, i - k), min(i, len(ring)))]

    def reviewed(self):
        d = self.eng.__dict__.get("hybrid_reviewed")
        if d is None:
            d = self.eng.__dict__["hybrid_reviewed"] = {}
        return d

    def review_candidates(self, scope):
        """Chatters with content the reviewer has not read (review fix 6):
        - scope "a": on the watchlist this run, with messages after their last review (or never reviewed);
        - scope "b": also any chatter with a toxic line this run after their last review.
        Each: {"ch", "uid", "n", "last_ts", "at", "lines": [(msg id, seq)], "listed"}: n, last_ts and at are the snapshot a
        review marks as read (review fix 5: what arrives later is the next press's)."""
        from . import engine as E
        eng, since = self.eng, self.since
        ts = self._display_ts()
        rv = self.reviewed()
        bots = getattr(eng, "owner_bots", None) or ()
        now = time.time()
        out = []
        for ch in list(self.status.get("channels") or ()):
            for u in eng.channel_users(ch):
                if not u.toxic_n and not u.alerted and not E._review_bad(u):
                    continue
                if u.uid in bots or (since is not None and u.last_seen < since):
                    continue
                st = rv.get((ch, u.uid))
                after_n = st.get("n", 0) if st else 0
                after_ts = st.get("last_ts", 0) if st else 0
                if u.n <= after_n:
                    continue
                cache = eng.__dict__.get("_notice_p") or {}
                p = (cache.get((ch, u.uid)) or (None, None))[1] if u.toxic_n else None
                listed = E._notice_reasons(u, p, ts, getattr(eng, "person_review", False), since=since) is not None
                # PH3 (audit 2026-11-10): a flagged line that left the 40 recent messages is still held as evidence
                # (FLAGS_PERSIST) and the live model still re-reads it: the review sees and re-reads it too
                held_msgs = list(u.msgs) + list(getattr(u, "evidence", None) or ())
                new_toxic = scope == "b" and any(
                    m.get("ts", 0) > after_ts and (since is None or m.get("ts", 0) >= since)
                    and E._level(m.get("toxicity")) != "none" for m in held_msgs)
                if not (listed or new_toxic):
                    continue
                lines = [(mid, seq) for mid, n, seq in self._queued.get((ch, u.uid), ())
                         if after_n < n <= u.n and eng.held_message(ch, u.uid, mid) is not None]
                out.append({"ch": ch, "uid": u.uid, "n": u.n, "last_ts": u.msgs[-1].get("ts", 0) if u.msgs else 0,
                            "at": now, "lines": lines, "listed": listed})
        return out

    def _reviewer_times(self):
        """(person s, a, b, label) for the reviewer on this machine: its own speed check, else the tested tier's."""
        rv = self.ai.get("reviewer")
        sp = picker.speed_entry(self.machine, rv)          # PH1: an entry without its numbers is skipped
        if sp:
            return sp["person_s"], sp["a"], sp["b"], f"measured here {sp['at'][:10]}"
        s = picker.setup(rv) or {}
        tr = self._tier()
        t = (s.get("tiers") or {}).get(tr.get("proxy") or "A") or (s.get("tiers") or {}).get("A")
        if t:
            return t["person_s"], t["a"], t["b"], f"tested on {picker.catalogue()['tiers'][tr.get('proxy') or 'A']['short']}"
        return 26.0, 0.0, 0.0, "a default of 26 s per chatter"

    def _swap_estimate(self):
        """Seconds of the two swaps of one batch, from this machine's newest measurements of these two models
        (BASELINE 34). A record without its models (before 2026-11-10: `swap`, `swap_cached`) does not count: the box
        showed 8 s, measured on the projector build, and the swap took 155 s.
        - Reviewer in memory (read through held files, so its load does not touch the share): the last batch here that
          started that way (SWAP_KEY with read_ahead); before there is one, the last loads here of both models plus 2 s
          of unloads (the safe side: a load may have come from the share), else the study's figure.
        - Otherwise the longest of: the last batch here, the last loads, and the reviewer's unread part at the share
          speed its read measured plus the live model's load. A swap measured with the reviewer cached never stands in
          for a cold one, and a cold one never for a cached one."""
        sw = picker.catalogue()["swap"]
        lo = sw["unload_s"][0] * 2 + sw["load_s"][0] + sw["live_back_s"][0] + (sw["first_answer_s"][0] - 7)
        hi = sw["unload_s"][1] * 2 + sw["load_s"][1] + sw["live_back_s"][1] + (sw["first_answer_s"][1] - 7)
        study = {"s": round((lo + hi) / 2), "range": [round(lo), round(hi)], "label": "tested on 12 GB Nvidia"}
        lv, rv = self.ai.get("live"), self.ai.get("reviewer")
        last = self.machine.get(SWAP_KEY) or {}
        last = last if last.get("swap_s") is not None and last.get("live") == lv and last.get("reviewer") == rv else {}
        loads = self.machine.get("loads") or {}
        both = None
        if (loads.get(lv) or {}).get("s") is not None and (loads.get(rv) or {}).get("s") is not None:
            both = (loads[lv]["s"] + loads[rv]["s"] + 2,
                    f"last loads here {min(str(loads[lv].get('at', '')), str(loads[rv].get('at', '')))[:10]}")
        pf = self.prefetch_summary() if hasattr(self, "prefetch") else {}
        batch = (last["swap_s"], f"last review here {str(last.get('at', ''))[:10]}") if last else None
        if pf.get("state") == "ready":
            pick = batch if last.get("read_ahead") else both
            return {"s": round(pick[0]), "label": pick[1], "cached": True} if pick else {**study, "cached": True}
        cands = [c for c in (batch, both) if c]
        rst = (getattr(self, "prefetch", None) or {}).get(rv) or {}
        if rst.get("share_mb_s") and rst.get("total"):
            left = max(rst["total"] - rst.get("done", 0), 0)
            cands.append((left / (rst["share_mb_s"] * 1e6) + ((loads.get(lv) or {}).get("s") or 0) + 2,
                          f"reviewer from disk at {rst['share_mb_s']:.0f} MB/s"))
        if not cands:
            return study
        s, label = max(cands, key=lambda c: c[0])
        return {"s": round(s), "label": label}

    def review_summary(self, force=False):
        """The button's numbers for both scopes, at most every 5 s: chatters, new lines, minutes, the live pause."""
        now = time.time()
        if not force and now - self._rv_cache[0] < 5 and self._rv_cache[1] is not None:
            return self._rv_cache[1]
        out = {"basis": "", "swap": {}}
        try:
            P, a, b, lab = self._reviewer_times()
            sw = self._swap_estimate()
            out.update(basis=lab, swap=sw)
            for scope in ("a", "b"):
                c = self.review_candidates(scope)
                work = sum(picker.review_seconds(P, a, b, len(x["lines"])) for x in c)
                out[scope] = {"chatters": len(c), "lines": sum(len(x["lines"]) for x in c),
                              "seconds": round(work + (sw["s"] if c else 0)), "work_s": round(work)}
        except Exception as e:
            from .serve import _log_error
            _log_error(self.eng, "ai_task_errors", e, "counting the chatters to review")
        self._rv_cache = (now, out, None)
        return out

    # ------------------------------------------------------------------ the hybrid review batch
    def start_review(self, body):
        if getattr(self, "ai_blocked", None):        # ramblock / an unknown cap: the AI is off for this run, and why
            raise Busy(f"AI off for this run: {self.ai_blocked}")
        if self.ai.get("mode") != "hybrid" or not self.ai.get("reviewer"):
            raise Busy("the review button is for a hybrid setup (a live model and a reviewer)")
        why = self.ram_block(self.ai)                # the same reason a start gives (ramblock)
        if why:
            raise Busy(why)
        rcap = self._cap(self.ai["reviewer"])
        if not rcap["known"]:                    # third review 2: never load the reviewer uncapped
            raise Busy(f"{self.ai['reviewer']}: {rcap['source']}")
        if self._ai_job is not None and not self._ai_job.done():
            raise Busy(f"wait: {self.ai_task.get('what')} is running")
        if self.stopping or self.finishing:
            raise Busy("the live queue is finishing a Stop: review after it has stopped and started again")
        live_run = self.status.get("mode") == "live" and self.status.get("running")
        replay_run = self.status.get("mode") == "replay"
        if not (live_run or replay_run) or self.review_q is None:
            raise Busy("start a session first: the review reads this session's chatters")
        behind = (self._llm_status() or {}).get("behind_hi_s") or 0
        if behind > REVIEW_LIVE_LAG_S:           # review fix 11: a batch now would stop after one chatter per swap
            raise Busy(f"the live AI is {round(behind / 60)} min behind (its oldest person read or flagged line): let it "
                       f"catch up below {REVIEW_LIVE_LAG_S // 60} min first, or a review now would stop after one chatter")
        scope = body.get("scope") or REVIEW_SCOPE_DEFAULT
        if scope not in ("a", "b"):
            raise ValueError("scope must be a (the watchlist) or b (also every chatter with a toxic line)")
        cands = self.review_candidates(scope)
        if not cands:
            raise Busy("nothing new to review")
        return self._run_task("review", f"reviewing {len(cands)} chatters with {self.ai['reviewer']}",
                              self._review_batch(scope, cands))

    def _review_stop(self, gen):
        return self._ai_cancel or self.shutting or gen != self.run_gen

    async def _review_batch(self, scope, cands):
        live, rev = self.ai["live"], self.ai["reviewer"]
        P, a, b, lab = self._reviewer_times()
        est = sum(picker.review_seconds(P, a, b, len(c["lines"])) for c in cands) + self._swap_estimate()["s"]
        task = self.ai_task
        task.update(total=len(cands), done=0, failed=0, est_s=round(est), scope=scope)
        tm = {}
        res = {"scope": scope, "chatters": len(cands), "reviewed": 0, "failed": 0, "lines": 0, "model": rev}
        res["models_read_ahead"] = self.prefetch_summary().get("state") == "ready"
        gen = self.run_gen
        t_hold = time.time()
        q0 = self._llm_status() or {}
        load0, others = _loadavg(), []
        self.hold_llm("review")
        unloaded_live = loaded_rev = False
        try:
            self._step("pausing the live AI after its call in progress")
            await self.llm_idle()
            tm["drain_s"] = round(time.time() - t_hold, 1)
            # review fix 3: a Stop, Cancel or shutdown before the swap needs no swap at all
            if self._review_stop(gen):
                res["cancelled"] = True
                return res
            unloaded_live = True
            tm["unload_live_s"] = round(await self._unload(live, f"unloading {live}"), 1)
            if self._review_stop(gen):
                res["cancelled"] = True
                return res
            tm["load_reviewer_s"] = round(await self._load(rev, f"loading {rev}"), 1)
            loaded_rev = True
            fails_in_row = 0
            t_rev = time.time()
            for i, c in enumerate(cands):
                if self._review_stop(gen):
                    res["cancelled"] = True
                    break
                behind = (self._llm_status() or {}).get("behind_hi_s") or 0
                if i and behind > REVIEW_LIVE_LAG_S:     # keep the live model current: the rest waits for the next press
                    res["stopped_for_live"] = {"left": len(cands) - i, "live_behind_s": behind}
                    break
                self._step(f"reviewing chatter {i + 1} of {len(cands)}")
                t0 = time.time()
                ok, why, n_lines = await self._review_chatter(c, rev, gen)
                await asyncio.sleep(OTHER_SAMPLE_DELAY_S)      # the reviewer idle: what still runs is other work
                others.append(_running_now())
                if i == 0:
                    tm["first_chatter_s"] = round(time.time() - t0, 1)
                if ok:
                    res["reviewed"] += 1
                    res["lines"] += n_lines             # the lines re-read (PH6: not +1 per chatter)
                    task["done"] += 1
                    fails_in_row = 0
                elif why == "cancelled":
                    res["cancelled"] = True
                    break
                else:
                    res["failed"] += 1
                    task["failed"] += 1
                    fails_in_row += 1
                    res["last_error"] = why
                    if fails_in_row >= 2:
                        res["error"] = f"stopped: the reviewer failed twice in a row ({why})"
                        break
                left = len(cands) - i - 1
                per = (time.time() - t_rev) / (i + 1)
                task["eta_s"] = round(left * per + self._swap_estimate()["s"] / 2)
            tm["review_s"] = round(time.time() - t_rev, 1)
        except Exception as e:
            res["error"] = f"{type(e).__name__}: {e}"[:300]
        finally:
            try:
                if not self.shutting and unloaded_live:
                    if loaded_rev or rev in (await self._gpu(llm.running_models)):
                        tm["unload_reviewer_s"] = round(await self._unload(rev, f"unloading {rev}"), 1)
                    tm["load_live_s"] = round(await self._load(live, f"loading {live} back"), 1)
            except Exception as e:
                res["swap_back_error"] = f"{type(e).__name__}: {e}"[:300]
            finally:
                self.release_llm("review")
            tm["outage_s"] = round(time.time() - t_hold, 1)
            res["timings"] = tm
            q1 = self._llm_status() or {}
            res["queue_before"] = {"reads": q0.get("reads"), "lines": q0.get("lines")}
            res["queue_after"] = {"reads": q1.get("reads"), "lines": q1.get("lines"), "behind_s": q1.get("behind_s")}
            res["estimate_s"] = round(est)
            res["basis"] = lab
            # review fix 8: a swap is measured only when the reviewer really loaded (else it would look short)
            thr = quiet_threshold()
            others.sort()
            res["load"] = {"before": load0, "threshold": thr, "other_median": others[len(others) // 2] if others else None}
            quiet = load0 is not None and load0 < thr and (not others or others[len(others) // 2] < thr)
            if not self.shutting and loaded_rev and "load_live_s" in tm:
                # the swap estimate's own record (BASELINE 34): every batch's, quiet or not (a busy machine's swap is
                # the longer, the safe side), with its two models so another build's never stands in
                sw = tm.get("unload_live_s", 0) + tm["load_reviewer_s"] + tm.get("unload_reviewer_s", 0) + tm["load_live_s"]
                self.machine[SWAP_KEY] = {"swap_s": round(sw), "at": _now(), "live": live, "reviewer": rev,
                                          "quiet": quiet, "read_ahead": res["models_read_ahead"], **tm}
                self._save_machine()
                asyncio.ensure_future(self._catchup_watch(res))
            self._log({"type": "hybrid_review", "ts": self.eng.clock, **{k: v for k, v in res.items()}})
            self._rv_cache = (0.0, None, None)
        return res

    async def _catchup_watch(self, res, limit_s=3600):
        """After a batch: how long until the live queue's oldest person read or flagged line is under 1 min old again
        (recorded on the result the page shows)."""
        t0 = time.time()
        while time.time() - t0 < limit_s:
            q = self._llm_status() or {}
            if (q.get("behind_hi_s") or 0) < 60:
                res["catchup_s"] = round(time.time() - t0, 1)
                return
            await asyncio.sleep(1.0)
        res["catchup_s"] = None

    async def _review_chatter(self, c, rev, gen):
        """(True, "", lines re-read) when both reads came back; (False, why, 0) when not, and then nothing marks them
        reviewed.
        What is marked read is the press-time snapshot (c["n"], c["last_ts"]): messages that arrived since are the next
        press's (review fix 5)."""
        eng = self.eng
        ch, uid = c["ch"], c["uid"]
        u = eng.users.get((ch, uid))
        if u is None:
            return False, "gone", 0
        em = eng.emote_names(ch)
        msgs = {str(mid): eng.held_message(ch, uid, mid) for mid, _ in c["lines"]}     # recent or evidence (PH3)
        msgs = {k: m for k, m in msgs.items() if m is not None}
        lines = [(mid, seq) for mid, seq in c["lines"] if str(mid) in msgs]
        covered = set()
        try:
            for k in range(0, len(lines), 12):
                if self._review_stop(gen):
                    return False, "cancelled", 0
                chunk = lines[k:k + 12]
                ctx = self._context(ch, chunk[0][1])
                texts = [f"{u.display}: {msgs[str(mid)]['text']}" for mid, _ in chunk]
                out, _, _ = await self._call(lambda: llm.rate_toxicity_window(ctx, texts, emotes=em, model=rev))
                if gen != self.run_gen:
                    return False, "cancelled", 0
                for (mid, _), r in zip(chunk, out):
                    if not r:
                        continue
                    upd = eng.apply_llm_toxicity(ch, uid, mid, r)
                    if upd:
                        upd["toxicity"]["reviewer"] = rev
                        covered.add(str(mid))
                        self._log({**upd, "ts": eng.clock, "by": "reviewer"})
                        self.queue.append(self._ui(upd))
                eng.stats["hybrid_lines_reread"] += len(chunk)
            if self._review_stop(gen):
                return False, "cancelled", 0
            p = eng.profile(ch, uid)
            ctx = list(eng.rooms[ch].context)[-20:]
            r, _, _ = await self._call(lambda: llm.review(p, ctx, emotes=em, model=rev))
        except Exception as e:
            eng.stats["hybrid_review_errors"] += 1
            return False, f"{type(e).__name__}: {e}"[:200], 0
        if gen != self.run_gen:
            return False, "cancelled", 0
        r = {**r, "model": rev, "by": "reviewer"}
        old = getattr(u, "review", None) or {}
        eng.apply_person_review(ch, uid, r, eng.clock)
        if u.review is not None:
            # review fix 12: the reviewer's read does not spend the live reads (REVIEW_MAX), so the live model still
            # reads the chatter on their next toxic message; it is counted apart
            u.review = {**u.review, "by": rev, "reads": old.get("reads", 0),
                        "reviewer_reads": old.get("reviewer_reads", 0) + 1}
        self._log({"type": "review", "kind": "person", "by": "reviewer", "channel": ch, "user_id": uid, "ts": eng.clock, **r})
        self.queue.append({"type": "person_read", "channel": ch, "user_id": uid, "ts": eng.clock,
                           "label": r.get("label"), "rationale": r.get("rationale", ""), "by": rev})
        self._drop_covered(ch, uid, covered, c.get("at"))
        self.reviewed()[(ch, uid)] = {"n": c["n"], "last_ts": c.get("last_ts", 0), "model": rev,
                                      "at": int(time.time() * 1000), "label": r.get("label"), "lines": len(lines)}
        eng.stats["hybrid_reviewed"] += 1
        return True, "", len(lines)

    def _drop_covered(self, ch, uid, ids, before=None):
        """The live queue's work the reviewer just did: those lines, and that chatter's person read if it was queued
        before the press-time snapshot (`before`, wall s; review fix 5: a read asked for since covers newer messages and
        stays)."""
        n = 0
        keep = [x for x in self.tox_pending if not (x[2].get("channel") == ch and x[2].get("user_id") == uid
                                                    and str(x[2].get("id")) in ids)]
        n += len(self.tox_pending) - len(keep)
        self.tox_pending[:] = keep
        q = self.review_q
        if q is not None:
            mine = [e for e in q.items if e[2] == "person" and e[3].get("channel") == ch and e[3].get("user_id") == uid]
            old = [e for e in mine if before is None or e[4] <= before]
            for e in old:
                q.items.remove(e)
            n += len(old)
            if len(old) == len(mine):
                self.person_queued.discard((ch, uid))
            else:                                 # a newer read still waits: keep it marked pending
                self.eng.mark_review_pending(ch, uid)
        self.eng.stats["hybrid_live_items_covered"] += n
        return n


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _loadavg():
    """The 1-minute load average (Linux), for the speed check's record; None elsewhere."""
    try:
        return float(open("/proc/loadavg").read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _close(fd):
    try:
        os.close(fd)
    except OSError:
        pass


def quiet_threshold():
    """The load under which a timing counts (owner 2026-11-07: a timing taken under load is never stored): a tenth of the
    logical CPUs, at least 1 (3.2 on this machine's 32 threads, the gate's 3 for studies; a model's own CPU threads run
    during its calls, so "during" is sampled while it is idle, see _running_now)."""
    return max(1.0, 0.1 * (os.cpu_count() or 8))


def _running_now():
    """Threads running on the machine right now (/proc/loadavg's "running", host-wide also inside a container), less
    this reader. Sampled just after a model call, while the model server idles, it is the other work's load."""
    try:
        return max(0, int(open("/proc/loadavg").read().split()[3].split("/")[0]) - 1)
    except (OSError, ValueError, IndexError):
        return 0
