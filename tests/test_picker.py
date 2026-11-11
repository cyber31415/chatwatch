"""The model picker, speed before Start and the hybrid review (docs/DASHBOARD.md "Choosing the AI", "Hybrid: the
reviewer"; owner 2026-11-04 and 2026-11-07). The LLM is mocked: a fake Ollama that holds one model at a time."""
import asyncio
import json
import math
import os
import time

import pytest

from chatwatch import aictl, llm, picker, serve
from chatwatch.engine import Engine
from chatwatch.serve import Hub, ReadQueue

from test_dashboard import _http
from test_scenarios import msg, room

PAGE = {"X-Chatwatch": "1"}
NV12 = {"host": "box", "gpu_vendor": "nvidia", "gpu_name": "RTX 4070 Ti", "vram_mb": 12282, "ram_mb": 96000,
        "cores": 16, "threads": 32}
AMD16 = {"host": "amdbox", "gpu_vendor": "amd", "gpu_name": "Navi 32", "vram_mb": 16368, "ram_mb": 96000,
         "cores": 8, "threads": 16}
LOW = {"toxicity": "low", "target": "chatter", "kind": "insult"}
BAD = {"label": "bad_faith", "rationale": "keeps insulting", "confidence": "high", "rules": ["R1"]}


class FakeOllama:
    """One model loaded at a time (OLLAMA_MAX_LOADED_MODELS=1); every call recorded in order."""

    def __init__(self, monkeypatch, installed=("gemma4mtp:31b-n3-text", "gemma4mtp:12b-n2", "gemma4:31b-it-qat",
                                                 "gpt-oss-safeguard:20b"), version="0.40.11", log=None, fit=None):
        self.loaded, self.calls, self.installed, self.ver = [], [], list(installed), version
        self.log, self.fit = log, fit or {}
        self.fail_warm = set()
        self.abort_until = {}            # model -> calls to fail with HTTP 500 (a runner abort) while its cap is above
        self.oom_above = {}              # model -> a load fails with HTTP 500 (out of GPU memory) while its cap is above
        self.up = True
        m = monkeypatch
        m.setattr(llm, "warm", self.warm)
        m.setattr(llm, "unload", self.unload)
        m.setattr(llm, "running_models", lambda timeout=10: self._up() or list(self.loaded))
        m.setattr(llm, "installed_models", lambda timeout=10: self._up() or list(self.installed))
        m.setattr(llm, "version", lambda timeout=5: self._up() or self.ver)
        m.setattr(llm, "review", self.review)
        m.setattr(llm, "rate_toxicity_window", self.window)
        m.setattr(llm, "reachable", lambda timeout=5: self.up)
        m.setattr(aictl, "CALL_WAITS", (0,))
        m.setattr(aictl, "UNLOAD_WAIT_S", 1)

    def _up(self):
        if not self.up:
            raise ConnectionRefusedError("cw-ollama is not answering")

    def warm(self, model, timeout=600):
        self._up()
        if model in self.fail_warm:
            raise ConnectionResetError("the runner died while loading")
        cap = llm.num_gpu(model)
        self.calls.append(("warm", model, cap))
        self.loaded = [model]
        if self.log and model in self.fit and cap is None:
            with open(self.log, "a") as f:
                f.write(f'msg="starting llama-server" -ngl\nload_tensors: offloaded {self.fit[model][0]}/{self.fit[model][1]} '
                        "layers to GPU\nload_tensors: offloaded 5/5 layers to GPU\n")
        lim = self.oom_above.get(model)
        if lim is not None and (cap is None or cap > lim):     # as Ollama: the offload line, then the failed load
            self.loaded = []
            import urllib.error
            raise urllib.error.HTTPError("http://x", 500, "Internal Server Error", {}, None)

    def unload(self, model, timeout=60):
        self._up()
        self.calls.append(("unload", model))
        self.loaded = [m for m in self.loaded if m != model]

    def _check(self, model, kind):
        self._up()
        if model not in self.loaded:
            raise AssertionError(f"{kind} call to {model} while {self.loaded} is loaded")
        self.calls.append((kind, model))
        lim = self.abort_until.get(model)
        if lim is not None and (llm.num_gpu(model) is None or llm.num_gpu(model) > lim):
            import urllib.error
            if self.log:                 # as Ollama logs a runner abort (the check reads it there: PH2)
                with open(self.log, "a") as f:
                    f.write("CUDA error: out of memory\nllama-server terminated\n")
            raise urllib.error.HTTPError("http://x", 500, "runner aborted", {}, None)

    def review(self, p, ctx, emotes=frozenset(), model=None, timeout=90):
        self._check(model or llm.MODEL, "person")
        time.sleep(0.002)
        return {**BAD, "model": model or llm.MODEL}

    def window(self, ctx, lines, emotes=frozenset(), model=None, timeout=180, **kw):
        self._check(model or llm.MODEL, "window")
        time.sleep(0.001 * (1 + len(lines)))
        return [LOW] * len(lines)


@pytest.fixture(autouse=True)
def _machines(tmp_path, monkeypatch):
    monkeypatch.setenv("CW_MACHINES_DIR", str(tmp_path / "machines"))
    monkeypatch.setattr(aictl, "_loadavg", lambda: 0.5)          # a quiet machine (timings are then stored)
    monkeypatch.setattr(aictl, "_running_now", lambda: 0)
    monkeypatch.setattr(aictl, "OTHER_SAMPLE_DELAY_S", 0)
    monkeypatch.setattr(llm, "NUM_GPU_BY_MODEL", {})
    monkeypatch.setattr(llm, "NUM_GPU", "")
    monkeypatch.setattr(llm, "MODEL", "gemma4mtp:31b-n3-text")


def _hub(specs=NV12, mode="live", choice=None, log=None):
    eng = Engine()
    eng.person_review = True
    t = room(eng)
    hub = Hub(eng)
    hub.log_sessions = False
    hub.status["mode"] = mode
    hub.review_q = ReadQueue()
    hub.tox_per_min = 1
    hub.ai_init(specs=specs, host=specs["host"], ollama_log=log)
    if choice:
        hub.machine["choice"] = choice
    hub.startup_choice()
    return eng, hub, t


async def _until(cond, s=5.0):
    end = time.time() + s
    while not cond() and time.time() < end:
        await asyncio.sleep(0.01)
    return cond()


# ------------------------------------------------------------------------------------------------ specs and rows
def test_specs_classes_and_tiers():
    s = picker.specs_from_env({"CW_SPEC_GPU_VENDOR": "amd", "CW_SPEC_VRAM_MB": "16368", "CW_SPEC_RAM_MB": "94000",
                               "CW_SPEC_CORES": "8", "CW_SPEC_GPU_NAME": "Navi 32", "CW_HOST": "amdbox"})
    c = picker.classes(s)
    assert (c["vram"], c["ram"]) == ("16+", "96+")
    tr = picker.tier(c, "amd")            # owner 2026-11-07: meets or exceeds the 12 GB tier -> its numbers, estimated
    assert tr["key"] == "A" and tr["measured"] and tr["estimate"] and "meets or exceeds" in tr["why"]
    nv = picker.tier(picker.classes(NV12), "nvidia")
    assert nv["key"] == "A" and nv["measured"] and not nv["estimate"]
    assert picker.tier({"vram": "none", "ram": "16"}, "none")["key"] == "F16"
    assert picker.tier({"vram": "8", "ram": "32"}, "nvidia")["key"] == "E"
    t16 = picker.tier({"vram": "16+", "ram": "16"}, "nvidia")       # ramfit: the GPU tier, each model gated by its RAM part
    assert t16["key"] == "A" and t16["ram_short"] and t16["estimate"] and not t16.get("below_tested")
    # owner 2026-11-09: 16 GB is the lowest RAM option; "<16" is gone
    assert picker.RAM_CHOICES[0] == "16" and "<16" not in picker.RAM_CHOICES
    t8 = picker.tier({"vram": "8", "ram": "16"}, "nvidia")
    assert t8["key"] == "E" and t8["ram_short"] and "below every tested" not in t8["why"]
    # the dropdown overrides what was read
    c = picker.classes(NV12, {"vram": "8", "ram": "32"})
    assert (c["vram"], c["ram"]) == ("8", "32") and c["overridden"]["vram"]
    assert picker.classes({"gpu_vendor": None, "vram_mb": None, "ram_mb": None})["vram"] is None   # unknown: ask
    # a machine read under 16 GB shows the lowest option; its free RAM read still gates each model
    assert picker.ram_class(8 * 1024) == "16" and picker.ram_class(15.5 * 1024) == "16"
    c = picker.classes({"gpu_vendor": "none", "vram_mb": 0, "ram_mb": 8 * 1024}, {"ram": "<16"})   # an old saved "<16"
    assert c["ram"] == "16" and not c["overridden"]["ram"]
    # with no free-RAM read, a machine under 16 GB is judged on its own total (three quarters free), not the class's 12
    m = picker.memory({"gpu_vendor": "none", "vram_mb": 0, "ram_mb": 7800})
    assert m["ram"] == 5.7 and m["ram_assumed"] and m["ram_shown"] is None and m["ram_total"] == 7.6
    e4 = next(r for r in picker.rows({"vram": "none", "ram": "16"}, "none", set(), "0.40.11", None, m)
              if r["id"] == "gemma4mtp:e4b-n4")
    assert e4["ram"]["state"] == "short" and e4["ram"]["assumed"] and not e4["recommended"]
    assert not picker.memory({"gpu_vendor": "none", "vram_mb": 0, "ram_mb": 15800})["ram_assumed"]   # 16 GB: as before
    t = picker.tier(picker.classes({"gpu_vendor": "none", "vram_mb": 0, "ram_mb": 7800}), "none")
    assert t["key"] == "F16" and t["estimate"] and "tested on this kind" not in t["why"] and "less RAM" in t["why"]


def test_rows_on_this_machine_recommend_the_mtp_31b_with_its_cap_and_tested_rate():
    rows = picker.rows(picker.classes(NV12), "nvidia", {"gemma4mtp:31b-n3-text", "gemma4mtp:12b-n2"}, "0.40.11")
    top = rows[0]
    assert top["id"] == "gemma4mtp:31b-n3-text" and top["recommended"] and top["installed"]
    assert top["num_gpu"] == {"value": 30, "known": True, "source": "study (12 GB Nvidia)", "how": "study"}
    assert top["speed"]["keepup"] == 40 and top["speed"]["this_hardware"] and top["speed"]["label"] == "tested on 12 GB Nvidia"
    assert not top["speed"]["estimate"]                    # the tested machine itself: "tested on", not "est."
    assert top["quality"]["over_no_ai"] == 66                # 117.3 / 70.8 - 1 (owner 2026-11-07: + % over No AI)
    assert next(r for r in rows if r["id"] == "gpt-oss-safeguard:20b")["quality"]["over_no_ai"] == 41
    assert 20 <= top["review_s"] <= 32                    # about 23 s per chatter (the text build; 25 for the projector build)
    twelve = next(r for r in rows if r["id"] == "gemma4mtp:12b-n2")
    assert twelve["num_gpu"]["value"] is None and twelve["num_gpu"]["known"] and twelve["speed"]["keepup_plus"]
    assert twelve["quality"]["over_no_ai"] == 50
    plain = next(r for r in rows if r["id"] == "gemma4:31b-it-qat")
    assert not plain["installed"] and plain["install"] == "bin/ollama.sh install gemma4:31b-it-qat"
    hy = picker.hybrid_rows(picker.classes(NV12), "nvidia", {"gemma4mtp:31b-n3-text", "gemma4mtp:12b-n2"})
    assert hy[0]["live"] == "gemma4mtp:12b-n2" and hy[0]["reviewer"] == "gemma4mtp:31b-n3-text" and hy[0]["installed"]
    assert hy[0]["quality"]["b_over_no_ai"] == 62 and hy[0]["quality"]["a_over_no_ai"] == 54


def test_a_bigger_or_other_vendor_card_gets_the_met_tiers_estimate_and_cap():
    rows = picker.rows(picker.classes(AMD16), "amd", {"gemma4mtp:31b-n3-text"}, "0.40.11")
    top = rows[0]
    assert top["id"] == "gemma4mtp:31b-n3-text" and not top["speed"]["measured"] and top["speed"]["estimate"]
    assert top["speed"]["keepup"] == 40 and top["speed"]["this_hardware"]
    assert top["num_gpu"] == {"value": 30, "known": True, "source": "study (12 GB Nvidia)", "how": "study"}   # the safe default
    # an Ollama without the MTP path blocks the MTP builds and says why
    rows = picker.rows(picker.classes(AMD16), "amd", {"gemma4mtp:31b-n3-text"}, "0.12.3")
    assert "0.40.11" in next(r for r in rows if r["mtp"])["blocked"]
    # a laptop without a GPU gets the CPU picks only
    ids = [r["id"] for r in picker.rows({"vram": "none", "ram": "16"}, "none", set(), "0.40.11")]
    assert ids[0] == "gemma4mtp:e4b-n4" and "gemma4mtp:31b-n3-text" not in ids
    # no GPU, a machine read with 8 GB (5 GB free): the lowest option's E4B + MTP is listed but not recommended (short)
    r8 = picker.rows({"vram": "none", "ram": "16"}, "none", set(), "0.40.11", None, {"vram": None, "ram": 5.0, "ram_total": 8.0})
    e4 = next(r for r in r8 if r["id"] == "gemma4mtp:e4b-n4")
    assert e4["ram"]["state"] == "short" and not e4["recommended"] and e4["ram_demoted"]


def test_ollama_log_fit_and_aborts(tmp_path):
    p = tmp_path / "o.log"
    p.write_text('load_tensors: offloaded 29/61 layers to GPU\nload_tensors: offloaded 5/5 layers to GPU\n')
    L = picker.OllamaLog(str(p))
    assert L.fit(L.since(0)) == (29, 61) and picker.cap_from_fit((29, 61)) == 26
    assert picker.cap_from_fit((49, 49)) is None and picker.cap_from_fit(None) is None
    off = L.size()
    with open(p, "a") as f:
        f.write("CUDA error: out of memory\nllama-server terminated\n")
    assert L.aborts(L.since(off)) == 2 and L.aborts(L.since(0)) == 2
    assert not picker.OllamaLog(str(tmp_path / "missing")).available()


def test_the_keepup_table_reproduces_the_studys_own_figures():
    """The speed check reads msgs/min from the grid; on the study's own measured service times it gives the figure
    the simulator gives them directly (within one display step)."""
    g = json.load(open(picker.GRID_FILE))
    for s, v in g["validation"].items():
        got = picker.step_down(picker.keepup_from(v["P"], v["W12"]))
        steps = list(picker.STEPS)
        own = min(v["own"], 150)
        assert abs(steps.index(min(got, 150)) - steps.index(own)) <= 1, (s, got, v["own"])


def test_summarize_is_speed2s_method():
    s = picker.summarize([6.0, 7.0, 8.0], [3.0, 2.9], [20.0, 19.0, 21.0])
    assert s["person_s"] == 7.0 and abs(s["b"] - (20.0 - 2.95) / 11) < 1e-3 and s["window12_s"] == 20.0
    assert s["keepup"] in picker.STEPS


def test_the_machine_file_and_its_env_sidecar(tmp_path):
    m = picker.save_machine("box", NV12, {"choice": {"mode": "hybrid", "live": "gemma4mtp:12b-n2",
                                                     "reviewer": "gemma4mtp:31b-n3-text"},
                                          "caps": {"gemma4mtp:31b-n3-text": 26, "gemma4mtp:12b-n2": None}})
    env = open(os.path.join(picker.machines_dir(), "box.env")).read()
    assert "CW_AI_MODE=hybrid" in env and "CW_LIVE_MODEL=gemma4mtp:12b-n2" in env and "CW_LLM_NUM_GPU=\n" in env
    assert "CW_REVIEW_NUM_GPU=26" in env and f"CW_MACHINE_GPU={picker.gpu_sig(NV12)}" in env
    assert picker.load_machine("box", NV12)["choice"]["live"] == "gemma4mtp:12b-n2"
    assert picker.load_machine("box", AMD16) == {}          # the same host name, another GPU: not this machine's
    with pytest.raises(ValueError):
        picker.save_machine("box", NV12, {"choice": {"mode": "single", "live": "x; rm -rf /"}})


def test_each_model_carries_its_own_cap_and_format(monkeypatch):
    got = []
    monkeypatch.setattr(llm, "_post", lambda body, timeout: got.append(body) or {"message": {"content": json.dumps(
        {"label": "mild", "rules": [], "confidence": "low", "rationale": "",
         "ratings": [{"n": 1, "toxicity": "none", "target": "", "kind": ""}]})}})
    monkeypatch.setattr(llm, "NUM_GPU", "26")               # gpu.env's cap for the live 31B
    monkeypatch.setattr(llm, "NUM_GPU_BY_MODEL", {"gemma4mtp:12b-n2": None, "gemma4mtp:31b-n3-text": 26})
    llm.rate_toxicity_window([], ["x: hi"], model="gemma4mtp:12b-n2")
    llm.rate_toxicity_window([], ["x: hi"], model="gemma4mtp:31b-n3-text")
    assert "num_gpu" not in got[0]["options"] and got[1]["options"]["num_gpu"] == 26
    llm.rate_toxicity_window([], ["x: hi"], model="gemma4mtp:e4b-n4")      # E4B re-reads think (its study config)
    assert got[2]["think"] is True and got[2]["options"]["num_predict"] == 6000
    llm.review({"display": "x", "recent_messages": [{"text": "hi"}], "messages": 1, "first_msg_seen": True,
                "account_age_days": 3.0, "sub_months": 0, "friction": {"instigated": 0, "joined_heated_room": 0}},
               [], model="gemma4mtp:e4b-n4")
    assert got[3]["think"] is False
    llm.rate_toxicity_window([], ["x: hi"], model="qwen3.6:35b-a3b-q4_K_M")
    assert got[4]["options"]["presence_penalty"] == 1.5 and got[4]["options"]["temperature"] == 0.7


# ------------------------------------------------------------------------------------------------ select
def test_select_only_while_stopped_swaps_the_model_and_saves_the_choice(monkeypatch):
    fo = FakeOllama(monkeypatch)
    fo.loaded = ["gemma4mtp:31b-n3-text"]
    eng, hub, t = _hub()

    async def go():
        hub.status["running"] = True
        with pytest.raises(aictl.Busy):
            hub.start_select({"mode": "single", "live": "gemma4mtp:12b-n2"})
        hub.status["running"] = False
        hub.start_select({"mode": "hybrid", "live": "gemma4mtp:12b-n2", "reviewer": "gemma4mtp:31b-n3-text"})
        await hub.ai_task_done()
    asyncio.run(go())
    assert hub.ai_last["select"].get("ok"), hub.ai_last
    assert fo.loaded == ["gemma4mtp:12b-n2"] and ("unload", "gemma4mtp:31b-n3-text") in fo.calls
    assert ("warm", "gemma4mtp:12b-n2", None) in fo.calls          # the 12B fits whole: no cap
    assert llm.MODEL == "gemma4mtp:12b-n2" and llm.num_gpu("gemma4mtp:31b-n3-text") == 30
    m = picker.load_machine("box", NV12)
    assert m["choice"] == {"mode": "hybrid", "live": "gemma4mtp:12b-n2", "reviewer": "gemma4mtp:31b-n3-text"}
    assert picker.cap_value(m["caps"]["gemma4mtp:31b-n3-text"]) == 30


def test_no_ai_turns_the_llm_work_off(monkeypatch):
    fo = FakeOllama(monkeypatch)
    fo.loaded = ["gemma4mtp:31b-n3-text"]
    eng, hub, t = _hub()

    async def go():
        hub.start_select({"mode": "none"})
        await hub.ai_task_done()
    asyncio.run(go())
    assert hub.ai_off and fo.loaded == [] and not eng.person_review and hub.tox_per_min == 0
    serve.handle(hub, msg("rude", "you are trash and everyone hates you", t + 1))
    assert hub.review_q.empty() and not hub.tox_pending and hub._next_job(time.time()) is None
    assert hub._status()["llm_queue"] is None


def test_a_missing_model_or_an_old_ollama_is_refused_with_the_command(monkeypatch):
    fo = FakeOllama(monkeypatch, installed=("gemma4mtp:31b-n3-text",), version="0.39.0")
    eng, hub, t = _hub()

    async def go():
        hub.start_select({"mode": "single", "live": "gemma4mtp:12b-n2"})
        await hub.ai_task_done()
        a = hub.ai_last["select"]["error"]
        hub.start_select({"mode": "single", "live": "gemma4mtp:31b-n3-text"})
        await hub.ai_task_done()
        return a, hub.ai_last["select"]["error"]
    a, b = asyncio.run(go())
    assert "bin/ollama.sh install gemma4mtp:12b-n2" in a and "0.40.11" in b


# ------------------------------------------------------------------------------------------------ speed check
def test_the_speed_check_measures_and_caches_per_model_and_machine(monkeypatch):
    fo = FakeOllama(monkeypatch)
    fo.loaded = ["gemma4mtp:31b-n3-text"]
    monkeypatch.setattr(aictl, "CHECK_BUDGET_S", 0)
    eng, hub, t = _hub()

    async def go():
        hub.start_speed({"model": "gemma4mtp:12b-n2"})
        await hub.ai_task_done()
    asyncio.run(go())
    r = hub.ai_last["speed"]
    assert not r.get("error"), r
    assert r["n_person"] >= aictl.CHECK_MIN["person"] and r["aborts"] == 0 and r["keepup"] in picker.STEPS
    assert fo.loaded == ["gemma4mtp:31b-n3-text"]                 # the live model is back
    sp = picker.load_machine("box", NV12)["speed"]["gemma4mtp:12b-n2"]
    assert sp["person_s"] == r["person_s"]
    row = next(x for x in picker.rows(picker.classes(NV12), "nvidia", {"gemma4mtp:12b-n2"}, "0.40.11", hub.machine)
               if x["id"] == "gemma4mtp:12b-n2")
    assert row["speed"]["measured"] and row["speed"]["label"].startswith("measured here ")


def test_the_check_finds_an_mtp_cap_from_ollamas_fit_and_steps_down_on_aborts(monkeypatch, tmp_path):
    log = str(tmp_path / "ollama.log")
    open(log, "w").close()
    fo = FakeOllama(monkeypatch, log=log, fit={"gemma4mtp:31b-n3-text": (34, 61)})
    fo.abort_until["gemma4mtp:31b-n3-text"] = 29                  # aborts above 29 layers on this card
    monkeypatch.setattr(aictl, "CHECK_BUDGET_S", 0)
    # a tier whose cap the study did not find (say a card class it never had): the check finds it
    monkeypatch.setitem(picker.setup("gemma4mtp:31b-n3-text")["tiers"]["A"], "num_gpu_known", False)
    eng, hub, t = _hub(specs=AMD16, log=log)
    hub.ai = {"mode": "single", "live": "gemma4mtp:31b-n3-text", "reviewer": None}

    async def go():
        hub.start_speed({"model": "gemma4mtp:31b-n3-text"})
        await hub.ai_task_done()
    asyncio.run(go())
    r = hub.ai_last["speed"]
    assert not r.get("error"), r
    assert r["fit"] == [34, 61] and r["num_gpu"] == 29 and r["aborts"] == 0       # 31 aborted -> 29
    assert any("stepping down to 29" in n for n in r["notes"])
    assert picker.cap_value(hub.machine["caps"]["gemma4mtp:31b-n3-text"]) == 29
    assert picker.cap_for("gemma4mtp:31b-n3-text", hub._tier(), "amd", hub.machine)["source"] == "found here"


# ------------------------------------------------------------------------------------------------ hybrid review
HYB = {"mode": "hybrid", "live": "gemma4mtp:12b-n2", "reviewer": "gemma4mtp:31b-n3-text"}


def _hybrid(monkeypatch, mode="live"):
    fo = FakeOllama(monkeypatch)
    fo.loaded = ["gemma4mtp:12b-n2"]
    eng, hub, t = _hub(mode=mode, choice=HYB)
    hub.machine.setdefault("caps", {})["gemma4mtp:31b-n3-text"] = 26
    llm.NUM_GPU_BY_MODEL.update({"gemma4mtp:31b-n3-text": 26, "gemma4mtp:12b-n2": None})
    hub.status.update(running=True, channels=["cozy"])
    hub.llm_open = True
    return fo, eng, hub, t


def _chatters(hub, eng, t, listed=("alpha", "bravo"), flagged=("charlie", "delta", "echo")):
    for i, who in enumerate(listed + flagged):
        serve.handle(hub, msg(who, "you are trash and nobody likes you", t + i))
        serve.handle(hub, msg(who, "honestly the worst player i have ever seen", t + i + 0.5))
    for who in listed:                                       # a bad-faith person read puts them on the watchlist
        u = next(u for u in eng.users.values() if u.login == who)
        u.review = {"label": "bad_faith", "rationale": "x", "reads": 1, "n": u.n, "ts": 0, "pending": False}


def test_review_counts_new_content_per_scope_and_one_press_is_one_swap(monkeypatch):
    fo, eng, hub, t = _hybrid(monkeypatch)
    _chatters(hub, eng, t)
    s = hub.review_summary(force=True)
    assert s["a"]["chatters"] == 2 and s["b"]["chatters"] == 5 and s["b"]["lines"] >= 5 and s["b"]["seconds"] > 30
    queued_before = len(hub.tox_pending)
    assert queued_before > 0

    async def go():
        hub.start_review({})                                 # the default scope: b (owner, 2026-11-07)
        await hub.ai_task_done()
    asyncio.run(go())
    r = hub.ai_last["review"]
    assert r["scope"] == "b" and r["reviewed"] == 5 and not r.get("error"), r
    swaps = [c for c in fo.calls if c[0] in ("warm", "unload")]
    assert swaps == [("unload", "gemma4mtp:12b-n2"), ("warm", "gemma4mtp:31b-n3-text", 26),
                     ("unload", "gemma4mtp:31b-n3-text"), ("warm", "gemma4mtp:12b-n2", None)]
    reads = [c for c in fo.calls if c[0] in ("person", "window")]
    assert reads and all(c[1] == "gemma4mtp:31b-n3-text" for c in reads)
    assert not hub.llm_hold and fo.loaded == ["gemma4mtp:12b-n2"]
    assert len(hub.reviewed()) == 5 and all(v["model"] == "gemma4mtp:31b-n3-text" for v in hub.reviewed().values())
    assert len(hub.tox_pending) < queued_before                # the live queue's covered lines are gone
    assert set(r["timings"]) >= {"unload_live_s", "load_reviewer_s", "unload_reviewer_s", "load_live_s", "outage_s"}
    rec = hub.machine[aictl.SWAP_KEY]
    assert rec["swap_s"] >= 0 and rec["live"] == "gemma4mtp:12b-n2" and rec["reviewer"] == "gemma4mtp:31b-n3-text"
    assert set(hub.machine["loads"]) == {"gemma4mtp:31b-n3-text", "gemma4mtp:12b-n2"}      # each model's last load here
    hub.machine["loads"] = {k: {"s": 200.0, "at": "2999-01-01T00:00:00Z"} for k in hub.machine["loads"]}
    assert hub._swap_estimate()["s"] == 402                  # loads longer than the batch: their sum, plus unloads
    # nothing new: nothing to review; a new message brings back that chatter only
    assert hub.review_summary(force=True)["b"]["chatters"] == 0
    with pytest.raises(aictl.Busy):
        hub.start_review({})
    serve.handle(hub, msg("charlie", "still the worst player i have seen", t + 30))
    s = hub.review_summary(force=True)
    assert s["b"]["chatters"] == 1 and s["b"]["lines"] == 1      # charlie only, his one new line
    assert [eng.users[(c["ch"], c["uid"])].login for c in hub.review_candidates("b")] == ["charlie"]


def test_the_watchlist_only_scope_reviews_the_listed_chatters(monkeypatch):
    fo, eng, hub, t = _hybrid(monkeypatch)
    _chatters(hub, eng, t)

    async def go():
        hub.start_review({"scope": "a"})
        await hub.ai_task_done()
    asyncio.run(go())
    assert hub.ai_last["review"]["reviewed"] == 2
    assert {eng.users[k].login for k in hub.reviewed()} == {"alpha", "bravo"}


def test_cancel_stops_at_a_safe_point_swaps_back_and_marks_nothing_half_done(monkeypatch):
    fo, eng, hub, t = _hybrid(monkeypatch)
    _chatters(hub, eng, t)
    real = fo.review

    def slow(*a, **k):
        time.sleep(0.2)
        return real(*a, **k)
    fo.review = slow
    monkeypatch.setattr(llm, "review", slow)

    async def go():
        hub.start_review({})
        await _until(lambda: (hub.ai_task or {}).get("done", 0) >= 1)
        hub.cancel_ai_task()
        await hub.ai_task_done()
    asyncio.run(go())
    r = hub.ai_last["review"]
    assert r.get("cancelled") and 1 <= r["reviewed"] < 5
    assert len(hub.reviewed()) == r["reviewed"]
    assert fo.loaded == ["gemma4mtp:12b-n2"] and not hub.llm_hold


def test_stop_during_a_review_cancels_it_swaps_back_then_finishes_the_queue(monkeypatch):
    fo, eng, hub, t = _hybrid(monkeypatch)
    _chatters(hub, eng, t)
    real = fo.review

    def slow(*a, **k):
        time.sleep(0.15)
        return real(*a, **k)
    monkeypatch.setattr(llm, "review", slow)
    order = []
    monkeypatch.setattr(serve, "save_engine_async", lambda e, fork=False: order.append("save") or asyncio.sleep(0))

    async def go():
        worker = asyncio.create_task(hub.reviewer())
        hub.start_review({})
        await _until(lambda: (hub.ai_task or {}).get("done", 0) >= 1)
        with pytest.raises(RuntimeError):
            hub.start_live(["cozy"])                         # (already running)
        await hub.stop_live()
        worker.cancel()
    asyncio.run(go())
    r = hub.ai_last["review"]
    assert r.get("cancelled") and fo.loaded == ["gemma4mtp:12b-n2"] and not hub.llm_hold
    assert hub.review_q.empty() and not hub.tox_pending     # the live queue then finished (with the live model)
    assert all(c[1] == "gemma4mtp:12b-n2" for c in fo.calls[fo.calls.index(("warm", "gemma4mtp:12b-n2", None)):]
               if c[0] in ("person", "window"))


def test_a_shutdown_during_a_review_skips_the_swap_and_keeps_the_choice(monkeypatch):
    fo, eng, hub, t = _hybrid(monkeypatch)
    _chatters(hub, eng, t)
    real = fo.review

    def slow(*a, **k):
        time.sleep(0.15)
        return real(*a, **k)
    monkeypatch.setattr(llm, "review", slow)
    monkeypatch.setattr(serve, "save_engine_async", lambda e, fork=False: asyncio.sleep(0))

    async def go():
        hub.start_review({})
        await _until(lambda: (hub.ai_task or {}).get("done", 0) >= 1)
        await hub.stop_live(finish=False)
        await _until(lambda: hub.ai_task is None)
    asyncio.run(go())
    assert not hub.llm_hold and hub.ai["live"] == "gemma4mtp:12b-n2"
    assert ("warm", "gemma4mtp:12b-n2", None) not in fo.calls      # no swap back during a shutdown
    assert len(hub.reviewed()) == hub.ai_last["review"]["reviewed"] < 5


def test_the_review_is_refused_while_a_stop_finishes_the_queue(monkeypatch):
    fo, eng, hub, t = _hybrid(monkeypatch)
    _chatters(hub, eng, t)
    hub.finishing = True
    with pytest.raises(aictl.Busy, match="finishing a Stop"):
        hub.start_review({})
    hub.finishing = False
    hub.ai = {"mode": "single", "live": "gemma4mtp:31b-n3-text", "reviewer": None}
    with pytest.raises(aictl.Busy, match="hybrid"):
        hub.start_review({})


def test_an_unreachable_ollama_fails_safe_and_says_so(monkeypatch):
    fo, eng, hub, t = _hybrid(monkeypatch)
    _chatters(hub, eng, t)
    fo.fail_warm.add("gemma4mtp:31b-n3-text")

    async def go():
        hub.start_review({})
        await hub.ai_task_done()
    asyncio.run(go())
    r = hub.ai_last["review"]
    assert "could not load gemma4mtp:31b-n3-text" in r["error"] and r["reviewed"] == 0 and not hub.reviewed()
    assert fo.loaded == ["gemma4mtp:12b-n2"] and not hub.llm_hold
    # the server stops answering mid-batch: the chatter in progress is not marked, the worker is released
    fo.fail_warm.clear()

    def down(*a, **k):
        fo.up = False
        raise ConnectionRefusedError("refused")
    monkeypatch.setattr(llm, "review", down)

    async def go2():
        hub.start_review({})
        await hub.ai_task_done()
    asyncio.run(go2())
    r = hub.ai_last["review"]
    assert r["reviewed"] == 0 and r["failed"] >= 1 and "swap_back_error" in r and not hub.llm_hold
    assert not hub.reviewed()


def test_the_endpoints(monkeypatch):
    fo = FakeOllama(monkeypatch)
    eng, hub, t = _hub()

    async def go():
        from aiohttp.test_utils import TestClient, TestServer
        app = serve.make_app(hub, port=8765)
        async with TestClient(TestServer(app), headers={"Host": "127.0.0.1:8765"}) as c:
            r = await c.get("/api/ai")
            j = await r.json()
            assert r.status == 200 and j["rows"][0]["id"] == "gemma4mtp:31b-n3-text" and j["ollama"]["version"] == "0.40.11"
            assert (await c.post("/api/ai/select", json={"mode": "none"})).status == 403       # no X-Chatwatch
            r = await c.post("/api/ai/select", json={"mode": "bogus"}, headers=PAGE)
            assert r.status == 400
            r = await c.post("/api/review", json={}, headers=PAGE)
            assert r.status == 409 and "hybrid" in (await r.json())["error"]
            r = await c.get("/api/ai?vram=8&ram=32")
            assert (await r.json())["tier"]["key"] == "E"
            s = await (await c.get("/api/status")).json()
            assert s["ai"]["mode"] == "single" and s["ai"]["scope_default"] == "b"
    asyncio.run(go())


def test_the_hybrids_models_are_read_into_the_page_cache(monkeypatch, tmp_path):
    """Swaps from the share are slow when the models are not cached (about 160 s for the 31B, BASELINE 34); in hybrid
    mode both models' blob files are held open and the reviewer's are read ahead. The estimate stays on this machine's
    measurements (a read-ahead alone does not shorten it)."""
    root = tmp_path / "ollama"
    blobs = root / "models" / "blobs"
    blobs.mkdir(parents=True)
    for tag, d in (("gemma4mtp:31b-n3-text", "a" * 64), ("gemma4mtp:12b-n2", "b" * 64)):
        name, t = tag.split(":")
        man = root / "models" / "manifests-v2" / "ollama.com" / "library" / name / t
        man.parent.mkdir(parents=True, exist_ok=True)
        (blobs / f"sha256-{d}").write_bytes(b"x" * 2_000_000)
        man.write_text(json.dumps({"layers": [{"digest": f"sha256:{d}", "size": 2_000_000},
                                              {"digest": "sha256:" + "c" * 64, "size": 64}]}))       # tiny: skipped
    monkeypatch.setenv("CW_OLLAMA_MODELS", str(root))
    assert [os.path.basename(p) for p, _ in picker.blob_files("gemma4mtp:31b-n3-text")] == ["sha256-" + "a" * 64]
    assert picker.blob_files("nosuch:model") == []
    fo, eng, hub, t = _hybrid(monkeypatch)
    hub.machine[aictl.SWAP_KEY] = {"swap_s": 352, "at": "2026-11-07T23:04:16Z", "live": "gemma4mtp:12b-n2",
                                   "reviewer": "gemma4mtp:31b-n3-text"}
    assert hub._swap_estimate()["s"] == 352 and hub.prefetch_summary()["state"] == "cold"

    async def go():
        await hub._prefetch("gemma4mtp:12b-n2", read=False)
        await hub._prefetch("gemma4mtp:31b-n3-text")
    asyncio.run(go())
    assert hub.prefetch_summary()["state"] == "ready"
    # the 352 s swap was measured with the reviewer not in memory: it no longer applies (no loads here: the study's)
    est = hub._swap_estimate()
    assert est["cached"] and est["label"] == "tested on 12 GB Nvidia" and est["s"] < 60
    assert set(hub._pf_fds) == {"gemma4mtp:31b-n3-text", "gemma4mtp:12b-n2"}     # held open (the share keeps its cache)
    assert hub.ai_status()["prefetch"]["state"] == "ready"


def test_a_batch_stops_to_keep_the_live_model_current(monkeypatch):
    """Once the live queue's oldest person read or flagged line has waited REVIEW_LIVE_LAG_S, the batch swaps back;
    the chatters left are not marked and the next press covers them."""
    fo, eng, hub, t = _hybrid(monkeypatch)
    _chatters(hub, eng, t)
    real = hub._llm_status
    late = {"on": False}                          # the live queue falls behind once the batch has started
    monkeypatch.setattr(hub, "_llm_status", lambda now=None: {**(real(now) or {}),
                                                              **({"behind_hi_s": aictl.REVIEW_LIVE_LAG_S + 1} if late["on"] else {})})

    async def go():
        hub.start_review({})
        late["on"] = True
        await hub.ai_task_done()
        late["on"] = False
    asyncio.run(go())
    r = hub.ai_last["review"]
    assert r["reviewed"] == 1 and r["stopped_for_live"]["left"] == 4 and len(hub.reviewed()) == 1
    assert fo.loaded == ["gemma4mtp:12b-n2"] and not hub.llm_hold
    assert hub.review_summary(force=True)["b"]["chatters"] == 4


# ------------------------------------------------------------------------------------------------ review fixes (2026-11-08)
def test_fix1_a_reviewed_lines_context_is_its_own_channels(monkeypatch):
    """Line numbers are per channel and gapless, so the 6 lines before a line are that channel's (two channels
    interleaved used to give another channel's lines, or the chatter's own later ones)."""
    eng, hub, t = _hub()
    for i in range(10):
        for ch in ("cozy", "other"):
            hub.track_message({"channel": ch, "user_id": "u", "id": f"{ch}{i}", "text": f"{ch} line {i}",
                               "display": "x", "n": i}, None)
    seq = hub._ring_seq["cozy"]                    # the 10th cozy line
    ctx = hub._context("cozy", seq)
    assert ctx == [f"x: cozy line {i}" for i in range(3, 9)]
    assert all("other" not in c for c in ctx)


def test_fix2_the_fit_is_waited_for_and_an_mtp_cap_is_never_none(monkeypatch, tmp_path):
    log = str(tmp_path / "ollama.log")
    open(log, "w").close()
    fo = FakeOllama(monkeypatch)
    monkeypatch.setattr(aictl, "CHECK_BUDGET_S", 0)
    monkeypatch.setitem(picker.setup("gemma4mtp:31b-n3-text")["tiers"]["A"], "num_gpu_known", False)
    real_warm = fo.warm

    def slow_log_warm(model, timeout=600):          # the copier writes the offload line 1.5 s after the load
        real_warm(model, timeout)
        if llm.num_gpu(model) is None:
            import threading
            threading.Timer(1.5, lambda: open(log, "a").write("load_tensors: offloaded 49/49 layers to GPU\n")).start()
    monkeypatch.setattr(llm, "warm", slow_log_warm)
    eng, hub, t = _hub(specs=AMD16, log=log)

    async def go():
        hub.start_speed({"model": "gemma4mtp:31b-n3-text"})
        await hub.ai_task_done()
    asyncio.run(go())
    r = hub.ai_last["speed"]
    assert not r.get("error"), r
    assert r["fit"] == [49, 49] and r["num_gpu"] == 49 and picker.cap_value(hub.machine["caps"]["gemma4mtp:31b-n3-text"]) == 49   # a number
    # without the log copy the check refuses: no uncapped MTP run, nothing stored, the old cap and model back
    eng, hub, t = _hub(specs=AMD16, log=str(tmp_path / "missing.log"))
    hub.machine["caps"] = {}                      # (the first part found one: forget it)
    llm.NUM_GPU_BY_MODEL.pop("gemma4mtp:31b-n3-text", None)

    n0 = len(fo.calls)

    async def go2():
        hub.start_speed({"model": "gemma4mtp:31b-n3-text"})
        await hub.ai_task_done()
    asyncio.run(go2())
    assert "Ollama's log is not readable" in hub.ai_last["speed"]["error"]
    assert "gemma4mtp:31b-n3-text" not in (hub.machine.get("caps") or {})
    assert not [c for c in fo.calls[n0:] if c[0] in ("person", "window")]          # nothing ran uncapped


def test_fix3_a_stop_before_the_swap_swaps_nothing_and_chat_stops_at_once(monkeypatch):
    fo, eng, hub, t = _hybrid(monkeypatch)
    _chatters(hub, eng, t)
    gate = {"open": False}

    async def slow_idle():                          # a live call in flight
        while not gate["open"]:
            await asyncio.sleep(0.01)
    monkeypatch.setattr(hub, "llm_idle", slow_idle)

    async def go():
        hub.start_review({})
        await asyncio.sleep(0.05)
        hub.cancel_ai_task()
        gate["open"] = True
        await hub.ai_task_done()
    asyncio.run(go())
    assert hub.ai_last["review"].get("cancelled") and not hub.llm_hold
    assert not [c for c in fo.calls if c[0] in ("warm", "unload")]          # no swap out, no swap back
    # a Stop while a batch reviews: chat stops (stopping, not running, live task cancelled) before the batch ends
    fo2, eng2, hub2, t2 = _hybrid(monkeypatch)
    _chatters(hub2, eng2, t2)
    real = fo2.review
    monkeypatch.setattr(llm, "review", lambda *a, **k: time.sleep(0.8) or real(*a, **k))
    warm2 = fo2.warm
    monkeypatch.setattr(llm, "warm", lambda m, timeout=600: (time.sleep(0.5) if m == "gemma4mtp:12b-n2" else None) or warm2(m, timeout))
    monkeypatch.setattr(serve, "save_engine_async", lambda e, fork=False: asyncio.sleep(0))
    seen = {}

    async def go2():
        hub2.live_task = asyncio.ensure_future(asyncio.sleep(3600))
        worker = asyncio.create_task(hub2.reviewer())
        hub2.start_review({})
        await _until(lambda: (hub2.ai_task or {}).get("done", 0) >= 1)
        stop = asyncio.ensure_future(hub2.stop_live())
        await _until(lambda: hub2.stopping and not hub2.status["running"])
        seen.update(stopping=hub2.stopping, running=hub2.status["running"], live_cancelled=hub2.live_task is None,
                    batch_running=hub2.ai_task is not None)
        await stop
        worker.cancel()
    asyncio.run(go2())
    assert seen == {"stopping": True, "running": False, "live_cancelled": True, "batch_running": True}, seen
    assert fo2.loaded == ["gemma4mtp:12b-n2"]


def test_fix4_the_cards_llm_tools_refuse_while_the_gpu_is_busy_or_the_ai_is_off(monkeypatch):
    fo = FakeOllama(monkeypatch)
    eng, hub, t = _hub()
    serve.handle(hub, msg("rude", "you are trash", t + 1))
    uid = next(u.uid for u in eng.users.values() if u.login == "rude")

    async def go():
        from aiohttp.test_utils import TestClient, TestServer
        async with TestClient(TestServer(serve.make_app(hub, port=8765)), headers={"Host": "127.0.0.1:8765"}) as c:
            out = []
            hub.hold_llm("review")
            for path in ("/api/llm_review", "/api/draft_warning"):
                r = await c.post(path, json={"channel": "cozy", "uid": uid}, headers=PAGE)
                out.append((r.status, (await r.json())["error"]))
            hub.release_llm("review")
            hub.ai["mode"] = "none"
            r = await c.post("/api/llm_review", json={"channel": "cozy", "uid": uid}, headers=PAGE)
            out.append((r.status, (await r.json())["error"]))
            return out
    out = asyncio.run(go())
    assert [s for s, _ in out] == [409, 409, 409] and "paused for a model swap" in out[0][1] and "No AI" in out[2][1]
    assert not [c for c in fo.calls if c[0] in ("person", "window")]


def test_fix5_lines_arriving_during_a_batch_are_the_next_presss(monkeypatch):
    fo, eng, hub, t = _hybrid(monkeypatch)
    _chatters(hub, eng, t)
    cands = hub.review_candidates("b")
    charlie = next(c for c in cands if eng.users[(c["ch"], c["uid"])].login == "charlie")
    serve.handle(hub, msg("charlie", "and you still cannot play at all", t + 20))     # after the press
    u = eng.users[(charlie["ch"], charlie["uid"])]
    assert u.n == charlie["n"] + 1
    hub._enqueue(0, "person", {"channel": charlie["ch"], "user_id": charlie["uid"]})   # a live read asked for since
    queued_after = time.time()

    async def go():
        hub.hold_llm("x")                         # (no live worker in this test)
        ok = await hub._review_chatter(charlie, "gemma4mtp:31b-n3-text", hub.run_gen) if fo.loaded.append("gemma4mtp:31b-n3-text") is None else None
        return ok
    asyncio.run(go())
    st = hub.reviewed()[(charlie["ch"], charlie["uid"])]
    assert st["n"] == charlie["n"]                                     # the snapshot, not the newer count
    assert [c for c in hub.review_candidates("b") if c["uid"] == charlie["uid"]]   # the newer line: next press
    assert any(e[2] == "person" and e[3]["user_id"] == charlie["uid"] for e in hub.review_q.items)  # kept
    assert queued_after > charlie["at"]


def test_fix6_a_harmless_line_after_a_review_does_not_bring_a_flagged_chatter_back(monkeypatch):
    fo, eng, hub, t = _hybrid(monkeypatch)
    _chatters(hub, eng, t, listed=())
    monkeypatch.setattr(llm, "review", lambda p, ctx, emotes=frozenset(), model=None, timeout=90:
                        fo._check(model, "person") or {"label": "mild", "rationale": "", "confidence": "low", "rules": []})

    async def go():
        hub.start_review({"scope": "b"})
        await hub.ai_task_done()
    asyncio.run(go())
    assert hub.ai_last["review"]["reviewed"] == 3
    serve.handle(hub, msg("charlie", "what game is this", t + 40))       # harmless
    assert hub.review_summary(force=True)["b"]["chatters"] == 0
    serve.handle(hub, msg("delta", "you are garbage and a clown", t + 41))  # a new toxic line
    assert [eng.users[(c["ch"], c["uid"])].login for c in hub.review_candidates("b")] == ["delta"]


def test_fix7_no_cap_is_stored_explicitly_and_gpu_envs_cap_does_not_leak(monkeypatch):
    fo = FakeOllama(monkeypatch)
    fo.loaded = ["gemma4mtp:31b-n3-text"]
    monkeypatch.setattr(llm, "NUM_GPU", "26")                   # gpu.env's first-run cap, for the 31B
    eng, hub, t = _hub()

    async def go():
        hub.start_select({"mode": "single", "live": "gpt-oss-safeguard:20b"})
        await hub.ai_task_done()
    asyncio.run(go())
    assert hub.ai_last["select"].get("ok"), hub.ai_last
    assert ("warm", "gpt-oss-safeguard:20b", None) in fo.calls          # not 26
    assert hub.machine["caps"]["gpt-oss-safeguard:20b"] == {"value": None, "tier": "A", "how": "set"}      # explicit, with its tier
    env = open(os.path.join(picker.machines_dir(), "box.env")).read()
    assert "CW_LIVE_MODEL=gpt-oss-safeguard:20b\nCW_LLM_NUM_GPU=\n" in env
    # the next start restores "no cap" from the file
    llm.NUM_GPU_BY_MODEL.clear()
    eng2, hub2, t2 = _hub()
    assert llm.num_gpu("gpt-oss-safeguard:20b") is None and hub2.ai["live"] == "gpt-oss-safeguard:20b"


def test_fix8_a_reviewer_that_never_loaded_records_no_swap(monkeypatch):
    fo, eng, hub, t = _hybrid(monkeypatch)
    _chatters(hub, eng, t)
    fo.fail_warm.add("gemma4mtp:31b-n3-text")

    async def go():
        hub.start_review({})
        await hub.ai_task_done()
    asyncio.run(go())
    assert "could not load" in hub.ai_last["review"]["error"]
    assert aictl.SWAP_KEY not in hub.machine and "swap" not in hub.machine and "swap_cached" not in hub.machine


def test_fix9_a_failed_switch_or_check_restores_the_live_model_and_its_cap(monkeypatch):
    fo = FakeOllama(monkeypatch)
    fo.loaded = ["gemma4mtp:31b-n3-text"]
    eng, hub, t = _hub()
    llm.NUM_GPU_BY_MODEL["gemma4mtp:31b-n3-text"] = 26
    fo.fail_warm.add("gemma4mtp:12b-n2")

    async def go():
        hub.start_select({"mode": "single", "live": "gemma4mtp:12b-n2"})
        await hub.ai_task_done()
    asyncio.run(go())
    assert "could not load" in hub.ai_last["select"]["error"]
    assert fo.loaded == ["gemma4mtp:31b-n3-text"] and hub.ai["live"] == "gemma4mtp:31b-n3-text"
    assert llm.num_gpu("gemma4mtp:31b-n3-text") == 26 and "gemma4mtp:12b-n2" not in llm.NUM_GPU_BY_MODEL
    # a speed check that fails leaves the old cap and the live model
    fo.fail_warm.clear()
    fo.abort_until["gemma4mtp:31b-n3-text"] = 0                       # aborts at any cap: not stable
    monkeypatch.setattr(aictl, "CHECK_BUDGET_S", 0)

    async def go2():
        hub.start_speed({"model": "gemma4mtp:31b-n3-text"})
        await hub.ai_task_done()
    asyncio.run(go2())
    assert "not stable" in hub.ai_last["speed"]["error"]
    assert llm.num_gpu("gemma4mtp:31b-n3-text") == 26 and fo.loaded == ["gemma4mtp:31b-n3-text"]


def test_fix10_with_the_ai_off_a_kept_queue_is_kept_not_dropped(monkeypatch):
    fo = FakeOllama(monkeypatch)
    eng, hub, t = _hub()
    serve.handle(hub, msg("rude", "you are trash and everyone hates you", t + 1))
    n = hub.review_q.qsize() + len(hub.tox_pending)
    assert n > 0
    hub.apply_choice({"mode": "none"})
    monkeypatch.setattr(serve, "save_engine_async", lambda e, fork=False: asyncio.sleep(0))
    hub.status["running"] = True
    asyncio.run(hub.stop_live())
    assert hub.review_q.qsize() + len(hub.tox_pending) == n and eng.stats["llm_dropped_at_stop"] == 0


def test_fix11_review_refuses_while_the_live_ai_is_far_behind(monkeypatch):
    fo, eng, hub, t = _hybrid(monkeypatch)
    _chatters(hub, eng, t)
    real = hub._llm_status
    monkeypatch.setattr(hub, "_llm_status", lambda now=None: {**(real(now) or {}), "behind_hi_s": aictl.REVIEW_LIVE_LAG_S + 60})
    with pytest.raises(aictl.Busy, match="min behind"):
        hub.start_review({})


def test_fix12_the_reviewers_read_does_not_spend_the_live_read_budget(monkeypatch):
    fo, eng, hub, t = _hybrid(monkeypatch)
    _chatters(hub, eng, t, listed=())
    u = next(u for u in eng.users.values() if u.login == "charlie")
    before = (u.review or {}).get("reads", 0)

    async def go():
        hub.start_review({})
        await hub.ai_task_done()
    asyncio.run(go())
    assert u.review["reads"] == before and u.review["reviewer_reads"] == 1 and u.review["label"] == "bad_faith"


def test_fix13_the_chat_count_is_not_gpu_work(monkeypatch):
    fo = FakeOllama(monkeypatch)
    eng, hub, t = _hub()
    gate = {"open": False}

    async def slow_count(ch, secs):
        while not gate["open"]:
            await asyncio.sleep(0.01)
        return {"channel": ch, "msgs_per_min": 42.0, "messages": 42, "seconds": 60, "note": ""}
    monkeypatch.setattr(hub, "_chat_rate", slow_count)

    async def go():
        hub.start_chatrate({"channel": "somechannel"})
        await asyncio.sleep(0.02)
        busy = hub._status()["llm_busy"]
        hub.start_select({"mode": "none"})          # not refused by the count
        await hub.ai_task_done()
        st = hub.ai_status()["rate_task"]
        gate["open"] = True
        await hub._rate_job
        return busy, st
    busy, st = asyncio.run(go())
    assert busy is False and st["kind"] == "chatrate" and hub.ai_last["select"].get("ok")
    assert hub.ai_last["chatrate"]["msgs_per_min"] == 42.0 and hub.rate_task is None


def test_a_check_on_a_busy_machine_is_kept_and_says_so(monkeypatch):
    """Owner 2026-11-07: the check shows what this machine does right now; the estimate is only a placeholder. So a
    result taken while other work ran replaces the estimate too, and says so beside the number."""
    fo = FakeOllama(monkeypatch)
    fo.loaded = ["gemma4mtp:31b-n3-text"]
    monkeypatch.setattr(aictl, "CHECK_BUDGET_S", 0)
    monkeypatch.setattr(aictl, "_loadavg", lambda: aictl.quiet_threshold() + 5)
    eng, hub, t = _hub()

    async def go():
        hub.start_speed({"model": "gemma4mtp:12b-n2"})
        await hub.ai_task_done()
    asyncio.run(go())
    r = hub.ai_last["speed"]
    assert r["busy"] and not r["quiet"] and any("other work was running" in n for n in r["notes"])
    saved = (hub.machine.get("speed") or {})["gemma4mtp:12b-n2"]
    assert saved["busy"] and saved["keepup"] and not hub.machine.get("loads")      # model loads: still quiet-only
    assert aictl.quiet_threshold() >= 1


def test_a_live_session_watches_one_channel(monkeypatch):
    """Owner 2026-11-07: one channel per live session; another channel means Stop, then Start."""
    eng, hub, t = _hub()

    async def go():
        from aiohttp.test_utils import TestClient, TestServer
        async with TestClient(TestServer(serve.make_app(hub, port=8765)), headers={"Host": "127.0.0.1:8765"}) as c:
            r = await c.post("/api/start", json={"channels": "cozy, other"}, headers=PAGE)
            return r.status, (await r.json())["error"]
    status, err = asyncio.run(go())
    assert status == 400 and err == "one channel at a time: Stop, then Start the other" and not hub.status["running"]
    monkeypatch.setattr("sys.argv", ["serve", "--channels", "cozy,other"])
    monkeypatch.setattr(serve, "take_lock", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no lock first")))
    with pytest.raises(SystemExit, match="one channel at a time"):
        serve.main()


# ------------------------------------------------------------------------------------------------ second review (2026-11-08)
def test_r2_1_a_cap_stored_on_one_tier_does_not_apply_on_another(monkeypatch):
    """GPU undetected -> tier F32 -> picking the 12B + MTP must not store "no cap"; once the dropdown says 8 GB (tier E)
    the study's cap 36 applies, never an uncapped MTP build (H3's abort case)."""
    fo = FakeOllama(monkeypatch)
    nogpu = {**NV12, "gpu_vendor": None, "vram_mb": None}
    eng, hub, t = _hub(specs=nogpu)
    hub.machine["override"] = {"vram": "none", "ram": "32"}
    assert hub._tier()["key"] == "F32"

    async def go():
        hub.start_select({"mode": "single", "live": "gemma4mtp:12b-n2"})
        await hub.ai_task_done()
    asyncio.run(go())
    assert hub.ai_last["select"].get("ok"), hub.ai_last
    assert "gemma4mtp:12b-n2" not in (hub.machine.get("caps") or {})          # an MTP build's "none" is not stored
    hub.machine["override"] = {"vram": "8", "ram": "32"}
    cap = picker.cap_for("gemma4mtp:12b-n2", hub._tier(), None, hub.machine)
    assert hub._tier()["key"] == "E" and cap["value"] == 36 and cap["source"].startswith("study")
    # a cap found on tier A is ignored on tier E; a plain old-format "none" for an MTP build is never trusted
    m = {"caps": {"gemma4mtp:31b-n3-text": picker.cap_entry(26, "A"), "gemma4mtp:12b-n2": None}}
    assert picker.stored_cap(m, "gemma4mtp:31b-n3-text", "E") == (False, None)
    assert picker.stored_cap(m, "gemma4mtp:31b-n3-text", "A") == (True, 26)
    assert picker.stored_cap(m, "gemma4mtp:12b-n2", "A") == (False, None)
    assert picker.stored_cap({"caps": {"gpt-oss-safeguard:20b": picker.cap_entry(None, "A")}},
                             "gpt-oss-safeguard:20b", "A") == (True, None)    # a plain model's explicit "no cap"


def test_r2_3_a_replay_keeps_the_cards_llm_tools(monkeypatch):
    """A replay (or a server without --llm-review) starts with the AI "off" for the work queue, but the user card's
    tools work as on master; only a live run with No AI refuses them."""
    eng = Engine()
    hub = Hub(eng)
    hub.status["mode"] = "replay"
    hub.ai_init(specs=NV12, host="box")
    hub.startup_choice()                        # review_q is None: mode none
    assert hub.ai_off and hub.llm_tools_busy() is None
    hub.status["mode"] = "live"
    assert "No AI" in hub.llm_tools_busy()


def test_r2_4_the_batch_samples_other_work_after_the_reviewer_idles(monkeypatch):
    fo, eng, hub, t = _hybrid(monkeypatch)
    _chatters(hub, eng, t)
    monkeypatch.setattr(aictl, "OTHER_SAMPLE_DELAY_S", 0.05)
    ends, gaps = [], []
    real = fo.review

    def review(*a, **k):
        r = real(*a, **k)
        ends.append(time.time())
        return r
    monkeypatch.setattr(llm, "review", review)
    monkeypatch.setattr(aictl, "_running_now", lambda: (gaps.append(time.time() - ends[-1]) if ends else None) or 0)

    async def go():
        hub.start_review({})
        await hub.ai_task_done()
    asyncio.run(go())
    assert gaps and min(gaps) >= 0.045


# ------------------------------------------------------------------------------------------------ third review (2026-11-08)
def test_r3_1_the_env_sidecar_carries_only_a_cap_valid_for_the_current_tier(monkeypatch):
    fo = FakeOllama(monkeypatch)
    eng, hub, t = _hub()
    hub.machine["choice"] = {"mode": "hybrid", "live": "gpt-oss-safeguard:20b", "reviewer": "gemma4mtp:31b-n3-text"}
    hub.machine["caps"] = {"gemma4mtp:31b-n3-text": picker.cap_entry(26, "A"), "gpt-oss-safeguard:20b": picker.cap_entry(None, "A")}
    env = lambda: open(os.path.join(picker.machines_dir(), "box.env")).read()
    hub._save_machine()
    assert "CW_REVIEW_NUM_GPU=26" in env() and "CW_LLM_NUM_GPU=\n" in env()
    hub.machine["override"] = {"vram": "8", "ram": "32"}         # the dropdown changed the tier, nothing re-picked
    hub._save_machine()
    assert "CW_REVIEW_NUM_GPU=15" in env() and "CW_LLM_NUM_GPU=\n" in env()   # tier E's study cap; the 20b: none
    monkeypatch.setitem(picker.setup("gemma4mtp:31b-n3-text")["tiers"]["E"], "num_gpu_known", False)
    hub._save_machine()
    assert "CW_REVIEW_NUM_GPU=unknown" in env()                      # start.sh does not warm it uncapped


def test_r3_2_an_mtp_build_on_a_gpu_below_every_tier_is_never_run_uncapped(monkeypatch):
    # ramfit (2026-11-08): every GPU class now meets a GPU tier's VRAM; a GPU below every tier (no 8 GB tier here) keeps
    # this safety path
    monkeypatch.setattr(picker, "TIER_ORDER", tuple(t for t in picker.TIER_ORDER if t[0] != "E"))
    fo = FakeOllama(monkeypatch)
    small = {**NV12, "vram_mb": 8192, "ram_mb": 12000, "ram_avail_mb": 40000}   # an 8 GB card: below every tier (RAM read
    # with room, so only the cap decides here; ramblock tests the RAM)
    eng, hub, t = _hub(specs=small, choice={"mode": "single", "live": "gemma4mtp:31b-n3-text"})
    tr = hub._tier()
    assert tr["key"] is None and tr["gpu"]
    cap = picker.cap_for("gemma4mtp:31b-n3-text", tr, "nvidia", hub.machine)
    assert cap == {"value": None, "known": False, "source": picker.NEEDS_CHECK, "how": "unknown"}
    assert hub.ai_off and "press Check Speed once" in hub.ai_status()["blocked"]       # this run: AI off, the page says why
    assert picker.cap_for("gpt-oss-safeguard:20b", tr, "nvidia", hub.machine)["known"]  # a plain model: Ollama's fit
    # a cap found here (by the check) under this tier lets it run
    hub.machine["caps"] = {"gemma4mtp:31b-n3-text": picker.cap_entry(12, None)}
    eng2, hub2, t2 = _hub(specs=small, choice={"mode": "single", "live": "gemma4mtp:31b-n3-text"})
    hub2.machine["caps"] = hub.machine["caps"]
    hub2.startup_choice()
    assert not hub2.ai_off and llm.num_gpu("gemma4mtp:31b-n3-text") == 12 and hub2.ai_status()["blocked"] is None
    # a hybrid whose reviewer has no cap here: Review refuses before any swap
    fo2, eng3, hub3, t3 = _hybrid(monkeypatch)
    hub3.specs = small
    hub3.machine["caps"] = {}
    _chatters(hub3, eng3, t3)
    with pytest.raises(aictl.Busy, match="press Check Speed once"):
        hub3.start_review({})


# ------------------------------------------------------------------------------------------------ one list (owner 2026-11-07)
def _shown(cls, vendor="nvidia"):
    inst = {s["id"] for s in picker.catalogue()["setups"]}
    singles = picker.rows(cls, vendor, inst, "0.40.11")
    hyb = picker.hybrid_rows(cls, vendor, inst, None, singles)
    s, h = picker.shown_rows(singles, hyb)
    return [r["id"] for r in s], [x["id"] for x in h]


def test_the_page_lists_mtp_builds_only_and_nothing_dominated():
    """Owner 2026-11-07: no plain builds (their MTP twins are as good and faster), no setup another listed one beats on
    quality and speed, per tier; the 20b and Qwen are not listed on any machine (owner: they fall behind the 12B + MTP)."""
    for cls, vendor in (({"vram": "12", "ram": "96+"}, "nvidia"), ({"vram": "16+", "ram": "96+"}, "amd")):
        s, h = _shown(cls, vendor)
        assert s == ["gemma4mtp:31b-n3-text", "gemma4mtp:12b-n2"], s
        assert h == ["gemma4mtp:12b-n2+gemma4mtp:31b-n3-text"], h
    s, h = _shown({"vram": "8", "ram": "32"})
    assert s == ["gemma4mtp:31b-n3-text", "gemma4mtp:12b-n2"] and h == ["gemma4mtp:12b-n2+gemma4mtp:31b-n3-text"]
    s, h = _shown({"vram": "none", "ram": "16"}, "none")
    assert s == ["gemma4mtp:e4b-n4"] and h == []                          # the 16 GB laptop keeps what fits
    s, h = _shown({"vram": "none", "ram": "32"}, "none")
    assert s == ["gemma4mtp:12b-n2", "gemma4mtp:e4b-n4"] and h == []      # 32 GB laptop: no 20b or Qwen (owner); E4B + MTP is faster


def test_no_ai_round_trips_through_the_machine_file(monkeypatch):
    fo = FakeOllama(monkeypatch)
    fo.loaded = ["gemma4mtp:31b-n3-text"]
    eng, hub, t = _hub()

    async def go():
        hub.start_select({"mode": "none"})
        await hub.ai_task_done()
    asyncio.run(go())
    assert hub.ai_last["select"].get("ok") and hub.ai_off
    assert "CW_AI_MODE=none" in open(os.path.join(picker.machines_dir(), "box.env")).read()
    eng2, hub2, t2 = _hub()                       # the next start: the saved choice, AI off
    assert hub2.ai_off and not eng2.person_review and hub2._status()["llm_queue"] is None


def test_a_slow_ollama_keeps_the_last_installed_list(monkeypatch):
    """AMD desktop 2026-11-07: right after a switch Ollama answered /api/tags too slowly (the shared store was busy with
    a 20 GB load), and every row turned "not installed". The last list it gave now stands."""
    _, hub, _ = _hub()
    monkeypatch.setattr(llm, "installed_models", lambda timeout=10: ["gemma4mtp:31b-n3-text", "gemma4mtp:12b-n2"])
    assert hub._installed() == {"gemma4mtp:31b-n3-text", "gemma4mtp:12b-n2"}

    def slow(timeout=10):
        raise TimeoutError("timed out")
    monkeypatch.setattr(llm, "installed_models", slow)
    assert hub._installed() == {"gemma4mtp:31b-n3-text", "gemma4mtp:12b-n2"}


def test_a_bigger_card_finds_its_own_cap_instead_of_the_tier_safe_one(monkeypatch, tmp_path):
    """AMD desktop 2026-11-07: the 16 GB card ran 31B + MTP on the 12 GB tier's safe cap (26 then) (~10 msgs/min). On a card
    above the tested one, Check speed reads Ollama's own fit for this card and keeps fit - 3 (here 40/61 -> 37)."""
    log = str(tmp_path / "ollama.log")
    open(log, "w").close()
    fo = FakeOllama(monkeypatch)
    monkeypatch.setattr(aictl, "CHECK_BUDGET_S", 0)
    real_warm = fo.warm

    def fit_log_warm(model, timeout=600):
        real_warm(model, timeout)
        if llm.num_gpu(model) is None:                 # the uncapped load: Ollama's own fit for this card
            open(log, "a").write("load_tensors: offloaded 40/61 layers to GPU\n")
    monkeypatch.setattr(llm, "warm", fit_log_warm)
    eng, hub, t = _hub(specs=AMD16, log=log)
    assert picker.cap_for("gemma4mtp:31b-n3-text", hub._tier(), "amd", hub.machine)["value"] == 30     # the safe default

    async def go():
        hub.start_speed({"model": "gemma4mtp:31b-n3-text"})
        await hub.ai_task_done()
    asyncio.run(go())
    r = hub.ai_last["speed"]
    assert not r.get("error"), r
    assert r["fit"] == [40, 61] and r["num_gpu"] == picker.cap_from_fit((40, 61))
    assert picker.cap_value(hub.machine["caps"]["gemma4mtp:31b-n3-text"]) == r["num_gpu"] > 30


# ------------------------------------------------------------------------------------------------ vramcap (owner 2026-11-07)
M31, M12, E4B = "gemma4mtp:31b-n3-text", "gemma4mtp:12b-n2", "gemma4mtp:e4b-n4"
A_GB = 11935 / 1024              # tier A as read here: 1,925 MiB free + 10,010 MiB held by cw-ollama's runner
E_GB = A_GB - 4.9                # tier E: the 4.9 GB placeholder
WARM = {"vram_held_mb": 10010}   # a read with cw-ollama's model loaded (as the points were read)


def _est(tag, v, cold=False):
    return picker.estimate_cap(tag, v, cold)["value"]


def test_the_vram_estimate_reproduces_the_measured_caps():
    """"We see what VRAM is free and we can estimate the layers based on that": the measured points come back exactly
    (HARDWARE_TIERS section 3; the text 31B's 30 on A, FACTS section 4), less free VRAM gives fewer layers, never more than
    the model has."""
    assert _est(M31, E_GB) == 15 and _est(M31, A_GB) == 30 and _est(M12, E_GB) == 36
    assert picker.estimate_cap(M12, A_GB)["value"] is None                   # fits whole, as measured
    assert _est(M31, 9900 / 1024) == 20 < 30                                 # a 12 GB card with 2 GB held by the desktop
    assert _est(M31, 15600 / 1024) == 37                                     # a 16 GB card: above the measured range
    assert _est(M31, 40) == 61 and _est(M31, 1.0) == 1                       # clamped to [1, layers]
    for cold in (False, True):
        last = {M31: 0, M12: 0}
        for k in range(10, 300):                                             # 1.0 .. 29.9 GB: never down as VRAM grows
            for tag, L in ((M31, 61), (M12, 49)):
                v = _est(tag, k / 10, cold)
                v = L if v is None else v
                assert 1 <= v <= L and v >= last[tag], (tag, k, cold)
                last[tag] = v
    g31, how = picker.gb_per_layer(M31)
    g12, how12 = picker.gb_per_layer(M12)
    assert how == "measured slope" and abs(g31 - 4.9 / 15) < 1e-9
    assert how12.startswith("weights per layer x 1.5") and 0.2 < g12 < 0.23
    assert picker.estimate_cap(E4B, 8.0)["value"] is None and picker.estimate_cap(E4B, 5.0) is None
    assert picker.estimate_cap("gemma4:31b-it-qat", 11.7) is None          # only MTP builds have points


def test_review1_a_reserve_in_gb_a_margin_and_no_cap_at_a_known_failure():
    """Review 2026-11-08 #1: a read with nothing held loses COLD_RESERVE_GB (the points were read with the runner's own
    context inside "held"); the margin is max(2, ceil(0.75 GB / GB per layer)) layers; a cap that failed at a point is
    never reached until the VRAM clears that point by 0.75 GB; the page's figure is rounded down. W2 (sweep 2026-11-09):
    the reserve is for the line only; a measured point is reached by the read itself, cold or held."""
    assert _est(M31, A_GB - 0.01, cold=True) == 24 < 30                      # cold off the point: the line, 0.75 GB less
    assert _est(M31, A_GB, cold=True) == 30                                  # cold at the point: its measured cap
    assert picker.estimate_cap(M31, A_GB, True)["how"] == "measured"
    assert picker.estimate_cap(M12, A_GB, True)["value"] is None             # the 12B's "whole" point, reached cold
    assert _est(M12, A_GB - 0.01, cold=True) == 49 and _est(M12, A_GB - 0.01) == 49
    assert picker.memory({**NV12, "vram_avail_mb": 11935})["vram_cold"]     # not told: treated as cold
    assert not picker.memory({**NV12, "vram_avail_mb": 11935, **WARM})["vram_cold"]
    for k in range(0, 76):                                                   # E .. E + 0.75 GB
        v = E_GB + k / 100
        assert _est(M12, v) <= 36 and _est(M31, v) <= 15, v                 # 38 and 16 failed at E
    assert _est(M12, 7.7) == 36                                              # a real 8 GB card: not 38 (the cliff)
    g12 = picker.gb_per_layer(M12)[0]
    assert _est(M12, 9.0) == math.floor(36 + (9.0 - E_GB) / g12) - max(2, math.ceil(0.75 / g12)) == 42
    assert _est(M31, 12.0) == 30                                             # 33 failed at A: not above 30 near it
    assert picker.memory({**NV12, "vram_avail_mb": 11935, "ram_avail_mb": 90701})["vram_shown"] == 11.6   # down
    assert picker.memory({**NV12, "vram_avail_mb": 11935, "ram_avail_mb": 90701})["ram_shown"] == 88.5


def _mem(mb, held=10010, ram=90700):
    return picker.memory({**NV12, "vram_avail_mb": mb, "vram_held_mb": held, "ram_avail_mb": ram})


def test_review2_a_found_cap_is_lowered_along_its_own_line():
    """Review #2: a cap found on this card, read again with less VRAM, comes down by this card's own drop (the drop over
    the GB per layer), not to the catalogue's line: found 29 with 15.6 GB, now 13.9 GB -> 29 - ceil(1.7 / 0.327) = 23."""
    amd = picker.tier({"vram": "16+", "ram": "96+"}, "amd")
    m = {"caps": {M31: picker.cap_entry(29, amd["key"], "found", 15.6)}}
    c = picker.cap_for(M31, amd, "amd", m, _mem(int(13.9 * 1024) + 1))
    g = picker.gb_per_layer(M31)[0]
    assert c["how"] == "lowered" and c["value"] == 29 - math.ceil((15.6 - 13.9) / g) == 23
    assert picker.estimate_cap(M31, 13.9)["value"] > 23                      # the catalogue's line would have said more
    assert picker.cap_for(M31, amd, "amd", m, _mem(int(15.25 * 1024)))["value"] == 29     # within 0.4 GB: it stands
    assert picker.cap_for(M31, amd, "amd", m, _mem(int(17.0 * 1024)))["value"] == 29      # more: never raised


def test_review3_a_cap_this_machine_has_is_never_raised_by_the_estimate(monkeypatch):
    """Review #3: gpu.env's cap, a set or found cap and a legacy entry are this machine's; the estimate never raises
    them, it only lowers them when the VRAM is clearly lower than when they were set (the legacy entry and gpu.env by
    the measured point with that cap). The estimate fills in only where this machine has no cap."""
    A = picker.tier({"vram": "12", "ram": "96+"}, "nvidia")
    amd = picker.tier({"vram": "16+", "ram": "96+"}, "amd")
    legacy = {"caps": {M31: {"value": 30, "tier": "A"}}}
    assert picker.cap_for(M31, A, "nvidia", legacy, _mem(11935))["value"] == 30
    low = picker.cap_for(M31, A, "nvidia", legacy, _mem(9900))               # the desktop took 2 GB
    assert low["how"] == "lowered" and low["value"] == 30 - math.ceil((A_GB - 9900 / 1024) / (4.9 / 15)) == 23
    assert "lowered from 30" in low["source"]
    sset = {"caps": {M31: picker.cap_entry(26, amd["key"], "set", 11.655)}}
    assert picker.cap_for(M31, amd, "amd", sset, _mem(15600))["value"] == 26   # not raised to the estimate (37)
    assert picker.cap_for(M31, amd, "amd", {}, _mem(15600))["value"] == 37     # no cap here: the estimate
    # gpu.env's CW_LLM_NUM_GPU=26 on a machine with no saved choice: kept, even on a bigger card
    FakeOllama(monkeypatch)
    monkeypatch.setattr(llm, "NUM_GPU", "26")
    eng, hub, t = _hub(specs={**AMD16, "vram_avail_mb": 15600, "vram_held_mb": 9000, "ram_avail_mb": 70000})
    assert llm.num_gpu(M31) == 26 and hub._cap(M31)["how"] == "env"
    top = asyncio.run(hub.ai_info())["rows"][0]
    assert top["num_gpu"]["value"] == 26 and top["num_gpu"]["source"] == "gpu.env"
    # with no gpu.env cap, Use this stores the estimate as this machine's cap; a later start with more VRAM keeps it
    monkeypatch.setattr(llm, "NUM_GPU", "")
    eng, hub, t = _hub(specs={**AMD16, "vram_avail_mb": 15600, "vram_held_mb": 9000, "ram_avail_mb": 70000})
    assert hub._cap(M31)["how"] == "estimate" and hub._cap(M31)["value"] == 37

    async def go():
        hub.start_select({"mode": "single", "live": M31})
        await hub.ai_task_done()
    asyncio.run(go())
    assert hub.machine["caps"][M31] == {"value": 37, "tier": "A", "how": "set", "avail_gb": round(15600 / 1024, 3)}
    eng2, hub2, t2 = _hub(specs={**AMD16, "vram_avail_mb": 16000, "vram_held_mb": 9000, "ram_avail_mb": 70000})
    assert llm.num_gpu(M31) == 37 and hub2._cap(M31)["how"] == "set"       # the same request start to start


def test_review4_an_estimated_cap_is_not_warmed_from_last_starts_file(monkeypatch):
    """Review #4: the .env carries "estimate" for a cap the server works out at each start; start.sh does not warm
    it with last start's figure (the first call loads it)."""
    FakeOllama(monkeypatch)
    eng, hub, t = _hub(specs={**AMD16, "vram_avail_mb": 15600, "vram_held_mb": 9000, "ram_avail_mb": 70000})
    hub.machine["choice"] = {"mode": "hybrid", "live": M12, "reviewer": M31}
    hub.machine["caps"] = {M31: picker.cap_entry(30, "A", "found", 15.2)}
    hub._save_machine()
    env = open(os.path.join(picker.machines_dir(), "amdbox.env")).read()
    assert "CW_LLM_NUM_GPU=estimate" in env and "CW_REVIEW_NUM_GPU=30" in env   # the 12B: estimated to fit whole
    sh = open(os.path.join(os.path.dirname(__file__), "..", "start.sh")).read()
    assert 'CW_LLM_NUM_GPU:-}" == estimate ]]; then CW_LLM_NUM_GPU=""; SKIP_WARM=2' in sh


def test_review5_a_model_with_no_cap_spills_what_the_card_cannot_hold():
    """Review #5: a model on Ollama's own fit (no cap) puts in RAM what does not fit in the VRAM available."""
    no_cap = {"known": True, "value": None, "source": "Ollama's own fit"}
    size = 13.8e9 / 1024 ** 3                     # the catalogue's download size
    assert picker.ram_need_gb("gpt-oss-safeguard:20b", no_cap, {"vram": 7.5}) == round(size - 7.5 + 1.0, 1)
    assert picker.ram_need_gb("gpt-oss-safeguard:20b", no_cap, {"vram": None, "vram_total": 12.0}) == round(size - 12 + 1, 1)
    assert picker.ram_need_gb("gpt-oss-safeguard:20b", no_cap, {"vram": 23.0}) == 1.0
    assert picker.ram_need_gb(M12, {"known": True, "value": None, "source": "fits whole in 11.6 GB available"},
                              {"vram": 11.6}) == 1.0                          # an MTP build: estimated to fit whole


def test_review6_the_start_scripts_vram_read(tmp_path):
    """Review #6: a failing nvidia-smi reads as 0 under set -euo pipefail (no exit); with two Nvidia cards only the
    held memory on the card whose free memory is used counts; DRM fdinfo values without a unit are bytes."""
    import subprocess
    sh = open(os.path.join(os.path.dirname(__file__), "..", "start.sh")).read()
    fns = sh[sh.index("detect_specs() {"):sh.index("\nmeasure_avail\n")]
    stub = tmp_path / "t.sh"
    stub.write_text("set -euo pipefail\n" + fns + """
podman() { printf 'HPID\\n111\\n222\\n'; }
nvidia-smi() {
  case "$*" in
    *query-gpu*) printf '8192, 7000, GPU-small\\n12282, 1925, GPU-big\\n' ;;
    *query-compute-apps*) printf '111, 10010, GPU-big\\n222, 3000, GPU-small\\n333, 500, GPU-big\\n' ;;
  esac
}
CW_SPEC_GPU_VENDOR=nvidia; measure_avail; echo "ok $CW_SPEC_VRAM_AVAIL_MB $CW_SPEC_VRAM_HELD_MB"
nvidia-smi() { return 9; }
measure_avail; echo "fail $CW_SPEC_VRAM_AVAIL_MB $CW_SPEC_VRAM_HELD_MB"
""")
    out = subprocess.run(["bash", str(stub)], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert "ok 11935 10010" in out.stdout and "fail 0 0" in out.stdout
    awk = sh[sh.index("line=\"$(awk '/^drm-pdev:/"):]
    awk = awk[awk.index("'") + 1:]
    awk = awk[:awk.index("' \\")]
    fd = tmp_path / "fdinfo"
    for body, kib in (("drm-memory-vram:\t10737418240", 10485760), ("drm-memory-vram:\t10485760 KiB", 10485760),
                      ("drm-resident-vram0:\t10240 MiB", 10485760)):
        fd.write_text(f"drm-driver:\tamdgpu\ndrm-pdev:\t0000:03:00.0\ndrm-client-id:\t7\n{body}\n")
        r = subprocess.run(["awk", awk, str(fd)], capture_output=True, text=True, timeout=10)
        assert r.stdout.split() == ["0000:03:00.0", "7", str(kib)], (body, r.stdout, r.stderr)


def test_the_start_read_reaches_the_cap_the_rows_and_the_check(monkeypatch, tmp_path):
    """Plumbing: start.sh's CW_SPEC_VRAM_AVAIL_MB / _HELD_MB / CW_SPEC_RAM_AVAIL_MB -> specs -> the cap the worker
    uses, the page's rows and memory line; Check speed still reads Ollama's fit and reports the estimate beside it."""
    env = {"CW_SPEC_GPU_VENDOR": "nvidia", "CW_SPEC_VRAM_MB": "12282", "CW_SPEC_RAM_MB": "93000",
           "CW_SPEC_VRAM_AVAIL_MB": "9933", "CW_SPEC_VRAM_HELD_MB": "10010", "CW_SPEC_RAM_AVAIL_MB": "90700",
           "CW_HOST": "box"}
    sp = picker.specs_from_env(env)
    assert sp["vram_avail_mb"] == 9933 and sp["vram_held_mb"] == 10010 and sp["ram_avail_mb"] == 90700
    mem = picker.memory(sp)
    assert (mem["vram_shown"], mem["ram_shown"], mem["ram_total"], mem["vram_cold"]) == (9.7, 88.5, 90.8, False)
    log = str(tmp_path / "ollama.log")
    open(log, "w").close()
    fo = FakeOllama(monkeypatch)
    monkeypatch.setattr(aictl, "CHECK_BUDGET_S", 0)
    real_warm = fo.warm

    def fit_log_warm(model, timeout=600):
        real_warm(model, timeout)
        if llm.num_gpu(model) is None:
            open(log, "a").write("load_tensors: offloaded 27/61 layers to GPU\n")
    monkeypatch.setattr(llm, "warm", fit_log_warm)
    eng, hub, t = _hub(specs={**NV12, "vram_avail_mb": 9933, "vram_held_mb": 10010, "ram_avail_mb": 90700}, log=log)
    assert llm.num_gpu(M31) == 21                                   # no cap here: the estimate
    info = asyncio.run(hub.ai_info())
    assert info["memory"]["vram_shown"] == 9.7
    top = info["rows"][0]
    assert top["id"] == M31 and top["num_gpu"]["value"] == 21 and top["num_gpu"]["source"] == "estimated from 9.7 GB available"

    async def go():
        hub.start_speed({"model": M31})
        await hub.ai_task_done()
    asyncio.run(go())
    r = hub.ai_last["speed"]
    assert not r.get("error"), r
    assert r["fit"] == [27, 61] and r["num_gpu"] == 24 and r["estimate"]["num_gpu"] == 21
    assert any("Ollama's fit 27/61 layers, cap 24; the estimate from 9.7 GB available: 21" in n for n in r["notes"])
    assert hub.machine["caps"][M31] == {"value": 24, "tier": "A", "how": "found", "avail_gb": 9.7}
    assert hub.machine["speed"][M31]["estimate"]["num_gpu"] == 21


def _rows_for(specs):
    sp = {"host": "x", "cores": 8, "threads": 16, "gpu_name": None, **specs}
    cls = picker.classes(sp)
    inst = {s["id"] for s in picker.catalogue()["setups"]}
    mem = picker.memory(sp)
    singles = picker.rows(cls, sp["gpu_vendor"], inst, "0.40.11", None, mem)
    hyb = picker.hybrid_rows(cls, sp["gpu_vendor"], inst, None, singles, mem)
    s, h = picker.shown_rows(singles, hyb)
    return {r["id"]: r for r in s}, {x["id"]: x for x in h}


def test_free_ram_flags_a_model_that_does_not_fit():
    """Owner: "we need a separate system free memory read too ... when models dont fit entirely on cards or we are
    using no gpu memory at all". The part in RAM plus headroom (4 GB or 10% of the RAM) must fit in what is free."""
    assert picker.ram_need_gb(M31, {"known": True, "value": 26, "source": "x"}) == 10.4        # 35 of 61 layers in RAM
    assert picker.ram_need_gb(M31, {"known": True, "value": 30, "source": "x"}) == 9.4         # 31 of 61 (the text build)
    assert picker.ram_need_gb(M12, {"known": True, "value": None, "source": "no GPU"}) == 8.5  # the whole model
    assert picker.ram_need_gb(M12, {"known": True, "value": None, "source": "fits whole"}) == 1.0
    # no GPU, 16 GB, 8 GB free: the E4B + MTP (6.9 GB + 4) does not fit; it is shown, flagged, not recommended
    s, h = _rows_for({"gpu_vendor": "none", "vram_mb": 0, "ram_mb": 15800, "ram_avail_mb": 8192})
    assert list(s) == [E4B] and s[E4B]["ram"]["state"] == "short" and not s[E4B]["recommended"]
    # no GPU, 32 GB, 11 GB free: the larger 12B + MTP is flagged; the E4B + MTP (only just fits) is recommended instead
    s, h = _rows_for({"gpu_vendor": "none", "vram_mb": 0, "ram_mb": 31800, "ram_avail_mb": 11264})
    assert s[M12]["ram"]["state"] == "short" and not s[M12]["recommended"]
    assert s[E4B]["recommended"] and s[E4B]["ram"]["state"] == "tight"
    # the same laptop with room: the 12B + MTP as before
    s, h = _rows_for({"gpu_vendor": "none", "vram_mb": 0, "ram_mb": 31800, "ram_avail_mb": 26000})
    assert s[M12]["recommended"] and s[M12]["ram"]["state"] == "ok"
    # a 12 GB card, 32 GB RAM with 12 GB free: the 31B's 31 CPU layers (9.4 GB + 4) do not fit; the hybrid (the larger
    # of its two, they swap) neither; the 12B + MTP fits whole on the card and is recommended
    s, h = _rows_for({"gpu_vendor": "nvidia", "vram_mb": 12282, "ram_mb": 32000, "vram_avail_mb": 11935, **WARM,
                      "ram_avail_mb": 12288})
    assert s[M31]["ram"]["state"] == "short" and not s[M31]["recommended"]
    assert s[M12]["recommended"] and s[M12]["ram"]["state"] == "ok"
    hy = h[f"{M12}+{M31}"]
    assert hy["ram"]["state"] == "short" and hy["ram"]["need_gb"] == 9.4 and not hy["recommended"]
    # this machine (93 GB, 88.6 free): everything fits, the hybrid keeps both files cached for quick swaps
    s, h = _rows_for({**NV12, "vram_avail_mb": 11935, **WARM, "ram_avail_mb": 90700})
    assert s[M31]["recommended"] and all(r["ram"]["state"] == "ok" for r in s.values())
    assert h[f"{M12}+{M31}"]["ram"]["readahead"] and h[f"{M12}+{M31}"]["recommended"]
    # 30 GB free of 93: the hybrid runs but cannot also keep both files (~26 GB) in the page cache
    s, h = _rows_for({**NV12, "vram_avail_mb": 11935, **WARM, "ram_avail_mb": 30000})
    assert h[f"{M12}+{M31}"]["ram"]["state"] != "short" and not h[f"{M12}+{M31}"]["ram"]["readahead"]
    # not read (an old start script): no RAM verdict, as before
    s, h = _rows_for({**NV12})
    assert s[M31]["ram"] is None and s[M31]["recommended"]


def test_rereview_a_machines_cap_comes_down_along_its_own_line_only():
    """Re-review 2026-11-08: a cap this machine has already carries its margin; a drop lowers it by the drop over the GB
    per layer only (not the estimate's margin and cold reserve on top), and only past 0.4 GB (a desktop's VRAM moves
    0.2-0.4 GB routinely). Stored 26 at 11.655 GB: 11.45 stays 26, 11.25 -> 24 (a 0.405 GB drop is a bit more than one
    0.327 GB layer), 11.2 -> 24, the 2 GB desktop -> 19 (the text build's line, 4.9 GB over 15 layers)."""
    A = picker.tier({"vram": "12", "ram": "96+"}, "nvidia")
    g = 4.9 / 15
    for how in ("set", "found"):
        m = {"caps": {M31: picker.cap_entry(26, "A", how, A_GB)}}
        for held in (10010, 0):                                    # warm or cold: the cold reserve is not applied here
            assert picker.cap_for(M31, A, "nvidia", m, _mem(int(11.45 * 1024), held))["value"] == 26
            assert picker.cap_for(M31, A, "nvidia", m, _mem(int(11.25 * 1024), held))["value"] == 26 - math.ceil(0.405 / g) == 24
            assert picker.cap_for(M31, A, "nvidia", m, _mem(int(11.2 * 1024), held))["value"] == 26 - math.ceil(0.455 / g) == 24
            two = picker.cap_for(M31, A, "nvidia", m, _mem(int((A_GB - 2.0) * 1024), held))
            assert two["value"] == 26 - math.ceil(2.0 / g) == 19 and two["how"] == "lowered"
    legacy = {"caps": {M31: {"value": 30, "tier": "A"}}}             # by the measured point with that cap (30 on A)
    assert picker.cap_for(M31, A, "nvidia", legacy, _mem(int(11.25 * 1024)))["value"] == 28


# ------------------------------------------------------------------------------------------------ ramfit (2026-11-08)
def _list(cls, vendor="nvidia", mem=None):
    inst = {s["id"] for s in picker.catalogue()["setups"]}
    singles = picker.rows(cls, vendor, inst, "0.40.11", None, mem)
    hyb = picker.hybrid_rows(cls, vendor, inst, None, singles, mem)
    s, h = picker.shown_rows(singles, hyb)
    return {r["id"]: r for r in s}, [x["id"] for x in h]


def test_a_gpu_with_little_ram_lists_each_model_whose_ram_part_fits():
    """A GPU with less RAM than its tier was tested with used to fall to "no GPU" and hide the 12B + MTP, which fits whole
    on a 12/16 GB card (~1 GB of RAM) and at 36 layers on 8 GB (~3 GB). Now each model is listed where its own RAM part
    fits the class's nominal free RAM (or the free RAM read), with the GPU tier's speed as an estimate."""
    for v in ("12", "16+"):
        s, h = _list({"vram": v, "ram": "16"})
        assert list(s) == [M12] and not h and s[M12]["recommended"]
        assert s[M12]["quality"]["over_no_ai"] == 50 and s[M12]["speed"]["keepup_plus"] and s[M12]["speed"]["estimate"]
        assert s[M12]["ram"]["need_gb"] == 1.0
    s, h = _list({"vram": "8", "ram": "16"})
    assert list(s) == [M12] and not h and s[M12]["num_gpu"]["value"] == 36 and s[M12]["ram"]["need_gb"] == 2.8
    # the 31B and the hybrid still need 32 GB: their CPU part (~11-14 GB) plus headroom does not fit 16 GB
    for v in ("8", "12", "16+"):
        s, h = _list({"vram": v, "ram": "16"})
        assert M31 not in s and not h
        s, h = _list({"vram": v, "ram": "32"})
        assert M31 in s and h                                     # unchanged at the tested RAM
    # a GPU machine read under 16 GB (the lowest option, "16", owner 2026-11-09): its free RAM read decides
    s, h = _list({"vram": "8", "ram": "16"}, mem={"vram": None, "ram": 5.0, "ram_total": 8.0})
    assert M12 not in s and list(s) == [E4B] and s[E4B]["recommended"]   # the 12B: 2.8 GB + 4 headroom > 5 GB
    s, h = _list({"vram": "8", "ram": "16"}, mem={"vram": None, "ram": 9.0, "ram_total": 12.0})
    assert list(s)[0] == M12 and s[M12]["recommended"]                  # a 12 GB machine read with 9 GB free: it fits
    # a free RAM read replaces the class's nominal: 12 GB card, "16 GB" class with only 4 GB free -> nothing fits
    s, h = _list({"vram": "12", "ram": "16"}, mem={"vram": None, "ram": 4.0, "ram_total": 15.5})
    assert not s
    # no-GPU rows unchanged
    s, h = _list({"vram": "none", "ram": "16"}, "none")
    assert list(s) == [E4B] and s[E4B]["recommended"]
    # nothing fits on a GPU machine below its tier's RAM: only No AI, and the note says why
    note = picker.ram_fit_note({"vram": "12", "ram": "16"}, "nvidia", mem={"vram": None, "ram": 4.0, "ram_total": 15.5})
    assert note.startswith("nothing fits in the RAM: the smallest option") and "4 GB free" in note
    assert picker.ram_fit_note({"vram": "12", "ram": "16"}, "nvidia") is None
    assert picker.ram_fit_note({"vram": "none", "ram": "16"}, "none") is None   # the tested 16 GB laptop: not RAM-gated
    # every combination with something runnable has a recommendation (owner 2026-11-08)
    for v in ("none", "8", "12", "16+"):
        for r in picker.RAM_CHOICES:
            s, h = _list({"vram": v, "ram": r}, "none" if v == "none" else "nvidia")
            if s:
                assert sum(x["recommended"] for x in s.values()) == 1, (v, r)


# ------------------------------------------------------------------------------------------------ ramblock (2026-11-08)
def test_a_saved_choice_whose_ram_part_no_longer_fits_does_not_start_silently(monkeypatch):
    """As the unknown-cap case: the run starts with the AI off, the saved choice is kept, and the page says why and what
    fits; the hybrid's Review refuses with the same reason; a free-RAM read with room unblocks it at the next start."""
    FakeOllama(monkeypatch)
    r16 = {**NV12, "ram_mb": 15800}                                   # a 12 GB card, now 16 GB of RAM, no read
    eng, hub, t = _hub(specs=r16, choice={"mode": "single", "live": M31})
    want = ("Gemma 4 31B + MTP needs ~14 GB free RAM, 12 assumed free with 16 GB of RAM: pick another "
            "(recommended: Gemma 4 12B + MTP) or free memory")
    assert hub.ai_off and hub.ai_status()["blocked"] == want and not eng.person_review
    assert hub.machine["choice"] == {"mode": "single", "live": M31}           # kept
    eng, hub, t = _hub(specs={**r16, "ram_avail_mb": 11264}, choice={"mode": "single", "live": M31})   # a read: its figure
    assert hub.ai_off and hub.ai_status()["blocked"].startswith("Gemma 4 31B + MTP needs ~14 GB free RAM, 11 available:")
    eng, hub, t = _hub(specs={**r16, "ram_avail_mb": 20000}, choice={"mode": "single", "live": M31})   # room: it starts
    assert not hub.ai_off and hub.ai_status()["blocked"] is None and llm.num_gpu(M31) == 30
    # the hybrid saved on 16 GB: the 31B reviewer does not fit, so it does not start; Review refuses with the reason
    eng, hub, t = _hub(specs=r16, choice=HYB)
    assert hub.ai_off and hub.ai_status()["blocked"] == want
    with pytest.raises(aictl.Busy, match="AI off for this run: Gemma 4 31B \\+ MTP needs ~14 GB free RAM"):
        hub.start_review({})
    # a hybrid running when the RAM read now says no room for the reviewer: Review refuses before any swap
    fo, eng, hub, t = _hybrid(monkeypatch)
    hub.specs = {**hub.specs, "ram_avail_mb": 14000}                 # 13.6 GB free of 93: the 12B fits, the 31B not
    _chatters(hub, eng, t)
    with pytest.raises(aictl.Busy, match="Gemma 4 31B \\+ MTP needs ~20 GB free RAM, 13.6 available"):
        hub.start_review({})
    # nothing fits at all: the reason says so; No AI saved is never blocked
    eng, hub, t = _hub(specs={**r16, "ram_avail_mb": 3000}, choice={"mode": "single", "live": M12})
    assert hub.ai_off and "(recommended: No AI)" in hub.ai_status()["blocked"]
    eng, hub, t = _hub(specs={**r16, "ram_avail_mb": 3000}, choice={"mode": "none"})
    assert hub.ai_status()["blocked"] is None


# ------------------------------------------------------------------------------------------------ re-read reach (sweep 2026-11-09)
def test_the_page_shows_how_far_the_re_reads_reach_beside_the_keep_up(monkeypatch):
    """ASSESSMENT_20261109 sections 3 and 6: the keep-up (~40 for the 31B) counts person reads and flagged lines only; the
    low-priority re-reads thin from about 20 (live_quality_now section 4: 98% at 20, 80% at 40). Each row carries
    "reread_all" (95% of queued lines re-read) and "reread_most" (80%: the switch to the hybrid); a hybrid's live model
    its figure with review presses; a speed check reads both off the same grid as its keep-up."""
    cat = picker.catalogue()
    a = picker.setup(M31)["tiers"]["A"]
    assert (a["keepup"], a["reread_all"], a["reread_most"]) == (40, 20, 40) and a["reread_source"]
    assert (picker.setup(M31)["tiers"]["E"]["reread_all"], picker.setup(M31)["tiers"]["E"]["reread_most"]) == (15, 25)
    assert picker.setup(M12)["tiers"]["A"]["reread_all"] == 200
    h = next(x for x in cat["hybrids"] if x["live"] == M12)
    assert h["live_reread"]["A"] == {"all": 60, "most": 75} and h["live_reread"]["E"] == {"all": 40, "most": 50}
    rows = picker.rows(picker.classes(NV12), "nvidia", {M31, M12}, "0.40.11")
    r31 = next(r for r in rows if r["id"] == M31)
    assert (r31["speed"]["keepup"], r31["speed"]["reread_all"], r31["speed"]["reread_most"]) == (40, 20, 40)
    hy = picker.hybrid_rows(picker.classes(NV12), "nvidia", {M31, M12}, singles=rows)[0]
    assert hy["live_speed"]["reread_all"] == 60 and hy["live_speed"]["keepup_plus"]
    # a speed check: the same grid as its keep-up (within one display step of the setup's own figure)
    s = picker.summarize([6.3] * 9, [2.7, 2.7], [18.4, 18.4])
    assert s["keepup"] == 40 and 15 <= s["reread_all"] <= 20 and 35 <= s["reread_most"] <= 40
    rr = picker.reread_from(1.02, 0.23 + 12 * 0.201)
    assert rr["all"] >= 100 and rr["most"] >= 150                     # the 12B + MTP: re-reads everything far out
    # a check stored before the table gets its rates from its own service times
    m = {"speed": {M31: {"keepup": 30, "person_s": 7.09, "a": 1.5, "b": 1.7, "at": "2026-11-08T01:24:30Z"}}}
    sp = next(r for r in picker.rows(picker.classes(NV12), "nvidia", {M31}, "0.40.11", m) if r["id"] == M31)["speed"]
    assert sp["measured"] and sp["reread_all"] is not None and sp["reread_all"] <= 20


# ------------------------------------------------------------------------------------------------ T5 (sweep 2026-11-09)
def test_t5_the_session_log_records_the_ai_the_run_uses(monkeypatch, tmp_path):
    """meta.json recorded CW_LLM_MODEL from the environment; it now records the page's choice: the model the worker
    calls (llm.MODEL), the mode, the reviewer and their caps, and why the AI is off when a saved choice could not run."""
    from chatwatch import session
    FakeOllama(monkeypatch)
    monkeypatch.setenv("CW_LLM_MODEL", "gpt-oss-safeguard:20b")          # gpu.env's old model: not what runs
    monkeypatch.setattr(session.Session, "_prepare", lambda self: None)
    eng, hub, t = _hub(specs={**NV12, "vram_avail_mb": 11935, "vram_held_mb": 10010, "ram_avail_mb": 90700})
    hub.apply_choice({"mode": "hybrid", "live": M12, "reviewer": M31})
    a = hub.session_ai()
    assert (a["mode"], a["live"], a["reviewer"]) == ("hybrid", M12, M31) and a["num_gpu"] == {M12: None, M31: 30}
    s = session.Session(["cozy"], ["x"], eng, base=str(tmp_path), ai=a)
    s.close(dict(eng.stats))
    meta = json.load(open(os.path.join(s.dir, "meta.json")))
    assert meta["llm_model"] == M12 and meta["ai"] == json.loads(json.dumps(a))
    hub.apply_choice({"mode": "single", "live": M31})
    assert hub.session_ai() == {"mode": "single", "live": M31, "reviewer": None, "num_gpu": {M31: 30}}
    hub.apply_choice({"mode": "none"})
    assert hub.session_ai()["mode"] == "none" and hub.session_ai()["live"] is None
    hub.ai_blocked = "a reason"
    assert hub.session_ai()["blocked"] == "a reason"
    # Start passes it (the server builds the session there); an old caller keeps the environment's model
    assert "ai=self.session_ai()" in open(serve.__file__).read()
    s2 = session.Session(["cozy"], ["x"], eng, base=str(tmp_path / "old"))
    s2.close(dict(eng.stats))
    assert json.load(open(os.path.join(s2.dir, "meta.json")))["llm_model"] == "gpt-oss-safeguard:20b"


def test_amdfit_a_fit_load_out_of_gpu_memory_steps_down_until_a_load_holds(monkeypatch, tmp_path):
    """The AMD desktop 2026-11-10: Ollama's own fit for the text 31B (47/61) left out the MTP drafter and the uncapped
    load failed with HTTP 500 (out of GPU memory); Check Speed stopped there. Now the fit is read from the log anyway and
    the capped loads step down 2 layers at a time until one holds."""
    log = str(tmp_path / "ollama.log")
    open(log, "w").close()
    fo = FakeOllama(monkeypatch, log=log, fit={"gemma4mtp:31b-n3-text": (47, 61)})
    fo.oom_above["gemma4mtp:31b-n3-text"] = 42                    # 47 and 44 run out of memory, 42 loads
    monkeypatch.setattr(aictl, "CHECK_BUDGET_S", 0)
    eng, hub, t = _hub(specs=AMD16, log=log)
    hub.ai = {"mode": "single", "live": "gemma4mtp:31b-n3-text", "reviewer": None}

    async def go():
        hub.start_speed({"model": "gemma4mtp:31b-n3-text"})
        await hub.ai_task_done()
    asyncio.run(go())
    r = hub.ai_last["speed"]
    assert not r.get("error"), r
    assert r["fit"] == [47, 61] and r["num_gpu"] == 42 and r["aborts"] == 0
    assert any("ran out of GPU memory" in n for n in r["notes"]) and any("stepping down to 42" in n for n in r["notes"])
    assert picker.cap_value(hub.machine["caps"]["gemma4mtp:31b-n3-text"]) == 42
    assert [c[2] for c in fo.calls if c[0] == "warm" and c[1] == "gemma4mtp:31b-n3-text"][-1] == 42


def test_amdfit_a_model_that_never_loads_fails_the_check_and_keeps_the_old_cap(monkeypatch, tmp_path):
    log = str(tmp_path / "ollama.log")
    open(log, "w").close()
    fo = FakeOllama(monkeypatch, log=log, fit={"gemma4mtp:31b-n3-text": (47, 61)})
    fo.oom_above["gemma4mtp:31b-n3-text"] = 1                     # nothing fits
    monkeypatch.setattr(aictl, "CHECK_BUDGET_S", 0)
    eng, hub, t = _hub(specs=AMD16, log=log)
    hub.ai = {"mode": "single", "live": "gemma4mtp:12b-n2", "reviewer": None}
    before = dict(hub.machine.get("caps") or {})

    async def go():
        hub.start_speed({"model": "gemma4mtp:31b-n3-text"})
        await hub.ai_task_done()
    asyncio.run(go())
    r = hub.ai_last["speed"]
    assert "could not load gemma4mtp:31b-n3-text" in (r.get("error") or ""), r
    assert (hub.machine.get("caps") or {}) == before and "gemma4mtp:31b-n3-text" not in (hub.machine.get("speed") or {})
