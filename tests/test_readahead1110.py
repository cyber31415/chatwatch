"""The hybrid's read-ahead and the Review box's swap estimate (2026-11-10, BASELINE 34, data/study/readahead/RESULT.md).
Measured on this machine's CIFS store: a blob's cached pages go when its last open handle closes, so the files are held
open first; the old read-ahead (cat, then open) kept nothing of the 31B and slowed the live 12B ~100x while it read at
line rate, so only the reviewer's files are read, paced, and paused while a model loads. The estimate never shows a
swap shorter than this machine's newest measurement of the same two models."""
import asyncio
import json
import os
import time

from chatwatch import aictl, picker

from test_picker import _hybrid, _machines  # noqa: F401  (_machines: autouse)

M31, M12 = "gemma4mtp:31b-n3-text", "gemma4mtp:12b-n2"
ROOT = os.path.join(os.path.dirname(__file__), "..")


def _store(monkeypatch, tmp_path, size=2_000_000):
    root = tmp_path / "ollama"
    blobs = root / "models" / "blobs"
    blobs.mkdir(parents=True)
    for tag, d in ((M31, "a" * 64), (M12, "b" * 64)):
        name, t = tag.split(":")
        man = root / "models" / "manifests-v2" / "ollama.com" / "library" / name / t
        man.parent.mkdir(parents=True, exist_ok=True)
        (blobs / f"sha256-{d}").write_bytes(b"x" * size)
        man.write_text(json.dumps({"layers": [{"digest": f"sha256:{d}", "size": size}]}))
    monkeypatch.setenv("CW_OLLAMA_MODELS", str(root))
    return {M31: str(blobs / ("sha256-" + "a" * 64)), M12: str(blobs / ("sha256-" + "b" * 64))}


def _reads(monkeypatch, hub):
    """Every pread: (the file read, were all of the hybrid's files of that model already held open then)."""
    seen, real = [], os.pread

    def pread(fd, n, off):
        path = os.readlink(f"/proc/self/fd/{fd}")
        seen.append((path, fd in sum(hub._pf_fds.values(), [])))
        return real(fd, n, off)
    monkeypatch.setattr(aictl.os, "pread", pread)
    return seen


def _loop_once(hub):
    """One pass of prefetch_loop (it then waits PREFETCH_EVERY_S)."""
    async def go():
        task = asyncio.create_task(hub.prefetch_loop())
        for _ in range(500):
            await asyncio.sleep(0.01)
            if hub.prefetch_summary()["state"] == "ready":
                break
        task.cancel()
    asyncio.run(go())


# ------------------------------------------------------------------------------------------------ the read-ahead
def test_ra1_both_models_held_open_first_and_only_the_reviewer_is_read(monkeypatch, tmp_path):
    paths = _store(monkeypatch, tmp_path)
    fo, eng, hub, t = _hybrid(monkeypatch)
    seen = _reads(monkeypatch, hub)
    _loop_once(hub)
    assert hub.prefetch_summary()["state"] == "ready"
    assert set(hub._pf_fds) == {M31, M12}                                    # both held
    assert seen and {p for p, _ in seen} == {paths[M31]}                     # never the live model's file
    assert all(held for _, held in seen)                                     # read through a handle already held
    assert hub.prefetch[M31]["done"] == 2_000_000 and hub.prefetch[M12]["total"] == 0


def test_ra2_a_reread_keeps_the_same_handles_and_leaving_the_hybrid_closes_them(monkeypatch, tmp_path):
    _store(monkeypatch, tmp_path)
    fo, eng, hub, t = _hybrid(monkeypatch)
    _loop_once(hub)
    fds = {k: list(v) for k, v in hub._pf_fds.items()}

    async def again():
        await hub._prefetch(M31)
    asyncio.run(again())
    assert hub._pf_fds == fds                       # no close and reopen (a close would drop the cached pages)
    hub.ai = {"mode": "single", "live": M12, "reviewer": None}
    hub._prefetch_letgo()
    assert hub._pf_fds == {} and hub.prefetch == {}
    for fd in sum(fds.values(), []):
        try:
            os.fstat(fd)
            raise AssertionError(f"fd {fd} still open")
        except OSError:
            pass


def test_ra3_a_read_from_the_share_is_paced_and_one_from_memory_is_not(monkeypatch, tmp_path):
    _store(monkeypatch, tmp_path)
    fo, eng, hub, t = _hybrid(monkeypatch)
    monkeypatch.setattr(aictl, "PREFETCH_CHUNK", 100_000)
    monkeypatch.setattr(aictl, "PREFETCH_MB_S", 10)            # 20 chunks of 0.1 MB at 10 MB/s: 0.2 s
    monkeypatch.setattr(aictl, "PREFETCH_MEM_S", -1)           # every chunk counts as from the share

    def timed():
        t0 = time.time()
        asyncio.run(hub._prefetch(M31))
        return time.time() - t0
    assert timed() >= 0.18
    monkeypatch.setattr(aictl, "PREFETCH_MEM_S", 10)           # every chunk counts as from memory
    assert timed() < 0.15
    assert hub.prefetch[M31]["state"] == "ready"


def test_ra4_the_read_waits_while_a_model_loads(monkeypatch, tmp_path):
    _store(monkeypatch, tmp_path)
    fo, eng, hub, t = _hybrid(monkeypatch)
    hub.hold_llm("review")                                     # a review swap is loading a model

    async def go():
        task = asyncio.create_task(hub._prefetch(M31))
        await asyncio.sleep(0.3)
        assert hub.prefetch[M31]["state"] == "reading" and hub.prefetch[M31]["done"] == 0
        hub.release_llm("review")
        await asyncio.wait_for(task, 5)
    asyncio.run(go())
    assert hub.prefetch[M31]["state"] == "ready" and hub.prefetch[M31]["done"] == 2_000_000


def test_ra5_read_off_holds_only(monkeypatch, tmp_path):
    _store(monkeypatch, tmp_path)
    fo, eng, hub, t = _hybrid(monkeypatch)
    monkeypatch.setattr(aictl, "PREFETCH_READ", False)
    seen = _reads(monkeypatch, hub)
    _loop_once(hub)
    assert seen == [] and set(hub._pf_fds) == {M31, M12}
    # the reviewer is held but not read: not "in memory" (the page keeps saying the first swap is slow)
    assert hub.prefetch[M31]["state"] == "held" and hub.prefetch[M12]["state"] == "ready"
    assert hub.prefetch_summary()["state"] == "cold"


def test_ra5b_while_the_reviewer_is_read_the_estimate_counts_its_load_from_the_share(monkeypatch, tmp_path):
    _store(monkeypatch, tmp_path)
    fo, eng, hub, t = _hybrid(monkeypatch)
    hub.machine["loads"] = {M31: {"s": 11.3, "at": "2026-11-10T16:06:27Z"}, M12: {"s": 4.4, "at": "2026-11-10T16:12:27Z"}}
    # 18.6 GB being read, the share measured at 117 MB/s: ~159 s for the reviewer, plus the live model's 4.4 s
    hub.prefetch = {M12: {"state": "ready", "total": 0, "at": "x"},
                    M31: {"state": "reading", "total": 18_605_861_728, "done": 1e9, "share_mb_s": 117.0}}
    est = hub._swap_estimate()
    # only the unread 17.6 GB comes from the share: 150.4 + 4.4 + 2
    assert est == {"s": 157, "label": "reviewer from disk at 117 MB/s"}
    hub.prefetch[M31].update(state="ready", at="y")            # read: the measurements again
    assert hub._swap_estimate()["s"] == 18
    # the page's cold line treats it as a measured figure
    page = open(os.path.join(ROOT, "web", "index.html")).read()
    assert "|reviewer from disk)/.test(sw.label" in page


def test_ra5c_the_read_measures_the_share_speed_without_its_pauses(monkeypatch, tmp_path):
    _store(monkeypatch, tmp_path)
    fo, eng, hub, t = _hybrid(monkeypatch)
    monkeypatch.setattr(aictl, "PREFETCH_CHUNK", 100_000)
    monkeypatch.setattr(aictl, "PREFETCH_MB_S", 1000)
    monkeypatch.setattr(aictl, "PREFETCH_MEM_S", -1)           # every chunk counts as from the share
    real = os.pread

    def slow(fd, n, off):                                      # 0.1 MB in ~0.05 s: about 2 MB/s
        time.sleep(0.05)
        return real(fd, n, off)
    monkeypatch.setattr(aictl.os, "pread", slow)
    asyncio.run(hub._prefetch(M31))
    assert 1.0 <= hub.prefetch[M31]["share_mb_s"] <= 2.1


def test_ra6_no_cat_subprocess_is_left():
    src = open(os.path.join(ROOT, "src", "chatwatch", "aictl.py")).read()
    assert "create_subprocess_exec" not in src and '"cat"' not in src


# ------------------------------------------------------------------------------------------------ the swap estimate
def test_ra7_the_estimate_is_never_shorter_than_the_newest_measurement_of_these_models(monkeypatch):
    fo, eng, hub, t = _hybrid(monkeypatch)
    # 2026-11-10 as it was: the 8 s cached swap of the projector build (no models in the record), today's loads
    hub.machine.update({
        "swap_cached": {"swap_s": 8, "at": "2026-11-08T01:34:40Z"},
        "swap": {"swap_s": 5, "at": "2026-11-08T01:00:00Z"},
        "loads": {M31: {"s": 11.3, "at": "2026-11-10T16:06:27Z"}, M12: {"s": 4.4, "at": "2026-11-10T16:12:27Z"}}})
    est = hub._swap_estimate()
    assert est["s"] == 18 and est["label"] == "last loads here 2026-11-10"      # the old records do not count
    # the batch at 17:06 (load 153.9 s): longer than the loads, so it counts
    hub.machine[aictl.SWAP_KEY] = {"swap_s": 156, "at": "2026-11-10T17:06:36Z", "live": M12, "reviewer": M31}
    assert hub._swap_estimate() == {"s": 156, "label": "last review here 2026-11-10"}
    # another pair's batch (the projector build) does not stand in
    hub.machine[aictl.SWAP_KEY] = {"swap_s": 7, "at": "2026-11-10T18:00:00Z", "live": M12,
                                   "reviewer": "gemma4mtp:31b-n3"}
    assert hub._swap_estimate()["s"] == 18
    # loads of only one of the two: the batch alone, else the study's figure
    hub.machine["loads"] = {M31: {"s": 11.3, "at": "2026-11-10T16:06:27Z"}}
    hub.machine.pop(aictl.SWAP_KEY)
    assert hub._swap_estimate()["label"] == "tested on 12 GB Nvidia"


def test_ra7b_a_cached_swap_and_a_cold_one_never_stand_in_for_each_other(monkeypatch):
    fo, eng, hub, t = _hybrid(monkeypatch)
    loads = {M31: {"s": 11.3, "at": "2026-11-10T16:06:27Z"}, M12: {"s": 4.4, "at": "2026-11-10T16:12:27Z"}}
    cold = {"swap_s": 236, "at": "2026-11-10T19:00:00Z", "live": M12, "reviewer": M31, "read_ahead": False}
    hot = {"swap_s": 9, "at": "2026-11-10T20:00:00Z", "live": M12, "reviewer": M31, "read_ahead": True}
    ready = {M12: {"state": "ready", "total": 0, "at": "x"}, M31: {"state": "ready", "total": 1, "at": "x"}}
    # a batch pressed mid-read took 236 s; the reviewer is now in memory: the next swap is not that one
    hub.machine.update({"loads": loads, aictl.SWAP_KEY: cold})
    hub.prefetch = dict(ready)
    assert hub._swap_estimate() == {"s": 18, "label": "last loads here 2026-11-10", "cached": True}
    hub.machine[aictl.SWAP_KEY] = hot                          # a batch that started in memory: that one
    assert hub._swap_estimate() == {"s": 9, "label": "last review here 2026-11-10", "cached": True}
    # the reviewer not in memory (held only): the 9 s cached swap is not the estimate, the loads are
    hub.prefetch = {M12: ready[M12], M31: {"state": "held", "total": 0}}
    assert hub._swap_estimate()["s"] == 18
    hub.machine[aictl.SWAP_KEY] = cold
    assert hub._swap_estimate()["s"] == 236


def test_ra7c_a_reread_of_a_reviewer_in_memory_stays_ready_unless_it_reads_the_share(monkeypatch, tmp_path):
    _store(monkeypatch, tmp_path)
    fo, eng, hub, t = _hybrid(monkeypatch)
    monkeypatch.setattr(aictl, "PREFETCH_CHUNK", 100_000)
    monkeypatch.setattr(aictl, "PREFETCH_MB_S", 1000)
    asyncio.run(hub._prefetch(M12, read=False))
    monkeypatch.setattr(aictl, "PREFETCH_MEM_S", 10)           # every chunk from memory
    asyncio.run(hub._prefetch(M31))
    states, real = [], os.pread

    def watch(fd, n, off):
        states.append(hub.prefetch_summary()["state"])
        return real(fd, n, off)
    monkeypatch.setattr(aictl.os, "pread", watch)
    asyncio.run(hub._prefetch(M31))                            # the 20-minute re-read, all from memory
    assert states and set(states) == {"ready"}
    states.clear()
    monkeypatch.setattr(aictl, "PREFETCH_MEM_S", -1)           # its pages were dropped: from the share
    asyncio.run(hub._prefetch(M31))
    assert states[0] == "ready" and set(states[1:]) == {"reading"}
    assert hub.prefetch_summary()["state"] == "ready"


def test_ra8_a_batch_on_a_busy_machine_is_recorded_for_the_estimate(monkeypatch):
    from test_picker import _chatters
    fo, eng, hub, t = _hybrid(monkeypatch)
    _chatters(hub, eng, t)
    monkeypatch.setattr(aictl, "_loadavg", lambda: 9.0)        # not quiet (the 17:04 batch: 3.68 > 3.2)
    hub.machine["loads"] = {M31: {"s": 11.3, "at": "2026-11-10T16:06:27Z"}, M12: {"s": 4.4, "at": "2026-11-10T16:12:27Z"}}

    async def go():
        hub.start_review({})
        await hub.ai_task_done()
    asyncio.run(go())
    r = hub.ai_last["review"]
    assert not r.get("error"), r
    rec = hub.machine[aictl.SWAP_KEY]
    assert rec["quiet"] is False and rec["live"] == M12 and rec["reviewer"] == M31 and "load_reviewer_s" in rec
    # a quiet-only figure is not touched by it (Check speed's loads keep their rule)
    assert hub.machine["loads"][M31]["s"] == 11.3
    saved = picker.load_machine(hub.host, hub.specs)
    assert saved[aictl.SWAP_KEY]["reviewer"] == M31
