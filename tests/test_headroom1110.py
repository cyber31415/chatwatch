"""Room for the rest of the machine (owner 2026-11-10: a 16 GB AMD card went sluggish during a review with a Twitch stream
open): the AI's CPU part leaves CW_KEEP_CORES cores free (CW_LLM_NUM_THREAD, sent with every load and call), and on a card
that drives a screen a model that fills the card leaves picker.DESKTOP_RESERVE_GB of the VRAM free; a whole model stays
whole."""
import asyncio
import os

from chatwatch import aictl, llm, picker

from test_picker import AMD16, FakeOllama, _hub, _machines  # noqa: F401  (_machines: autouse)

ROOT = os.path.join(os.path.dirname(__file__), "..")
M31, M12 = "gemma4mtp:31b-n3-text", "gemma4mtp:12b-n2"
AMD_SCREEN = {**AMD16, "vram_avail_mb": 15360, "gpu_display": True}      # 15.0 GB available, a screen on it


# ------------------------------------------------------------------------------------------------ CPU threads
def test_hr1_the_thread_count_goes_with_every_load_and_call(monkeypatch):
    monkeypatch.setenv("CW_LLM_NUM_THREAD", "6")
    assert llm._load_options(M31)["num_thread"] == 6
    assert llm._tune({"model": M31, "messages": [], "options": {}}, M31)["options"]["num_thread"] == 6
    assert llm._tune({"model": M31, "messages": [], "options": {"temperature": 0}}, M31, sampling=False)["options"]["num_thread"] == 6
    monkeypatch.delenv("CW_LLM_NUM_THREAD")
    assert "num_thread" not in llm._load_options(M31)                     # unset: Ollama's own (every core)
    assert "num_thread" not in llm._tune({"model": M31, "messages": [], "options": {}}, M31)["options"]


def test_hr2_the_launchers_keep_two_cores_and_tell_the_screen():
    live = open(os.path.join(ROOT, "start.sh")).read()
    assert 'KEEP="${CW_KEEP_CORES:-2}"' in live and "CW_LLM_NUM_THREAD=$(( CW_SPEC_CORES > KEEP" in live
    assert "drm_screen()" in live and "CW_SPEC_GPU_DISPLAY" in live
    assert 'NT="${CW_LLM_NUM_THREAD:-}"' in open(os.path.join(ROOT, "bin", "ollama.sh")).read()   # the warm load too
    assert "CW_LLM_NUM_THREAD" in open(os.path.join(ROOT, "bin", "cw")).read()


# ------------------------------------------------------------------------------------------------ the desktop's VRAM
def test_hr3_the_reserve_applies_only_to_a_card_with_a_screen():
    assert picker.specs_from_env({"CW_SPEC_GPU_DISPLAY": "1"})["gpu_display"] is True
    assert picker.specs_from_env({"CW_SPEC_GPU_DISPLAY": "0"})["gpu_display"] is False
    assert picker.specs_from_env({})["gpu_display"] is None
    assert picker.memory(AMD_SCREEN)["reserve_gb"] == 1.5
    assert picker.memory({**AMD_SCREEN, "gpu_display": False})["reserve_gb"] == 0
    assert picker.memory({**AMD_SCREEN, "gpu_display": None})["reserve_gb"] == 0


def test_hr4_a_found_cap_that_fills_the_card_is_lowered_by_the_reserve_and_a_whole_model_is_not():
    tr = picker.tier(picker.classes(AMD_SCREEN), "amd")
    machine = {"caps": {M31: picker.cap_entry(44, tr["key"], "found", 14.735),       # a 16 GB AMD card's, 2026-11-10
                        M12: picker.cap_entry(49, tr["key"], "found", 14.452)}}
    c = picker.cap_for(M31, tr, "amd", machine, picker.memory(AMD_SCREEN))
    # 15.0 GB now, 1.5 kept: 13.5 usable against 14.735 when found -> ceil(1.235 / 0.327) = 4 layers fewer
    assert c["value"] == 40 and c["how"] == "lowered" and "1.5 GB kept for the desktop" in c["source"]
    assert picker.cap_for(M31, tr, "amd", machine, picker.memory({**AMD_SCREEN, "gpu_display": False}))["value"] == 44
    assert picker.cap_for(M12, tr, "amd", machine, picker.memory(AMD_SCREEN))["value"] == 49     # whole stays whole


def test_hr5_a_cap_found_with_the_reserve_is_not_lowered_twice():
    tr = picker.tier(picker.classes(AMD_SCREEN), "amd")
    machine = {"caps": {M31: picker.cap_entry(39, tr["key"], "found", 15.0, 1.5)}}
    assert picker.cap_for(M31, tr, "amd", machine, picker.memory(AMD_SCREEN))["value"] == 39


def test_hr6_the_estimate_keeps_the_reserve_unless_the_model_fits_whole():
    tr = picker.tier(picker.classes(AMD_SCREEN), "amd")
    with_screen = picker.cap_for(M31, tr, "amd", {}, picker.memory(AMD_SCREEN))
    without = picker.cap_for(M31, tr, "amd", {}, picker.memory({**AMD_SCREEN, "gpu_display": False}))
    assert with_screen["value"] == picker.estimate_cap(M31, 13.5, True)["value"] < without["value"]     # a cold read (nothing held)
    assert "1.5 GB kept for the desktop" in with_screen["source"]
    assert picker.cap_for(M12, tr, "amd", {}, picker.memory(AMD_SCREEN))["value"] is None        # the 12B: whole


def test_hr7_check_speed_takes_the_reserve_off_ollamas_fit(monkeypatch, tmp_path):
    log = str(tmp_path / "ollama.log")
    open(log, "w").close()
    fo = FakeOllama(monkeypatch, log=log, fit={M31: (47, 61)})
    monkeypatch.setattr(aictl, "CHECK_BUDGET_S", 0)
    eng, hub, t = _hub(specs=AMD_SCREEN, log=log)
    hub.ai = {"mode": "single", "live": M31, "reviewer": None}

    async def go():
        hub.start_speed({"model": M31})
        await hub.ai_task_done()
    asyncio.run(go())
    r = hub.ai_last["speed"]
    assert not r.get("error"), r
    assert r["fit"] == [47, 61] and r["num_gpu"] == 39                # 47 - 3, then 5 layers (1.5 GB / 0.327) fewer
    assert any("so 1.5 GB of the GPU stays free for the desktop" in n for n in r["notes"])
    e = hub.machine["caps"][M31]
    assert e["value"] == 39 and e["reserve_gb"] == 1.5
    assert picker.cap_for(M31, hub._tier(), "amd", hub.machine, hub._mem())["value"] == 39       # and it stands
