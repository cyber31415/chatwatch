"""The text-only 31B in the live tool (owner 2026-11-09; data/study/quant/CONCLUSION.md "The text-only 31B", FACTS.md
section 4): gemma4mtp:31b-n3-text replaces gemma4mtp:31b-n3 everywhere the tool picks a 31B (the live default, the
hybrid's reviewer, the picker), a choice saved with the old id resolves to the new build, and a cap stored for the old
build is carried as a floor. The old tag itself stays installed and unchanged (the teacher study uses it)."""
import asyncio
import json
import os
import re

import pytest

from chatwatch import aictl, llm, picker

from test_llm_models import _bodies
from test_picker import AMD16, NV12, FakeOllama, _hub

OLD, NEW, M12 = "gemma4mtp:31b-n3", "gemma4mtp:31b-n3-text", "gemma4mtp:12b-n2"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A_GB = 11935 / 1024


@pytest.fixture(autouse=True)
def _machines(tmp_path, monkeypatch):
    monkeypatch.setenv("CW_MACHINES_DIR", str(tmp_path / "machines"))
    monkeypatch.setattr(aictl, "_loadavg", lambda: 0.5)
    monkeypatch.setattr(llm, "NUM_GPU_BY_MODEL", {})
    monkeypatch.setattr(llm, "NUM_GPU", "")
    monkeypatch.setattr(llm, "MODEL", NEW)


def _mem(mb, held=10010, ram=90700, specs=NV12):
    return picker.memory({**specs, "vram_avail_mb": mb, "vram_held_mb": held, "ram_avail_mb": ram})


def _write_machine(tmp_path, host, m):
    d = tmp_path / "machines"
    d.mkdir(exist_ok=True)
    (d / f"{host}.json").write_text(json.dumps(m))


# ------------------------------------------------------------------------------------------------ the catalogue
def test_the_catalogue_lists_the_text_build_in_place_of_the_projector_build():
    ids = [s["id"] for s in picker.catalogue()["setups"]]
    assert NEW in ids and OLD not in ids
    assert picker.resolve(OLD) == NEW and picker.resolve(NEW) == NEW and picker.resolve(M12) == M12
    s = picker.setup(NEW)
    assert s["mtp"] and s["parent"] == "gemma4:31b-it-qat" and s["modelfile"] == "Modelfile.31b.n3-text"
    # quality and gates carry over (same weights, drafter, prompts): HARDWARE_TIERS section 1's figures, cited
    assert s["quality"]["pct"] == 117.3 and s["gates"] == {"severe": "pass", "checklist": 0.983}
    assert s["quality"]["source"]
    # tiers A and E: the text build's own quiet-window runs, at 30 and 15 layers
    a, e = s["tiers"]["A"], s["tiers"]["E"]
    assert (a["run"], a["num_gpu"], a["person_s"], a["window12_s"], a["keepup"]) == ("q31t_mtp3:QA", 30, 6.32, 18.4, 40)
    assert (e["run"], e["num_gpu"], e["person_s"], e["keepup"]) == ("q31t_mtp3_e:E", 15, 8.55, 20)
    assert "build" not in a and "build" not in e
    assert s["vram_fit"]["points"] == [[6.755, 15, 16], [11.655, 30, 33]]
    # every hybrid reviews with the text build; the effective-hybrid table names the same pair
    cat = picker.catalogue()
    assert {h["reviewer"] for h in cat["hybrids"]} == {NEW}
    assert cat["hybrid_effective"]["pair"] == [M12, NEW]
    # the 12B's one-point estimate is unchanged by the move (the overhead ratio is pinned to the projector build's)
    g12, how = picker.gb_per_layer(M12)
    assert how.startswith("weights per layer x 1.55") and 0.2 < g12 < 0.23


def test_the_page_says_where_each_tier_was_measured():
    row = next(r for r in picker.rows({"vram": "8", "ram": "32"}, "nvidia", {NEW}, "0.40.11") if r["id"] == NEW)
    assert row["speed"]["label"] == "tested on 8 GB Nvidia" and row["num_gpu"]["value"] == 15
    row = picker.rows(picker.classes(NV12), "nvidia", {NEW}, "0.40.11")[0]
    assert row["id"] == NEW and row["speed"]["label"] == "tested on 12 GB Nvidia" and row["num_gpu"]["value"] == 30


# ------------------------------------------------------------------------------------------------ saved choices
def test_an_old_machine_file_resolves_to_the_text_build_and_carries_its_caps():
    saved = {"choice": {"mode": "single", "live": OLD}, "caps": {OLD: {"value": 26, "tier": "A"}},
            "speed": {OLD: {"keepup": 30}}}
    m, changed = picker.migrate_machine(saved)
    assert changed and m["choice"] == {"mode": "single", "live": NEW}
    assert m["caps"][NEW] == {"value": 26, "tier": "A", "how": "carried", "from": OLD}
    assert m["caps"][OLD] == {"value": 26, "tier": "A"} and NEW not in m["speed"]     # the old build's numbers stay its own
    assert picker.migrate_machine(m) == (m, False)                                   # once
    desk = {"choice": {"mode": "hybrid", "live": M12, "reviewer": OLD},
            "caps": {OLD: {"value": 40, "tier": "A", "how": "found", "avail_gb": 14.358}, M12: {"value": 49, "tier": "A"}}}
    m, changed = picker.migrate_machine(desk)
    assert changed and m["choice"] == {"mode": "hybrid", "live": M12, "reviewer": NEW}
    assert m["caps"][NEW] == {"value": 40, "tier": "A", "avail_gb": 14.358, "how": "carried", "from": OLD}
    own = {"choice": {"mode": "single", "live": NEW}, "caps": {OLD: {"value": 26, "tier": "A"},
                                                             NEW: {"value": 28, "tier": "A", "how": "found"}}}
    assert picker.migrate_machine(own) == (own, False)                               # its own cap wins
    assert picker.migrate_machine({}) == ({}, False)


def test_a_carried_cap_is_a_floor_never_a_ceiling():
    A = picker.tier(picker.classes(NV12), "nvidia")
    amd = picker.tier(picker.classes(AMD16), "amd")
    m, _ = picker.migrate_machine({"caps": {OLD: {"value": 26, "tier": "A"}}})
    c = picker.cap_for(NEW, A, "nvidia", m, _mem(11935))             # this machine: the text build's own 30
    assert (c["value"], c["how"]) == (30, "estimate")
    c = picker.cap_for(NEW, A, "nvidia", m, _mem(11000))             # less VRAM: the estimate is below 26, 26 stands
    assert (c["value"], c["how"], c["source"]) == (26, "carried", f"carried from {OLD}")
    assert picker.cap_for(NEW, A, "nvidia", m, picker.memory(NV12))["value"] == 30      # no VRAM read: the study's 30
    m2, _ = picker.migrate_machine({"caps": {OLD: {"value": 40, "tier": "A", "how": "found", "avail_gb": 14.358}}})
    c = picker.cap_for(NEW, amd, "amd", m2, _mem(int(14.358 * 1024), specs=AMD16))
    assert (c["value"], c["how"]) == (40, "carried")                 # above the estimate there (35): kept
    c = picker.cap_for(NEW, amd, "amd", m2, _mem(int(13.5 * 1024), specs=AMD16))
    assert c["how"] == "lowered" and c["value"] == 37                # less VRAM than when it was found: lowered as usual
    # a carried cap from another tier is ignored, as any stored cap
    assert picker.cap_for(NEW, picker.tier({"vram": "8", "ram": "32"}, "nvidia"), "nvidia", m, None)["value"] == 15


def test_a_start_with_an_old_saved_choice_runs_the_text_build_and_saves_it(monkeypatch, tmp_path):
    fo = FakeOllama(monkeypatch)
    _write_machine(tmp_path, "box", {"choice": {"mode": "single", "live": OLD}, "caps": {OLD: {"value": 26, "tier": "A"}},
                                     "gpu": picker.gpu_sig(NV12)})
    eng, hub, t = _hub(specs={**NV12, "vram_avail_mb": 11935, "vram_held_mb": 10010, "ram_avail_mb": 90700})
    assert hub.ai == {"mode": "single", "live": NEW, "reviewer": None} and llm.MODEL == NEW
    assert llm.num_gpu(NEW) == 30 and not hub.ai_off
    saved = picker.load_machine("box", NV12)
    assert saved["choice"] == {"mode": "single", "live": NEW} and saved["caps"][NEW]["how"] == "carried"
    env = (tmp_path / "machines" / "box.env").read_text()
    assert f"CW_LIVE_MODEL={NEW}\n" in env and OLD + "\n" not in env and "CW_LLM_NUM_GPU=estimate" in env
    # the hybrid saved on the AMD desktop: the reviewer is the text build with the found cap carried
    _write_machine(tmp_path, "amdbox", {"choice": {"mode": "hybrid", "live": M12, "reviewer": OLD},
                                       "caps": {OLD: {"value": 40, "tier": "A", "how": "found", "avail_gb": 14.358},
                                                M12: {"value": 49, "tier": "A", "how": "found", "avail_gb": 14.358}},
                                       "gpu": picker.gpu_sig(AMD16)})
    eng, hub, t = _hub(specs={**AMD16, "vram_avail_mb": int(14.358 * 1024), "vram_held_mb": 9000, "ram_avail_mb": 80000})
    assert hub.ai == {"mode": "hybrid", "live": M12, "reviewer": NEW} and llm.num_gpu(NEW) == 40
    env = (tmp_path / "machines" / "amdbox.env").read_text()
    assert f"CW_REVIEW_MODEL={NEW}\n" in env and "CW_REVIEW_NUM_GPU=40\n" in env
    # Use this on the page keeps the carried cap carried (W2, sweep 2026-11-09: never frozen as "set"), with its VRAM
    hub.ai = {"mode": "single", "live": M12, "reviewer": None}
    _select(hub, NEW)
    assert hub.machine["caps"][NEW] == {"value": 40, "tier": "A", "how": "carried", "avail_gb": 14.358, "from": OLD}


def test_gpu_envs_old_live_model_means_the_text_build(monkeypatch):
    FakeOllama(monkeypatch)
    monkeypatch.setattr(llm, "MODEL", OLD)                 # CW_LLM_MODEL=gemma4mtp:31b-n3, CW_LLM_NUM_GPU=26, no machine file
    monkeypatch.setattr(llm, "NUM_GPU", "26")
    eng, hub, t = _hub(specs={**NV12, "vram_avail_mb": 11935, "vram_held_mb": 10010, "ram_avail_mb": 90700})
    assert llm.MODEL == NEW and hub.ai["live"] == NEW
    c = hub._cap(NEW)
    assert (c["value"], c["how"]) == (30, "estimate") and llm.num_gpu(NEW) == 30   # 26 was the projector build's
    monkeypatch.setattr(llm, "MODEL", OLD)
    eng, hub, t = _hub(specs={**NV12, "vram_avail_mb": 11000, "vram_held_mb": 10010, "ram_avail_mb": 90700})
    assert (hub._cap(NEW)["value"], hub._cap(NEW)["how"]) == (26, "carried")       # a floor where the estimate is lower


# ------------------------------------------------------------------------------------------------ scripts
def test_live_sh_maps_the_same_replaced_ids_as_the_catalogue():
    sh = open(os.path.join(ROOT, "start.sh")).read()
    block = sh[sh.index('case "${CW_LIVE_MODEL:-}" in'):]
    block = block[:block.index("esac")]
    pairs = dict(re.findall(r"^  ([\w.:-]+)\)\n(?:.*\n)*?    CW_LIVE_MODEL=([\w.:-]+) CW_LLM_NUM_GPU=estimate", block, re.M))
    assert pairs == {k: v["by"] for k, v in picker.catalogue()["replaced"].items()}


def test_ollama_sh_builds_the_text_twin_from_the_parents_own_blob():
    sh = open(os.path.join(ROOT, "bin", "ollama.sh")).read()
    rx = re.search(r'\[\[ "\$tag" =~ (\S+) \]\]', sh).group(1)
    for tag, want in ((NEW, ("31b", "3", "-text")), (OLD, ("31b", "3", "")), ("gemma4mtp:26b-a4b-n2", ("26b-a4b", "2", "")),
                      ("gemma4mtp:e4b-n4-text", ("e4b", "4", "-text"))):
        m = re.match(rx.replace("[0-9]", r"\d"), tag)
        assert m and (m.group(1), m.group(2), m.group(3) or "") == want, tag
    assert re.match(rx, "gpt-oss-safeguard:20b") is None
    assert 'ollama show --modelfile "$parent"' in sh and 'tmf="Modelfile.$size.n$n-text"' in sh
    assert "/root/.ollama/models/blobs/sha256-0374" not in sh.split("studied=")[0]      # no store-local path baked in


def test_the_text_builds_requests_differ_from_the_projector_builds_only_in_the_tag_and_the_cap(monkeypatch):
    old = _bodies(monkeypatch, OLD, "26")
    new = _bodies(monkeypatch, NEW, "30")
    for o, n in zip(old, new):
        assert o["model"] == OLD and n["model"] == NEW
        assert o["options"].pop("num_gpu") == 26 and n["options"].pop("num_gpu") == 30
        assert {k: v for k, v in o.items() if k != "model"} == {k: v for k, v in n.items() if k != "model"}


# ------------------------------------------------------------------------------------------------ W2 (sweep 2026-11-09)
OLD_MACHINE = {"choice": {"mode": "single", "live": OLD}, "caps": {OLD: {"value": 26, "tier": "A"}}}   # a machine file saved before the text build
COLD = {**NV12, "vram_avail_mb": 11942, "vram_held_mb": 0, "ram_avail_mb": 88938}           # the read of 2026-11-09


def _select(hub, live):
    async def go():
        hub.start_select({"mode": "single", "live": live})
        await hub.ai_task_done()
    asyncio.run(go())
    assert hub.ai_last["select"].get("ok"), hub.ai_last


def test_w2_this_machine_comes_up_at_30_on_a_cold_read_and_on_a_held_one():
    """W2: with nothing loaded, the card read 11,942 MiB free (held 0); with the runner loaded it read 1,925 free + 10,010
    held = 11,935 MiB, the tier-A point where 30 ran with 0 aborts. A cold read reaches the measured point by the read
    itself; the cold reserve applies only to the line off the points. The carried 26 stays the floor below it."""
    A = picker.tier(picker.classes(NV12), "nvidia")
    m, _ = picker.migrate_machine(OLD_MACHINE)
    cold = picker.cap_for(NEW, A, "nvidia", m, picker.memory(COLD))
    assert (cold["value"], cold["how"]) == (30, "estimate") and cold["estimate"]["how"] == "measured"
    held = picker.cap_for(NEW, A, "nvidia", m, _mem(11935))
    assert (held["value"], held["how"]) == (30, "estimate")
    assert picker.cap_for(NEW, A, "nvidia", {}, picker.memory(COLD))["value"] == 30           # no file: the same 30
    # below the point, cold or held: the carried 26 is the floor (the line gives 24 cold, 26 held at 11.52 GB)
    for held_mb in (0, 10010):
        c = picker.cap_for(NEW, A, "nvidia", m, _mem(11800, held=held_mb))
        assert (c["value"], c["how"], c["from"]) == (26, "carried", OLD), held_mb
    # the safety side is unchanged: 33 failed at the point, so nothing above 30 until the VRAM clears it by 0.75 GB
    assert picker.estimate_cap(NEW, 12.3, cold=True)["value"] == 30
    assert picker.estimate_cap(NEW, 12.3)["value"] == 30


def test_w2_a_cold_start_runs_30_and_the_page_shows_the_cap_it_runs(monkeypatch, tmp_path):
    FakeOllama(monkeypatch)
    _write_machine(tmp_path, "box", {**OLD_MACHINE, "gpu": picker.gpu_sig(NV12)})
    eng, hub, t = _hub(specs=COLD)
    assert hub.ai["live"] == NEW and llm.num_gpu(NEW) == 30
    top = asyncio.run(hub.ai_info())["rows"][0]
    assert top["id"] == NEW and top["num_gpu"]["value"] == 30 and top["speed"]["keepup"] == 40


def test_w2_use_this_on_a_carried_cap_keeps_it_carried(monkeypatch, tmp_path):
    """W2: a session that ran on the carried 26 (VRAM lower that day) and pressed "Use this" stores 26 as carried, not as
    "set": the next start with the VRAM back at the point gets the text build's 30, not a frozen 26."""
    FakeOllama(monkeypatch)
    _write_machine(tmp_path, "box", {**OLD_MACHINE, "gpu": picker.gpu_sig(NV12)})
    eng, hub, t = _hub(specs={**COLD, "vram_avail_mb": 11500})
    assert (hub._cap(NEW)["value"], hub._cap(NEW)["how"]) == (26, "carried")
    _select(hub, NEW)
    e = picker.load_machine("box", NV12)["caps"][NEW]
    assert (e["value"], e["how"], e["from"], e["avail_gb"]) == (26, "carried", OLD, round(11500 / 1024, 3))
    eng, hub, t = _hub(specs=COLD)
    assert (hub._cap(NEW)["value"], hub._cap(NEW)["how"]) == (30, "estimate") and llm.num_gpu(NEW) == 30
    _select(hub, NEW)                                     # Use this now stores the text build's own 30
    assert hub.machine["caps"][NEW]["how"] == "set" and hub.machine["caps"][NEW]["value"] == 30
    # a cap found by Check speed is this machine's own and stays found
    hub.machine["caps"][NEW] = picker.cap_entry(30, "A", "found", 11942 / 1024)
    _select(hub, NEW)
    assert hub.machine["caps"][NEW]["how"] == "found"
