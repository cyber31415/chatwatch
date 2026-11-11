"""The model picker (owner, 2026-11-04 and 2026-11-07; docs/DASHBOARD.md "Choosing the AI").

"The program can run without AI"; "a few system specs pick the models"; "before a user hits start, there needs to be
an estimate or even better we have hard numbers that show what the message/minute processing speed will be". This
module is the part with no server state:
- the machine's specs (./start.sh reads them on the host and passes CW_SPEC_*), their classes and the study tier
  they match, with the page's dropdown as an override;
- the catalogue (assets/models.json, written by data/study/picker/build_catalogue.py from docs/HARDWARE_TIERS.md and
  the study's runs), filtered and ranked for a machine: one row per setup with its tested numbers, labelled with the
  tier they were tested on, or with this machine's own speed check when there is one;
- the speed check's sample (synthetic chat, no real text), its arithmetic (speed2's method: person-read mean, window
  a + b*k from the medians of k = 1 and 12) and the keep-up table it reads msgs/min from (assets/keepup_grid.json:
  H4's lossless simulator over a grid of service times);
- the per-machine file (data/state/machines/<host>.json, and the .env sidecar ./start.sh reads): the choice, each
  model's GPU layer cap on this machine, the speed checks, the last review batch's swap times;
- reading Ollama's log, as ./start.sh's tail copies it: the layers Ollama put on the GPU, and runner aborts.
Units (owner, 2026-11-04 10:11 am EST): live chat in msgs/min; reviews in time per chatter."""
import json
import math
import os
import random
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ASSETS = os.path.join(HERE, "assets")
CATALOGUE_FILE = os.path.join(ASSETS, "models.json")
GRID_FILE = os.path.join(ASSETS, "keepup_grid.json")
VRAM_CHOICES = ("none", "8", "12", "16+")
# owner 2026-11-08: 48 and 64 as two choices; owner 2026-11-09: 16 GB is the lowest option ("<16" removed: it could mean
# 2 or 12 GB, and nothing tested runs with no GPU there)
RAM_CHOICES = ("16", "32", "48", "64", "96+")
MTP_MIN_VERSION = (0, 40, 11)          # Ollama's MTP `DRAFT` path (HARDWARE_TIERS section 3)
# Display steps for a measured keep-up: a number between two of them shows as the lower one ("about 40")
STEPS = (0, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 600, 800)
CAP_MARGIN = 3                         # an MTP build's cap: Ollama's fit minus this (H3: fit 29 -> 26 on 12 GB)
TAG = re.compile(r"^[A-Za-z0-9._/-]+(:[A-Za-z0-9._-]+)?$")
_cat = _grid = None


def machines_dir():
    return os.environ.get("CW_MACHINES_DIR") or os.path.join(ROOT, "data", "state", "machines")


def catalogue():
    global _cat
    if _cat is None:
        _cat = json.load(open(CATALOGUE_FILE))
    return _cat


def setup(tag):
    return next((s for s in catalogue()["setups"] if s["id"] == tag), None)


def is_mtp(tag):
    s = setup(tag)
    return bool(s["mtp"]) if s else tag.startswith("gemma4mtp:")


# ------------------------------------------------------------------------------------------------ replaced setups
# A setup the catalogue no longer lists because another build replaced it (catalogue "replaced"; owner 2026-11-09: the
# text-only 31B, gemma4mtp:31b-n3-text, replaces gemma4mtp:31b-n3 in the live tool: the same weights, drafter and
# prompts without the vision projector the tool never uses, data/study/quant/FACTS.md section 4). A choice saved with the
# old id (this machine's file, its .env, gpu.env) means the new one. The old tag itself stays installed and unchanged
# (the pipeline's teacher study uses it).
def resolve(tag):
    """The id that replaced a saved setup id; any other tag as it is."""
    r = (catalogue().get("replaced") or {}).get(tag) if isinstance(tag, str) else None
    return r["by"] if r else tag


def migrate_machine(m):
    """A machine file written before a setup was replaced -> (the file as the picker reads it now, changed?). The choice
    names the new id. A cap stored for the old id moves to the new one as "carried" (cap_for: a floor, never a ceiling:
    the two builds hold the same VRAM at the same cap, FACTS section 4, so a cap that ran the old one runs the new one),
    unless the new id has its own. The old id's speed checks and loads stay under the old id (another build's numbers)."""
    rep = catalogue().get("replaced") or {}
    if not isinstance(m, dict) or not rep:
        return m, False
    m, changed = dict(m), False
    c = dict(m.get("choice") or {})
    for k in ("live", "reviewer"):
        if c.get(k) in rep:
            c[k], changed = rep[c[k]]["by"], True
    if changed:
        m["choice"] = c
    caps = dict(m.get("caps") or {})
    for old, r in rep.items():
        e = caps.get(old)
        if isinstance(e, dict) and r["by"] not in caps and e.get("value") is not None:
            caps[r["by"]] = {**{k: e[k] for k in ("value", "tier", "avail_gb") if k in e}, "how": "carried",
                             "from": old}
            changed = True
    if changed:
        m["caps"] = caps
    return m, changed


# ------------------------------------------------------------------------------------------------ specs
def specs_from_env(env=None):
    """What ./start.sh read on the host (nvidia-smi; the largest AMD card under /sys/class/drm; /proc/meminfo; lscpu).
    A value it could not read is None; the page then asks for it (the dropdowns)."""
    env = os.environ if env is None else env

    def num(k):
        try:
            return int(float(env.get(k, "") or ""))
        except ValueError:
            return None
    vendor = (env.get("CW_SPEC_GPU_VENDOR") or "").strip().lower() or None
    return {"host": env.get("CW_HOST") or None, "gpu_vendor": vendor, "gpu_name": (env.get("CW_SPEC_GPU_NAME") or "")[:80] or None,
            "vram_mb": num("CW_SPEC_VRAM_MB"), "ram_mb": num("CW_SPEC_RAM_MB"), "cores": num("CW_SPEC_CORES"),
            "threads": num("CW_SPEC_THREADS"), "vram_avail_mb": num("CW_SPEC_VRAM_AVAIL_MB"),
            "ram_avail_mb": num("CW_SPEC_RAM_AVAIL_MB"), "vram_held_mb": num("CW_SPEC_VRAM_HELD_MB"),
            "ollama_gpu": (env.get("CW_GPU") or env.get("CW_SPEC_OLLAMA_GPU") or None),
            # the card also drives a screen (./start.sh: nvidia-smi display_active; an AMD card's connected outputs;
            # unknown counts as yes there): its desktop keeps DESKTOP_RESERVE_GB (owner 2026-11-10)
            "gpu_display": {"1": True, "0": False}.get((env.get("CW_SPEC_GPU_DISPLAY") or "").strip()),
            # Linux inside Windows (WSL2): ./start.sh reads it from the kernel name; the desktop is outside its memory
            "wsl": (env.get("CW_SPEC_WSL") or "").strip() == "1"}


def vram_class(mb, vendor=None):
    if not mb or vendor in (None, "none", "cpu"):
        return "none" if vendor in ("none", "cpu") or mb == 0 else None
    if mb < 7000:
        return "none"                  # under 8 GB: nothing tested fits; the CPU tiers apply
    if mb < 10500:
        return "8"
    if mb < 14500:
        return "12"
    return "16+"


def ram_class(mb):
    if not mb:
        return None
    gb = mb / 1024
    # under 16 GB read: the lowest option, "16"; the free RAM read still gates each model (a short one is not
    # recommended), and the panel shows what was read
    return ("16" if gb < 26 else "32" if gb < 44 else "48" if gb < 56 else "64" if gb < 80
            else "96+")


def classes(specs, override=None):
    """The specs as the dropdowns' classes; an override (the page's dropdown) wins over what was read."""
    o = override or {}
    v = o.get("vram") if o.get("vram") in VRAM_CHOICES else vram_class(specs.get("vram_mb"), specs.get("gpu_vendor"))
    r = o.get("ram") if o.get("ram") in RAM_CHOICES else ram_class(specs.get("ram_mb"))
    mb = specs.get("ram_mb")
    return {"vram": v, "ram": r, "overridden": {k: k in o and o[k] in (VRAM_CHOICES if k == "vram" else RAM_CHOICES)
                                                for k in ("vram", "ram")},
            # read under the lowest option (16 GB) and not set by the dropdown: never "tested on this kind of machine"
            "ram_below": bool(mb) and mb / 1024 < 14 and o.get("ram") not in RAM_CHOICES}


# The tested tiers, highest first, with what a machine needs to meet each (owner, 2026-11-07: "if the machine has specs
# that meet what we tested then we should be perfectly fine to just put an estimated speed number"). The 12 GB tiers
# A-D share their numbers. The GPU vendor does not block a tier.
TIER_ORDER = (("A", "12", "32"), ("E", "8", "32"), ("F32", "none", "32"), ("F16", "none", "16"))
# The RAM a dropdown class is taken to have free for a model when nothing was read (a GPU machine below its GPU tier's
# RAM, ramfit 2026-11-08): three quarters of the class (the system, the desktop and the browser take the rest).
RAM_NOMINAL_GB = {"16": 12.0, "32": 24.0, "48": 36.0, "64": 48.0, "96+": 72.0}
RAM_CLASS_GB = {"16": 16, "32": 32, "48": 48, "64": 64, "96+": 96}


def tier(cls, vendor=None):
    """The highest tested tier this machine meets or exceeds (GPU memory and RAM both at least the tier's): its numbers
    are shown as this machine's estimate. {"key", "proxy" (the tier the numbers come from), "measured" (False only
    below every tested tier), "estimate" (the machine is not the tested one exactly), "label", "why"}.
    A GPU with less RAM than its GPU tier was tested with (ramfit, 2026-11-08: a 12 GB card with 16 GB of RAM fell to
    "no GPU" and hid the 12B + MTP that fits whole on the card) gets that GPU tier, marked "ram_short": rows() then
    lists each model only where its own system-RAM part fits (RAM_NOMINAL_GB of the class, or the free RAM read)."""
    v, r = cls.get("vram"), cls.get("ram")
    T = catalogue()["tiers"]
    if v is None or r is None:
        return {"key": None, "proxy": None, "measured": False, "estimate": False, "label": "unknown",
                "why": "pick your GPU memory and RAM"}
    vi, ri = VRAM_CHOICES.index(v), RAM_CHOICES.index(r)
    for k, tv, trr in TIER_ORDER:
        if tv == "none" and v != "none":
            # a GPU never takes a no-GPU tier (with 16 GB the lowest RAM option every machine meets F16): a GPU below
            # every GPU tier keeps the "below every tested machine" path, where an MTP build is never run uncapped
            continue
        if vi >= VRAM_CHOICES.index(tv) and ri >= RAM_CHOICES.index(trr):
            exact = v == T[k]["vram"] and (T[k]["gpu"] == "none" or (vendor or "none") == T[k]["gpu"])
            if cls.get("ram_below"):
                return {"key": k, "proxy": k, "measured": True, "estimate": True, "label": T[k]["label"],
                        "why": f"estimated from the tested {T[k]['short']} machine ({T[k]['label']}): this one has less "
                               "RAM than the lowest option, so each model is checked against its RAM; Check Speed "
                               "measures this one"}
            return {"key": k, "proxy": k, "measured": True, "estimate": not exact, "label": T[k]["label"],
                    "why": (f"tested on this kind of machine ({T[k]['label']})" if exact else
                            f"estimated from the tested {T[k]['short']} machine ({T[k]['label']}): this one meets or "
                            "exceeds it") + "; Check Speed measures this one"}
        if v != "none" and tv != "none" and vi >= VRAM_CHOICES.index(tv):
            # the GPU meets this tier, the RAM does not: the GPU tier's numbers, each model gated by its RAM part
            return {"key": k, "proxy": k, "measured": True, "estimate": True, "label": T[k]["label"],
                    "ram_short": True,
                    "why": f"estimated from the tested {T[k]['short']} machine ({T[k]['label']}) with less RAM"
                           "; a model is listed only where its system-RAM part fits; Check Speed measures this one"}
    return {"key": None, "proxy": "F16", "measured": False, "estimate": False, "label": "below every tested machine",
            "gpu": v != "none",             # a GPU below the tested ones: an MTP build there needs a cap found here
            "why": f"below every tested machine (the smallest: {T['F16']['label']}): the figures are not tested here"}


def parse_version(v):
    try:
        return tuple(int(x) for x in re.findall(r"\d+", v or "")[:3])
    except ValueError:
        return ()


# ------------------------------------------------------------------------------------------------ keep-up table
def _grid_file():
    global _grid
    if _grid is None:
        _grid = json.load(open(GRID_FILE))
    return _grid


# PH8 (audit 2026-11-10): a check's keep-up is worked out as the tier figures are. Each grid cell holds the highest
# simulated rate that kept up (a floor: the edge lies between it and the next rate tried), so a lookup between floors,
# floored again, read up to one step low twice (e4b + MTP on CPU 16 GB: tier 20, check 10). The lookup now runs over each
# cell's edge midpoint (the geometric middle of its passing and next rate) and the result is floored once, on the steps
# the tier figures use (keepup_setups.py's 10 / 20 / 40 / 75 / 100 / 150, "150+" at the top; 5 below 10). On the
# catalogue's own service times this gives every listed (MTP) setup's tier figure (19 of 21), against 9 of 21 before;
# the re-read reach 35 of 42 (the rest one step low), against 21 of 42. The two cells off, both never listed, go opposite
# ways (PH-R2-2, audit round 2): qwen on 8 GB reads one step LOW (150 -> 100) and the plain gemma4:12b-it-qat on 8 GB one
# step HIGH (75 -> 100), the unsafe direction for keep-up. No listed setup reads above its tier figure
# (test_ph8 asserts it); the grid's 15 validation cells read none above, one below.
KEEPUP_STEPS = (0, 5, 10, 20, 40, 75, 100, 150)


def keepup_step(x):
    return max(s for s in KEEPUP_STEPS if s <= x + 1e-9)


def _edge_mid(grid, rates):
    """Each cell's floor rate -> the geometric middle of it and the next rate tried (the top rate stays)."""
    def mid(v):
        if v <= 0:
            return math.sqrt(0.5 * rates[0])
        nxt = next((r for r in rates if r > v), None)
        return v if nxt is None else math.sqrt(v * nxt)
    return [[mid(v) for v in row] for row in grid]


_mids = {}


def _grid_mid(key, rates_key):
    if key not in _mids:
        g = _grid_file()
        _mids[key] = _edge_mid(g[key], g[rates_key]) if g.get(key) else None
    return _mids[key]


def keepup_from(P, W12):
    """msgs/min (one channel) at which person reads and flagged lines keep a p95 lag under 10 min, for these measured
    service times: bilinear in log P and log W12 over the H4-simulated grid (assets/keepup_grid.json), each cell at its
    edge midpoint (PH8)."""
    return _lookup(_grid_mid("grid", "rates"), P, W12)


# How far the re-reads reach (sweep 2026-11-09; ASSESSMENT_20261109 sections 3 and 6). The keep-up above counts person
# reads and flagged lines only; the priority 1-3 re-reads, which carry most of the 31B's gain, age out after 15 min and
# thin well below it (the 31B on 12 GB: keep-up 40, every line re-read to 20). data/study/picker/reread.py, per setup and
# over the keep-up grid's cells: "all" = the busiest rate with 95% of queued lines re-read (the page's "every line
# re-read to ~N"); "most" = 80%, where the hybrid starts to beat the single 31B (live_quality_now section 4: at 40 msgs/min
# the 31B re-reads 80% and still leads by the owner's 10:1 rule; at 75, 47%, the hybrid catches clearly more).
REREAD_STEPS = (0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 75, 100, 150, 200, 300)


def reread_from(P, W12):
    """{"all", "most"} msgs/min for these measured service times (None when the grid asset has no re-read table)."""
    g = _grid_file()
    if not g.get("reread_grid_all"):
        return None
    step = lambda x: max(s for s in REREAD_STEPS if s <= x + 1e-9)        # noqa: E731
    return {"all": step(_lookup(_grid_mid("reread_grid_all", "reread_rates"), P, W12)),
            "most": step(_lookup(_grid_mid("reread_grid_most", "reread_rates"), P, W12))}


def _lookup(g, P, W12):
    pg, wg = _grid_file()["p_grid"], _grid_file()["w_grid"]

    def pos(v, ax):
        v = min(max(v, ax[0]), ax[-1])
        for i in range(len(ax) - 1):
            if v <= ax[i + 1]:
                return i, (math.log(v) - math.log(ax[i])) / (math.log(ax[i + 1]) - math.log(ax[i]))
        return len(ax) - 2, 1.0
    i, fi = pos(P, pg)
    j, fj = pos(W12, wg)
    k = lambda a, b: math.log(g[a][b] + 1)
    v = (k(i, j) * (1 - fi) * (1 - fj) + k(i + 1, j) * fi * (1 - fj) + k(i, j + 1) * (1 - fi) * fj
         + k(i + 1, j + 1) * fi * fj)
    return max(0.0, math.exp(v) - 1)


def step_down(x):
    return max(s for s in STEPS if s <= x + 1e-9)


def review_seconds(P, a, b, k_lines):
    """One chatter's review: the person read plus the re-read of their k new lines in windows of up to 12."""
    k = max(0, int(k_lines))
    return P + (math.ceil(k / 12) * a + b * k if k else 0.0)


# ------------------------------------------------------------------------------------------------ rows for the page
def over_no_ai(pct):
    """A test Accuracy (% of the 20b) as the page shows it: whole percent better than No AI, the lowest-quality option and
    the same on every machine (owner 2026-11-07: "the baseline be the lowest quality option then all higher quality
    options have +% improvement over the baseline"). 117.3 -> 66; the 20b (100) -> 41."""
    return round(100 * pct / catalogue()["no_ai"]["pct"] - 100)


def install_cmd(tag):
    return f"bin/ollama.sh install {tag}"


SPEED_KEYS = ("at", "person_s", "a", "b")     # a stored speed check is usable only with these (PH1, audit 2026-11-10)


def valid_speed(sp):
    """A stored speed check with its numbers (a check that measured nothing once stored an entry without them)."""
    if not isinstance(sp, dict) or not isinstance(sp.get("at"), str):
        return False
    return all(isinstance(sp.get(k), (int, float)) and not isinstance(sp.get(k), bool) and sp[k] >= 0
               for k in SPEED_KEYS[1:])


def speed_entry(machine, tag):
    """This machine's speed check of the model, or None (none, or one without its numbers)."""
    sp = ((machine or {}).get("speed") or {}).get(tag)
    return sp if valid_speed(sp) else None


def check_numbers(P, a, b):
    """A speed check's keep-up and re-read reach from its service times, worked out as the tier figures are (PH8):
    {"keepup_raw", "keepup", "keepup_plus", "reread_all", "reread_most"}."""
    k = keepup_from(P, a + 12 * b)
    s = keepup_step(k)
    rr = reread_from(P, a + 12 * b) or {}
    return {"keepup_raw": round(k, 1), "keepup": s, "keepup_plus": s >= KEEPUP_STEPS[-1],
            "reread_all": rr.get("all"), "reread_most": rr.get("most")}


def _tier_numbers(s, tr, machine):
    """(numbers, tier entry): this machine's speed check if any, else the met tier's numbers as this machine's
    estimate ("est.", owner 2026-11-07). Below every tested tier the smallest tier's figure is "not tested here".
    A check's keep-up and re-read figures are worked out again from its stored service times (check_numbers), so a check
    stored before PH8 reads the same way as a new one."""
    sp = speed_entry(machine, s["id"])
    t = s["tiers"].get(tr["proxy"]) if tr.get("proxy") else None
    if sp:
        rr = check_numbers(sp["person_s"], sp["a"], sp["b"])
        return ({"keepup": rr["keepup"], "keepup_plus": rr["keepup_plus"], "person_s": sp["person_s"],
                 "a": sp["a"], "b": sp["b"], "reread_all": rr.get("reread_all"), "reread_most": rr.get("reread_most"),
                 "measured": True, "label": f"measured here {sp['at'][:10]}" + (" (other work running)" if sp.get("busy") else "")},
                t)
    if t is None:
        return None, None
    tl = catalogue()["tiers"][tr["proxy"]]["short"]
    return ({"keepup": t["keepup"], "keepup_plus": t["keepup_plus"], "person_s": t["person_s"], "a": t["a"],
             "b": t["b"], "reread_all": t.get("reread_all"), "reread_most": t.get("reread_most"),
             "measured": False, "estimate": bool(tr.get("estimate")),
             # a tier measured on the build this setup replaced says so (the 31B on 8 GB: the projector build's run)
             "label": f"tested on {tl}" + (f" ({t['build_note']})" if t.get("build_note") else ""),
             "this_hardware": bool(tr["measured"])}, t)


# ------------------------------------------------------------------------------------------------ the cap from free VRAM
# Owner 2026-11-07: "we see what VRAM is free and we can estimate the layers based on that". ./start.sh reads, on the
# host at start, the VRAM available for models (the card's free memory plus what cw-ollama itself holds) and passes it as
# CW_SPEC_VRAM_AVAIL_MB (and CW_SPEC_VRAM_HELD_MB, what cw-ollama held). An MTP build's cap is then estimated from its
# measured points (catalogue "vram_fit": [available GiB, the cap that ran with 0 runner aborts, the cap that failed
# there or None]), review of 2026-11-08:
# - the points were read with the runner loaded (its CUDA context inside "held"): a read with nothing held is cut by
#   COLD_RESERVE_GB first for the line (the part of the estimate that is not measured);
# - but a measured point is reached by the read itself, cold or held (W2, sweep 2026-11-09): a cold read is the card's
#   free memory, the same quantity a held read gives as free + held. Measured on tier A's own card: held 11,935 MiB
#   (1,925 free + 10,010 held, 2026-11-08), cold 11,942 MiB (0 held, 2026-11-09), so nvidia-smi's per-process figure
#   holds the runner's whole footprint and the reserve, taken off a cold read at the point, only pushed this machine
#   from its measured 30 down to 24 (below the carried 26). The reserve stays on the line between and beyond points;
# - the line through the points (one point: a per-layer size; see gb_per_layer), less a margin of
#   max(EST_MARGIN, ceil(MARGIN_GB / GB per layer)) layers, never below a measured cap the VRAM reaches (so a measured
#   point gives its own cap back) and never at or above a cap that failed at a point the VRAM does not clear by MARGIN_GB;
# - clamped to [1, layers] (a model that fits whole gets every layer); measured "fits whole" from that point up.
EST_MARGIN = 2                         # layers, at least, off the line
MARGIN_GB = 0.75                       # the same margin in GB (4 layers of the 12B, 2 of the 31B)
COLD_RESERVE_GB = 0.75                 # a read with nothing held lacks the runner's own context (~0.3-0.6 GB)
# Owner 2026-11-10 (a 16 GB AMD card went sluggish during a review with a Twitch stream open: the 31B filled the card to
# Ollama's fit less 3 layers): on a card that also drives a screen, a model that does not fit whole is capped so that
# this much of the VRAM read at start stays free for the desktop (the browser, video decoding, the compositor). A model
# that fits whole is left whole (it leaves room of its own: the 12B on 12 GB keeps about 3 GB free).
DESKTOP_RESERVE_GB = float(os.environ.get("CW_VRAM_RESERVE_GB") or 1.5)
AVAIL_DROP_GB = 0.4                    # a cap set for this machine stands until the VRAM is this much lower than then
                                       # (a desktop's VRAM moves 0.2-0.4 GB routinely)


def _floor1(x):
    return math.floor(x * 10 + 1e-9) / 10


def avail_gb(specs):
    """The VRAM available for models in GiB, exact (None when not read: no GPU, an old start script, a failed read)."""
    mb = (specs or {}).get("vram_avail_mb")
    if not mb or mb <= 0 or (specs or {}).get("gpu_vendor") in (None, "none", "cpu"):
        return None
    return mb / 1024


def _overhead():
    """GiB per GPU layer over weights per layer: the catalogue's "vram_overhead" (the projector 31B's two caps, 0.445 /
    0.288, pinned when the text build replaced it: 2026-11-09), else from the setups with two cap points."""
    pinned = (catalogue().get("vram_overhead") or {}).get("ratio")
    if pinned:
        return pinned
    r = []
    for s in catalogue()["setups"]:
        f = s.get("vram_fit") or {}
        pts = sorted(p[:2] for p in f.get("points", ()) if p[1] is not None)
        if len(pts) >= 2 and f.get("layers") and pts[-1][1] != pts[0][1]:
            gb_per_layer = (pts[-1][0] - pts[0][0]) / (pts[-1][1] - pts[0][1])
            r.append(gb_per_layer / (f["weights_gib"] / f["layers"]))
    return sum(r) / len(r) if r else None


def gb_per_layer(tag):
    """(GiB per GPU layer, how it was found) for an MTP build with VRAM points, else (None, "")."""
    f = (setup(tag) or {}).get("vram_fit") or {}
    pts = sorted(p[:2] for p in f.get("points", ()) if p[1] is not None)
    if len(pts) >= 2 and pts[-1][1] != pts[0][1]:
        return (pts[-1][0] - pts[0][0]) / (pts[-1][1] - pts[0][1]), "measured slope"
    ov = _overhead()
    if f.get("layers") and f.get("weights_gib") and ov:
        return f["weights_gib"] / f["layers"] * ov, f"weights per layer x {ov:.2f}"
    return None, ""


def point_avail(tag, cap):
    """The available VRAM at which this cap was measured for the model (None if it is not one of its points)."""
    for p in ((setup(tag) or {}).get("vram_fit") or {}).get("points", ()):
        if p[1] is not None and cap is not None and int(p[1]) == int(cap):
            return p[0]
    return None


def estimate_cap(tag, avail, cold=False):
    """The cap for an MTP build at `avail` GiB available (cold: nothing was held at the read), or None when it cannot be
    estimated (no points; below the measured range with no layer count). {"value" (None: fits whole, no cap), "how":
    "measured" / "estimated" / "fits whole", "gb_per_layer", "eff_gb"}."""
    f = (setup(tag) or {}).get("vram_fit")
    if not f or avail is None:
        return None
    r = float(avail)                                 # the read: what reaches a measured point (see above, W2)
    v = r - (COLD_RESERVE_GB if cold else 0.0)       # the line's figure: a cold read keeps the reserve
    L = f.get("layers")
    whole = [p[0] for p in f["points"] if p[1] is None]
    if whole and r >= min(whole) - 1e-9:
        return {"value": None, "how": "fits whole", "gb_per_layer": None, "eff_gb": round(v, 3)}
    pts = sorted((p[0], p[1], p[2] if len(p) > 2 else None) for p in f["points"] if p[1] is not None)
    g, _ = gb_per_layer(tag)
    if not pts or not L or not g:
        return None
    # the line: linear between the points, the per-layer size beyond them
    if v <= pts[0][0]:
        line = pts[0][1] + (v - pts[0][0]) / g
    elif v >= pts[-1][0]:
        line = pts[-1][1] + (v - pts[-1][0]) / g
    else:
        (x0, c0, _), (x1, c1, _) = next((a, b) for a, b in zip(pts, pts[1:]) if a[0] <= v <= b[0])
        line = c0 + (c1 - c0) * (v - x0) / (x1 - x0)
    x = math.floor(line + 1e-9) - max(EST_MARGIN, math.ceil(MARGIN_GB / g - 1e-9))
    reached = [c for (px, c, _) in pts if px <= r + 1e-9]
    if reached:
        x = max(x, max(reached))                     # a measured cap the VRAM reaches stands
    for px, c, fail in pts:
        if fail is not None and v < px + MARGIN_GB:  # not clear of a known failure: stay at its proven cap
            x = min(x, c) if r >= px - 1e-9 else min(x, fail - 1)
    x = max(1, min(L, x))
    return {"value": x, "how": "measured" if reached and x == max(reached) else "estimated",
            "gb_per_layer": round(g, 3), "eff_gb": round(v, 3)}


# ------------------------------------------------------------------------------------------------ free system RAM
# Owner 2026-11-07: "we need a separate system free memory read too, because this is just as important when models dont
# fit entirely on cards or we are using no gpu memory at all". ./start.sh reads, at start, the RAM available for models:
# MemAvailable plus the anonymous memory cw-ollama's processes already hold (a loaded model's CPU part; its mapped file
# pages are page cache, already inside MemAvailable). /api/ps is not used for it: for an MTP build (llama.cpp runner) it
# reports 1.6 GB for the 31B, which holds 9.5 GB of RAM. The part of a setup that sits in RAM must fit with headroom.
RAM_BUF_GB = 1.0          # the runner's own buffers beside the CPU layers. Check: the 31B + MTP at 26 of 61 layers on
                          # 12 GB is estimated at 10.1 + 1.0 = 11.1 GB; its runner held 9.5 GB (2026-11-08), so this errs high
RAM_HEADROOM_GB = 4.0     # at least this, or 10% of the RAM, kept free beside the model: the dashboard itself starts
RAM_HEADROOM_SHARE = 0.10  # after the read (its ML tier), and the desktop and browser grow
# Windows (WSL2, first Windows test 2026-11-10): the desktop and the browser run outside the Linux VM, whose memory cap
# (.wslconfig) already leaves them room, so inside it only the dashboard itself is kept free beside the model
RAM_HEADROOM_WSL_GB = 2.0


def ram_avail_gb(specs):
    mb = (specs or {}).get("ram_avail_mb")
    return mb / 1024 if mb and mb > 0 else None


def memory(specs):
    """What the start read, in GiB: "vram" / "ram" exact (None when not read), "vram_shown" / "ram_shown" rounded DOWN to
    0.1 for the page, "vram_cold" (nothing held by cw-ollama at the read, or not told), "vram_total", "ram_total"."""
    s = specs or {}
    v, r = avail_gb(s), ram_avail_gb(s)
    held = s.get("vram_held_mb")
    total = round(s["ram_mb"] / 1024, 1) if s.get("ram_mb") else None
    assumed = False
    if r is None and total is not None and total < 14:
        # owner 2026-11-09: 16 GB is the lowest RAM option, so a machine read under it shows "16"; with no free-RAM
        # read its own total decides, three quarters of it taken as free (as for a class), never the 16 GB class's 12
        r, assumed = round(0.75 * total, 1), True
    reserve = DESKTOP_RESERVE_GB if (s.get("gpu_display") and v is not None and DESKTOP_RESERVE_GB > 0) else 0.0
    return {"vram": v, "vram_shown": None if v is None else _floor1(v), "vram_cold": not held, "reserve_gb": reserve,
            "vram_total": round(s["vram_mb"] / 1024, 1) if s.get("vram_mb") else None,
            "ram": r, "ram_shown": None if r is None or assumed else _floor1(r), "ram_assumed": assumed,
            "ram_total": total, "wsl": bool(s.get("wsl"))}


def _sizes(tag):
    """(weights GiB, drafter GiB, layers or None) of a catalogue setup."""
    s = setup(tag) or {}
    f = s.get("vram_fit") or {}
    total = (s.get("download_gb") or 0) * 1e9 / 1024 ** 3
    w = f.get("weights_gib") or total
    return w, max(0.0, total - w), f.get("layers")


def ram_need_gb(tag, cap, mem=None):
    """GiB of RAM the model takes at this cap (cap_for's answer): the whole model on no GPU; the layers off the GPU at a
    cap; with no cap, what does not fit in the VRAM available (Ollama's own fit spills the rest to the CPU), only the
    buffers when it fits. None when it cannot be told (a cap still to be found)."""
    if not cap or not cap.get("known"):
        return None
    w, d, L = _sizes(tag)
    if cap.get("source") == "no GPU":
        return round(w + d + RAM_BUF_GB, 1)
    v = cap.get("value")
    if v is None:
        m = mem or {}
        room = m.get("vram") if m.get("vram") is not None else m.get("vram_total")
        if is_mtp(tag) or room is None:
            return round(RAM_BUF_GB, 1)          # an MTP build with no cap: measured or estimated to fit whole
        return round(max(0.0, w + d - room) + RAM_BUF_GB, 1)
    if not L:
        return round(RAM_BUF_GB, 1)
    return round(w * max(0, L - v) / L + RAM_BUF_GB, 1)


def ram_check(need, mem):
    """{"need_gb", "headroom_gb", "avail_gb", "state"}: "short" (avail < need + headroom: not recommended), "tight"
    (under twice the headroom to spare: it only just fits) or "ok"; None when either side is unknown."""
    a = (mem or {}).get("ram")
    if need is None or a is None:
        return None
    h = (RAM_HEADROOM_WSL_GB if (mem or {}).get("wsl")
         else round(max(RAM_HEADROOM_GB, RAM_HEADROOM_SHARE * ((mem or {}).get("ram_total") or 0)), 1))
    state = "short" if a < need + h else "tight" if a < need + 2 * h else "ok"
    return {"need_gb": need, "headroom_gb": h, "avail_gb": _floor1(a), "state": state,
            "assumed": bool((mem or {}).get("ram_assumed"))}


_MISSING = object()
NEEDS_CHECK = "press Check Speed once for this model first: it sets how much of the model goes on this GPU"
UNKNOWN_CAP = "unknown"        # the .env value for an MTP build whose cap is not known on this tier
ESTIMATE_CAP = "estimate"      # the .env value for a cap the server estimates at start (./start.sh does not warm it)


def cap_entry(value, tier_key, how=None, avail=None, reserve=0.0):
    """How a cap is stored in the machine file: with the tier it was found or chosen for (second review, 2026-11-08: a
    cap stored on one tier, say "no GPU", must not apply after the GPU dropdown changes the tier). how: "found" (Check
    speed: Ollama's fit, confirmed by 0 runner aborts), "set" (Use this without a check) or "env" (gpu.env's, never
    stored); avail: the GiB available for models when it was found or set."""
    e = {"value": None if value is None else int(value), "tier": tier_key}
    if how:
        e["how"] = how
    if avail is not None:
        e["avail_gb"] = round(avail, 3)
    if reserve:
        e["reserve_gb"] = reserve              # found with the desktop's reserve already taken off (owner 2026-11-10)
    return e


def _entry(machine, tag):
    return ((machine or {}).get("caps") or {}).get(tag, _MISSING)


def stored_cap(machine, tag, tier_key):
    """(found, value): this machine's stored cap for the model on this tier. An entry from another tier is ignored. A
    plain value (written before 2026-11-08) counts only as a number for an MTP build: "no cap" is never trusted for one."""
    e = _entry(machine, tag)
    if e is _MISSING:
        return False, None
    if isinstance(e, dict):
        if e.get("tier") != tier_key:
            return False, None
        v = e.get("value")
    else:
        v = e
    if v is None and is_mtp(tag):
        return False, None
    return True, (None if v is None else int(v))


def cap_value(e):
    """A stored entry's value (either format)."""
    return e.get("value") if isinstance(e, dict) else e


CAP_SOURCE = {"found": "found here", "set": "set here", "env": "gpu.env", "carried": "carried from the build this one replaced"}


def cap_for(tag, tr, vendor, machine, mem=None):
    """This machine's GPU layer cap for the model: {"value", "known", "source", "how"}; how: "found" (Check speed here),
    "set" (Use this here, or a cap stored before 2026-11-08), "env" (gpu.env's first-run cap), "carried" (a cap of the
    build this one replaced, migrate_machine), "lowered", "estimate", "study", "none" (no cap), "unknown". mem:
    picker.memory() (the VRAM available at start).
    - A cap known for this machine (found, set, gpu.env) is never raised by the estimate (review 2026-11-08: the owner's
      26 here is the study's setting, request for request). It is lowered only when the VRAM available now is more than
      AVAIL_DROP_GB below what it was then (the value stored with it, else the measured point with that cap): to
      the cap - ceil(the drop / GB per layer), along its own line only (the cap already has its margin: the estimate's
      margin and cold reserve are not applied on top; re-review 2026-11-08).
    - A carried cap is a floor, not a ceiling: it ran the replaced build, which holds the same VRAM at the same cap; the
      new build's own figure (the estimate, else the tier's study cap) wins where it is higher (the text-only 31B: 30
      on the 12 GB tier against the projector build's 26).
    - Else, for an MTP build with VRAM points, the estimate from the VRAM available (owner 2026-11-07): it fills in where
      this machine has no cap, e.g. a bigger card (above the tier's safe default) or a new one.
    - Else the met tier's (a bigger card gets the tested tier's cap, e.g. 26 for the 31B on a 16 GB card: fewer layers
      on the GPU than fit, so it works; the optional check may raise it). An MTP build with no cap known for the tier is
      found by the check (Ollama's fit minus CAP_MARGIN, confirmed by 0 runner aborts)."""
    m = mem or {}
    avail, shown = m.get("vram"), m.get("vram_shown")
    res = m.get("reserve_gb") or 0.0
    found, v = stored_cap(machine, tag, tr.get("key"))
    est = estimate_cap(tag, avail, m.get("vram_cold", True)) if is_mtp(tag) else None
    if res and est is not None and est["value"] is not None:
        # a model that does not fit whole: estimated from the VRAM less the desktop's reserve (owner 2026-11-10)
        est = estimate_cap(tag, avail - res, m.get("vram_cold", True))
        shown = f"{shown} GB available, {res:g} GB kept for the desktop; " if shown is not None else shown
    if found:
        e = _entry(machine, tag)
        how = (e.get("how") if isinstance(e, dict) else None) or "set"
        name = (f"carried from {e.get('from')}" if how == "carried" and e.get("from") else
                CAP_SOURCE[how] if how in CAP_SOURCE else "set here")
        then = (e.get("avail_gb") if isinstance(e, dict) else None) or point_avail(tag, v)
        g, _ = gb_per_layer(tag)
        layers = ((setup(tag) or {}).get("vram_fit") or {}).get("layers")
        now = avail
        if res and avail is not None and v is not None and not (layers and v >= layers):
            # the desktop's reserve on a cap that fills the card (not a whole model): the VRAM it may use now is the
            # read less the reserve; a cap found with the reserve already off (its entry says so) compares like with like
            now = avail - res
            if isinstance(e, dict) and e.get("reserve_gb") and then is not None:
                then = then - e["reserve_gb"]
        mine = {"value": v, "known": True, "how": how if how in CAP_SOURCE else "set", "source": name}
        if how == "carried":
            mine["from"] = e.get("from")             # _select keeps it carried (a floor), never "set"
        if v is not None and g and now is not None and then is not None and now < then - AVAIL_DROP_GB:
            low = v - math.ceil((then - now) / g - 1e-9)
            low = max(1, low)
            if low < v:
                mine = {"value": low, "known": True, "how": "lowered", "estimate": est, "was": v,
                        "source": f"lowered from {v}: {m.get('vram_shown')} GB available" + (
                            f", {res:g} GB kept for the desktop" if now != avail else "") + f"; {_floor1(then)} GB "
                            f"{'usable ' if now != avail else ''}when it was {name}"}
        if how == "carried":
            own = _own_cap(tag, tr, est, shown)
            if own["known"] and own["how"] in ("estimate", "study") and (
                    own["value"] is None or (mine["value"] is not None and own["value"] > mine["value"])):
                return own
        return mine
    return _own_cap(tag, tr, est, shown)


def _own_cap(tag, tr, est, shown):
    """cap_for's answer when this machine has no cap of its own for the model: the estimate, else the tier's."""
    if est is not None:
        if est["value"] is None:
            return {"value": None, "known": True, "how": "estimate", "estimate": est,
                    "source": f"fits whole in {shown} GB available"}
        return {"value": est["value"], "known": True, "how": "estimate", "estimate": est,
                "source": (f"estimated from {shown}".rstrip("; ") if isinstance(shown, str) else f"estimated from {shown} GB available")}
    s = setup(tag)
    if tr.get("key") is None and tr.get("gpu") and is_mtp(tag):
        # third review 2: a GPU below every tested tier: an MTP build is never run uncapped there; the cap must be found
        # here (Check speed reads Ollama's fit), stored under this tier (None)
        return {"value": None, "known": False, "source": NEEDS_CHECK, "how": "unknown"}
    if tr.get("key") in ("F16", "F32") or tr.get("key") is None:
        return {"value": None, "known": True, "source": "no GPU", "how": "none"}
    if s is None or not is_mtp(tag):
        return {"value": None, "known": True, "source": "Ollama's own fit", "how": "none"}
    t = s["tiers"].get(tr["key"])
    if t and t["num_gpu_known"]:
        return {"value": t["num_gpu"], "known": True, "how": "study",
                "source": f"study ({catalogue()['tiers'][tr['key']]['short']})"}
    return {"value": None, "known": False, "source": "not known for this tier: the speed check finds it", "how": "unknown"}


def _ram_view(cls, mem):
    """The RAM a model may use on a GPU machine below its tier's RAM: the free RAM read, else the class's nominal."""
    m = dict(mem or {})
    if m.get("ram") is None:
        m["ram"] = RAM_NOMINAL_GB.get(cls.get("ram"))
        m["ram_total"] = RAM_CLASS_GB.get(cls.get("ram"))
        m["ram_assumed"] = True
    if m.get("vram") is None and m.get("vram_total") is None:     # the card's room for a model on Ollama's own fit
        m["vram_total"] = {"8": 7.6, "12": 11.6, "16+": 15.6}.get(cls.get("vram"))
    return m


def rows(cls, vendor, installed, version, machine=None, mem=None):
    """The setups that fit this machine, recommended first, each with its numbers. installed: the cw-ollama tags (None
    when Ollama does not answer)."""
    tr = tier(cls, vendor)
    out = []
    ver = parse_version(version)
    for s in catalogue()["setups"]:
        nums, t = _tier_numbers(s, tr, machine) if tr["proxy"] else (None, None)
        if t is None and nums is None:
            continue
        if (tr["key"] is not None and not tr.get("ram_short")
                and RAM_CHOICES.index(cls["ram"]) < RAM_CHOICES.index(s.get("min_ram", "16"))):
            continue
        cap = cap_for(s["id"], tr, vendor, machine, mem)
        mv = _ram_view(cls, mem) if tr.get("ram_short") else mem
        ram = ram_check(ram_need_gb(s["id"], cap, mv), mv)
        if tr.get("ram_short") and (ram is None or ram["state"] == "short"):
            continue                   # below the tier's RAM: only a model whose own RAM part fits runs here
        short = bool(ram and ram["state"] == "short")
        default = tr["key"] in s.get("default_for", ())
        inst = installed is not None and s["id"] in installed
        blocked = ""
        if s["mtp"] and ver and ver < MTP_MIN_VERSION:
            blocked = f"needs Ollama {'.'.join(map(str, MTP_MIN_VERSION))} or later for MTP (this one: {version})"
        q = {**s["quality"], "over_no_ai": over_no_ai(s["quality"]["pct"])}
        out.append({"id": s["id"], "label": s["label"], "mtp": s["mtp"], "installed": inst,
                    "install": None if inst else install_cmd(s["id"]), "download_gb": s["download_gb"],
                    "quality": q, "gates": s["gates"], "speed": nums, "num_gpu": cap,
                    "recommended": default and not short, "ram_demoted": default and short, "ram": ram,
                    "blocked": blocked,
                    "note": s.get("note", ""),
                    "review_s": round(review_seconds(nums["person_s"], nums["a"], nums["b"],
                                                     catalogue()["review_load"]["b"]["lines_per_chatter"]), 1)})
    if tr.get("ram_short"):
        # owner 2026-11-08: "if a model can run without issue ... it is still listed and recommended": the best model whose
        # RAM part fits is recommended, "below every tested machine" stays a note on the figures
        ok = [r for r in out if r["mtp"] and r["ram"] and r["ram"]["state"] != "short" and not r["blocked"]]  # MTP: listed
        best = max(ok, key=lambda r: r["quality"]["pct"], default=None)
        for r in out:
            r["recommended"], r["ram_demoted"] = r is best, False
    out.sort(key=lambda r: (not r["recommended"], -r["quality"]["pct"], not r["mtp"]))
    return out


def ram_fit_note(cls, vendor, machine=None, mem=None):
    """Why nothing is listed on a machine below its tier's RAM (only No AI is left): the option that needs the least
    free RAM, what it needs (its RAM part + headroom) and what is free. None when something fits or the tier is not
    RAM-gated."""
    tr = tier(cls, vendor)
    if not tr.get("ram_short"):
        return None
    mv, best = _ram_view(cls, mem), None
    for s in catalogue()["setups"]:
        if not s["mtp"] or s.get("parent") is None or tr["proxy"] not in s["tiers"]:
            continue
        r = ram_check(ram_need_gb(s["id"], cap_for(s["id"], tr, vendor, machine, mem), mv), mv)
        if r is None:
            continue
        if r["state"] != "short":
            return None
        if best is None or r["need_gb"] < best[1]["need_gb"]:
            best = (s, r)
    if best is None:
        return None
    s, r = best
    read = (mem or {}).get("ram") is not None and not (mem or {}).get("ram_assumed")
    return (f"nothing fits in the RAM: the smallest option, {s['label']}, needs about "
            f"{math.ceil(r['need_gb'] + r['headroom_gb'])} GB free ({r['need_gb']:g} GB for the model, "
            f"{r['headroom_gb']:g} GB kept free); {r['avail_gb']:g} GB {'free' if read else 'assumed free for this RAM size'}")


def _hybrid_live_speed(sp, h, tr):
    """The hybrid's live model's figures, with its re-read rates as the hybrid runs it: a review press every 10 min
    pauses the live model, so its re-reads reach less far than as a single model (the 12B + MTP on 12 GB: every line
    re-read to 60 msgs/min, against 200 alone; reread.py). The tested tier's figure (a check measures the model alone)."""
    lr = (h.get("live_reread") or {}).get(tr.get("proxy")) if sp else None
    if not lr:
        return sp
    return {**sp, "reread_all": min(lr["all"], sp.get("reread_all") or lr["all"]),
            "reread_most": min(lr["most"], sp.get("reread_most") or lr["most"])}


def hybrid_rows(cls, vendor, installed, machine=None, singles=None, mem=None):
    tr = tier(cls, vendor)
    singles = {r["id"]: r for r in (singles if singles is not None else rows(cls, vendor, installed, None, machine, mem))}
    out = []
    for h in catalogue()["hybrids"]:
        if tr["key"] not in h["tiers"] or h["live"] not in singles or h["reviewer"] not in singles:
            continue
        lv, rv = singles[h["live"]], singles[h["reviewer"]]
        # one model in memory at a time (they swap): the RAM the larger one needs; the read-ahead of both files into the
        # page cache (quick swaps) is nice to have, not required
        needs = [x["ram"]["need_gb"] for x in (lv, rv) if x.get("ram")]
        ram = ram_check(max(needs), mem) if len(needs) == 2 else None
        if ram:
            cache = round((lv["download_gb"] + rv["download_gb"]) * 1e9 / 1024 ** 3, 1)
            ram["readahead_gb"] = cache
            ram["readahead"] = ram["avail_gb"] - ram["need_gb"] - ram["headroom_gb"] >= cache
        out.append({"id": f"{h['live']}+{h['reviewer']}", "live": h["live"], "reviewer": h["reviewer"],
                    "label": f"{lv['label']} live, {rv['label']} reviews",
                    "quality": {"a": h["a"], "b": h["b"], "a_over_no_ai": over_no_ai(h["a"][0]),
                                "b_over_no_ai": over_no_ai(h["b"][0])},
                    "gate_b": h["gate_b"], "live_speed": _hybrid_live_speed(lv["speed"], h, tr),
                    "review_s": rv["review_s"],
                    "installed": lv["installed"] and rv["installed"], "blocked": lv["blocked"] or rv["blocked"],
                    "num_gpu": {"live": lv["num_gpu"], "reviewer": rv["num_gpu"]}, "ram": ram})
    out.sort(key=lambda r: -r["quality"]["b"][0])
    if out:
        out[0]["recommended"] = not _short(out[0])
    return out


# ------------------------------------------------------------------------------------------------ what the page lists
# Owner 2026-11-07: "always have MTP utilized and just not even worry about listing the options without MTP"; "no reason
# to include gpt-oss-safeguard 20B or qwen3.6 35B-A3B, they both fall behind Gemma 4 12B". The catalogue keeps every
# setup for traceability; the page lists only:
# - no plain build of a model that has an MTP build (the MTP twin is the same quality and up to 2-3x faster);
# - no dominated setup: one is left out when another listed one that fits this tier is at least as good on quality and
#   at least as fast (keep-up msgs/min; a hybrid by its live model's), per tier, so a smaller machine still gets what fits.
def _plain_twins():
    return {s["parent"] for s in catalogue()["setups"] if s.get("mtp") and s.get("parent")}


def _short(r):
    return bool(r.get("ram") and r["ram"]["state"] == "short")


def _speed_key(sp):
    """A keep-up figure as a number to compare: "150+" (the study's top) counts as at least 150."""
    if not sp:
        return -1
    return sp["keepup"] + (0.5 if sp.get("keepup_plus") else 0)


def _dominated(r, others, q, sp):
    """r is left out when another row is at least as good on quality (q) and speed (sp), and better on one of them (or,
    when exactly equal, comes first in the catalogue). A setup short of free RAM never hides one that fits."""
    for o in others:
        if o is r:
            continue
        if _short(o) and not _short(r):
            continue
        if q(o) >= q(r) and sp(o) >= sp(r) and (q(o) > q(r) or sp(o) > sp(r) or o["_order"] < r["_order"]):
            return True
    return False


# Not listed on any machine (owner, 2026-11-07: "no reason to include gpt-oss-safeguard 20B or qwen3.6 35B-A3B, they
# both fall behind Gemma 4 12B"): kept in the catalogue for traceability only.
NOT_LISTED = ("gpt-oss-safeguard:", "qwen")


def shown_rows(singles, hybrids):
    """(singles, hybrids) as the page lists them (see above)."""
    plain = _plain_twins() | {r["id"] for r in singles if r["id"].startswith(NOT_LISTED)}
    plain |= {t for h in hybrids for t in (h["live"], h["reviewer"]) if t.startswith(NOT_LISTED)}
    order = {s["id"]: i for i, s in enumerate(catalogue()["setups"])}
    s1 = [{**r, "_order": order.get(r["id"], 99)} for r in singles if r["id"] not in plain]
    s2 = [r for r in s1 if not _dominated(r, s1, lambda x: x["quality"]["pct"], lambda x: _speed_key(x["speed"]))]
    if any(r.get("ram_demoted") for r in s1) and not any(r["recommended"] for r in s2):
        # the default setup is short of free RAM here: the best one that fits is recommended instead
        ok = [r for r in s2 if not _short(r)]
        if ok:
            best = max(ok, key=lambda r: (r["quality"]["pct"], -r["_order"]))
            s2 = [{**r, "recommended": r is best} for r in s2]
    h1 = [{**h, "_order": i} for i, h in enumerate(hybrids) if h["live"] not in plain and h["reviewer"] not in plain]
    h2 = [h for h in h1 if not _dominated(h, h1, lambda x: x["quality"]["b"][0], lambda x: _speed_key(x["live_speed"]))]
    strip = lambda rs: [{k: v for k, v in r.items() if k != "_order"} for r in rs]
    return strip(s2), strip(h2)


# ------------------------------------------------------------------------------------------------ the machine file
def machine_key(host):
    return re.sub(r"[^A-Za-z0-9._-]", "_", host or "unknown")[:64]


def gpu_sig(specs):
    return f"{specs.get('gpu_vendor') or 'none'}-{specs.get('vram_mb') or 0}"


def _aside_base(host, sig):
    """Where a host's file for one GPU waits while another GPU's file has the host's name (PH9, audit 2026-11-10)."""
    return os.path.join(machines_dir(), machine_key(host) + "." + machine_key(sig))


def _read_json(p):
    try:
        m = json.load(open(p))
    except (OSError, ValueError):
        return {}
    return m if isinstance(m, dict) else {}


def load_machine(host, specs=None):
    """This machine's file, or {} when there is none or it cannot be read. Two machines with one host name share the
    folder, and one machine reads another GPU when its driver fails to answer at a start: the GPU tells them apart. The
    file under the host's name is for the GPU that saved last; another GPU's file waits beside it (<host>.<gpu>.json,
    PH9) and is read when that GPU starts. A speed check without its numbers is dropped (PH1)."""
    m = _read_json(os.path.join(machines_dir(), machine_key(host) + ".json"))
    if specs is not None and m.get("gpu") and m["gpu"] != gpu_sig(specs):
        m = _read_json(_aside_base(host, gpu_sig(specs)) + ".json")
        if m.get("gpu") != gpu_sig(specs):
            return {}
    if isinstance(m.get("speed"), dict) and not all(valid_speed(v) for v in m["speed"].values()):
        m["speed"] = {k: v for k, v in m["speed"].items() if valid_speed(v)}
    return m


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_machine(host, specs, m, env_caps=None):
    """The JSON and the .env ./start.sh reads (key=value lines it parses, never sources). Tags are checked first.
    env_caps: model -> the cap valid for the machine's current tier (third review 1: a cap stored for another tier is not
    written), or UNKNOWN_CAP for an MTP build whose cap is not known here (./start.sh then does not warm it); without it
    the stored values are written."""
    m = {**m, "host": host, "gpu": gpu_sig(specs), "saved": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    c = m.get("choice") or {}
    for t in (c.get("live"), c.get("reviewer")):
        if t is not None and not TAG.match(t):
            raise ValueError(f"not a model tag: {t!r}")
    base = os.path.join(machines_dir(), machine_key(host))
    # PH9 (audit 2026-11-10): the host's file written for another GPU is set aside under that GPU (its .env with it),
    # never overwritten; this GPU's own aside copy, if any, is replaced by the file written now
    old = _read_json(base + ".json")
    if old.get("gpu") and old["gpu"] != m["gpu"]:
        aside = _aside_base(host, old["gpu"])
        os.replace(base + ".json", aside + ".json")
        if os.path.exists(base + ".env"):
            os.replace(base + ".env", aside + ".env")
    _write(base + ".json", json.dumps(m, indent=1))
    mine = _aside_base(host, m["gpu"])
    for ext in (".json", ".env"):
        try:
            os.remove(mine + ext)
        except OSError:
            pass
    caps = m.get("caps") or {}
    if env_caps is not None:
        cap = lambda t: ("" if env_caps.get(t) is None else env_caps[t] if env_caps[t] in (UNKNOWN_CAP, ESTIMATE_CAP)
                         else str(int(env_caps[t])))
    else:
        cap = lambda t: "" if t is None or cap_value(caps.get(t)) is None else str(int(cap_value(caps[t])))
    env = [f"# written by the dashboard's model picker ({m['saved']}); ./start.sh reads these keys only",
           f"CW_MACHINE_GPU={m['gpu']}", f"CW_AI_MODE={c.get('mode', 'single')}"]
    if c.get("live"):
        env += [f"CW_LIVE_MODEL={c['live']}", f"CW_LLM_NUM_GPU={cap(c['live'])}"]
    if c.get("reviewer"):
        env += [f"CW_REVIEW_MODEL={c['reviewer']}", f"CW_REVIEW_NUM_GPU={cap(c['reviewer'])}"]
    _write(base + ".env", "\n".join(env) + "\n")
    return m


# ------------------------------------------------------------------------------------------------ Ollama's log
OFFLOAD = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
ABORT = re.compile(r"llama-server terminated|runner terminated|out of memory|CUDA error|HIP error|ROCm error|"
                   r"GGML_ASSERT|signal: (aborted|killed|segmentation)", re.I)


class OllamaLog:
    """The lines ./start.sh's host-side tail copies out of `podman logs -f cw-ollama` (offload lines, runner starts and
    aborts only). None of it is chat. Missing file: the tail is not running (the check says so)."""

    def __init__(self, path):
        self.path = path

    def available(self):
        return bool(self.path) and os.path.exists(self.path)

    def size(self):
        try:
            return os.path.getsize(self.path)
        except (OSError, TypeError):
            return 0

    def since(self, offset):
        try:
            with open(self.path, "rb") as f:
                f.seek(offset)
                return f.read().decode("utf-8", "replace").splitlines()
        except (OSError, TypeError):
            return []

    @staticmethod
    def fit(lines):
        """(layers on the GPU, layers in all) of the main model in these lines (the drafter's line is the smaller)."""
        best = None
        for l in lines:
            m = OFFLOAD.search(l)
            if m:
                x, y = int(m.group(1)), int(m.group(2))
                if best is None or y > best[1]:
                    best = (x, y)
        return best

    @staticmethod
    def aborts(lines):
        return sum(1 for l in lines if ABORT.search(l) and "offloaded" not in l)


def cap_from_fit(fit):
    """Ollama's fit -> the cap: none when it fit whole, else the fit minus CAP_MARGIN (H3)."""
    if fit is None:
        return None
    x, y = fit
    return None if x >= y else max(1, x - CAP_MARGIN)


# ------------------------------------------------------------------------------------------------ the speed sample
_WORDS = {
    "hype": ["LETS GO", "W stream", "this is so good", "insane play", "clip that", "no way he hit that", "huge",
             "POG", "gg", "W chat", "that was clean", "actual cinema", "goated"],
    "talk": ["what game is this", "how long has he been live", "is this the new patch", "what rank is he",
             "did he beat the boss yet", "first time watching, this is fun", "what settings does he use",
             "the music is nice today", "who is he playing with", "is there a discord", "when is the next stream",
             "i missed the start what happened", "he should try the other route", "that build looks strong"],
    "banter": ["chat is cooked", "bro is lagging", "skill issue", "he is so washed lol", "ratio", "cope",
               "that was mid", "dogwater aim", "L take", "no shot that was legit", "bro forgot how to play"],
    "jab": ["you are actually terrible at this", "nobody likes this streamer", "get a real job", "worst stream ever",
            "this guy is a clown", "stop crying about it", "you are so annoying"],
}
_NAMES = ["pixelfox", "gamerdude42", "nightowl_7", "blueberrytea", "kappa_king", "silentstorm", "mapleleaf99",
          "tacotuesday", "rngesus", "lowkeyhigh", "frostbyte", "sunnyside", "quietwolf", "neonrider", "cozycat",
          "potatoaim", "ironclad", "zenmode", "chillvibes", "stormchaser"]


def _line(rng, kinds=("hype", "talk", "banter", "talk", "hype", "jab")):
    a = rng.choice(_WORDS[rng.choice(kinds)])
    return a if rng.random() < 0.6 else f"{a} {rng.choice(_WORDS[rng.choice(kinds)])}"


def speed_sample(seed=20261107, persons=12, windows=6):
    """Fixed synthetic chat (no real chatter's text) shaped like the study's speed runs: person reads with 12 recent
    messages and 20 lines of room, re-read windows of 12 lines with 6 lines of context, and 1-line windows."""
    rng = random.Random(seed)
    P = []
    for i in range(persons):
        name = _NAMES[i % len(_NAMES)]
        mix = ("jab", "banter", "talk") if i % 3 == 0 else ("hype", "talk", "banter")
        msgs = [{"text": _line(rng, mix)} for _ in range(12)]
        room = [(0, rng.choice(_NAMES), _line(rng)) for _ in range(20)]
        P.append(({"display": name, "recent_messages": msgs, "messages": 12 + i, "first_msg_seen": i % 4 == 0,
                   "account_age_days": float(30 + 40 * i), "sub_months": i % 5,
                   "friction": {"instigated": i % 2, "joined_heated_room": 0}}, room))
    W = []
    for _ in range(windows):
        ctx = [f"{rng.choice(_NAMES)}: {_line(rng)}" for _ in range(6)]
        W.append((ctx, [f"{rng.choice(_NAMES)}: {_line(rng)}" for _ in range(12)]))
    return P, W


def summarize(person_s, w1, w12):
    """speed2's numbers from the check's seconds: person-read mean and median; window a + b*k fitted on the medians
    of k = 1 and k = 12; the lossless keep-up (msgs/min) from the table."""
    med = lambda xs: sorted(xs)[len(xs) // 2] if len(xs) % 2 else sum(sorted(xs)[len(xs) // 2 - 1:len(xs) // 2 + 1]) / 2
    P = sum(person_s) / len(person_s)
    m1, m12 = med(w1), med(w12)
    b = max(0.0, (m12 - m1) / 11)
    a = max(0.0, m1 - b)
    return {"person_s": round(P, 2), "person_median_s": round(med(person_s), 2), "a": round(a, 3), "b": round(b, 4),
            "window12_s": round(m12, 2), **check_numbers(round(P, 2), round(a, 3), round(b, 4)),
            "n_person": len(person_s), "n_w1": len(w1), "n_w12": len(w12)}


# ------------------------------------------------------------------------------------------------ the page cache
def models_dir():
    """cw-ollama's store as this process sees it (the project's models/ollama; CW_OLLAMA_MODELS overrides)."""
    return os.environ.get("CW_OLLAMA_MODELS") or os.path.join(ROOT, "models", "ollama")


def blob_files(tag, root=None):
    """The blob files a model loads (weights, drafter, projector; layers over 1 MB), from its manifest in either of
    Ollama's layouts (manifests-v2/ollama.com/... for models created on 0.40, manifests/registry.ollama.ai/... for
    pulled ones). [] when the manifest is not found."""
    root = root or models_dir()
    name, _, t = tag.partition(":")
    t = t or "latest"
    lib = name if "/" in name else f"library/{name}"
    for base in (("manifests-v2", "ollama.com"), ("manifests", "registry.ollama.ai")):
        p = os.path.join(root, "models", *base, lib, t)
        try:
            man = json.load(open(p))
        except (OSError, ValueError):
            continue
        out = []
        for layer in man.get("layers", []):
            d = str(layer.get("digest", ""))
            if layer.get("size", 0) > 1_000_000 and re.match(r"^sha256:[0-9a-f]{64}$", d):
                out.append((os.path.join(root, "models", "blobs", "sha256-" + d[7:]), int(layer["size"])))
        return out
    return []
