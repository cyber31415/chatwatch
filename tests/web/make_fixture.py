"""The websocket batches and API answers a real dashboard server sends, for the page smoke test
(tests/web/smoke.js). Synthetic chat (no raw chat, no LLM), run through the real Engine and Hub, so the
page is tested against the server's actual record shapes.

    python tests/web/make_fixture.py OUT.json
"""
import asyncio
import json
import os
import sys
import time

sys.path[:0] = ["src", "tests"]
from chatwatch import serve  # noqa: E402
from chatwatch.engine import Engine  # noqa: E402
from test_scenarios import CALM_ROOM, msg  # noqa: E402


def main(out):
    eng = Engine()
    hub = serve.Hub(eng)
    real = time.time        # a live run's wall clock is its chat's: live snapshots age with the wall clock (LB4-4)
    time.time = lambda: eng.clock / 1000 if eng.clock else real()
    hub.log_sessions = False
    hub.status.update(mode="live", channels=["cozy"], running=True, connected=True,
                      session="20261029-170000_cozy")
    batches = []

    class Client:
        async def send_str(self, data):
            batches.append(json.loads(data))
    hub.clients = {Client()}
    toxic = ["you are trash", "worst streamer ever, you are trash", "you are a nigger", "i will kill you",
             "kys idiot", "you are a stupid idiot streamer"]
    t, uids = 0, {}
    for i in range(900):
        ch = "cozy"                     # a live session watches one channel (owner, 2026-11-07)
        if i % 37 == 5:
            who, text = f"heckler{i % 3}", toxic[i % len(toxic)]
        elif i % 53 == 7:
            who, text = f"copier{i % 4}", "you are a nigger lol"
        else:
            who, text = f"regular{i % 23}", CALM_ROOM[i % len(CALM_ROOM)] + f" #{i}"
        first = who.startswith("new")
        if i % 41 == 9:
            who, first = f"newbie{i}", True
        ev = {**msg(who, text, t, first=first, sub=6 if who.startswith("regular") else 0), "channel": ch}
        uids[(ch, who)] = ev["user_id"]
        serve.handle(hub, ev)
        t += 2
        if i % 15 == 14:
            hub._last_room = 0
            asyncio.run(hub._flush())
    # the owner features of 2026-11-01: 'oldtimer' is listed for 3 recent insults and has an older one beyond the
    # 40 recent messages (the watchlist card's dropdown); 'gamebot' is then given the owner's bot role (its lines
    # come as bot_msg records)
    for k, text in enumerate(["you are a stupid idiot streamer"] + [CALM_ROOM[j % len(CALM_ROOM)] + f" ~{j}" for j in range(45)]
                             + ["you are trash", "worst streamer ever, you are trash", "you are a stupid idiot streamer"]):
        for who, line in (("oldtimer", text), ("gamebot", f"!points giveaway round {k}")):
            ev = {**msg(who, line, t), "channel": "cozy"}
            uids[("cozy", who)] = ev["user_id"]
            serve.handle(hub, ev)
            t += 1
        if k % 15 == 14:
            hub._last_room = 0
            asyncio.run(hub._flush())
    eng.owner_bots = frozenset({uids[("cozy", "gamebot")]})
    for k in range(2):
        serve.handle(hub, {**msg("gamebot", f"!points giveaway, marked {k}", t), "channel": "cozy"})
        t += 1
    hub._last_room = 0
    asyncio.run(hub._flush())
    prof, evidence = {}, {}
    for (ch, who), uid in uids.items():
        if (who.startswith("heckler") or who in ("oldtimer", "gamebot")) and ch == "cozy":
            p = eng.profile(ch, uid, since=None)
            p["person_read"] = None
            p["evidence"] = eng.evidence(ch, uid)                 # as serve.py's /api/profile adds them
            p["owner_role"] = None
            prof[f"{ch}:{uid}"] = p
            evidence[f"{ch}:{uid}"] = {"channel": ch, "user_id": uid, "items": eng.evidence(ch, uid)}
    # the model picker (aictl.py): /api/ai as the server answers it on this machine (12 GB Nvidia) and on the owner's
    # AMD desktop (16 GB), and the status of a hybrid run with chatters to review. Ollama is mocked
    from chatwatch import llm  # noqa: E402
    os.environ.setdefault("CW_MACHINES_DIR", "/tmp/cw_machines_fixture")
    llm.installed_models = lambda timeout=10: ["gemma4mtp:31b-n3-text", "gemma4mtp:12b-n2", "gpt-oss-safeguard:20b"]
    llm.version = lambda timeout=5: "0.40.11"
    hub.review_q = serve.ReadQueue()
    ai = {}
    for name, specs in (("nv12", {"host": "box", "gpu_vendor": "nvidia", "gpu_name": "NVIDIA GeForce RTX 4070 Ti",
                                  "vram_mb": 12282, "ram_mb": 96000, "cores": 16, "threads": 32,
                                  "vram_avail_mb": 11935, "vram_held_mb": 10010, "ram_avail_mb": 90700}),
                        # the same card with the desktop holding 2 GB, and one with 32 GB RAM of which 12 GB is free
                        ("nv12busy", {"host": "box2", "gpu_vendor": "nvidia", "gpu_name": "NVIDIA GeForce RTX 4070 Ti",
                                      "vram_mb": 12282, "ram_mb": 96000, "cores": 16, "threads": 32,
                                      "vram_avail_mb": 9900, "vram_held_mb": 10010, "ram_avail_mb": 90700}),
                        ("nv12low", {"host": "box3", "gpu_vendor": "nvidia", "gpu_name": "NVIDIA GeForce RTX 4070 Ti",
                                     "vram_mb": 12282, "ram_mb": 32000, "cores": 16, "threads": 32,
                                     "vram_avail_mb": 11935, "vram_held_mb": 10010, "ram_avail_mb": 12288}),
                        ("f32low", {"host": "lap32b", "gpu_vendor": "none", "vram_mb": 0, "ram_mb": 31800, "cores": 8,
                                    "threads": 16, "ram_avail_mb": 11264}),
                        # a 12 GB card with 16 GB of RAM (ramfit: the GPU tier, each model where its RAM part fits)
                        ("f8", {"host": "lap8", "gpu_vendor": "none", "vram_mb": 0, "ram_mb": 7800, "cores": 4, "threads": 8}),
                        ("nv12r16", {"host": "box16", "gpu_vendor": "nvidia", "gpu_name": "NVIDIA GeForce RTX 4070 Ti",
                                     "vram_mb": 12282, "ram_mb": 15800, "cores": 8, "threads": 16}),
                        ("f16", {"host": "lap16", "gpu_vendor": "none", "vram_mb": 0, "ram_mb": 15800, "cores": 8,
                                 "ram_avail_mb": 13000,
                                 "threads": 16}),
                        ("f32", {"host": "lap32", "gpu_vendor": "none", "vram_mb": 0, "ram_mb": 31800, "cores": 8,
                                 "ram_avail_mb": 26000,
                                 "threads": 16}),
                        ("amd16", {"host": "amdbox", "gpu_vendor": "amd", "gpu_name": "Navi 32 [Radeon RX 7700 XT / 7800 XT]",
                                   "vram_mb": 16368, "ram_mb": 94208, "cores": 8, "threads": 16,
                                   "vram_avail_mb": 15600, "vram_held_mb": 9000, "ram_avail_mb": 72000}),
                        # an 8 GB Nvidia card (tier E: the text 31B at 15 layers, measured 2026-11-09)
                        ("nv8", {"host": "box8", "gpu_vendor": "nvidia", "gpu_name": "NVIDIA GeForce RTX 4060",
                                 "vram_mb": 8188, "ram_mb": 32000, "cores": 8, "threads": 16}),
                        # the AMD desktop's saved hybrid from before 2026-11-09 (the projector 31B reviewing, its found
                        # cap 40): the reviewer resolves to the text build, the cap carried
                        ("amd16old", {"host": "amdboxold", "gpu_vendor": "amd",
                                      "gpu_name": "Navi 32 [Radeon RX 7700 XT / 7800 XT]",
                                      "vram_mb": 16368, "ram_mb": 94208, "cores": 8, "threads": 16,
                                      "vram_avail_mb": 14702, "vram_held_mb": 9000, "ram_avail_mb": 72000}),
                        # W2 (sweep 2026-11-09): this machine's file as of 11-08 (the projector 31B at 26), read cold
                        # (nothing loaded: 11,942 MiB free, 0 held): the text build comes up at 30, not the carried 26
                        ("nv12cold", {"host": "boxcold", "gpu_vendor": "nvidia", "gpu_name": "NVIDIA GeForce RTX 4070 Ti",
                                      "vram_mb": 12282, "ram_mb": 96000, "cores": 16, "threads": 32,
                                      "vram_avail_mb": 11942, "vram_held_mb": 0, "ram_avail_mb": 88938}),
                        # a machine whose own Check speed of the 31B gave keep-up 40, every line re-read to 20, 80% to 35
                        ("nv12meas", {"host": "boxm", "gpu_vendor": "nvidia", "gpu_name": "NVIDIA GeForce RTX 4070 Ti",
                                      "vram_mb": 12282, "ram_mb": 96000, "cores": 16, "threads": 32,
                                      "vram_avail_mb": 11935, "vram_held_mb": 10010, "ram_avail_mb": 90700})):
        if name == "nv12cold":
            os.makedirs(os.environ["CW_MACHINES_DIR"], exist_ok=True)
            json.dump({"choice": {"mode": "single", "live": "gemma4mtp:31b-n3"},
                       "caps": {"gemma4mtp:31b-n3": {"value": 26, "tier": "A"}}, "gpu": "nvidia-12282"},
                      open(os.path.join(os.environ["CW_MACHINES_DIR"], "boxcold.json"), "w"))
        if name == "nv12meas":
            from chatwatch import picker
            os.makedirs(os.environ["CW_MACHINES_DIR"], exist_ok=True)
            sp = {**picker.summarize([6.3] * 9, [2.7, 2.7], [18.4, 18.4]), "at": "2026-11-09T22:00:00Z", "num_gpu": 30}
            json.dump({"choice": {"mode": "single", "live": "gemma4mtp:31b-n3-text"},
                       "caps": {"gemma4mtp:31b-n3-text": {"value": 30, "tier": "A", "how": "found", "avail_gb": 11.655}},
                       "speed": {"gemma4mtp:31b-n3-text": sp}, "gpu": "nvidia-12282"},
                      open(os.path.join(os.environ["CW_MACHINES_DIR"], "boxm.json"), "w"))
        if name == "amd16old":
            os.makedirs(os.environ["CW_MACHINES_DIR"], exist_ok=True)
            json.dump({"choice": {"mode": "hybrid", "live": "gemma4mtp:12b-n2", "reviewer": "gemma4mtp:31b-n3"},
                       "caps": {"gemma4mtp:31b-n3": {"value": 40, "tier": "A", "how": "found", "avail_gb": 14.358},
                                "gemma4mtp:12b-n2": {"value": 49, "tier": "A", "how": "found", "avail_gb": 14.358}},
                       "gpu": "amd-16368"}, open(os.path.join(os.environ["CW_MACHINES_DIR"], "amdboxold.json"), "w"))
        hub.ai_init(specs=specs, host=specs["host"])
        llm.MODEL = "gemma4mtp:31b-n3-text"
        hub.startup_choice()
        ai[name] = asyncio.run(hub.ai_info())
    hub.ai_init(specs={"host": "box", "gpu_vendor": "nvidia", "vram_mb": 12282, "ram_mb": 96000}, host="box")
    hub.machine["choice"] = {"mode": "hybrid", "live": "gemma4mtp:12b-n2", "reviewer": "gemma4mtp:31b-n3-text"}
    hub.startup_choice()
    ai["status_hybrid"] = hub.ai_status()
    json.dump({"batches": batches, "profiles": prof, "evidence": evidence,
               "uids": {"oldtimer": uids[("cozy", "oldtimer")], "gamebot": uids[("cozy", "gamebot")]},
               "ai": ai},
              open(out, "w"), default=serve._jsonable)
    print(f"{len(batches)} batches, {sum(len(b) for b in batches)} records, {len(prof)} profiles -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fixture.json")
