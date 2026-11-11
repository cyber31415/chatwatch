"""Model downloads from the page (owner 2026-11-10: "download everything except the AI models ... the No AI option will be
the starting point and if the system can support a model or models then the user will be able to download it using the
GUI"): src/chatwatch/install.py (the dashboard's side) and bin/install_helper.sh (the host's side), through the files in
data/state/install/."""
import json
import os
import shutil
import signal
import subprocess
import time
from collections import namedtuple
from pathlib import Path

import pytest

from chatwatch import install, picker

ROOT = Path(__file__).resolve().parents[1]
E4B = "gemma4mtp:e4b-n4"


@pytest.fixture(autouse=True)
def _dir(tmp_path, monkeypatch):
    d = tmp_path / "install"
    d.mkdir()
    monkeypatch.setenv("CW_INSTALL_DIR", str(d))
    return d


def _alive(d, age=0):
    (d / "alive").write_text(f"{int(time.time() - age)}\n")


def _status(d, **s):
    (d / "status.json").write_text(json.dumps({"error": "", "at": int(time.time()), **s}))


def _lines(d, *recs, tail=""):
    with open(d / "progress.jsonl", "a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
        f.write(tail)


# ------------------------------------------------------------------ the page's figure
def test_progress_sums_drafter_and_model_layers_and_waits_for_a_whole_line(_dir):
    _lines(_dir, {"phase": "drafter", "completed": 50_000_000, "total": 171_785_536},
           {"phase": "drafter", "completed": 171_785_536, "total": 171_785_536},
           {"phase": "pull", "model": "gemma4:e4b-it-qat"}, {"status": "pulling manifest"},
           {"status": "pulling a", "digest": "sha256:a", "total": 5_000_000_000, "completed": 1_000_000_000},
           {"status": "pulling b", "digest": "sha256:b", "total": 1_000_000, "completed": 1_000_000},
           tail='{"status": "pulling a", "digest": "sha256:a", "total": 5000000000, "comp')     # half written
    pr = install.Progress()
    pr.follow((E4B, 1))
    pr.read()
    v = pr.view(E4B)
    assert v["phase"] == "download" and v["done"] == 171_785_536 + 1_000_000_000 + 1_000_000
    assert v["total"] == 6_300_000_000              # the catalogue's 6.3 GB until the parts weigh more
    with open(_dir / "progress.jsonl", "a") as f:    # the rest of the half line arrives: read from where it stopped
        f.write('leted": 4000000000}\n{"phase": "build"}\n')
    pr.read()
    v = pr.view(E4B)
    assert v["phase"] == "build" and v["done"] == 171_785_536 + 4_000_000_000 + 1_000_000


def test_a_new_download_reads_its_file_from_the_start(_dir):
    _lines(_dir, {"phase": "pull", "model": "m"}, {"digest": "sha256:a", "total": 10, "completed": 10})
    pr = install.Progress()
    pr.follow(("x:1", 1))
    pr.read()
    (_dir / "progress.jsonl").write_text("")
    _lines(_dir, {"phase": "pull", "model": "m"}, {"digest": "sha256:z", "total": 4, "completed": 1})
    pr.follow((E4B, 2))
    pr.read()
    assert pr.layers == {"sha256:z": (1, 4)}


def test_state_helper_heartbeat_running_stale_and_pending(_dir):
    pr = install.Progress()
    assert install.state(pr)["helper"] is False
    _alive(_dir)
    _status(_dir, state="running", tag=E4B)
    _lines(_dir, {"phase": "pull", "model": "m"}, {"digest": "sha256:a", "total": 3_000_000_000, "completed": 1_500_000_000})
    s = install.state(pr)
    assert s["helper"] and s["state"] == "running" and s["done"] == 1_500_000_000 and s["phase"] == "download"
    _alive(_dir, age=60)                             # the helper went away mid-download (the dashboard was stopped)
    s = install.state(pr)
    assert s["state"] == "failed" and "press Download" in s["error"]
    _alive(_dir)
    _status(_dir, state="done", tag="gemma4mtp:12b-n2", at=int(time.time()) - 100)
    s = install.state(pr, pending=(E4B, time.time()))
    assert s["state"] == "starting" and s["tag"] == E4B
    _status(_dir, state="running", tag=E4B)          # the helper took it
    assert install.state(pr, pending=(E4B, time.time() - 0.5))["state"] == "running"


# ------------------------------------------------------------------ asking for a download
def test_request_refuses_unlisted_tags_no_helper_a_running_download_and_a_full_disk(_dir, monkeypatch):
    for bad in ("evil:1", "gemma4mtp:e4b-n4; rm -rf /", None, 7, "../x"):
        with pytest.raises(ValueError):
            install.request(bad)
    with pytest.raises(install.Refused, match=r"\./start\.sh .*bin/ollama\.sh install gemma4mtp:e4b-n4"):
        install.request(E4B)
    _alive(_dir)
    _status(_dir, state="running", tag="gemma4mtp:12b-n2")
    with pytest.raises(install.Refused, match="12b-n2 is downloading"):
        install.request(E4B)
    _status(_dir, state="done", tag="gemma4mtp:12b-n2")
    DU = namedtuple("DU", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: DU(100e9, 96e9, 4e9))
    with pytest.raises(install.Refused, match=r"needs about 8 GB free, 4 GB is free"):
        install.request(E4B)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: DU(100e9, 10e9, 90e9))
    assert install.request(E4B)["tag"] == E4B
    assert (_dir / "request").read_text() == E4B + "\n" and not (_dir / "request.tmp").exists()


def test_cancel_only_while_running(_dir):
    assert install.cancel() is False and not (_dir / "cancel").exists()
    _status(_dir, state="running", tag=E4B)
    assert install.cancel() is True and (_dir / "cancel").exists()


def test_every_listed_setup_has_a_size_for_the_disk_check():
    assert all(float(s.get("download_gb") or 0) > 0 for s in picker.catalogue()["setups"])


def test_the_hub_reports_the_download_and_starts_one(_dir, monkeypatch):
    from chatwatch import aictl, llm
    from test_picker import NV12, _hub
    monkeypatch.setenv("CW_MACHINES_DIR", str(_dir.parent / "machines"))
    monkeypatch.setattr(llm, "NUM_GPU_BY_MODEL", {})
    monkeypatch.setattr(llm, "NUM_GPU", "")
    monkeypatch.setattr(llm, "MODEL", "gemma4mtp:31b-n3-text")
    _, hub, _ = _hub(NV12)
    assert hub.ai_status()["install"]["helper"] is False
    with pytest.raises(aictl.Busy, match="start.sh"):
        hub.start_install({"model": E4B})
    _alive(_dir)
    DU = namedtuple("DU", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: DU(100e9, 10e9, 90e9))
    hub.start_install({"model": E4B})
    st = hub.ai_status()["install"]
    assert st["state"] == "starting" and st["tag"] == E4B and st["helper"]


# ------------------------------------------------------------------ the host's helper
def _proj(tmp_path, stub):
    """A copy of the helper beside a stand-in bin/ollama.sh and the real catalogue."""
    p = tmp_path / "proj"
    (p / "bin").mkdir(parents=True)
    (p / "src/chatwatch/assets").mkdir(parents=True)
    shutil.copy(ROOT / "bin/install_helper.sh", p / "bin/install_helper.sh")
    shutil.copy(ROOT / "src/chatwatch/assets/models.json", p / "src/chatwatch/assets/models.json")
    (p / "bin/ollama.sh").write_text(stub)
    return p


STUB = """#!/usr/bin/env bash
# stands in for bin/ollama.sh install: progress lines, then success (or a long wait, for the cancel tests)
echo "$@" > "$CW_INSTALL_PROGRESS.args"
printf '{"phase": "pull", "model": "m"}\\n{"digest": "sha256:a", "total": 100, "completed": 100}\\n' >> "$CW_INSTALL_PROGRESS"
[[ -f "$(dirname "$0")/../slow" ]] && sleep 30
[[ -f "$(dirname "$0")/../fail" ]] && { printf 'writing manifest \\033[K\\r\\033[?25lpulling m failed: no network \\033[K\\n' >&2; exit 1; }
exit 0
"""


def _wait(cond, s=15):
    end = time.time() + s
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return False


def _st(d):
    try:
        return json.loads((d / "status.json").read_text())
    except (OSError, ValueError):
        return {}


@pytest.fixture
def helper(tmp_path, _dir):
    procs = []

    def run(stub=STUB):
        p = _proj(tmp_path, stub)
        h = subprocess.Popen(["bash", str(p / "bin/install_helper.sh")], env={**os.environ, "CW_INSTALL_DIR": str(_dir)},
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        procs.append(h)
        assert _wait(lambda: (_dir / "alive").exists())
        return p, h
    yield run
    for h in procs:
        if h.poll() is None:
            os.killpg(h.pid, signal.SIGKILL)


def test_helper_installs_a_listed_tag_and_refuses_another(helper, _dir):
    p, h = helper()
    (_dir / "request").write_text("evil:latest\n")
    assert _wait(lambda: _st(_dir).get("state") == "failed")
    assert _st(_dir)["tag"] == "evil:latest" and "not a model chatwatch lists" in _st(_dir)["error"]
    (_dir / "request").write_text(E4B + "\n")
    assert _wait(lambda: _st(_dir).get("state") == "done" and _st(_dir).get("tag") == E4B)
    assert (_dir / "progress.jsonl.args").read_text().split() == ["install", E4B]
    assert '"digest": "sha256:a"' in (_dir / "progress.jsonl").read_text()


def test_helper_reports_a_failure_with_its_reason(helper, _dir):
    p, h = helper()
    (p / "fail").write_text("")
    (_dir / "request").write_text(E4B + "\n")
    assert _wait(lambda: _st(_dir).get("state") == "failed")
    err = _st(_dir)["error"]
    assert "pulling m failed: no network" in err and "\x1b" not in err and "[K" not in err and "[?25l" not in err, err


def test_helper_cancel_stops_the_download(helper, _dir):
    p, h = helper()
    (p / "slow").write_text("")
    (_dir / "request").write_text(E4B + "\n")
    assert _wait(lambda: _st(_dir).get("state") == "running")
    (_dir / "cancel").write_text("1\n")
    assert _wait(lambda: _st(_dir).get("state") == "cancelled", s=8)
    assert not (_dir / "cancel").exists()


def test_helper_stopped_with_the_dashboard_stops_its_download_and_says_so(helper, _dir):
    p, h = helper()
    (p / "slow").write_text("")
    (_dir / "request").write_text(E4B + "\n")
    assert _wait(lambda: _st(_dir).get("state") == "running")
    os.killpg(h.pid, signal.SIGTERM)                 # start.sh's exit: kill -- -PGID
    assert _wait(lambda: h.poll() is not None, s=8)
    assert _st(_dir)["state"] == "failed" and "stopped when chatwatch stopped" in _st(_dir)["error"]
    assert not (_dir / "alive").exists()
    # no stand-in left running (its own process group was stopped too)
    assert _wait(lambda: not any(str(p / "bin/ollama.sh") in c for c in _cmdlines()), s=5)


def _cmdlines():
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                out.append(open(f"/proc/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace"))
            except OSError:
                pass
    return out


# ------------------------------------------------------------------ review 2026-11-11
def test_cancel_while_starting_takes_the_request_back(_dir):
    _alive(_dir)
    install.request(E4B)
    assert install.cancel() is True and not (_dir / "request").exists() and not (_dir / "cancel").exists()


def test_a_second_request_waits_for_the_first_to_be_taken(_dir, monkeypatch):
    _alive(_dir)
    DU = namedtuple("DU", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: DU(100e9, 10e9, 90e9))
    install.request(E4B)
    with pytest.raises(install.Refused, match="another download is starting"):
        install.request("gemma4mtp:12b-n2")
    assert (_dir / "request").read_text() == E4B + "\n"


def test_a_status_cut_inside_a_character_still_reads(_dir):
    (_dir / "status.json").write_bytes(b'{"state": "failed", "tag": "x:1", "error": "spinner \xe2\xa0", "at": 1}')
    assert install.read_status()["state"] == "failed"


def test_the_heartbeat_and_status_are_moved_in_place_never_written_in_place():
    sh = (ROOT / "bin/install_helper.sh").read_text()
    assert 'date +%s > "$D/alive.tmp" && mv -f "$D/alive.tmp" "$D/alive"' in sh
    assert 'date +%s > "$D/alive"' not in sh


BUILD_STUB = """#!/usr/bin/env bash
# the download part done, then the build (inside the AI container: the helper cannot stop it, so it does not try)
printf '{"phase": "pull", "model": "m"}\\n{"phase": "build"}\\n' >> "$CW_INSTALL_PROGRESS"
sleep 3
exit 0
"""


def test_helper_does_not_cancel_the_build_and_reports_it_done(helper, _dir):
    p, h = helper(BUILD_STUB)
    (_dir / "request").write_text(E4B + "\n")
    assert _wait(lambda: _st(_dir).get("state") == "running")
    assert _wait(lambda: '"build"' in (_dir / "progress.jsonl").read_text())
    (_dir / "cancel").write_text("1\n")
    assert _wait(lambda: _st(_dir).get("state") == "done", s=10)


def test_helper_stopped_during_the_build_lets_it_finish(helper, _dir):
    p, h = helper(BUILD_STUB)
    (_dir / "request").write_text(E4B + "\n")
    assert _wait(lambda: '"build"' in (_dir / "progress.jsonl").read_text() if (_dir / "progress.jsonl").exists() else False)
    os.killpg(h.pid, signal.SIGTERM)
    assert _wait(lambda: h.poll() is not None, s=10)
    assert _st(_dir)["state"] == "done"


def test_helper_failure_reason_is_plain_ascii(helper, _dir):
    stub = STUB.replace("pulling m failed: no network", "pulling m failed: r\\xc3\\xa9seau \\xe2\\xa0\\x8b")
    p, h = helper(stub)
    (p / "fail").write_text("")
    (_dir / "request").write_text(E4B + "\n")
    assert _wait(lambda: _st(_dir).get("state") == "failed")
    err = _st(_dir)["error"]
    assert "pulling m failed: r" in err and err.isascii(), err
