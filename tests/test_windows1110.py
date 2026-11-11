"""Windows through WSL2 (first Windows test, 2026-11-10: an i7-13700H laptop, 16 GB, Intel graphics only, WSL given 12 GB):
inside WSL the Windows desktop and browser are outside Linux's memory, so the picker keeps only the dashboard's share free
beside a model (picker.RAM_HEADROOM_WSL_GB), and ./start.sh says when that is the case (CW_SPEC_WSL)."""
import os
import re
import subprocess
from pathlib import Path

from chatwatch import picker

# what ./start.sh read in the Windows test's Ubuntu (11,959 MB total, about 10.8 GB available before the dashboard)
WSL16 = {"CW_HOST": "winlap", "CW_SPEC_GPU_VENDOR": "none", "CW_SPEC_VRAM_MB": "0", "CW_SPEC_RAM_MB": "11959",
         "CW_SPEC_RAM_AVAIL_MB": "11093", "CW_SPEC_CORES": "10", "CW_SPEC_THREADS": "20", "CW_SPEC_WSL": "1"}


def _rows(env):
    specs = picker.specs_from_env(env)
    cls = picker.classes(specs, {})
    mem = picker.memory(specs)
    return specs, mem, picker.rows(cls, specs["gpu_vendor"], set(), "0.40.11", None, mem)


def test_start_sh_wsl_flag_is_read():
    assert picker.specs_from_env(WSL16)["wsl"] is True
    assert picker.specs_from_env({**WSL16, "CW_SPEC_WSL": "0"})["wsl"] is False
    assert picker.specs_from_env({k: v for k, v in WSL16.items() if k != "CW_SPEC_WSL"})["wsl"] is False


def test_inside_wsl_only_the_dashboard_is_kept_free_so_the_e4b_is_offered():
    specs, mem, rows = _rows(WSL16)
    assert mem["wsl"]
    e4b = next(r for r in rows if r["id"] == "gemma4mtp:e4b-n4")
    assert e4b["ram"]["headroom_gb"] == picker.RAM_HEADROOM_WSL_GB and e4b["ram"]["state"] != "short"
    assert e4b["recommended"]


def test_the_same_numbers_on_plain_linux_keep_the_desktop_headroom():
    _, mem, rows = _rows({**WSL16, "CW_SPEC_WSL": "0"})
    assert not mem["wsl"]
    assert all(r["id"] != "gemma4mtp:e4b-n4" or r["ram"]["headroom_gb"] == picker.RAM_HEADROOM_GB for r in rows)
    # a native 12 GB Linux box keeps 4 GB beside the model, so the E4B (about 6.6 GB) does not fit in 10.8 GB free
    assert not any(r["id"] == "gemma4mtp:e4b-n4" and r["ram"]["state"] != "short" for r in rows)


# ------------------------------------------------------------------ the Windows launcher (windows/, the release's zip)
ROOT = Path(__file__).resolve().parents[1]
WIN = ROOT / "windows"


def test_the_launcher_clones_this_release():
    ver = (ROOT / "VERSION").read_text().strip()
    m = re.search(r'^\$Version = "([^"]+)"', (WIN / "chatwatch.ps1").read_text(), re.M)
    assert m and m.group(1) == ver


def test_the_launcher_files_suit_windows_and_linux():
    ps1 = (WIN / "chatwatch.ps1").read_bytes()
    assert all(b < 128 for b in ps1), "Windows PowerShell 5.1 reads a file without a BOM as ANSI: ASCII only"
    cmd = (WIN / "Start chatwatch.cmd").read_bytes()
    assert b"chatwatch.ps1" in cmd and b"-ExecutionPolicy Bypass" in cmd
    sh = (WIN / "setup_wsl.sh").read_bytes()
    assert b"\r" not in sh and all(b < 128 for b in sh)
    assert subprocess.run(["bash", "-n", str(WIN / "setup_wsl.sh")]).returncode == 0
    attrs = (ROOT / ".gitattributes").read_text()
    assert "*.cmd -text" in attrs and "*.sh text eol=lf" in attrs and cmd.count(b"\r\n") == cmd.count(b"\n")


def test_the_launcher_installs_what_start_sh_needs_and_stops_wsl_after():
    sh = (WIN / "setup_wsl.sh").read_text()
    assert "iptables" in sh and "podman" in sh and 'clone -q --branch "v$ver"' in sh
    ps1 = (WIN / "chatwatch.ps1").read_text()
    assert "finally" in ps1 and "--terminate $Distro" in ps1 and "--shutdown" in ps1


def _git(*a, cwd=None):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=cwd, check=True, capture_output=True)


def test_the_launcher_keeps_chatwatch_at_its_release_and_puts_a_test_copy_back(tmp_path):
    """setup_wsl.sh's code step (review 2026-11-11, 6): a fresh clone at the tag; a test copy (-From) loses its git so the
    next normal start makes the folder the release again, in place, keeping data/."""
    import shutil
    if not shutil.which("git"):
        import pytest
        pytest.skip("git is not installed here")
    origin = tmp_path / "origin"
    origin.mkdir()
    _git("init", "-q", cwd=origin)
    (origin / "start.sh").write_text("release\n")
    (origin / ".gitignore").write_text("data/\n")
    _git("add", "-A", cwd=origin)
    _git("commit", "-qm", "r", cwd=origin)
    _git("tag", "v9.9.9", cwd=origin)
    app, test_copy = tmp_path / "app", tmp_path / "copy"
    test_copy.mkdir()
    (test_copy / "start.sh").write_text("test\n")
    env = {**os.environ, "CW_REPO_URL": str(origin), "CW_APP": str(app)}
    run = lambda *a: subprocess.run(["bash", str(WIN / "setup_wsl.sh"), "code", *a], env=env, capture_output=True, text=True)
    assert run("9.9.9").returncode == 0 and (app / "start.sh").read_text() == "release\n"
    (app / "data").mkdir()
    (app / "data" / "state.json").write_text("mine")
    r = run("9.9.9", str(test_copy))
    assert r.returncode == 0 and (app / "start.sh").read_text() == "test\n" and not (app / ".git").exists()
    r = run("9.9.9")
    assert r.returncode == 0, r.stderr
    assert (app / "start.sh").read_text() == "release\n" and (app / "data" / "state.json").read_text() == "mine"
