"""Model downloads from the page (owner 2026-11-10: "download everything except the AI models ... the No AI option will be
the starting point and if the system can support a model or models then the user will be able to download it using the
GUI"). The dashboard runs in a container and cannot run bin/ollama.sh, so ./start.sh runs bin/install_helper.sh beside it
on the host. They talk through files in data/state/install/:
  request         the page's ask: one model tag (written here, taken by the helper)
  cancel          stop the download in progress
  alive           the helper's heartbeat (a Unix time, every second)
  status.json     the helper's {"state": running|done|failed|cancelled, "tag", "error", "at"}
  progress.jsonl  bin/ollama.sh install's progress: Ollama's own pull lines ({"digest", "total", "completed"}), the
                  drafter's ({"phase": "drafter", "total", "completed"}) and the phase changes ({"phase": "pull"|"build"})
  install.log     bin/ollama.sh's output (its last lines are a failure's reason)
The helper takes only a tag the catalogue lists (src/chatwatch/assets/models.json), and so does this side."""
import json
import os
import shutil
import time

from . import picker

ALIVE_S = 10              # a heartbeat older than this: no helper (an old start script, --no-llm, or it was stopped)
PENDING_S = 8             # a request the helper has not taken within this is shown as not started
DISK_SPARE_GB = 2.0       # free disk kept beyond the download itself


class Refused(RuntimeError):
    """A download that cannot start now (the page shows the message)."""


def install_dir():
    return os.environ.get("CW_INSTALL_DIR") or os.path.join(picker.ROOT, "data", "state", "install")


def _p(name):
    return os.path.join(install_dir(), name)


def helper_alive(now=None):
    try:
        with open(_p("alive")) as f:
            t = float(f.read().strip() or 0)
    except (OSError, ValueError):
        return False
    return (now or time.time()) - t < ALIVE_S


def read_status():
    try:
        with open(_p("status.json"), encoding="utf-8", errors="replace") as f:
            s = json.load(f)
        return s if isinstance(s, dict) else {}
    except (OSError, ValueError):
        return {}


def store_path():
    """A folder on the disk the models go to: the Ollama store as the container sees it (bin/cw mounts it read-only at
    CW_OLLAMA_MODELS once it has models), else the default store in this folder, else this folder."""
    for c in (os.environ.get("CW_OLLAMA_MODELS"), os.path.join(picker.ROOT, "models", "ollama"), picker.ROOT):
        if c and os.path.isdir(c):
            return c
    return picker.ROOT


class Progress:
    """progress.jsonl, read as it grows (an offset kept; a new download starts the file again)."""

    def __init__(self):
        self.key = None
        self._reset()

    def _reset(self):
        self.off, self.phase, self.layers, self.drafter, self.model = 0, None, {}, None, None

    def follow(self, key):
        """A new download (another status key: tag and start time) reads the file from the start."""
        if key != self.key:
            self.key = key
            self._reset()

    def read(self):
        try:
            with open(_p("progress.jsonl"), "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size < self.off:
                    self._reset()
                f.seek(self.off)
                chunk = f.read(size - self.off)
        except OSError:
            return
        end = chunk.rfind(b"\n")
        if end < 0:
            return
        self.off += end + 1
        for line in chunk[:end].splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if not isinstance(d, dict):
                continue
            ph = d.get("phase")
            if ph:
                self.phase = ph
                if ph == "pull":
                    self.model = d.get("model")
                elif ph == "drafter":
                    self.drafter = (_int(d.get("completed")), _int(d.get("total")))
                continue
            if d.get("digest") and d.get("total"):
                self.layers[d["digest"]] = (_int(d.get("completed")), _int(d["total"]))

    def view(self, tag):
        """One figure for the page: bytes so far of the whole download (the drafter, then the model), against the
        catalogue's size until Ollama has said what each part weighs."""
        s = picker.setup(tag) or {}
        dd, dt = self.drafter or (0, 0)
        md = sum(c for c, _ in self.layers.values())
        mt = sum(t for _, t in self.layers.values())
        done = dd + md
        total = max(int(float(s.get("download_gb") or 0) * 1e9), dt + mt, done)
        return {"phase": "build" if self.phase == "build" else "download" if self.phase else "starting",
                "done": done, "total": total}


def _int(x):
    try:
        return max(0, int(x or 0))
    except (TypeError, ValueError):
        return 0


def state(prog, pending=None, now=None):
    """What the page shows: {"helper", "state", "tag", "error", "phase", "done", "total"}. pending: (tag, time) of a
    request this server wrote that the helper has not taken yet."""
    now = now or time.time()
    s, alive = read_status(), helper_alive(now)
    out = {"helper": alive, "state": s.get("state"), "tag": s.get("tag"), "error": s.get("error") or ""}
    if pending and now - pending[1] < PENDING_S and not (s.get("tag") == pending[0] and _num(s.get("at")) >= pending[1] - 1):
        return {**out, "state": "starting", "tag": pending[0], "error": "", "phase": "starting", "done": 0,
                "total": int(float((picker.setup(pending[0]) or {}).get("download_gb") or 0) * 1e9)}
    if out["state"] == "running":
        if not alive:
            out.update(state="failed", error="the download stopped when chatwatch stopped: press Download to go on")
        else:
            prog.follow((s.get("tag"), s.get("at")))
            prog.read()
            out.update(prog.view(s.get("tag")))
    return out


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def request(tag, now=None):
    """Ask the helper for a download. Raises ValueError (not a listed model) or Refused (the reason, for the page)."""
    s = picker.setup(tag) if isinstance(tag, str) and picker.TAG.match(tag) else None
    if not s:
        raise ValueError(f"not a model chatwatch lists: {tag!r}")
    if not helper_alive(now):
        raise Refused("downloads need chatwatch started with ./start.sh (without --no-llm); "
                      f"or, in a second terminal: {picker.install_cmd(tag)}")
    cur = read_status()
    if cur.get("state") == "running":
        raise Refused(f"wait: {cur.get('tag')} is downloading")
    if os.path.exists(_p("request")):          # asked for, not yet taken by the helper (it looks every second)
        raise Refused("wait: another download is starting")
    need = float(s.get("download_gb") or 0) + DISK_SPARE_GB
    try:
        free = shutil.disk_usage(store_path()).free / 1e9
    except OSError:
        free = None
    if free is not None and free < need:
        raise Refused(f"not enough disk space: {s['label']} needs about {need:.0f} GB free, {free:.0f} GB is free")
    os.makedirs(install_dir(), exist_ok=True)
    tmp = _p("request.tmp")
    with open(tmp, "w") as f:
        f.write(tag + "\n")
    os.replace(tmp, _p("request"))
    return {"tag": tag, "download_gb": s.get("download_gb")}


def cancel():
    """Stop the download: one not yet taken by the helper is taken back; a running one is stopped by the helper (except
    in its last seconds, the build, which it finishes)."""
    try:
        os.remove(_p("request"))
        return True
    except FileNotFoundError:
        pass
    if read_status().get("state") != "running":
        return False
    with open(_p("cancel"), "w") as f:
        f.write("1\n")
    return True
