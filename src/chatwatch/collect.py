"""Anonymous, read-only Twitch chat collector.

Logs in as justinfanNNNNN (no account, no token, cannot post) and appends every
raw IRC line it receives to data/raw/<YYYY-MM-DD>/<channel>[.<run>].log as
'<recv_ms> <line>'. Raw lines keep all tags, including the moderation events
(CLEARCHAT = timeout/ban, CLEARMSG = deleted message) that serve as weak
labels, and every message is stored before any deletion arrives, so deleted
messages are kept in full (docs/TRAINING_CHANNELS.md).

Used two ways:
  - as a component (Collector) by the pipeline control page (pipeline.py):
    start/stop, and change the channel set while running (JOIN/PART live)
  - from the command line:
        python -m chatwatch.collect --channels a,b,c [--hours 6] [--run r1]   (no status page: the old
        stand-alone page and env/collect.sh were cut on 2026-11-01; the log lines say what it does)

A `run` id makes each collection run write its own files, so two collectors
(e.g. during a handoff) never append to the same file. Readers treat
"somechannel.log" and "somechannel.r20261021.log" as the same channel and de-duplicate
overlapping lines by message id (replay.iter_events).
"""
import argparse
import asyncio
import os
import random
import signal
import time

import websockets

URL = "wss://irc-ws.chat.twitch.tv:443"
JOIN_BATCH = 20          # Twitch allows 20 JOINs / 10 s for unverified clients (a JOIN of 20 channels is 20)
JOIN_PAUSE = 10.5        # 205 channels: 11 batches, ~105 s (was 15 per 11 s: 143 s; audit PB-6)
FSYNC_EVERY = 300        # s: the open raw files are fsynced this often, off the event loop (N12): a write the
                         # share lost (a CIFS stall) then shows as a disk error instead of a silent NUL-filled spot
LOST_KEEP = 50           # status.lost_writes keeps this many of the newest


class RawSink:
    """Per-channel, per-UTC-day append-only files."""

    def __init__(self, root, run=""):
        self.root = root
        self.run = run
        self.files = {}
        self.counts = {}

    def write(self, chan, line, ms):
        day = time.strftime("%Y-%m-%d", time.gmtime(ms / 1000))
        key = (day, chan)
        f = self.files.get(key)
        if f is None:
            for k in [k for k in self.files if k[0] != day]:
                self.files.pop(k).close()
            d = os.path.join(self.root, day)
            os.makedirs(d, exist_ok=True)
            name = (chan or "_server") + (f".{self.run}" if self.run else "") + ".log"
            f = self.files[key] = open(os.path.join(d, name), "a", encoding="utf-8", buffering=1 << 16)
        f.write(f"{ms} {line}\n")
        self.counts[chan] = self.counts.get(chan, 0) + 1

    def flush(self):
        for f in self.files.values():
            f.flush()

    def fds(self):
        """(file name, fd) of every open file, for fsync_all in another thread."""
        return [(f.name, f.fileno()) for f in self.files.values() if not f.closed]

    @staticmethod
    def fsync_all(fds):
        """fsync each fd; (files synced, the errors that mean lost data (EIO, ENOSPC, ...) as "name: error").
        A file closed meanwhile (EBADF; EINVAL from an fd reused for something else) is neither."""
        import errno
        synced, bad = 0, []
        for name, fd in fds:
            try:
                os.fsync(fd)
                synced += 1
            except OSError as e:
                if e.errno not in (errno.EBADF, errno.EINVAL):
                    bad.append(f"{os.path.basename(name)}: {e}")
        return synced, bad

    def close(self):
        for f in self.files.values():
            f.close()
        self.files.clear()

    def reset(self):
        """Drop the open files after a write error; each is reopened by its next line."""
        for f in self.files.values():
            try:
                f.close()
            except OSError:
                pass
        self.files.clear()


def _channel_of(line):
    # '@tags :prefix CMD #chan ...' -> chan
    i = line.find(" #")
    if i < 0:
        return ""
    j = line.find(" ", i + 2)
    return line[i + 2: j if j > 0 else None].lower()


class Status:
    """Per-channel activity, connection and disk, for the pipeline page."""

    def __init__(self, channels, out_root, deadline=None):
        self.out_root = out_root
        self.started = time.time()
        self.deadline = deadline
        self.connected = False
        self.joined = 0
        self.reconnects = 0
        self.last_error = ""
        self.disk_errors = 0
        # the two disk states are kept apart (audit_20261101b PB5-1, PF5-3): writes_failing is true while a write fails
        # now (chat is not being saved); disk_error is that write error, else an fsync report ("the share lost written
        # chat": chat already written may be lost while new chat is saved), else ""
        self.writes_failing = False
        self.disk_error = ""
        # every fsync that said the share lost written chat, {"utc", "file", "error"}, kept while the collector runs
        # (the newest LOST_KEEP): the files tools/raw_integrity.py should check for NUL-filled spots (PB4-8)
        self.lost_writes = []
        self.ch = {}
        self.set_channels(channels)
        self._disk = (0, 0.0)
        self._walking = False

    def set_channels(self, channels):
        for c in channels:
            self.ch.setdefault(c, {"msgs": 0, "actions": 0, "deletes": 0, "last_msg": 0.0, "minutes": {}})
        for c in [c for c in self.ch if c not in channels]:
            del self.ch[c]

    def note(self, chan, line, now):
        c = self.ch.get(chan)
        if c is None:
            return
        if " PRIVMSG #" in line:
            c["msgs"] += 1
            c["last_msg"] = now
            m = int(now // 60)
            c["minutes"][m] = c["minutes"].get(m, 0) + 1
            if len(c["minutes"]) > 20:
                for k in sorted(c["minutes"])[:-15]:
                    del c["minutes"][k]
        elif " CLEARMSG #" in line:
            c["actions"] += 1
            c["deletes"] += 1
        elif " CLEARCHAT #" in line:
            c["actions"] += 1

    def disk_bytes(self):
        """Bytes of raw chat on disk, refreshed every 30 s by a worker thread: the walk of the share takes
        0.2-0.3 s and grows with the files, and status() runs on the collector's event loop (pipeline
        review L6). The first call walks once inline so the number is never missing."""
        if self._disk[1] == 0.0:
            self._walk()
        elif time.time() - self._disk[1] > 30 and not self._walking:
            import threading
            self._walking = True
            threading.Thread(target=self._walk, daemon=True, name="raw-disk-walk").start()
        return self._disk[0]

    def _walk(self):
        try:
            total = 0
            for root, _, files in os.walk(self.out_root):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:                  # rotated or removed while walking
                        pass
            self._disk = (total, time.time())
        finally:
            self._walking = False

    def snapshot(self):
        now = time.time()
        rows = []
        for name, c in self.ch.items():
            cur = int(now // 60)
            last5 = sum(v for k, v in c["minutes"].items() if cur - 5 < k <= cur)
            idle = now - c["last_msg"] if c["last_msg"] else None
            state = ("active" if last5 >= 25 else "quiet" if idle is not None and idle < 1800 else "offline")
            rows.append({"channel": name, "msgs": c["msgs"], "actions": c["actions"], "deletes": c["deletes"],
                         "per_min": round(last5 / 5, 1), "idle_s": round(idle) if idle is not None else None,
                         "state": state})
        rows.sort(key=lambda r: (-r["per_min"], -r["msgs"]))
        return {"now": now, "started": self.started, "deadline": self.deadline, "connected": self.connected,
                "joined": self.joined, "reconnects": self.reconnects, "last_error": self.last_error,
                "disk_errors": self.disk_errors, "disk_error": self.disk_error, "writes_failing": self.writes_failing,
                "lost_writes": list(self.lost_writes),
                "channels": rows, "total_msgs": sum(r["msgs"] for r in rows),
                "total_actions": sum(r["actions"] for r in rows),
                "total_deletes": sum(r["deletes"] for r in rows),
                "msgs_per_min": round(sum(r["per_min"] for r in rows), 1),
                "active": sum(r["state"] == "active" for r in rows), "disk_bytes": self.disk_bytes(),
                "out": self.out_root}


async def _pause(seconds):
    await asyncio.sleep(seconds)


async def _join_pause(seconds):
    await asyncio.sleep(seconds)


class Collector:
    """One anonymous connection; channels can be changed while running."""
    recv_timeout = 2              # s: how often the loop checks stop, the deadline and channel changes
    idle_s = 360                  # Twitch PINGs about every 5 min: 6 min of nothing is a dead connection

    def __init__(self, channels, out_root="data/raw", run="", deadline=None, stats_every=60, log=print, log_every=None):
        self.want = list(dict.fromkeys(channels))
        self.joined = set()
        self.sink = RawSink(out_root, run)
        self.status = Status(self.want, out_root, deadline)
        self.deadline = deadline
        self.run_id = run
        self.stats_every = stats_every
        self.log_every = stats_every if log_every is None else log_every   # the lines/channel log line (PF-25)
        self.log = log
        self._stop = asyncio.Event()
        self._dirty = asyncio.Event()
        self._unflushed = False          # lines written since the last flush
        self._sync_task = None           # the JOIN/PART pacing, beside the receive loop (PB-6)
        self._fsync_fut = None           # the fsync of the open raw files running in a thread (N12)
        self._write_error = ""           # a write failing now (PB5-1): cleared by a flush of new lines
        self._fsync_error = ""           # an fsync's report: cleared only by a clean fsync of at least one file (PB4-8)
        self._last_fsync = time.time()

    def stop(self):
        self._stop.set()

    def set_channels(self, channels):
        """Change the channel set; applied live with JOIN/PART."""
        self.want = list(dict.fromkeys(channels))
        self.status.set_channels(self.want)
        self._dirty.set()

    async def _sync(self, ws):
        add = [c for c in self.want if c not in self.joined]
        drop = [c for c in self.joined if c not in self.want]
        for i in range(0, len(drop), JOIN_BATCH):
            await ws.send("PART " + ",".join("#" + c for c in drop[i:i + JOIN_BATCH]))
        for c in drop:
            self.joined.discard(c)
        for i in range(0, len(add), JOIN_BATCH):
            await ws.send("JOIN " + ",".join("#" + c for c in add[i:i + JOIN_BATCH]))
            self.joined.update(add[i:i + JOIN_BATCH])
            self.status.joined = len(self.joined)
            if i + JOIN_BATCH < len(add):
                await _join_pause(JOIN_PAUSE)
        self.status.joined = len(self.joined)
        if add or drop:
            self.log(f"[collect] joined {len(self.joined)} channels (+{len(add)} / -{len(drop)})")

    async def run(self):
        backoff = 1.0
        last_stats = time.time()
        last_log = time.time()
        try:
            while not self._stop.is_set():
                try:
                    async with websockets.connect(URL, ping_interval=None, max_size=2 ** 20) as ws:
                        self.joined = set()
                        await ws.send("CAP REQ :twitch.tv/tags twitch.tv/commands")
                        await ws.send("PASS SCHMOOPIIE")
                        await ws.send(f"NICK justinfan{random.randint(10000, 99999)}")
                        # joining runs beside the receive loop (PB-6): channels already joined record,
                        # PINGs are answered and lines flushed while the rest join (~105 s for 205)
                        sync = self._sync_task = asyncio.create_task(self._sync(ws))
                        self.status.connected = True
                        backoff = 1.0
                        last_rx = time.time()
                        while not self._stop.is_set():
                            if self.deadline and time.time() > self.deadline:
                                self._stop.set()
                                break
                            if sync.done():
                                if sync.exception() is not None:
                                    raise sync.exception()       # a send on a dead connection: reconnect
                                if self._dirty.is_set():
                                    self._dirty.clear()
                                    sync = self._sync_task = asyncio.create_task(self._sync(ws))
                            try:
                                data = await asyncio.wait_for(ws.recv(), timeout=self.recv_timeout)
                                last_rx = time.time()
                            except asyncio.TimeoutError:
                                data = ""
                                if time.time() - last_rx > self.idle_s:     # a half-open connection
                                    raise ConnectionError(f"nothing received for {self.idle_s:.0f} s")
                            ms = int(time.time() * 1000)
                            for line in data.split("\r\n"):
                                if not line:
                                    continue
                                if line.startswith("PING"):
                                    await ws.send("PONG :tmi.twitch.tv")
                                    continue
                                if " RECONNECT" in line[:80]:
                                    raise ConnectionError("server asked to reconnect")
                                chan = _channel_of(line)
                                self._write(chan, line, ms)
                                self.status.note(chan, line, ms / 1000)
                            if time.time() - last_stats > self.stats_every:
                                last_stats = time.time()
                                self._flush()
                                if time.time() - last_log >= self.log_every:
                                    last_log = time.time()
                                    top = sorted(self.sink.counts.items(), key=lambda kv: -kv[1])[:8]
                                    self.log(f"[collect] {time.strftime('%H:%M:%S')} lines/channel: "
                                             + ", ".join(f"{c}={n}" for c, n in top))
                except (OSError, websockets.WebSocketException, ConnectionError) as e:
                    self._cancel_sync()
                    self.status.connected = False
                    if self._stop.is_set():
                        break
                    self.status.reconnects += 1
                    self.status.last_error = f"{time.strftime('%H:%M:%S')} {e}"[:200]
                    self.log(f"[collect] disconnected ({e}); retry in {backoff:.0f}s")
                    await _pause(backoff)
                    backoff = min(backoff * 2, 60)
        finally:                      # a stop that timed out cancels us: the files still get closed
            self._cancel_sync()
            self.status.connected = False
            try:
                self.sink.close()
            except OSError as e:
                self._disk_failed(e)
            self.log("[collect] stopped")

    def _cancel_sync(self):
        if self._sync_task is not None and not self._sync_task.done():
            self._sync_task.cancel()
        self._sync_task = None

    def _write(self, chan, line, ms):
        """A disk error (full disk, EIO) is not a disconnect: reconnecting would lose chat in a
        loop and hide the cause. It is reported once (log, status.disk_error, shown on the
        page) and counted; the line is lost, and the next line tries again."""
        try:
            self.sink.write(chan, line, ms)
            self._unflushed = True
        except OSError as e:
            self._disk_failed(e)

    def _flush(self):
        """Only a flush of new lines proves the disk works again (writes go to a 64 KB buffer). It clears a write
        failure only: an fsync report stays until a clean fsync (PB4-8, PB5-1)."""
        try:
            self.sink.flush()
        except OSError as e:
            self._disk_failed(e)
            return
        if self._write_error and self._unflushed:
            self._write_error = ""
            self._disk_state()
            self.log("[collect] disk writes work again")
        self._unflushed = False
        self._fsync_check()

    def _fsync_check(self, now=None):
        """Every FSYNC_EVERY s: report what the last fsync found, then start the next one in a worker thread
        (an fsync on the share can take seconds in a stall: never on the loop that answers PINGs). A flush
        only reaches the page cache; a write-back the share loses is reported to fsync alone (N12)."""
        now = time.time() if now is None else now
        fut = self._fsync_fut
        if fut is not None and fut.done():
            self._fsync_fut = None
            self._fsync_report(*(fut.result() if fut.exception() is None else (0, [str(fut.exception())])))
        if self._fsync_fut is None and now - self._last_fsync >= FSYNC_EVERY:
            self._last_fsync = now
            fds = self.sink.fds()
            try:
                self._fsync_fut = asyncio.get_running_loop().run_in_executor(None, RawSink.fsync_all, fds)
            except RuntimeError:                  # no running loop (a direct call): inline
                self._fsync_report(*RawSink.fsync_all(fds))

    def _fsync_report(self, synced, bad):
        """A flush reaches only the page cache, so a lost write-back shows here alone: it stays until a later fsync
        that synced at least one file is clean (a flush is not proof, and a tick with no open file, as after a write
        failure closed them all, proves nothing), and each file it named stays in status.lost_writes (audit_20261101b
        PB4-8, PB5-1). It never touches a write failure's state."""
        if bad:
            self.status.disk_errors += len(bad)
            self._fsync_error = self.status.last_error = \
                f"{time.strftime('%H:%M:%S')} disk: the share lost written chat ({bad[0]})"[:200]
            at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            for b in bad:
                name, _, err = b.partition(": ")
                self.status.lost_writes.append({"utc": at, "file": name, "error": err[:120]})
            del self.status.lost_writes[:-LOST_KEEP]
            self._disk_state()
            self.log(f"[collect] DISK ERROR on fsync, chat already written may be lost: {'; '.join(bad[:5])}")
        elif self._fsync_error and synced:
            self._fsync_error = ""
            self._disk_state()
            self.log("[collect] fsync is clean again; the files in lost_writes may hold NUL-filled spots")

    def _disk_failed(self, e):
        """A write failing now: chat is not being saved. It always shows as such (status.writes_failing, and its
        text in disk_error over an fsync report) and is logged once per run of failures (PB5-1)."""
        self.status.disk_errors += 1
        self.sink.reset()
        self._unflushed = False
        first = not self._write_error
        self._write_error = self.status.last_error = f"{time.strftime('%H:%M:%S')} disk: {e}"[:200]
        self._disk_state()
        if first:
            self.log(f"[collect] DISK ERROR, chat is not being saved: {e}")

    def _disk_state(self):
        self.status.writes_failing = bool(self._write_error)
        self.status.disk_error = self._write_error or self._fsync_error


# ---------------------------------------------------------------- channel lists and the command line
def group_key(title):
    """Short key for a "# --- <title>" section of a channel list (docs/TRAINING_CHANNELS.md)."""
    t = title.lower()
    return ("heldout" if "held out" in t else "calm" if "calm" in t else "harassment" if "harassment" in t else
            "political" if "political" in t else "gta" if "gta" in t else "drama" if "drama" in t else
            "rage" if "rage" in t else "competitive" if "competitive" in t else t)


def channel_groups(path):
    """{channel: group key} from the "# --- " section headings of a list file."""
    out, cur = {}, None
    for line in open(path):
        s = line.strip()
        if s.startswith("# ---"):
            cur = group_key(s[5:].strip())
        elif s and not s.startswith("#"):
            out.setdefault(s.split("#")[0].strip().lower(), cur)
    return out


def read_channel_file(path):
    chans = []
    for row in open(path):
        row = row.split("#", 1)[0].strip().lower()
        if row:
            chans.append(row)
    return list(dict.fromkeys(chans))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", default="")
    ap.add_argument("--channels-file")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--hours", type=float, default=0, help="0 = run until stopped")
    ap.add_argument("--until", type=float, default=0, help="stop at this unix time (overrides --hours)")
    ap.add_argument("--run", default="", help="run id: files become <channel>.<run>.log")
    a = ap.parse_args()
    chans = [c.strip().lower().lstrip("#") for c in a.channels.split(",") if c.strip()]
    if a.channels_file:
        chans += read_channel_file(a.channels_file)
    chans = list(dict.fromkeys(chans))
    if not chans:
        ap.error("no channels")
    deadline = a.until or (time.time() + a.hours * 3600 if a.hours else None)

    async def go():
        col = Collector(chans, a.out, a.run, deadline, log=lambda m: print(m, flush=True))
        loop = asyncio.get_running_loop()
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(s, col.stop)
            except NotImplementedError:
                pass
        await col.run()

    asyncio.run(go())


if __name__ == "__main__":
    main()
