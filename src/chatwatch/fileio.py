"""Small file helpers shared by the pipeline page, its jobs and the tools they run.

read_held_out: the held-out asset, normalised the way channel lists are read
(lower-case, inline "# ..." comments stripped), so a capitalised or commented
name can never stop a channel from being held out.
durable_write / atomic_json / atomic_text / dump_pickle: state files written whole or not at all, and
checked on disk before they replace the old file (docs/review/audit_20261029b N1).
load_json / load_pickle: read a state file, falling back to its previous copy (.prev) when it fails; the next
keep_prev save of that file then keeps the good .prev and sets the damaged file aside as <path>.damaged.
read_jsonl: label files read past a torn last line.

Why the check (N1, 2026-10-30): the data folder is a CIFS share. During a network stall a 16 MiB block of a
3.5 GB pickle became zeros, nothing reported an error, and os.replace installed the damaged file. A write is
now flushed and fsynced inside its `with` (a lost write-back surfaces as an error there), read back from the
server with the page cache dropped and compared with what was written, and only then put in place; the file
it replaces can be kept as <path>.prev.

Transient share errors (audit_20261101b PB4-2): while the share is mounted soft a stall surfaces as EAGAIN
(errno 11), EIO or a timeout on an open, write, fsync or read-back. durable_write and append_line try again
after RETRY_DELAYS (a few seconds in all) before they give up; every attempt is checked on disk as above.
"""
import errno
import hashlib
import json
import os
import re
import time

LOGIN = re.compile(r"^[a-z0-9_]{3,25}$")
PREV = ".prev"
DAMAGED = ".damaged"     # a current file that failed to load, set aside by the next save (one copy kept)
CHUNK = 4 << 20
# files whose last load in this process fell back to .prev (the current one did not load): their next
# keep_prev save must not rotate the damaged file over the good .prev (audit_20261101 PB3-6)
FELL_BACK = set()
# errors a share stall gives that a later attempt can get past (not a full disk, a permission or a missing folder)
TRANSIENT = {errno.EAGAIN, errno.EINTR, errno.EIO, errno.EBUSY, errno.ETIMEDOUT, errno.ESTALE, errno.ECONNRESET,
             errno.ECONNABORTED, errno.EHOSTDOWN, errno.EHOSTUNREACH, errno.ENETUNREACH, errno.ENETRESET}
RETRY_DELAYS = (0.5, 1.0, 2.0, 4.0)     # s between attempts: 5 attempts, 7.5 s of waiting at most (PB4-2)


class ReadBackMismatch(OSError):
    """The temp file read back from disk is not what was written (a write-back the share lost)."""


def transient(e):
    """Is this error one a share stall gives, worth another attempt?"""
    return isinstance(e, ReadBackMismatch) or (isinstance(e, OSError) and e.errno in TRANSIENT)


def retrying(fn, delays=None, on_retry=None):
    """fn(), tried again after each of delays (RETRY_DELAYS) while it raises a transient error; the last
    error is raised. on_retry(error, attempt) is told of each failed attempt that is tried again."""
    delays = RETRY_DELAYS if delays is None else delays
    for i in range(len(delays) + 1):
        try:
            return fn()
        except OSError as e:
            if not transient(e) or i == len(delays):
                raise
            if on_retry is not None:
                on_retry(e, i + 1)
            time.sleep(delays[i])


class Corrupt(RuntimeError):
    """A state file and its previous copy both failed to load (or it failed and there is no previous copy)."""

    def __init__(self, path, errors):
        self.path, self.errors = path, errors
        super().__init__(f"{path} cannot be loaded: " + "; ".join(errors))


def channel_name(line):
    """'  SomeChannel   # clean test' -> 'somechannel'; '' for blank and comment lines."""
    return line.split("#", 1)[0].strip().lower()


def read_held_out(path, missing_ok=False):
    """{channel} from the held-out asset. A missing asset is an error unless
    missing_ok (training without it would use the held-out channels)."""
    if missing_ok and not os.path.exists(path):
        return set()
    return {c for c in map(channel_name, open(path)) if c}


class _HashWriter:
    """What is written through it, counted and hashed on the way to the file."""

    def __init__(self, f):
        self.f, self.h, self.n = f, hashlib.sha256(), 0

    def write(self, b):
        self.h.update(b)
        # bytes, or a pickle.PickleBuffer (protocol 5 hands a numpy array of 64 KiB or more over as one, and it
        # has no len(): audit_20261101 PB3-4); nbytes is the byte count of either
        self.n += memoryview(b).nbytes
        return self.f.write(b)


def _drop_cache(fd):
    """Forget the file's cached pages, so the read-back that follows comes from the disk (the share's
    server), not from memory."""
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


def _fsync_dir(d):
    """Make the rename itself durable (a no-op where the file system refuses it, as CIFS may)."""
    try:
        fd = os.open(d or ".", os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def file_sha256(path, limit=None):
    """(sha256 hex, bytes read) of the file, or of its first `limit` bytes."""
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as f:
        while limit is None or n < limit:
            b = f.read(CHUNK if limit is None else min(CHUNK, limit - n))
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def durable_write(path, stream=None, to_path=None, check=None, keep_prev=False, tmp=None):
    """Write a file so that a damaged write never replaces a good file.

    stream(f): writes the content to f (a binary file-like object with write()); what it writes is hashed.
    to_path(tmp): a writer that wants a path (np.savez); give check() then, which must read it all back.
    Steps: the temp file is written inside `with`, flushed and fsynced; its cached pages are dropped and it
    is read back (stream: its size and sha256 must equal what was written; check(tmp), when given, must not
    raise); the current file becomes <path>.prev when keep_prev; os.replace; an fsync of the folder.
    A transient share error (EAGAIN, EIO, a lost write-back, ...: transient()) is tried again from the start
    after each of RETRY_DELAYS, so stream and to_path must be callable more than once (PB4-2).
    Any other failure, or the last attempt's, raises, and leaves the current file (and its .prev) as they were;
    the temp file is removed. Returns the sha256 of what was written (stream) or None."""
    if (stream is None) == (to_path is None):
        raise ValueError("durable_write takes stream or to_path")
    return retrying(lambda: _durable_write_once(path, stream, to_path, check, keep_prev, tmp))


def _durable_write_once(path, stream, to_path, check, keep_prev, tmp):
    d = os.path.dirname(path)
    tmp = tmp or f"{path}.tmp{os.getpid()}"
    digest = None
    try:
        if stream is not None:
            with open(tmp, "wb") as f:
                w = _HashWriter(f)
                stream(w)
                f.flush()
                os.fsync(f.fileno())
                _drop_cache(f.fileno())
            digest = w.h.hexdigest()
            on_disk, n = file_sha256(tmp)
            if n != w.n or on_disk != digest:
                raise ReadBackMismatch(f"{path}: the file read back from disk ({n:,} bytes) is not what was written "
                                       f"({w.n:,} bytes); the old file is kept")
        else:
            to_path(tmp)
            with open(tmp, "rb+") as f:
                os.fsync(f.fileno())
                _drop_cache(f.fileno())
        if check is not None:
            check(tmp)
        if keep_prev and os.path.exists(path):
            if os.path.abspath(path) in FELL_BACK:
                # the current file failed to load and .prev is the good copy this run started from: keep .prev,
                # and set the damaged one aside instead of rotating it over the good one (audit_20261101 PB3-6)
                os.replace(path, path + DAMAGED)
            else:
                os.replace(path, path + PREV)
        os.replace(tmp, path)
        FELL_BACK.discard(os.path.abspath(path))
        _fsync_dir(d)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:              # a stall again: the error that ended the attempt is what counts
                pass
    return digest


def append_line(path, line, delays=None):
    """Append one line (no newline in it) to a log file such as history.jsonl and fsync it (N1: a lost
    write-back is an error here, not a silent hole), trying again on a transient error (PB4-2). A failed attempt
    may still have reached the disk, so the next one first reads the file's tail back from the disk: it appends
    only when the line is not there (after a newline when the tail does not end in one: a NUL-filled spot), so
    the line is never there twice. Raises the last attempt's error."""
    data = line.encode() + b"\n"
    tried = []

    def once():
        prefix = b""
        if tried and os.path.exists(path):
            with open(path, "rb") as f:
                _drop_cache(f.fileno())
                f.seek(max(0, os.fstat(f.fileno()).st_size - max(1 << 16, 2 * len(data))))
                tail = f.read()
            if (b"\n" + data) in (b"\n" + tail):
                # the failed attempt's line is there: only its fsync is left to prove it reached the server
                fd = os.open(path, os.O_WRONLY | os.O_APPEND)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                return
            prefix = b"\n" if tail and not tail.endswith(b"\n") else b""
        tried.append(1)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            buf = prefix + data
            while buf:
                buf = buf[os.write(fd, buf):]
            os.fsync(fd)
        finally:
            os.close(fd)
    retrying(once, delays)


def atomic_json(path, obj, keep_prev=False, **kw):
    """json.dump through durable_write: a crash, a kill or a lost write leaves the old file or the new
    one, never half of one or a damaged one. keep_prev keeps the old one as <path>.prev (load_json)."""
    data = json.dumps(obj, **kw).encode()
    durable_write(path, stream=lambda f: f.write(data), keep_prev=keep_prev)


def atomic_text(path, text, keep_prev=False):
    data = text.encode()
    durable_write(path, stream=lambda f: f.write(data), keep_prev=keep_prev)


def load_with_prev(path, load, log, what):
    """load(p) for path, then path.prev; the first that loads. Corrupt when every existing one fails,
    FileNotFoundError when neither exists."""
    errors = []
    for p in (path, path + PREV):
        if not os.path.exists(p):
            if p == path:
                errors.append(f"{path} is missing")
            continue
        try:
            obj = load(p)
        except MemoryError:
            raise
        except Exception as e:                       # any unpickling or parsing error: try the previous copy
            errors.append(f"{p}: {type(e).__name__}: {str(e)[:200]}")
            continue
        if p != path:
            if os.path.exists(path):
                FELL_BACK.add(os.path.abspath(path))     # the next save keeps this good .prev (PB3-6)
            if log:
                saved = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(os.path.getmtime(p)))
                log(f"[{what or 'state'}] {errors[0]}; using the previous copy {p} (saved {saved})")
        else:
            FELL_BACK.discard(os.path.abspath(path))
        return obj
    if len(errors) == 1 and not os.path.exists(path):
        raise FileNotFoundError(path)
    raise Corrupt(path, errors)


def load_json(path, log=print, what=None):
    """json.load(path), or its previous copy with a log line when it fails (load_pickle)."""
    def load(p):
        with open(p, "rb") as f:
            return json.loads(f.read())
    return load_with_prev(path, load, log, what)


# A pickle written by dump_pickle ends with this trailer after pickle's own STOP (pickle.load ignores what
# follows): the sha256 and length of the pickle's bytes, so a load can tell a damaged file from a good one
# even where the damage would still unpickle (zeros inside an array).
TRAILER = b"\nCWSUM1 "


def dump_pickle(path, obj, keep_prev=False, protocol=None, tmp=None):
    import pickle

    def stream(w):
        pickle.dump(obj, w, protocol=pickle.HIGHEST_PROTOCOL if protocol is None else protocol)
        w.write(TRAILER + f"{w.h.copy().hexdigest()} {w.n}\n".encode())
    return durable_write(path, stream=stream, keep_prev=keep_prev, tmp=tmp)


class _HashReader:
    def __init__(self, f):
        self.f, self.h, self.n = f, hashlib.sha256(), 0

    def read(self, n=-1):
        b = self.f.read(n)
        self.h.update(b)
        self.n += len(b)
        return b

    def readinto(self, buf):
        k = self.f.readinto(buf)
        self.h.update(memoryview(buf)[:k])
        self.n += k
        return k

    def readline(self, n=-1):
        b = self.f.readline(n)
        self.h.update(b)
        self.n += len(b)
        return b


def read_pickle(path):
    """pickle.load(path), checked against the trailer dump_pickle writes (a file without one, written
    before 2026-11-01, is loaded unchecked). A mismatch raises ValueError."""
    import pickle
    with open(path, "rb") as f:
        r = _HashReader(f)
        obj = pickle.load(r)
        tail = f.read(len(TRAILER) + 100)
    if not tail:
        return obj
    if not tail.startswith(TRAILER):
        raise ValueError(f"{len(tail)}+ unexpected bytes after the pickle")
    try:
        want, n = tail[len(TRAILER):].split()
        n = int(n)
    except ValueError:
        raise ValueError("a damaged checksum trailer")
    if n != r.n or want.decode() != r.h.hexdigest():
        raise ValueError(f"checksum mismatch: the file is damaged ({r.n:,} pickle bytes, {n:,} written)")
    return obj


def load_pickle(path, log=print, what=None):
    """read_pickle(path), or its previous copy (path.prev) with a log line naming both when it fails.
    Corrupt when neither loads; FileNotFoundError when neither exists."""
    return load_with_prev(path, read_pickle, log, what)


def read_jsonl(path):
    """(rows, undecodable): every JSON line of a label file. A line cut off by a
    kill mid-write is skipped and counted instead of failing every later step."""
    rows, bad = [], 0
    for line in open(path):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            bad += 1
    return rows, bad


def check_npz(path):
    """Read every array of an .npz back (each member's CRC is checked): raises when the file is damaged
    (ReadBackMismatch, so durable_write writes it again: PB4-2)."""
    import zipfile
    import zlib
    import numpy as np
    try:
        with np.load(path) as z:
            for k in z.files:
                z[k]
    except (ValueError, EOFError, zipfile.BadZipFile, zlib.error) as e:
        raise ReadBackMismatch(f"{path}: the file read back from disk does not load ({type(e).__name__}: {e})") from e


def write_npz(path, compressed=True, keep_prev=False, **arrays):
    """np.savez(_compressed) through durable_write: the file is read back whole (check_npz) before it
    replaces anything. The temp file is hidden (.<name>.tmp<pid>.npz), so no chunk_*.npz glob sees it."""
    import numpy as np
    d, base = os.path.split(path)
    tmp = os.path.join(d, f".{base}.tmp{os.getpid()}.npz")
    save = np.savez_compressed if compressed else np.savez
    durable_write(path, to_path=lambda t: save(t, **arrays), check=check_npz, tmp=tmp, keep_prev=keep_prev)
