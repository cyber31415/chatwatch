"""emotes.py (audit 2026-10-29 LB-9, live_backend test gap 6): no network, the fetch is a fake."""
import json
import os
import threading
import time

from chatwatch import emotes
from chatwatch.emotes import EmoteSets

SEVENTV = {"emote_set": {"emotes": [{"name": "Okayge"}, {"name": "catJAM"}]}}
BTTV = {"channelEmotes": [{"code": "monkaS"}], "sharedEmotes": []}
FFZ = {"sets": {"1": {"emoticons": [{"name": "LULW"}]}}}


def _fake(monkeypatch, answers):
    calls = []

    def get(url, timeout=8):
        calls.append(url)
        for key, ans in answers.items():
            if key in url:
                return ans
        return emotes.NOT_THERE
    monkeypatch.setattr(emotes, "_get", get)
    return calls


def test_a_failed_provider_keeps_its_old_codes_and_is_retried_soon(tmp_path, monkeypatch):
    _fake(monkeypatch, {"7tv.io/v3/users": SEVENTV, "betterttv.net/3/cached/users": BTTV, "frankerfacez.com/v1/room": FFZ})
    es = EmoteSets(str(tmp_path))
    assert es.codes("42") >= {"Okayge", "catJAM", "monkaS", "LULW"}
    path = tmp_path / "42.json"
    d = json.load(open(path))
    d["fetched"] -= 7 * 3600                                       # due for a refresh
    json.dump(d, open(path, "w"))
    _fake(monkeypatch, {"7tv.io/v3/users": None, "betterttv.net/3/cached/users": BTTV, "frankerfacez.com/v1/room": FFZ})
    es2 = EmoteSets(str(tmp_path))
    assert es2.codes("42") >= {"Okayge", "catJAM", "monkaS", "LULW"}   # 7TV failed: its codes are kept
    d = json.load(open(path))
    assert d["failed"] == ["7tv"] and d["retry_at"] - time.time() <= emotes.RETRY_S + 5
    assert "42" in es2.retry


def test_a_404_is_an_empty_set_not_a_failure(tmp_path, monkeypatch):
    _fake(monkeypatch, {"7tv.io/v3/users": emotes.NOT_THERE, "betterttv.net/3/cached/users": BTTV,
                        "frankerfacez.com/v1/room": FFZ})
    es = EmoteSets(str(tmp_path))
    es.codes("43")
    d = json.load(open(tmp_path / "43.json"))
    assert "failed" not in d and d["counts"]["7tv"] == 0


def test_the_cache_is_written_atomically_and_read_without_fetching(tmp_path, monkeypatch):
    _fake(monkeypatch, {"7tv.io/v3/users": SEVENTV, "betterttv.net/3/cached/users": BTTV, "frankerfacez.com/v1/room": FFZ})
    EmoteSets(str(tmp_path)).codes("44")
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
    calls = _fake(monkeypatch, {})
    es = EmoteSets(str(tmp_path), fetch=False)                    # pipeline jobs and replays
    assert "Okayge" in es.codes("44") and not calls


def test_a_corrupt_file_never_breaks_a_room(tmp_path, monkeypatch):
    calls = _fake(monkeypatch, {})
    (tmp_path / "45.json").write_text('{"codes": ["Okay')        # torn write from an older version
    es = EmoteSets(str(tmp_path), fetch=False)
    assert es.codes("45") == es.global_codes and not calls
    es = EmoteSets(str(tmp_path), fetch=True)
    assert es.codes("45") == es.global_codes                       # refetched (nothing there)


def test_a_failing_fill_does_not_leave_the_room_pending(tmp_path, monkeypatch):
    _fake(monkeypatch, {})
    es = EmoteSets(str(tmp_path), background=True)

    def boom(*a):
        raise RuntimeError("disk gone")
    monkeypatch.setattr(es, "_load_or_fetch", boom)
    assert es.codes("46") == es.global_codes
    for _ in range(100):
        if "46" in es.sets:
            break
        time.sleep(0.01)
    assert "46" not in es.pending and es.sets["46"] == es.global_codes and es.errors == 1
    monkeypatch.setattr(es, "_load_or_fetch", lambda *a: {"Okayge"})
    es.retry["46"] = 0                                             # due: the next lookup refills in the background
    es.codes("46")
    for _ in range(100):
        if "Okayge" in es.sets["46"]:
            break
        time.sleep(0.01)
    assert "Okayge" in es.sets["46"] and threading.active_count() >= 1


# ------------------------------------------------------------------ audit 2026-10-29b LB2-15
def test_a_cache_write_that_fails_keeps_the_codes_just_fetched(tmp_path, monkeypatch):
    _fake(monkeypatch, {"7tv.io/v3/users": SEVENTV, "betterttv.net/3/cached/users": BTTV, "frankerfacez.com/v1/room": FFZ})
    es = EmoteSets(str(tmp_path))

    def full(path, d):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(emotes, "_write", full)
    assert es.codes("42") >= {"Okayge", "catJAM", "monkaS", "LULW"}       # not the global codes only
    assert es.errors == 1 and "42" in es.retry


def test_a_retry_time_from_an_earlier_process_is_kept_and_the_global_set_is_retried(tmp_path, monkeypatch):
    # an earlier run's 7TV failure: the file holds retry_at; this run must still retry the room when it is due
    _fake(monkeypatch, {"7tv.io/v3/users": None, "betterttv.net/3/cached/users": BTTV, "frankerfacez.com/v1/room": FFZ,
                        "7tv.io/v3/emote-sets/global": None})
    EmoteSets(str(tmp_path)).codes("42")
    es = EmoteSets(str(tmp_path), background=True)
    assert "global" in es.retry                                             # the global 7TV fetch failed
    es.sets["42"] = frozenset()                                              # as filled from the file
    es._load_or_fetch("42", emotes.CHANNEL, "42")
    assert "42" in es.retry                                                  # the file's retry_at, kept
    _fake(monkeypatch, {"7tv.io": SEVENTV, "betterttv.net/3/cached/users": BTTV, "frankerfacez.com/v1/room": FFZ})
    for k in ("global", "42"):                                               # both are due now
        d = json.load(open(tmp_path / f"{k}.json"))
        d["retry_at"] = es.retry[k] = time.time() - 1
        json.dump(d, open(tmp_path / f"{k}.json", "w"))
    es.codes("42")
    for _ in range(100):
        if not es.pending:
            break
        time.sleep(0.02)
    assert {"Okayge", "catJAM"} <= es.global_codes and {"Okayge", "catJAM"} <= es.codes("42")
