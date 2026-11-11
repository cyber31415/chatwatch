"""Unit tests for parsing, normalisation, cues and account age.
Run inside the container:  bin/cw python -m pytest -q tests (with pytest in the image)
"""
from pathlib import Path

import numpy as np

from chatwatch import accounts, cues, irc, text

ROOT = Path(__file__).resolve().parents[1]
GRID = np.round(np.arange(0.50, 0.951, 0.005), 3)


def test_parse_privmsg_tags():
    line = ("@badge-info=subscriber/14;badges=subscriber/12,vip/1;color=#FF0000;display-name=Foo;"
            "emotes=25:0-4;first-msg=1;flags=6-9:A.6/P.6;id=abc;mod=0;room-id=42;"
            "tmi-sent-ts=1700000000000;user-id=123 :foo!foo@foo.tmi.twitch.tv PRIVMSG #chan :Kappa you suck")
    ev = irc.to_event(line)
    assert ev["kind"] == "msg" and ev["channel"] == "chan" and ev["login"] == "foo"
    assert ev["first_msg"] and ev["vip"] and ev["badge_info"]["subscriber"] == "14"
    assert ev["emotes"][0][3] == "Kappa"
    assert ev["automod"] == {"A": 6, "P": 6}


def test_clearchat_timeout_and_ban():
    t = irc.to_event("@ban-duration=600;room-id=1;target-user-id=9;tmi-sent-ts=5 :tmi.twitch.tv CLEARCHAT #c :bad")
    b = irc.to_event("@room-id=1;target-user-id=9;tmi-sent-ts=5 :tmi.twitch.tv CLEARCHAT #c :bad")
    assert t["duration"] == 600 and b["duration"] == -1 and b["target_login"] == "bad"


def test_tag_unescape():
    assert irc.parse_tags(r"system-msg=hello\sworld\:x")["system-msg"] == "hello world;x"


def test_normalize_undoes_evasion():
    assert "idiot" in text.normalize("id1ot")
    assert "kys" in text.normalize("k y s")
    assert text.normalize("ѕtupid") == "stupid"          # Cyrillic s
    assert text.normalize("1v1 me") == "1v1 me"           # numbers left alone
    sig = text.evasion_signals("you​ suck")
    assert sig["zero_width"] == 1


def test_dup_bypass_char_is_not_evasion():
    # Twitch clients append U+E0000 / U+034F to dodge the duplicate filter
    assert text.evasion_signals("same message \U000e0000")["zero_width"] == 0


def test_evaded_cue_detected():
    raw = "who ev3n w4tches this"
    hits, evaded = cues.match(text.normalize(raw), text.plain_lower(raw))
    assert "dismiss_stream" in hits and "dismiss_stream" in evaded


def test_cues_basic():
    def hit(s):
        return set(cues.match(text.normalize(s))[0])
    assert "hedged_attack" in hit("no offense but your voice is so annoying")
    assert "loaded_question" in hit("genuine question why is he always like this")
    assert "newcomer" in hit("hi first time here, just found your stream")
    assert "scam" in hit("buy viewers on streamboo dot com")
    assert "hype" in hit("LETS GOOOO")
    assert not hit("what a nice play that was")


def test_account_age_monotonic_and_gap():
    now = 1792525099064
    ages = [accounts.age_days(i, now) for i in (50_000_000, 280_000_000, 400_000_000, 1_500_000_000)]
    assert ages == sorted(ages, reverse=True)
    # nothing lives in the Dec 2018 gap: both sides are days apart, not years
    assert abs(accounts.age_days(280_500_000, now) - accounts.age_days(400_000_000, now)) < 5
    assert accounts.age_days("notanid", now) is None


def test_every_form_of_the_n_word_is_a_slur():
    """Owner decision 2026-10-25 (FINDINGS): the -a form is the same root slur."""
    def hit(s):
        return set(cues.match(text.normalize(s))[0])
    for s in ("thats my nigga fr", "NIGGAS", "niggaz", "n1gga", "you are a nigger", "n i g g e r", "n i g g a"):
        assert "slur" in hit(s), s
    for s in ("snigger", "oo nga", "niger river"):
        assert "slur" not in hit(s), s


def test_twitch_labels_on_the_a_form_count():
    from chatwatch.signals import automod_levels
    ev = {"text": "fresh spray nigga", "automod": {"A": 6},
          "automod_spans": [(12, 16, {"A": 6})]}
    assert automod_levels(ev) == {"A": 6}
    ev2 = {"text": "shut up bitch", "automod": {"A": 6}, "automod_spans": [(8, 12, {"A": 6})]}
    assert automod_levels(ev2) == {"A": 6}


def test_benign_words_with_digits_are_not_evasion():
    raw = "L0L"
    _, evaded = cues.match(text.normalize(raw), text.plain_lower(raw))
    assert evaded == []


def test_overlapping_runs_are_deduplicated(tmp_path):
    """A handoff writes the same chat to two run files; readers must merge them."""
    from chatwatch.replay import iter_events
    d = tmp_path / "2026-10-21"
    d.mkdir()
    line = ("@id=abc;user-id=1;tmi-sent-ts=1000;display-name=A :a!a@a PRIVMSG #chan :hello")
    line2 = ("@id=def;user-id=2;tmi-sent-ts=2000;display-name=B :b!b@b PRIVMSG #chan :hi")
    (d / "chan.log").write_text(f"1000 {line}\n")
    (d / "chan.r2.log").write_text(f"1000 {line}\n2000 {line2}\n")
    evs = list(iter_events([str(d)], {"chan"}))
    assert [e["text"] for e in evs] == ["hello", "hi"]
    assert irc.channel_of_path(str(d / "chan.r2.log")) == "chan"


def test_comparison_and_leaving_cues_are_narrow():
    """FINDINGS 2026-10-25: "X is better" and "im out" were ~40 false alarms per set."""
    def hit(s):
        return set(cues.match(text.normalize(s))[0])
    assert "comparison" not in hit("Chile is better")
    assert "comparison" in hit("my friend is way better than you")
    assert "leaving_threat" not in hit("ight im out gn")
    assert "leaving_threat" in hit("im done watching this")


def test_threat_cue_needs_the_speaker_as_agent_v1():
    """RULE_THREAT_V1: "X gonna kill you" is a threat only from the speaker (I/we, or a bare opening "gonna");
    warnings, narration and coaching are not. The other alternatives and the chants are unchanged."""
    def hit(s):
        return "threat" in cues.match(text.normalize(s))[0]
    for s in ("SHES GONNA KILL YOU", "hes gonna kill you LOL", "they are going to kill you", "the boss gonna kill him",
              "you're gonna beat him", "no one's gonna find you", "running isnt gonna kill you"):
        assert not hit(s), s
    for s in ("i'll kill u", "i will find you", "im gonna shoot you", "i am going to hurt him", "we're gonna kill them",
              "@someone gonna kill you", "gonna find you", "KILL THEM ALL", "shoot them all", "i know where u live",
              "watch your back", "im swatting u", "I got a bullet for each of them", "gas and a match"):
        assert hit(s), s
    for s in ("imma kill you", "we will kill you"):       # V1 never adds a hit the old cue did not have
        assert not hit(s), s


def test_mlp_head_file_is_scored_like_the_tested_candidate(tmp_path, monkeypatch):
    """An MLP head file (head_mlp/candidate format) loads and scores as mean-over-nets + shift."""
    import json
    import numpy as np
    from chatwatch import toxicity
    rng = np.random.RandomState(0)
    d = 4
    nets = [{"W": [rng.randn(d, 3).tolist(), rng.randn(3, 1).tolist()], "b": [rng.randn(3).tolist(), [0.1]]} for _ in range(2)]
    head = {"kind": "mlp", "nets": nets, "mean": [0.0] * d, "scale": [1.0] * d, "shift": -0.2, "tox_keys": [], "threshold": 0.56}
    p = tmp_path / "head.json"
    p.write_text(json.dumps(head))
    monkeypatch.setenv("CW_TOX_HEAD", str(p))
    monkeypatch.setattr(toxicity, "_HEAD", None)
    x = rng.randn(d)
    want = np.mean([np.maximum(x @ np.array(n["W"][0]) + n["b"][0], 0) @ np.array(n["W"][1]) + 0.1 for n in nets]) - 0.2
    f = {"ml_emb": x[:-1]}                    # the layer score is the last feature
    got = toxicity.head_probability(f, x[-1])
    assert abs(got - 1 / (1 + np.exp(-want))) < 1e-9
    assert toxicity._head()["t"] == 0.56
    monkeypatch.setattr(toxicity, "_HEAD", None)


def _mlp_on(d, col, bias=0.0, threshold=0.575):
    """An MLP head file whose logit is feature `col` + bias (one net, one hidden unit kept in its linear part)."""
    W0 = [[1.0 if i == col else 0.0] for i in range(d)]
    return {"kind": "mlp", "nets": [{"W": [W0, [[1.0]]], "b": [[100.0], [-100.0 + bias]]}], "shift": 0.0,
            "mean": [0.0] * d, "scale": [1.0] * d, "tox_keys": [], "threshold": threshold}



def test_model_threat_skips_narration_n1():
    """RULE_N1: the model's threat score does not count on a threat narrated about someone else; it still counts
    with the speaker or a proxy as the agent, real-world words or a plain threat."""
    from chatwatch import toxicity
    for s in ("SHES GONNA KILL YOU", "this boss gonna kill you", "the dragon gonna kill you", "he wants you dead",
              "you said you were gonna kill him", "its not gonna hurt you its only a skin"):
        assert toxicity.threat_narrated(text.normalize(s)), s
    for s in ("im gonna kill you", "my boys gonna find you", "he gonna find you irl", "they know where u live",
              "i kill you", "kys", "i hope you die", "you are dead", "we r gonna jump u"):
        assert not toxicity.threat_narrated(text.normalize(s)), s
    f = {"cues": {}, "families": {}, "ml_toxicity": 0.1, "ml_threat": 0.95, "aimed_at_chatter": True, "target_chatter": "x",
         "addresses_streamer": False}
    assert "threat" not in toxicity.message_toxicity({**f, "norm": "shes gonna kill you"})["kinds"]
    assert "threat" in toxicity.message_toxicity({**f, "norm": "im gonna kill you"})["kinds"]
