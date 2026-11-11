"""Per-model request settings (llm._tune; docs/HARDWARE_TIERS.md Appendix B). The 20b's bodies are unchanged; Gemma 4
(gemma4:*, gemma4mtp:*) gets the frozen study settings of gemma31_final(_mtp3_h3): Google's sampling, thinking off,
and the per-machine GPU layer cap when CW_LLM_NUM_GPU is set (data/scratch/livesw/identity_gemma.py checks the full
bodies against the study's own, 0 of 180 differ)."""
import json

from chatwatch import llm

PROFILE = {"display": "x", "recent_messages": [{"text": "hi"}], "messages": 1, "first_msg_seen": True,
           "account_age_days": 3.0, "sub_months": 0, "friction": {"instigated": 0, "joined_heated_room": 0}}


def _bodies(monkeypatch, model, num_gpu=""):
    monkeypatch.setattr(llm, "NUM_GPU", num_gpu)
    got = []
    monkeypatch.setattr(llm, "_post", lambda body, timeout: got.append(body) or {"message": {"content": json.dumps(
        {"label": "mild", "rules": [], "confidence": "low", "rationale": "", "warranted": False, "warning": "",
         "why": "", "toxicity": "none", "ratings": [{"n": 1, "toxicity": "none", "target": "", "kind": ""}]})}})
    llm.review(PROFILE, [], model=model)
    llm.rate_toxicity_window([], ["x: hi"], model=model)
    llm.draft_warning(PROFILE, [], model=model)
    return got


def test_the_20b_bodies_are_unchanged(monkeypatch):
    person, window, draft = _bodies(monkeypatch, "gpt-oss-safeguard:20b")
    assert person["options"] == window["options"] == {"temperature": 0, "num_ctx": llm.NUM_CTX}
    assert person["think"] == window["think"] == llm.THINK and draft["options"]["temperature"] == 0.2


def test_gemma_gets_its_study_settings_and_the_machines_layer_cap(monkeypatch):
    person, window, draft = _bodies(monkeypatch, "gemma4mtp:31b-n3", "26")
    want = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "num_gpu": 26, "num_ctx": llm.NUM_CTX}
    assert person["options"] == window["options"] == want
    assert person["think"] is window["think"] is draft["think"] is False
    assert person["format"] == llm.SCHEMA and window["format"] == llm.WINDOW_SCHEMA
    assert draft["options"] == {"temperature": 0.2, "num_ctx": llm.NUM_CTX, "num_gpu": 26}   # its own temperature
    person, _, _ = _bodies(monkeypatch, "gemma4:31b-it-qat")
    assert "num_gpu" not in person["options"] and person["options"]["top_k"] == 64            # no cap: Ollama's fit


def test_warm_loads_with_the_calls_own_load_options(monkeypatch):
    got = []
    monkeypatch.setattr(llm, "_post", lambda body, timeout: got.append(body))
    monkeypatch.setattr(llm, "NUM_GPU", "26")
    llm.warm("gemma4mtp:31b-n3")
    assert got[0]["options"] == {"num_ctx": llm.NUM_CTX, "num_gpu": 26}
