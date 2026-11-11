"""What the public repository adds over the development copy: session logs off by default with a retention limit, the
dashboard on 127.0.0.1 unless told otherwise, pinned and checksum-verified model downloads, and an AI container that
needs no config file."""
import os
import time

from chatwatch import models, serve

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_old_session_logs_are_deleted_and_nothing_else(tmp_path):
    s = tmp_path / "data" / "sessions"
    old, new, other = s / "20200101-000000_somechannel", s / "29990101-000000_somechannel", s / "notes"
    for d in (old, new, other):
        d.mkdir(parents=True)
    (s / "owner_roles.json").write_text("{}")
    past = time.time() - 30 * 86400
    for p in (old, other):
        os.utime(p, (past, past))
    assert serve.prune_sessions(days=14, root=str(tmp_path)) == 1
    assert not old.exists() and new.exists() and other.exists() and (s / "owner_roles.json").exists()
    assert serve.prune_sessions(days=0, root=str(tmp_path)) == 0          # 0 keeps them all


def test_logging_is_opt_in_and_the_dashboard_listens_on_localhost():
    src = open(os.path.join(ROOT, "src", "chatwatch", "serve.py")).read()
    assert 'hub.log_sessions = bool(a.log) and not a.no_log' in src
    assert 'ap.add_argument("--host", default="127.0.0.1"' in src and "web.TCPSite(runner, a.host, a.port)" in src
    start = open(os.path.join(ROOT, "start.sh")).read()
    assert "--log)    LOG=(--log)" in start and '--host 0.0.0.0 "${LLM[@]}" "${AI[@]}" "${LOG[@]}"' in start


def test_encoder_downloads_are_pinned_and_checked(tmp_path, monkeypatch):
    assert len(models.TOX_REV) == 40 and len(models.EMB_REV) == 40
    assert all(f"unbiased-toxic-roberta-onnx/{f}" in models.SHA256 for f in models.TOX_FILES)
    assert all(f"all-MiniLM-L6-v2/{f}" in models.SHA256 for f in models.EMB_FILES)
    import urllib.request
    seen = []

    def fake(url, path):
        seen.append(url)
        open(path, "wb").write(b"not the model")
    monkeypatch.setattr(urllib.request, "urlretrieve", fake)
    dest = str(tmp_path / "unbiased-toxic-roberta-onnx")
    try:
        models._fetch(models.TOX_REPO, ["config.json"], dest, models.TOX_REV)
        raise AssertionError("a file that does not match its sha256 was accepted")
    except RuntimeError as e:
        assert "sha256" in str(e)
    assert seen == [f"https://huggingface.co/{models.TOX_REPO}/resolve/{models.TOX_REV}/config.json"]
    assert not os.listdir(dest)                                            # nothing left behind


def test_a_file_already_on_disk_that_does_not_match_is_refused(tmp_path):
    dest = tmp_path / "unbiased-toxic-roberta-onnx"
    dest.mkdir()
    (dest / "config.json").write_text("changed on disk")
    models._CHECKED.discard("unbiased-toxic-roberta-onnx/config.json")
    try:
        models._fetch(models.TOX_REPO, ["config.json"], str(dest), models.TOX_REV)
        raise AssertionError("a changed file on disk was used")
    except RuntimeError as e:
        assert "does not match its pinned sha256" in str(e)


def test_the_launcher_reads_its_settings_file_first_and_treats_another_gpus_file_as_none():
    start = open(os.path.join(ROOT, "start.sh")).read()
    assert start.index('CONF="$HOME/.config/chatwatch/gpu.env"') < start.index('PORT="${CW_PORT:-8765}"')
    assert '-z "${!k+set}" ]] && export "$k=$v"' in start                  # the environment's own value wins
    assert 'MENV_IGNORED=1' in start and '( ! -f "$MENV" || -n "${MENV_IGNORED:-}" )' in start
    assert 'IMAGE_TAG="chatwatch:0.9-beta-$(sha256sum "$HERE/Containerfile"' in start   # rebuilt when it changes
    for helper in ('bash "$HERE/bin/cw"', 'bash "$HERE/bin/ollama.sh" start', 'setsid bash "$HERE/bin/ollama_log_tail.sh"'):
        assert helper in start                                              # no exec bit needed on the helpers


def test_old_session_logs_are_aged_by_the_time_in_their_name(tmp_path):
    s = tmp_path / "data" / "sessions"
    recent_name = time.strftime("%Y%m%d-%H%M%S", time.gmtime(time.time() - 86400)) + "_somechannel"
    (s / recent_name).mkdir(parents=True)
    past = time.time() - 60 * 86400
    os.utime(s / recent_name, (past, past))                                # an old mtime does not make it old
    assert serve.prune_sessions(days=14, root=str(tmp_path)) == 0 and (s / recent_name).exists()


def test_the_ai_container_needs_no_config_file_and_checks_its_drafters():
    sh = open(os.path.join(ROOT, "bin", "ollama.sh")).read()
    assert 'STORE="${CW_OLLAMA_STORE:-$PROJ/models/ollama}"' in sh and 'PORT="${CW_OLLAMA_PORT:-11435}"' in sh
    assert "refusing" not in sh and "/home/" not in sh
    import json
    d = json.load(open(os.path.join(ROOT, "models", "drafts", "drafters.json")))
    for size, mf in (("31b", "Modelfile.31b.n3"), ("12b", "Modelfile.12b.n2"), ("e4b", "Modelfile.e4b.n4")):
        assert len(d[size]["sha256"]) == 64 and "/resolve/" in d[size]["url"] and "/main/" not in d[size]["url"]
        text = open(os.path.join(ROOT, "models", "drafts", mf)).read()
        assert f"DRAFT /root/.ollama/drafts/{d[size]['file']}" in text
