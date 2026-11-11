"""The live server's console stays quiet (owner 2026-11-10): no huggingface/tokenizers fork warning."""
import os
import subprocess
import sys

ENCODE_THEN_FORK = ("import os, warnings\n"
                    "warnings.simplefilter('ignore')\n"
                    "from tokenizers import Tokenizer, models, pre_tokenizers\n"
                    "t = Tokenizer(models.WordLevel({'a': 0, '[UNK]': 1}, unk_token='[UNK]'))\n"
                    "t.pre_tokenizer = pre_tokenizers.Whitespace()\n"
                    "t.encode_batch(['a a a'] * 2000)\n"
                    "os.fork() or os._exit(0)\n")


def _run(code, **extra):
    env = {k: v for k, v in os.environ.items() if k != "TOKENIZERS_PARALLELISM"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**env, "PYTHONPATH": "src", **extra}, check=True)
    return out.stdout + out.stderr


def test_serve_turns_tokenizer_parallelism_off_before_any_encode():
    assert _run("import os, chatwatch.serve; print(os.environ.get('TOKENIZERS_PARALLELISM'))").strip() == "false"


def test_an_explicit_setting_still_wins():
    code = "import os, chatwatch.serve; print(os.environ.get('TOKENIZERS_PARALLELISM'))"
    assert _run(code, TOKENIZERS_PARALLELISM="true").strip() == "true"


def test_a_batched_encode_then_a_fork_warns_without_it_and_is_quiet_with_it():
    assert "huggingface/tokenizers" in _run(ENCODE_THEN_FORK)                  # the warning the owner saw
    assert "huggingface/tokenizers" not in _run("import chatwatch.serve\n" + ENCODE_THEN_FORK)
