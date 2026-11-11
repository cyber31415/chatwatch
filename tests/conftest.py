"""The suite runs from the repository root, whatever directory pytest is started in.

The tools under test read their files relative to the root (they are run from /work: tools/RUBRIC_VERSION,
src/chatwatch/assets/..., data/...), and some read them at import. Tests that need a scratch data/ folder
chdir to tmp_path themselves (monkeypatch.chdir, undone after the test). Tests that read the real data
use ROOT, never the current directory (docs/review/tests_docs.md 2b)."""
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


@pytest.fixture(autouse=True)
def _no_retry_waits(monkeypatch):
    """fileio tries a transient share error again after RETRY_DELAYS (PB4-2); tests that make a write fail on
    purpose would wait for each one. The attempts are the same, the waits are zero."""
    from chatwatch import fileio
    monkeypatch.setattr(fileio, "RETRY_DELAYS", (0, 0, 0, 0))
