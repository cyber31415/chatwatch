# Tests
- `python -m pytest -q` (Python 3.12 with the packages in `Containerfile`, plus pytest), or in the container:
  `bin/cw sh -c 'pip install -q --target /tmp/t pytest && PYTHONPATH=/tmp/t:/work/src python -m pytest -q'`.
  No network, GPU or model download is needed.
- `tests/web/run.sh OUT_DIR` runs the page's browser checks (Playwright in its container; set `PW_MODULES` to a
  node_modules folder holding playwright@1.48.2, see the script's header). Screenshots land in `OUT_DIR/shots`.
