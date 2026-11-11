#!/usr/bin/env bash
# Smoke test of the dashboard page (web/index.html) in a real browser, no network:
#   1. tests/web/make_fixture.py (the chatwatch image) runs synthetic chat through the real Engine and Hub and saves
#      the batches the server would send;
#   2. tests/web/smoke.js (the playwright image) loads the page with the websocket and the API mocked from
#      that fixture, clicks, scrolls and types, and prints PASS/FAIL per check.
#   PW_MODULES=/local/disk/node_modules tests/web/run.sh [OUT_DIR]
# PW_MODULES: a node_modules holding playwright@1.48.2, on local disk. Install it once with
#   podman run --rm -v DIR:/pw -w /pw mcr.microsoft.com/playwright:v1.48.2-jammy npm i playwright@1.48.2
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$(mktemp -d)}"
PW_MODULES="${PW_MODULES:-${CW_PW_NODE_MODULES:?set PW_MODULES (or CW_PW_NODE_MODULES) to a node_modules folder with playwright@1.48.2}}"
mkdir -p "$OUT/shots"
podman run --rm --userns=keep-id --network none -e CUDA_VISIBLE_DEVICES= -v "$HERE":/work:ro -v "$OUT":/out \
  -w /work "${CW_IMAGE:-chatwatch:0.9-beta-$(sha256sum "$HERE/Containerfile" | cut -c1-8)}" python tests/web/make_fixture.py /out/fixture.json
podman run --rm --userns=keep-id --network none -e HOME=/tmp -e NODE_PATH=/pw -v "$PW_MODULES":/pw:ro \
  -v "$HERE/tests/web":/t:ro -v "$HERE/web":/web:ro -v "$OUT":/out mcr.microsoft.com/playwright:v1.48.2-jammy \
  node /t/smoke.js /out/fixture.json /web /out/shots
