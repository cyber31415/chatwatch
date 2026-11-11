#!/usr/bin/env bash
# Model downloads from the page. The dashboard runs in a container and cannot run bin/ollama.sh, so ./start.sh runs this
# beside it on the host (in its own process group, stopped when the dashboard stops). The page writes a model tag to
# data/state/install/request; this runs `bin/ollama.sh install <tag>` for a tag the catalogue lists (nothing else), with
# its progress in progress.jsonl, its output in install.log and the outcome in status.json; the file `cancel` stops it.
# A heartbeat (`alive`, every second) tells the page that downloads are possible. src/chatwatch/install.py is the other
# side. Every file the page reads is written whole and then moved in place, so it never reads half of one.
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
D="${CW_INSTALL_DIR:-$PROJ/data/state/install}"
CAT="$PROJ/src/chatwatch/assets/models.json"
BUILD_WAIT_S=90               # a stop during "Setting it up" waits this long for the build (seconds) before cutting it
mkdir -p "$D"

beat() { date +%s > "$D/alive.tmp" && mv -f "$D/alive.tmp" "$D/alive"; }
status() {                    # state tag error (the error: printable ASCII only, so the JSON is always valid)
  local e
  # shellcheck disable=SC1003   # (tr deletes the quote and the backslash)
  e="$(printf '%s' "$3" | LC_ALL=C tr -cd '\40-\176' | tr -d '"\\' | cut -c1-300)"
  printf '{"state": "%s", "tag": "%s", "error": "%s", "at": %s}\n' "$1" "${2//[\"\\]/}" "$e" "$(date +%s)" > "$D/status.json.tmp"
  mv -f "$D/status.json.tmp" "$D/status.json"
}
# the last step, `ollama create` inside the AI container: stopping the podman client here would not stop it there, so
# a Cancel is not taken then (the page hides it) and a stop waits for it (it takes seconds)
building() { [[ "$(tail -n 1 "$D/progress.jsonl" 2>/dev/null)" == '{"phase": "build"}' ]]; }

pid="" tag=""
# stopped with the dashboard (start.sh's exit): the download in progress stops too, and says so
on_stop() {
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    for _ in $(seq 1 "$BUILD_WAIT_S"); do
      if building && kill -0 "$pid" 2>/dev/null; then sleep 1; else break; fi
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null
      status failed "$tag" "stopped when chatwatch stopped: press Download to go on"
    elif wait "$pid"; then
      status "done" "$tag" ""
    else
      status failed "$tag" "stopped when chatwatch stopped: press Download to go on"
    fi
  fi
  rm -f "$D/alive"
  exit 0
}
trap on_stop TERM INT HUP

rm -f "$D/request" "$D/cancel"
# a download left "running" by a helper that was stopped (the dashboard's Ctrl+C): Ollama keeps what it got, so the next
# Download goes on from there
if grep -q '"state": "running"' "$D/status.json" 2>/dev/null; then
  status failed "$(sed -n 's/.*"tag": "\([^"]*\)".*/\1/p' "$D/status.json")" "stopped when chatwatch stopped: press Download to go on"
fi

while true; do
  beat
  if [[ -f "$D/request" ]]; then
    rm -f "$D/cancel"                 # before taking the request: a Cancel for it that comes after is not lost
    tag="$(head -c 200 "$D/request" | tr -d '[:space:]')"
    rm -f "$D/request"
    if [[ ! "$tag" =~ ^[A-Za-z0-9._-]+:[A-Za-z0-9._-]+$ ]] || ! grep -qF "\"id\": \"$tag\"" "$CAT"; then
      status failed "$tag" "not a model chatwatch lists"
      continue
    fi
    : > "$D/progress.jsonl"
    : > "$D/install.log"
    status running "$tag" ""
    # its own process group (setsid), so a cancel stops curl and podman with it
    CW_INSTALL_PROGRESS="$D/progress.jsonl" setsid bash "$PROJ/bin/ollama.sh" install "$tag" < /dev/null > "$D/install.log" 2>&1 &
    pid=$!
    cancelled=0
    while kill -0 "$pid" 2>/dev/null; do
      beat
      if [[ -f "$D/cancel" ]]; then
        rm -f "$D/cancel"
        if ! building; then
          kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null
          cancelled=1
        fi
      fi
      sleep 1
    done
    wait "$pid"
    rc=$?
    pid=""
    if (( cancelled )); then
      status cancelled "$tag" ""
    elif (( rc == 0 )); then
      status "done" "$tag" ""
    else
      # its last two lines, without the terminal codes `ollama create` prints (seen in the Windows test, 2026-11-10)
      status failed "$tag" "$(sed 's/\x1b\[[0-9;?]*[A-Za-z]//g' "$D/install.log" | tr '\r' '\n' | grep -v '^\s*$' | tail -n 2 | tr '\n' ' ')"
    fi
  fi
  sleep 1
done
