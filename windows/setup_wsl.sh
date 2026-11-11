#!/usr/bin/env bash
# chatwatch on Windows: the part inside WSL's Ubuntu. windows/chatwatch.ps1 runs it; it is not meant to be run by hand.
#   setup_wsl.sh setup <version> [folder]   podman and its helpers, then chatwatch <version> from GitHub into ~/chatwatch
#                                           (or a copy of [folder], for testing a version not yet released)
#   setup_wsl.sh start [start.sh args]      ./start.sh in ~/chatwatch (Ctrl+C stops it, and its AI container)
# Exit 3 from setup: systemd was just turned on; chatwatch.ps1 restarts Ubuntu and runs setup again.
set -uo pipefail
REPO_URL="${CW_REPO_URL:-https://github.com/cyber31415/chatwatch.git}"     # (another: for tests)
APP="${CW_APP:-$HOME/chatwatch}"
say()  { printf '\n==> %s\n' "$*"; }
fail() { printf '\n*** %s\n' "$*" >&2; exit 1; }
SUDO=sudo; [[ "$(id -u)" == 0 ]] && SUDO=""

setup() {
  local ver="${1:?version}" from="${2:-}"
  grep -qi microsoft /proc/sys/kernel/osrelease || fail "this is not WSL"
  [[ "$(id -u)" != 0 ]] || fail "Ubuntu's default user is root: chatwatch runs podman as a normal user. In Ubuntu run: adduser NAME; then in PowerShell: wsl --manage Ubuntu-24.04 --set-default-user NAME"

  # rootless podman needs systemd for its user session; Ubuntu 24.04 on WSL normally has it on
  if [[ "$(ps -p 1 -o comm=)" != systemd ]]; then
    say "turning on systemd in /etc/wsl.conf (podman needs it); Ubuntu restarts after this step"
    if grep -qE '^\s*systemd\s*=' /etc/wsl.conf 2>/dev/null; then
      $SUDO sed -i -E 's/^\s*systemd\s*=.*/systemd=true/' /etc/wsl.conf
    elif grep -qE '^\s*\[boot\]' /etc/wsl.conf 2>/dev/null; then
      $SUDO sed -i -E '/^\s*\[boot\]/a systemd=true' /etc/wsl.conf
    else
      printf '[boot]\nsystemd=true\n' | $SUDO tee -a /etc/wsl.conf >/dev/null
    fi
    exit 3
  fi

  # the packages the Linux test machine runs chatwatch with (Ubuntu 24.04: podman 4.9.3, netavark, crun). iptables:
  # netavark 1.4 sets up chatwatch's network with it but does not depend on it, and WSL's Ubuntu lacks it
  local need=(podman git curl ca-certificates uidmap catatonit netavark aardvark-dns passt slirp4netns fuse-overlayfs
              iptables dbus-user-session) missing=() p
  for p in "${need[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p"); done
  if (( ${#missing[@]} )); then
    say "installing podman and its helpers (sudo asks for your Ubuntu password)"
    $SUDO apt-get update -q || fail "apt-get update failed"
    $SUDO apt-get install -y -q "${missing[@]}" || fail "apt-get install failed"
  fi
  podman info --format '{{.Version.Version}}' >/dev/null || fail "podman does not work for this user (see above)"
  get_code "$ver" "$from"
}

# chatwatch <version> into $APP: a clone at the tag v<version>, kept at it; or a test copy of a folder
get_code() {
  local ver="${1:?version}" from="${2:-}"
  if [[ -n "$from" ]]; then
    [[ -f "$from/start.sh" ]] || fail "no chatwatch in $from"
    say "copying chatwatch from $from (a test copy; its data and models in $APP stay)"
    mkdir -p "$APP"
    cp -a "$from/." "$APP/" || fail "the copy failed"
    rm -rf "$APP/.git"                # no longer a release: the next normal start puts the release back (below)
  elif [[ -d "$APP/.git" ]]; then
    if [[ "$(git -C "$APP" describe --tags --exact-match 2>/dev/null)" != "v$ver" ]]; then
      say "updating chatwatch to $ver"          # -f: only its own files change; data/ and models/ are not tracked
      git -C "$APP" fetch -q --tags origin && git -c advice.detachedHead=false -C "$APP" checkout -q -f "v$ver" \
        || echo "could not update to $ver (no network?): starting the version already here"
    fi
  elif [[ -d "$APP" ]]; then
    # a folder without git (a test copy): made a clone in place, so its own files become the release; data/ and models/
    # (not tracked) stay
    say "putting chatwatch $ver into $APP (your data and models there stay)"
    { git -C "$APP" init -q && { git -C "$APP" remote add origin "$REPO_URL" 2>/dev/null || true; } \
      && git -C "$APP" fetch -q --tags origin \
      && git -c advice.detachedHead=false -C "$APP" checkout -q -f "v$ver"; } || fail "the download failed (no network?)"
  else
    say "downloading chatwatch $ver into $APP"
    git -c advice.detachedHead=false clone -q --branch "v$ver" "$REPO_URL" "$APP" || fail "the download failed (no network?)"
  fi
}

start() {
  cd "$APP" || fail "chatwatch is not set up: run chatwatch.ps1 again"
  ./start.sh "$@"
}

case "${1:-}" in
  setup) shift; setup "$@" ;;
  code) shift; get_code "$@" ;;               # the code step alone (tests)
  start) shift; start "$@" ;;
  *) echo "usage: $0 setup <version> [folder] | start [args]" >&2; exit 2 ;;
esac
