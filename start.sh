#!/usr/bin/env bash
# Start the chatwatch dashboard.
#   ./start.sh [--no-llm] [--log] [extra serve args...]
#   CW_PORT=8766 ./start.sh        # if another program already uses 8765
#   --no-llm   start with the AI off (the page can turn it on)
#   --log      keep a session log of this run in data/sessions/ (raw chat lines with usernames, and the reads);
#              off by default; logs older than CW_SESSION_KEEP_DAYS (14) are deleted at start
# Then open http://127.0.0.1:8765 (or your CW_PORT), type a channel and press Start. Ctrl+C stops the dashboard; the
# AI container (cw-ollama) keeps its model in memory for the next start: `bin/ollama.sh stop` frees it.
# One dashboard at a time: the server takes data/state/dashboard.lock itself and removes it on exit; a lock left by
# this machine with no cw-live container running is stale and is removed here (CW_FORCE=1 starts anyway).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# settings from ~/.config/chatwatch/gpu.env (NAME=value lines), exported for the helpers; the environment's own win
CONF="$HOME/.config/chatwatch/gpu.env"
if [[ -f "$CONF" ]]; then
  while IFS='=' read -r k v; do
    v="${v%%#*}"; v="${v%"${v##*[![:space:]]}"}"; v="${v#\"}"; v="${v%\"}"; v="${v#\'}"; v="${v%\'}"
    [[ "$k" =~ ^CW_[A-Z0-9_]+$ && -z "${!k+set}" ]] && export "$k=$v"
  done < <(grep -E '^CW_[A-Z0-9_]+=' "$CONF")
fi
HOST="${HOSTNAME:-$(uname -n)}"                     # this machine's name (no `hostname` command needed)
PORT="${CW_PORT:-8765}"
IMAGE_TAG="chatwatch:0.9-beta-$(sha256sum "$HERE/Containerfile" | cut -c1-8)"
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  echo "refusing to start: another program on this machine already listens on 127.0.0.1:$PORT." >&2
  echo "Use a free port instead, e.g.:  CW_PORT=8766 ./start.sh" >&2
  exit 1
fi
LOCK="$HERE/data/state/dashboard.lock"
if [[ -f "$LOCK" && "${CW_FORCE:-}" != 1 ]]; then
  read -r host pid since port < "$LOCK" || true
  if [[ "${host:-}" == "$HOST" ]] && ! podman container exists cw-live; then
    echo "[live] removing a stale lock from this machine (no cw-live container is running)"
    rm -f "$LOCK"
  else
    echo "refusing to start: a dashboard already runs on machine '${host:-?}' (since ${since:-?}, port ${port:-?})." >&2
    echo "Stop that one first (Ctrl+C in its terminal). If that machine is off or crashed:" >&2
    echo "  CW_FORCE=1 ./start.sh" >&2
    exit 1
  fi
fi
# its own image, built here on first run (about a minute; needs the network once)
if ! podman image exists "$IMAGE_TAG"; then
  echo "[live] building the $IMAGE_TAG image (about a minute; again only when the Containerfile changes)..."
  podman build -q -f "$HERE/Containerfile" -t "$IMAGE_TAG" "$HERE" >/dev/null
fi
LLM=(--llm-review --person-review)   # the AI's reads of each chatter make the Watchlist
AI=() NO_LLM="" LOG=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-llm) AI=(--ai none); NO_LLM=1; shift ;;     # this run starts with the AI off (the page can turn it on)
    --log)    LOG=(--log); shift ;;                  # a session log of this run (off by default)
    *) break ;;
  esac
done

# ---- this machine's specs, for the page's model picker: read here on the host,
# passed as CW_SPEC_*; the page's dropdowns override them. The GPU: nvidia-smi, else the AMD card with the most memory
# under /sys/class/drm if it has at least 4 GB (an integrated GPU's small share is not used), else none.
detect_specs() {
  local out best=0 dev="" d b
  CW_SPEC_GPU_VENDOR=none CW_SPEC_GPU_NAME="" CW_SPEC_VRAM_MB=0 CW_SPEC_GPU_DISPLAY=1
  if command -v nvidia-smi >/dev/null 2>&1 &&
     out="$(nvidia-smi --query-gpu=memory.total,name --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)" &&
     [[ "$out" =~ ^[0-9]+ ]]; then
    CW_SPEC_GPU_VENDOR=nvidia CW_SPEC_VRAM_MB="${out%%,*}" CW_SPEC_GPU_NAME="${out#*, }"
    # does that card drive a screen (a desktop shares it)? its DRM outputs, found by its PCI address (owner 2026-11-10)
    b="$(nvidia-smi --query-gpu=memory.total,pci.bus_id --format=csv,noheader 2>/dev/null | sort -rn | head -1 || true)"
    b="$(printf '%s' "${b##*, }" | tr 'A-F' 'a-f')"; b="${b: -12}"
    for d in /sys/class/drm/card[0-9]*; do
      [[ -e "$d/device" && "$(basename "$(readlink -f "$d/device")")" == "$b" ]] && CW_SPEC_GPU_DISPLAY="$(drm_screen "$(basename "$d")")"
    done
  else
    for d in /sys/class/drm/card*/device; do
      [[ -r "$d/mem_info_vram_total" && "$(cat "$d/vendor" 2>/dev/null)" == 0x1002 ]] || continue
      b="$(cat "$d/mem_info_vram_total" 2>/dev/null || echo 0)"
      if [[ "$b" =~ ^[0-9]+$ ]] && (( b > best )); then best=$b dev=$d; fi
    done
    if (( best >= 4 * 1024 * 1024 * 1024 )); then
      CW_SPEC_GPU_VENDOR=amd CW_SPEC_VRAM_MB=$(( best / 1048576 )) CW_AMD_DEV="$dev"
      CW_SPEC_GPU_DISPLAY="$(drm_screen "$(basename "$(dirname "$dev")")")"     # does it drive a screen? (owner 2026-11-10)
      CW_SPEC_GPU_NAME="$(cat "$dev/product_name" 2>/dev/null || true)"
      if [[ -z "$CW_SPEC_GPU_NAME" ]] && command -v lspci >/dev/null 2>&1; then
        # "03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 32 [Radeon RX 7700 XT / 7800 XT]
        # (rev c8)": drop the address and class (the address has colons too), then prefer the bracketed marketing name
        CW_SPEC_GPU_NAME="$(lspci -s "$(basename "$(readlink -f "$dev")")" 2>/dev/null | head -1 \
          | sed -E 's/^[0-9a-fA-F:.]+ [^:]*: //; s/ \(rev [^)]*\)$//')"
        br="$(printf '%s' "$CW_SPEC_GPU_NAME" | sed -n 's/.*\[\(Radeon[^]]*\)\].*/AMD \1/p')"
        [[ -n "$br" ]] && CW_SPEC_GPU_NAME="$br"
      fi
      CW_SPEC_GPU_NAME="${CW_SPEC_GPU_NAME:-AMD GPU}"
    fi
  fi
  CW_SPEC_RAM_MB="$(awk '/^MemTotal:/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 0)"
  CW_SPEC_THREADS="$(nproc 2>/dev/null || echo 0)"
  CW_SPEC_CORES="$( (lscpu -p=core,socket 2>/dev/null || true) | grep -v '^#' | sort -u | wc -l)"
  [[ "$CW_SPEC_CORES" =~ ^[1-9] ]] || CW_SPEC_CORES="$CW_SPEC_THREADS"
  CW_SPEC_GPU_NAME="$(printf '%s' "$CW_SPEC_GPU_NAME" | tr -cd 'A-Za-z0-9 ._()/+-' | cut -c1-80)"
  [[ "$CW_SPEC_GPU_VENDOR" == none ]] && CW_SPEC_GPU_DISPLAY=0
  export CW_SPEC_GPU_VENDOR CW_SPEC_GPU_NAME CW_SPEC_VRAM_MB CW_SPEC_RAM_MB CW_SPEC_THREADS CW_SPEC_CORES CW_SPEC_GPU_DISPLAY
}
# 1 when the DRM card ($1, e.g. card1) has a connected screen output, or when that cannot be told (none listed): its
# desktop then keeps some GPU memory (picker.DESKTOP_RESERVE_GB); 0 when it lists outputs and none is connected
drm_screen() {
  local c any=0
  for c in /sys/class/drm/"$1"-*/status; do
    [[ -r "$c" && "$c" != *Writeback* ]] || continue
    any=1; [[ "$(cat "$c" 2>/dev/null)" == connected ]] && { echo 1; return; }
  done
  (( any == 1 )) && echo 0 || echo 1
}

# ---- the memory AVAILABLE for models, read at start (owner 2026-11-07: "we see what VRAM is free and we can estimate
# the layers based on that"; "we need a separate system free memory read too"). What cw-ollama itself holds counts as
# available (it is ours to unload): VRAM = the card's free memory + cw-ollama's processes' GPU memory (nvidia-smi per
# process; AMD: their DRM fdinfo); RAM = MemAvailable + cw-ollama's anonymous memory (a loaded model's CPU part; its
# mapped file pages are page cache, already in MemAvailable). Ollama's /api/ps is not used: for an MTP build (llama.cpp
# runner) it reports 1.6 GB for a 31B that holds 10 GB of VRAM and 9.5 GB of RAM. The container cannot see the GPU, so
# this is read once here; the page says "at start". A part that cannot be read counts as 0 held (the safe side).
ollama_pids() { podman top cw-ollama hpid 2>/dev/null | tail -n +2 | grep -E '^[0-9]+$' || true; }
measure_avail() {
  local free=0 held=0 p pids line pid mb uuid gpu="" total used kb id seen="" ra=0 d pdev
  pids="$(ollama_pids)"
  CW_SPEC_VRAM_AVAIL_MB=0
  if [[ "$CW_SPEC_GPU_VENDOR" == nvidia ]]; then
    # the card the specs describe (the largest): its free memory, and cw-ollama's processes on that card only
    line="$(nvidia-smi --query-gpu=memory.total,memory.free,uuid --format=csv,noheader,nounits 2>/dev/null \
            | sort -t, -k1,1 -rn | head -1 || true)"
    free="$(awk -F', *' '{print int($2)}' <<< "$line")"
    gpu="$(awk -F', *' '{print $3}' <<< "$line")"
    while IFS=', ' read -r pid mb uuid; do
      [[ "$pid" =~ ^[0-9]+$ && "$mb" =~ ^[0-9]+$ && "$uuid" == "$gpu" ]] || continue
      grep -qx "$pid" <<< "$pids" && held=$(( held + mb ))
    done < <(nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader,nounits 2>/dev/null || true)
    [[ "$free" =~ ^[0-9]+$ && -n "$gpu" ]] && CW_SPEC_VRAM_AVAIL_MB=$(( free + held ))
  elif [[ "$CW_SPEC_GPU_VENDOR" == amd && -n "${CW_AMD_DEV:-}" ]]; then
    total="$(cat "$CW_AMD_DEV/mem_info_vram_total" 2>/dev/null || echo 0)"
    used="$(cat "$CW_AMD_DEV/mem_info_vram_used" 2>/dev/null || echo 0)"
    pdev="$(basename "$(readlink -f "$CW_AMD_DEV")")"
    for p in $pids; do
      for f in /proc/"$p"/fdinfo/*; do
        # one DRM client per id on this card: its VRAM in KiB (drm-memory-vram on amdgpu, drm-resident-vram on newer
        # kernels; the value carries its unit, and one without a unit is bytes)
        line="$(awk '/^drm-pdev:/ {d=$2} /^drm-client-id:/ {c=$2} /^drm-(memory|resident)-vram0?:/ && !v {v=$2; u=$3}
                     END {if (c != "" && v != "") print d, c, int(u == "KiB" ? v : u == "MiB" ? v * 1024 : u == "GiB" ? v * 1048576 : v / 1024)}' \
                "$f" 2>/dev/null || true)"
        [[ -n "$line" ]] || continue
        read -r d id kb <<< "$line"
        [[ "$d" == "$pdev" && " $seen " != *" $id "* && "$kb" =~ ^[0-9]+$ ]] || continue
        seen="$seen $id"; held=$(( held + kb / 1024 ))
      done
    done
    if [[ "$total" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ ]] && (( total > 0 )); then
      CW_SPEC_VRAM_AVAIL_MB=$(( (total - used) / 1048576 + held ))
    fi
  fi
  CW_SPEC_VRAM_HELD_MB=$held        # 0: nothing held at the read (the page's estimate keeps a reserve for the runner)
  for p in $pids; do
    kb="$(awk '/^(RssAnon|RssShmem):/ {s += $2} END {print s + 0}' /proc/"$p"/status 2>/dev/null || echo 0)"
    [[ "$kb" =~ ^[0-9]+$ ]] && ra=$(( ra + kb ))
  done
  kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
  [[ "$kb" =~ ^[0-9]+$ ]] || kb=0
  CW_SPEC_RAM_AVAIL_MB=$(( (kb + ra) / 1024 ))
  export CW_SPEC_VRAM_AVAIL_MB CW_SPEC_VRAM_HELD_MB CW_SPEC_RAM_AVAIL_MB
}
detect_specs
measure_avail
echo "[live] this machine: GPU ${CW_SPEC_GPU_VENDOR} ${CW_SPEC_GPU_NAME} (${CW_SPEC_VRAM_MB} MB, ${CW_SPEC_VRAM_AVAIL_MB} MB available for models), RAM ${CW_SPEC_RAM_MB} MB (${CW_SPEC_RAM_AVAIL_MB} MB available), ${CW_SPEC_CORES} cores / ${CW_SPEC_THREADS} threads"
# Room for the rest of the machine (owner 2026-11-10: "the tool should run while other things are running"): the AI's
# CPU part uses the physical cores less CW_KEEP_CORES (default 2; 0: every core, Ollama's own), and on a card that
# drives a screen a model that fills the card leaves CW_VRAM_RESERVE_GB (default 1.5) free (picker.DESKTOP_RESERVE_GB).
KEEP="${CW_KEEP_CORES:-2}"
if [[ -z "${CW_LLM_NUM_THREAD:-}" && "$KEEP" =~ ^[0-9]+$ && "$KEEP" -gt 0 && "$CW_SPEC_CORES" =~ ^[0-9]+$ && "$CW_SPEC_CORES" -gt 0 ]]; then
  CW_LLM_NUM_THREAD=$(( CW_SPEC_CORES > KEEP ? CW_SPEC_CORES - KEEP : 1 ))
fi
export CW_LLM_NUM_THREAD="${CW_LLM_NUM_THREAD:-}"
echo "[live] kept free for the rest of the machine: $(( CW_SPEC_CORES - ${CW_LLM_NUM_THREAD:-$CW_SPEC_CORES} )) of ${CW_SPEC_CORES} cores$( [[ "$CW_SPEC_GPU_DISPLAY" == 1 ]] && echo ", ${CW_VRAM_RESERVE_GB:-1.5} GB of GPU memory for the desktop")"

# The AI is chosen on the page and saved per machine in data/state/machines/<host>.env (key=value lines read here,
# never sourced; ignored when written for another GPU). With nothing saved, the dashboard starts with the AI off and
# the page lists what fits this machine. ~/.config/chatwatch/gpu.env may set CW_LIVE_MODEL (an Ollama tag) and, for an
# MTP build, CW_LLM_NUM_GPU (its GPU layer cap) as a first-run default.
MDIR="$HERE/data/state/machines"
HOSTKEY="$(printf '%s' "$HOST" | sed 's/[^A-Za-z0-9._-]/_/g' | cut -c1-64)"
MENV="$MDIR/$HOSTKEY.env"
# PH9 (audit 2026-11-10): when another GPU saved last under this host name (two machines with one name, or a start whose
# GPU read failed), this GPU's file waits beside it as <host>.<vendor>-<MB>.env (picker.save_machine)
SIGKEY="$(printf '%s' "$CW_SPEC_GPU_VENDOR-$CW_SPEC_VRAM_MB" | sed 's/[^A-Za-z0-9._-]/_/g' | cut -c1-64)"
if [[ -f "$MDIR/$HOSTKEY.$SIGKEY.env" ]] && ! grep -qxF "CW_MACHINE_GPU=$CW_SPEC_GPU_VENDOR-$CW_SPEC_VRAM_MB" "$MENV" 2>/dev/null; then
  MENV="$MDIR/$HOSTKEY.$SIGKEY.env"
fi
AI_MODE=single
if [[ -f "$MENV" ]]; then
  declare -A M=()
  while IFS='=' read -r k v; do
    case "$k" in
      CW_MACHINE_GPU|CW_AI_MODE|CW_LIVE_MODEL|CW_LLM_NUM_GPU|CW_REVIEW_MODEL|CW_REVIEW_NUM_GPU)
        [[ "$v" =~ ^[A-Za-z0-9._:/+-]*$ ]] && M[$k]="$v" ;;
    esac
  done < "$MENV"
  if [[ -n "${M[CW_MACHINE_GPU]:-}" && "${M[CW_MACHINE_GPU]}" != "$CW_SPEC_GPU_VENDOR-$CW_SPEC_VRAM_MB" ]]; then
    echo "[live] $MENV was written for another GPU (${M[CW_MACHINE_GPU]}): ignored; pick the AI on the page"
    MENV_IGNORED=1
  else
    AI_MODE="${M[CW_AI_MODE]:-single}"
    if [[ -n "${M[CW_LIVE_MODEL]:-}" ]]; then CW_LIVE_MODEL="${M[CW_LIVE_MODEL]}" CW_LLM_NUM_GPU="${M[CW_LLM_NUM_GPU]:-}"; fi
  fi
fi
[[ ${#AI[@]} -gt 0 ]] && AI_MODE=none
FIRST=""
if [[ ( ! -f "$MENV" || -n "${MENV_IGNORED:-}" ) && -z "${CW_LIVE_MODEL:-}" && -z "$NO_LLM" ]]; then
  AI=(--ai none) AI_MODE=none FIRST=1                # nothing chosen yet on this machine: pick it on the page
fi
# A setup the catalogue replaced (src/chatwatch/assets/models.json "replaced"; tests/test_text31.py keeps this list equal
# to it): the old id, from gpu.env or the machine file, means the new build. Its cap was set for the old build, so the
# server works the cap out at start (the old one is carried as a floor: picker.cap_for), and nothing is warmed here
# with the old figure. Owner 2026-11-09: the text-only 31B (no vision projector; data/study/quant/FACTS.md section 4).
case "${CW_LIVE_MODEL:-}" in
  gemma4mtp:31b-n3)
    echo "[live] gemma4mtp:31b-n3 is replaced by gemma4mtp:31b-n3-text (the same model without the vision projector)"
    CW_LIVE_MODEL=gemma4mtp:31b-n3-text CW_LLM_NUM_GPU=estimate ;;
esac
# a cap the page could not vouch for on this card (an MTP build, below every tested tier, not yet checked): never warm it
# uncapped; the server starts with the AI off and the page says to run Check speed (third review 2)
SKIP_WARM=0
if [[ "${CW_LLM_NUM_GPU:-}" == unknown ]]; then CW_LLM_NUM_GPU=""; SKIP_WARM=1; fi
# a cap the server estimates from the VRAM read above (no cap set for this machine yet): the server works it out at
# this start, so it is not warmed here with last start's figure; the first call loads it (review 2026-11-08)
if [[ "${CW_LLM_NUM_GPU:-}" == estimate ]]; then CW_LLM_NUM_GPU=""; SKIP_WARM=2; fi
export CW_LLM_MODEL="${CW_LIVE_MODEL:-gemma4mtp:31b-n3-text}" CW_LLM_NUM_GPU="${CW_LLM_NUM_GPU:-}" CW_GPU="${CW_GPU:-}"

# ---- Ollama's log, for the speed check: the layers Ollama puts on the GPU and runner aborts only (no chat). A host-side
# tail started here, in its own process group, and stopped on exit (Ctrl+C included); one per machine (a tail left by a
# killed run is stopped first).
OLOG="$MDIR/$HOSTKEY.ollama.log" TPID="$MDIR/$HOSTKEY.ollama.tail.pid"
TAIL_PGID=""
# FE-6 (audit 2026-11-10): the copy goes with its copier, so a run without one (--no-llm) never reads a stale copy
stop_tail() { if [[ -n "$TAIL_PGID" ]]; then kill -- "-$TAIL_PGID" 2>/dev/null || true; fi; rm -f "$TPID" "$OLOG"; TAIL_PGID=""; }
start_tail() {
  mkdir -p "$MDIR"
  if [[ -f "$TPID" ]]; then
    read -r old < "$TPID" || true
    if [[ "${old:-}" =~ ^[0-9]+$ ]] && ps -o args= -p "$old" 2>/dev/null | grep -q "ollama_log_tail.sh"; then
      kill -- "-$old" 2>/dev/null || true
    fi
    rm -f "$TPID"
  fi
  : > "$OLOG"
  setsid bash "$HERE/bin/ollama_log_tail.sh" "$OLOG" </dev/null >/dev/null 2>&1 &
  TAIL_PGID=$!
  echo "$TAIL_PGID" > "$TPID"
  trap stop_tail EXIT
}

if [[ -z "$NO_LLM" ]]; then
  if ! podman container exists cw-ollama || [[ "$(podman inspect -f '{{.State.Running}}' cw-ollama)" != "true" ]]; then
    bash "$HERE/bin/ollama.sh" start >/dev/null
    for _ in $(seq 1 30); do curl -s -m 2 -o /dev/null "127.0.0.1:${CW_OLLAMA_PORT:-11435}/api/version" && break; sleep 1; done
  fi
  start_tail
  if [[ "$SKIP_WARM" == 2 ]]; then
    echo "[live] $CW_LLM_MODEL: its GPU layer cap is estimated from the free VRAM by the server; it loads on the first call"
  elif [[ "$SKIP_WARM" == 1 ]]; then
    echo "[live] $CW_LLM_MODEL needs a GPU layer cap for this card: not loaded; press Check Speed on the page"
  elif [[ "$AI_MODE" != none ]]; then
    echo "[live] loading $CW_LLM_MODEL into VRAM (about a minute the first time)..."
    bash "$HERE/bin/ollama.sh" warm "$CW_LLM_MODEL" >/dev/null || echo "[live] the warm-up did not finish; the first call loads it"
  elif [[ -n "$FIRST" ]]; then
    echo "[live] no AI chosen yet on this machine: pick one on the page (it lists what fits; Check Speed measures it)"
  else
    echo "[live] AI: off (chosen on the page); the page can turn it on"
  fi
fi
[[ -z "$TAIL_PGID" ]] && rm -f "$OLOG"      # no copier this run: a last run's copy must not pass for a live one (FE-6)
echo "[live] open http://127.0.0.1:$PORT and choose a channel"
CW_HOST="$HOST" CW_PORT="$PORT" CW_IMAGE="$IMAGE_TAG" CW_NAME=cw-live \
  CW_OLLAMA_LOG="${TAIL_PGID:+/work/data/state/machines/$HOSTKEY.ollama.log}" bash "$HERE/bin/cw" python -m chatwatch.serve \
  --ml --host 0.0.0.0 "${LLM[@]}" "${AI[@]}" "${LOG[@]}" "$@"
