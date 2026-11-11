#!/usr/bin/env bash
# chatwatch's own Ollama, in its own container (cw-ollama), with its models in models/ollama/ of this folder
# (CW_OLLAMA_STORE for another place). It listens on 127.0.0.1 only; the dashboard reaches it by name on the cw-net
# network (http://cw-ollama:11434).
#   bin/ollama.sh start             start the server
#   bin/ollama.sh install <tag>     install a model the page lists as "not installed": a plain model is pulled; an MTP
#                                   build (gemma4mtp:<size>-n<N>) pulls its parent (gemma4:<size>-it-qat), downloads its
#                                   drafter (models/drafts/drafters.json: pinned, sha256-checked) and is created from
#                                   models/drafts/Modelfile.<size>.n<N>; its text-only twin (gemma4mtp:<size>-n<N>-text:
#                                   the same weights and drafter, no vision projector) is created from the parent's own
#                                   weights file
#                                   (the page's Download button runs this through bin/install_helper.sh)
#   bin/ollama.sh pull <model>      a plain `ollama pull`
#   bin/ollama.sh warm [model]      load a model into memory now
#   bin/ollama.sh version | stop
# Settings (environment, or ~/.config/chatwatch/gpu.env):
#   CW_GPU=nvidia|amd|cpu   (default: nvidia when nvidia-smi works, amd when /dev/kfd exists, else cpu)
#   CW_OLLAMA_PORT=11435    (the host port; 11435 so it never meets an Ollama of your own on 11434)
#   CW_OLLAMA_IMAGE=<image> (default: ollama/ollama:0.40.11, or 0.40.11-rocm for AMD)
set -euo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
STORE="${CW_OLLAMA_STORE:-$PROJ/models/ollama}"
CONF="$HOME/.config/chatwatch/gpu.env"
# the caller's GPU layer cap (start.sh passes the picked model's, set but empty for "no cap") wins over the file's
if [[ -n "${CW_LLM_NUM_GPU+set}" ]]; then CALLER_NG="$CW_LLM_NUM_GPU"; fi
if [[ -f "$CONF" ]]; then                     # NAME=value lines; a variable already set wins
  while IFS='=' read -r k v; do
    v="${v%%#*}"; v="${v%"${v##*[![:space:]]}"}"; v="${v#\"}"; v="${v%\"}"; v="${v#\'}"; v="${v%\'}"
    [[ "$k" =~ ^CW_[A-Z0-9_]+$ && -z "${!k+set}" ]] && export "$k=$v"
  done < <(grep -E '^CW_[A-Z0-9_]+=' "$CONF")
fi
if [[ -n "${CALLER_NG+set}" ]]; then CW_LLM_NUM_GPU="$CALLER_NG"; fi
if [[ -z "${CW_GPU:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then CW_GPU=nvidia
  elif [[ -e /dev/kfd ]] && awk '$1 >= 4294967296 {f=1} END {exit !f}' /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null; then
    CW_GPU=amd                                 # an AMD card with at least 4 GB (an integrated GPU's share is not used)
  else CW_GPU=cpu; fi
fi
GPU="$CW_GPU"
PORT="${CW_OLLAMA_PORT:-11435}"
mkdir -p "$STORE/drafts"
C="${CW_OLLAMA_CONTAINER:-cw-ollama}"
OLLAMA_VERSION="0.40.11"                     # 0.40.11 or later: the MTP `DRAFT` line
case "$GPU" in
  amd)    IMAGE="${CW_OLLAMA_IMAGE:-docker.io/ollama/ollama:$OLLAMA_VERSION-rocm}"
          DEV=(--device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt seccomp=unconfined
               --security-opt label=disable) ;;
  nvidia) IMAGE="${CW_OLLAMA_IMAGE:-docker.io/ollama/ollama:$OLLAMA_VERSION}"
          DEV=(--device nvidia.com/gpu=all --security-opt label=disable) ;;
  cpu)    IMAGE="${CW_OLLAMA_IMAGE:-docker.io/ollama/ollama:$OLLAMA_VERSION}"
          DEV=(--security-opt label=disable) ;;
  *) echo "CW_GPU must be nvidia, amd or cpu" >&2; exit 1 ;;
esac

# The page's downloads (bin/install_helper.sh sets CW_INSTALL_PROGRESS): progress lines for src/chatwatch/install.py.
prog() { if [[ -n "${CW_INSTALL_PROGRESS:-}" ]]; then printf '%s\n' "$1" >> "$CW_INSTALL_PROGRESS"; fi; }
# a plain `ollama pull`; for the page, through Ollama's API, its progress lines kept (stopping curl stops the pull;
# Ollama keeps the parts it has, so the next pull goes on from there)
pull_model() {
  local m="$1" line last=""
  if [[ -z "${CW_INSTALL_PROGRESS:-}" ]]; then podman exec "$C" ollama pull "$m"; return; fi
  echo "pulling $m ..."
  prog "{\"phase\": \"pull\", \"model\": \"$m\"}"
  while IFS= read -r line; do
    printf '%s\n' "$line" >> "$CW_INSTALL_PROGRESS"
    last="$line"
  done < <(curl -sN 127.0.0.1:"$PORT"/api/pull -d "{\"model\": \"$m\"}")
  if [[ "$last" == *'"success"'* ]]; then return 0; fi
  echo "pulling $m failed: ${last:-no answer from the AI container}" >&2
  return 1
}

# an MTP build's drafter: downloaded once into the store's drafts/ folder, checked against its pinned sha256
fetch_drafter() {
  local size="$1" spec file url sum got bytes cpid
  spec="$(sed -n "/\"$size\": {/,/}/p" "$PROJ/models/drafts/drafters.json")"
  file="$(sed -n 's/.*"file": "\([^"]*\)".*/\1/p' <<< "$spec")"
  url="$(sed -n 's/.*"url": "\([^"]*\)".*/\1/p' <<< "$spec")"
  sum="$(sed -n 's/.*"sha256": "\([0-9a-f]*\)".*/\1/p' <<< "$spec")"
  if [[ -z "$file" || -z "$url" || -z "$sum" ]]; then
    echo "no drafter listed for $size in models/drafts/drafters.json" >&2; return 1
  fi
  if [[ -f "$STORE/drafts/$file" ]] && [[ "$(sha256sum "$STORE/drafts/$file" | cut -d' ' -f1)" == "$sum" ]]; then return 0; fi
  echo "downloading the $size drafter ($file) ..."
  if [[ -n "${CW_INSTALL_PROGRESS:-}" ]]; then
    bytes="$(sed -n 's/.*"size": \([0-9]*\).*/\1/p' <<< "$spec")"
    curl -fsL --retry 3 -o "$STORE/drafts/$file.part" "$url" &
    cpid=$!
    while kill -0 "$cpid" 2>/dev/null; do
      prog "{\"phase\": \"drafter\", \"completed\": $(stat -c %s "$STORE/drafts/$file.part" 2>/dev/null || echo 0), \"total\": ${bytes:-0}}"
      sleep 1
    done
    if ! wait "$cpid"; then rm -f "$STORE/drafts/$file.part"; echo "downloading $file failed" >&2; return 1; fi
    prog "{\"phase\": \"drafter\", \"completed\": ${bytes:-0}, \"total\": ${bytes:-0}}"
  else
    curl -fL --retry 3 -o "$STORE/drafts/$file.part" "$url"
  fi
  got="$(sha256sum "$STORE/drafts/$file.part" | cut -d' ' -f1)"
  if [[ "$got" != "$sum" ]]; then
    rm -f "$STORE/drafts/$file.part"; echo "$file: sha256 $got, expected $sum; not installed" >&2; return 1
  fi
  mv "$STORE/drafts/$file.part" "$STORE/drafts/$file"
}

case "${1:-start}" in
  start)
    podman network exists cw-net || podman network create cw-net >/dev/null
    podman run -d --name "$C" --replace --network cw-net \
      "${DEV[@]}" \
      -v "$STORE":/root/.ollama \
      -e OLLAMA_KEEP_ALIVE=-1 -e OLLAMA_HOST=0.0.0.0:11434 -e OLLAMA_CONTEXT_LENGTH=8192 -e OLLAMA_MAX_LOADED_MODELS=1 \
      -p 127.0.0.1:"$PORT":11434 "$IMAGE" ;;
  pull) podman exec "$C" ollama pull "$2" ;;
  warm) # the load options must be the dashboard's own (llm._load_options: num_ctx, the GPU layer cap CW_LLM_NUM_GPU
        # and the CPU threads CW_LLM_NUM_THREAD), or its first call loads the model again
    NG="${CW_LLM_NUM_GPU:-}" NT="${CW_LLM_NUM_THREAD:-}"
    curl -s --max-time 1200 127.0.0.1:"$PORT"/api/generate \
      -d "{\"model\": \"${2:?usage: $0 warm <model>}\", \"keep_alive\": -1, \"options\": {\"num_ctx\": 8192${NG:+, \"num_gpu\": $NG}${NT:+, \"num_thread\": $NT}}}" >/dev/null && \
      podman exec "$C" ollama ps ;;
  install)
    tag="${2:?usage: $0 install <model tag>}"
    if ! podman container exists "$C" || [[ "$(podman inspect -f '{{.State.Running}}' "$C")" != true ]]; then
      echo "the AI container ($C) is not running: start it with ./start.sh (or bin/ollama.sh start) first" >&2; exit 1
    fi
    if [[ "$tag" =~ ^gemma4mtp:([a-z0-9-]+)-n([0-9]+)(-text)?$ ]]; then
      size="${BASH_REMATCH[1]}" n="${BASH_REMATCH[2]}" text="${BASH_REMATCH[3]}"
      parent="gemma4:$size-it-qat" mf="Modelfile.$size.n$n"
      if [[ ! -f "$PROJ/models/drafts/$mf" ]]; then
        echo "no models/drafts/$mf: $tag is not one of chatwatch's builds" >&2; exit 1
      fi
      fetch_drafter "$size"
      cp "$PROJ/models/drafts/$mf" "$STORE/drafts/$mf"
      podman exec "$C" ollama show "$parent" >/dev/null 2>&1 || pull_model "$parent"
      prog '{"phase": "build"}'
      if [[ -z "$text" ]]; then
        podman exec "$C" sh -c "cd /root/.ollama/drafts && ollama create $tag -f $mf"
      else
        # The text-only twin: Ollama counts the vision projector when it fits a partly offloaded model, then runs the
        # projector on the CPU; without it the 31B fits about 4 more layers on a 12 GB card. A Modelfile cannot drop a
        # layer of its FROM model, so the build starts from the parent's weights file (the first FROM of its Modelfile;
        # the projector follows), keeping the parent's renderer, parser and sampling, and the drafter and N of $mf.
        tmf="Modelfile.$size.n$n-text"
        pmf="$(podman exec "$C" ollama show --modelfile "$parent")"
        mapfile -t froms < <(awk '$1 == "FROM" {print $2}' <<< "$pmf")
        blob="${froms[0]:-}"
        if [[ ! "$blob" =~ ^/root/\.ollama/models/blobs/sha256-[0-9a-f]{64}$ ]]; then
          echo "$parent: no weights file in its Modelfile (got '${blob:-nothing}'); not created" >&2; exit 1
        fi
        bsize="$(podman exec "$C" stat -c %s "$blob")"
        for f in "${froms[@]:1}"; do       # the weights are the largest file; anything else is the projector
          if (( $(podman exec "$C" stat -c %s "$f") >= bsize )); then
            echo "$parent: the first FROM ($blob) is not its largest file; not created" >&2; exit 1
          fi
        done
        base="$(cat "$PROJ/models/drafts/$mf")"
        { echo "# written by bin/ollama.sh install $tag ($(date -u +%FT%TZ)): $parent's weights without its vision"
          echo "# projector, with the drafter and N of $mf"
          echo "FROM $blob"
          grep '^DRAFT ' <<< "$base"
          grep -E '^(RENDERER|PARSER|PARAMETER) ' <<< "$pmf"
          grep '^PARAMETER draft_num_predict ' <<< "$base"
        } > "$STORE/drafts/$tmf"
        case "$size" in      # the 31B weights chatwatch was measured with (Ollama's gemma4:31b-it-qat)
          31b) studied=0374ce7b0124db9ba96fc649e835c531223ee224a497ce88a374baaea10932ec ;;
          *) studied="" ;;
        esac
        if [[ -n "$studied" && "$blob" != */sha256-$studied ]]; then
          echo "note: $parent here is ${blob##*/}, not the measured sha256-$studied (Ollama updated it): the build follows it" >&2
        fi
        podman exec "$C" sh -c "cd /root/.ollama/drafts && ollama create $tag -f $tmf"
      fi
    else
      pull_model "$tag"
    fi ;;
  version) curl -s --max-time 5 127.0.0.1:"$PORT"/api/version; echo ;;
  stop) podman stop "$C" ;;
  *) echo "usage: $0 start | install <tag> | pull <model> | warm <model> | version | stop" >&2; exit 1 ;;
esac
