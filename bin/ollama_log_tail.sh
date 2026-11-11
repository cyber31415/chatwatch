#!/usr/bin/env bash
# Copy the lines of cw-ollama's log the model picker's speed check reads: the
# layers Ollama put on the GPU, runner starts and runner aborts. Nothing else (no chat is ever in that log anyway).
#   bin/ollama_log_tail.sh OUT_FILE        (start.sh starts it in its own process group and stops it on exit)
# podman cannot follow (-f) a journald log with the file events backend, so this polls every 2 s with --since/--until.
set -u
OUT="${1:?usage: $0 OUT_FILE}"
PAT='offloaded [0-9]+/[0-9]+ layers|starting llama-server|llama-server terminated|runner terminated|out of memory|CUDA error|HIP error|ROCm error|GGML_ASSERT|signal: '
since="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
while true; do
  sleep 2
  now="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
  podman logs --since "$since" --until "$now" cw-ollama 2>&1 | grep -E "$PAT" | cut -c1-600 >> "$OUT"
  since="$now"
done
