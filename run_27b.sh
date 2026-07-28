#!/usr/bin/env bash
# Qwen3.6-27B experiment launcher (July 28) — same machine, bigger mind.
# Everything else is unchanged: STREAM_MODE=document, VIDEO_MODE=multi, all
# gates live. Plain `python machine.py` still runs the 9B as before.
#
# Known rough edge for this experiment: ComfyUI drawing generation may fail
# while ~21GB is resident (the stop→generate→unload→restart handoff bracket
# is not built yet) — the drawing aborts, the machine carries on.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

export LLAMA_SERVER_BIN="$HOME/llama.cpp-27b/build/bin/llama-server"
export LLAMA_MODEL_PATH="$HOME/models/qwen3.6-27b-mtp/Qwen3.6-27B-Q4_K_M.gguf"
export LLAMA_MMPROJ_PATH="$HOME/models/qwen3.6-27b-mtp/mmproj-F16.gguf"
export LLAMA_CTX_SIZE=16384
export LLAMA_EXTRA_ARGS="--spec-type draft-mtp --spec-draft-n-max 2 -fa on"
export MODEL_NAME="qwen3.6:27b"

exec python machine.py "$@"
