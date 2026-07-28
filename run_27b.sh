#!/usr/bin/env bash
# Qwen3.6-27B experiment launcher (July 28) — same machine, bigger mind.
# Everything else is unchanged: STREAM_MODE=document, VIDEO_MODE=multi, all
# gates live. Plain `python machine.py` still runs the 9B as before.
#
# VRAM handoff around drawings is ALREADY BUILT and applies here unchanged:
# drawing.py unloads llama-server before ComfyUI allocates, and every query
# path restores it via ensure_server_up() — which frees ComfyUI's VRAM
# (/free) first, then restarts with retry (~6s warm reload for this model).
# 27B deltas vs 9B: thinner margins (21.3GB resident) and a longer reload
# silence — watch the first drawing cycle, expect it to just work.
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
