#!/usr/bin/env bash
# The 9B A/B arm (Aug 5). Config defaults are the 27B hybrid stack now, so this
# is the script that goes BACK — same machine, smaller mind, and the sampling
# the 9B was tuned for (it blooms purple above ~0.7 where the 27B needs 1.0).
# Everything else — gates, seam, frame count, inference gate — is model-agnostic
# and stays exactly as it is.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

export LLAMA_SERVER_BIN="$HOME/llama.cpp/build/bin/llama-server"
export LLAMA_MODEL_PATH="$HOME/models/qwen3.5-9b/Qwen3.5-9B-Q5_K_M.gguf"
export LLAMA_MMPROJ_PATH="$HOME/models/qwen3.5-9b/mmproj-F16.gguf"
export LLAMA_CTX_SIZE=65536
export MODEL_NAME="qwen3.5:9b"

export CAPTION_TEMP="${CAPTION_TEMP:-0.7}"
export CAPTION_TEMP_BORED="${CAPTION_TEMP_BORED:-0.6}"
export CAPTION_TOP_P="${CAPTION_TOP_P:-0.85}"
export CAPTION_MIN_P="${CAPTION_MIN_P:-0.0}"
export STREAM_WINDOW="${STREAM_WINDOW:-6}"
export STREAM_CONSOLIDATE_CHARS="${STREAM_CONSOLIDATE_CHARS:-800}"

exec python machine.py "$@"
