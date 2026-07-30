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

# The information budget (July 28): repetition is what the model can't see it
# already said. Six entries was a 9B relic; the 27B holds minutes of visible
# selfhood. Consolidation threshold scales with it — and with the model's
# actual verbosity: 6000 assumed ~250 chars/entry, but the 27B writes ~400,
# so the stream sat permanently over threshold and consolidation thrashed
# (~140 folds in 2h on July 30, churning the visible past and doubling the
# long-call load that correlates with server wedges). 12000 puts the
# threshold above the 24-entry steady state; folds become occasional again.
export STREAM_WINDOW=24
export STREAM_CONSOLIDATE_CHARS=12000

# Next experiment when ready: the world shape on a model that can follow it —
# uncomment for grounded + connected + naturally varied, or prefix at launch:
#   STREAM_MODE=world ./run_27b.sh
# export STREAM_MODE=world

exec python machine.py "$@"
