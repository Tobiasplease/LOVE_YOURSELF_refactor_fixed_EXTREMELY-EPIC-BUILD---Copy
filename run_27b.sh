#!/usr/bin/env bash
# Qwen3.6-27B launcher. NOTE (Aug 5): this is no longer required — the config
# defaults ARE this stack, so `python machine.py` runs the 27B hybrid setup.
# Kept as the explicit pin: it forces the 27B regardless of what the defaults
# later become, which is what the tmux launchers want. run_9b.sh is the A/B arm.
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
# MTP off by default since Aug 1: speculative decoding is meant to preserve the
# sampling distribution, but it is a variable the 9B never had, and the voice
# question is a distribution question. Re-enable to get the ~1.7x decode speed
# back once the register is settled: LLAMA_MTP=1 ./run_27b.sh
if [ "${LLAMA_MTP:-0}" = "1" ]; then
  export LLAMA_EXTRA_ARGS="--spec-type draft-mtp --spec-draft-n-max 2 -fa on"
else
  export LLAMA_EXTRA_ARGS="-fa on"
fi
export MODEL_NAME="qwen3.6:27b"

# SAMPLING (Aug 1): 0.7/top_p 0.85 was tuned to restrain a 9B; on a 27B it pins
# every token to the mode — measured as 72-76% "The ___" openings, 69%
# semicolons, lengths welded to 37-61 words, while vivid felt-state inputs
# ("blind but screaming internally") came out as polished literature. Temp 1.0
# with min_p instead of top_p: the tail is cut in proportion to the model's
# confidence, so variety arrives where the distribution is genuinely flat and
# no gibberish arrives where it is sharp. top_p 1.0 = off (min_p replaces it).
export CAPTION_TEMP="${CAPTION_TEMP:-1.0}"
export CAPTION_TEMP_BORED="${CAPTION_TEMP_BORED:-0.9}"
export CAPTION_TOP_P="${CAPTION_TOP_P:-1.0}"
export CAPTION_MIN_P="${CAPTION_MIN_P:-0.05}"

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

# THE SHAPE TRAVELS WITH THE SCRIPT (Aug 3). This used to rely on the caller
# prefixing STREAM_MODE=hybrid, so a bare ./run_27b.sh quietly ran document
# mode — the same "which invocation decides the config" trap that made a whole
# day of experiments unattributable. Hybrid is the landed shape: world ordering
# (perception last) plus a seam of the machine's own unfinished thought.
# Still overridable for A/B:  STREAM_MODE=world ./run_27b.sh
export STREAM_MODE="${STREAM_MODE:-hybrid}"

exec python machine.py "$@"
