#!/usr/bin/env bash
# Qwen3.8-27B launcher (Aug 17). Since Aug 19 these pins are also the CODE
# DEFAULTS (utils/llama_server.py, config MODEL_NAME) — bare `python
# machine.py` boots the same 3.8 stack; this script remains the explicit,
# self-documenting pin. run_27b.sh re-pins the parked 3.6 arm.
#
# 3.8 is a NEW ARCHITECTURE (hybrid Gated DeltaNet + gated attention), not a
# 3.6 retrain. It needs the llama.cpp-38 build — older CUDA builds (including
# llama.cpp-27b) run it without error but emit garbage: the DeltaNet CUDA path
# was only fixed around build 10450. Do not point this script at the old bin.
#
# VRAM handoff around drawings applies unchanged (drawing.py unload +
# ensure_server_up restore). Q4_K_M is 17.1GB on disk vs 3.6's ~16.5GB —
# expect resident within ~1GB of the 21.3GB the 3.6 arm holds; watch the
# first drawing cycle as usual.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

export LLAMA_SERVER_BIN="$HOME/llama.cpp-38/build/bin/llama-server"
export LLAMA_MODEL_PATH="$HOME/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf"
export LLAMA_MMPROJ_PATH="$HOME/models/qwen3.8-27b/mmproj-F16.gguf"
export LLAMA_CTX_SIZE=16384
# MTP: 3.8 trained nextn layers into the base weights; mainline draft-mtp
# should work IF the unsloth quant kept them — unverified, and MTP is off in
# the 3.6 arm anyway (distribution purity while the voice question is open).
# --image-min-tokens 1024 (Aug 17, first 3.8 evening): the server warns at
# every load that Qwen-VL needs >=1024 image tokens for GROUNDING — spatial
# relations, who moved where — and our camera frames were encoding at ~600
# (measured: react calls ~2780 tokens total, ~1800 across 3 frames). Room
# awareness was starved at the eye, not the mind. Costs ~1.6s extra prompt
# eval per react call. The 3.6 arm runs without it — add there too before
# any A/B that touches perception.
if [ "${LLAMA_MTP:-0}" = "1" ]; then
  export LLAMA_EXTRA_ARGS="--spec-type draft-mtp --spec-draft-n-max 2 -fa on --image-min-tokens 1024"
else
  export LLAMA_EXTRA_ARGS="-fa on --image-min-tokens 1024"
fi
export MODEL_NAME="qwen3.8:27b"

# SAMPLING: inherited verbatim from run_27b.sh so the A/B isolates the model.
# The 3.6 numbers were tuned against ITS distribution (mode-pinning, "The ___"
# openings); 3.8's distribution is unmeasured. Qwen's official non-thinking
# recommendation for 3.8 is temp 0.7 / top_p 0.8 / presence_penalty 1.5 —
# the presence penalty hints the model repeats more than 3.6 at low temp.
# Try it as an arm once the baseline is heard:
#   CAPTION_TEMP=0.7 CAPTION_TOP_P=0.8 CAPTION_MIN_P=0 ./run_38.sh
export CAPTION_TEMP="${CAPTION_TEMP:-1.0}"
export CAPTION_TEMP_BORED="${CAPTION_TEMP_BORED:-0.9}"
export CAPTION_TOP_P="${CAPTION_TOP_P:-1.0}"
export CAPTION_MIN_P="${CAPTION_MIN_P:-0.05}"

# Information budget: same as the 3.6 arm until 3.8's verbosity is measured.
export STREAM_WINDOW=24
export STREAM_CONSOLIDATE_CHARS=12000

export STREAM_MODE="${STREAM_MODE:-hybrid}"

exec python machine.py "$@"
