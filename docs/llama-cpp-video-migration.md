# llama.cpp Video Migration Plan

## Why

Ollama wraps llama.cpp but doesn't expose Qwen3.5's temporal video encoding (Conv3D super-frames + M-RoPE). Sending multiple images through Ollama treats them as unrelated pictures — the model can't perceive motion. The `llama-video` package patches llama.cpp's vision encoder to pair consecutive frames into 6-channel super-frames, giving the model genuine temporal perception.

Secondary benefit: direct llama-server gives us cleaner VRAM control (stop/start process vs. keep_alive API calls to a shared daemon).

## Architecture Change

### Before (Ollama)
```
Camera (30fps) → drop all but 1 frame per 10s → save to disk → base64 encode →
  Ollama /api/chat (single image) → caption
```

### After (llama-server + llama-video)
```
Camera (30fps) → frame buffer (ring buffer, ~2fps keyed by frame diff) →
  every 10s: pick last 10-20 frames → llama-video Preprocessor →
  super-frames (paired, 6-channel) + temporal M-RoPE →
  llama-server /v1/chat/completions → caption
```

## Components

### 1. llama-server setup
- Clone llama.cpp, apply llama-video patch (`video-support-20260424.patch`, pinned to commit `0adede8`)
- Build with CUDA: `cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release`
- Download Qwen3.5 9B GGUF + mmproj from HuggingFace
- Run: `./llama-server -m qwen3.5-9b.gguf --mmproj mmproj.gguf --jinja --ctx-size 65536 --port 8080`

### 2. utils/llama_server.py (replaces utils/ollama.py)
- Drop-in replacement for `query_ollama()` → `query_llama_server()`
- Uses /v1/chat/completions (OpenAI-compatible format)
- Supports both single-image (regular caption) and multi-frame video input
- For video: uses llama-video's Preprocessor to create super-frames + temporal metadata
- VRAM management: process lifecycle (start/stop) instead of keep_alive API

### 3. Frame buffer (new, in captioner/)
- Ring buffer collecting frames at ~2fps from the main camera loop
- Frame diff scoring to identify "interesting" frames
- On caption cycle: extract last N seconds of buffered frames
- Pass to llama-video Extractor/Preprocessor pipeline

### 4. VRAM handoff (update drawing/drawing.py)
- Before ComfyUI: stop llama-server process (or send unload command)
- After ComfyUI: restart llama-server, wait for model load
- Simpler than current Ollama keep_alive dance

## Dependencies

- llama-video: `pip install "llama-video[ui]"` (includes Extractor, Preprocessor, client)
- ffmpeg (already likely installed for other media handling)
- Qwen3.5 GGUF model + mmproj file (download from HuggingFace)
- CMake 3.21+, CUDA toolkit (already have for ComfyUI)

## Migration Path

1. Build patched llama-server alongside existing Ollama (both can coexist)
2. Write utils/llama_server.py with same API as utils/ollama.py
3. Add frame buffer to captioner
4. Test with single-image first (parity with current system)
5. Enable multi-frame video input
6. Update VRAM handoff in drawing.py
7. Once stable: remove Ollama dependency entirely

## Model Files Needed

- `Qwen3.5-9B-Instruct-GGUF` (or the 35B-A3B MoE for better quality at similar speed)
- Matching `mmproj-*.gguf` vision projector file
- Both from: https://huggingface.co/Qwen/Qwen3.5-9B-Instruct-GGUF (or similar)
