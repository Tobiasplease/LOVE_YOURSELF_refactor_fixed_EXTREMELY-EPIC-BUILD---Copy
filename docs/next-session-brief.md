# Next Session Brief — Memory Redesign + Super-frame Fix

## Current State (June 7, 2026)

**Branch:** `experimental/llama-cpp-video`

### What's Working Now
- **llama-server is the primary backend** — Ollama fully deprecated in this branch
- `machine.py` auto-starts llama-server on launch (falls back to Ollama if it fails)
- Model paths hardcoded: `~/models/qwen3.5-9b/Qwen3.5-9B-Q5_K_M.gguf` + `mmproj-F16.gguf`
- **Multi-image video mode is active** — sends 2-6 frames when motion detected (diff > 0.015)
- All ~20 `query_ollama` call sites migrated to `query_model()` via `utils/inference.py`
- Single model (Qwen3.5-9B) handles ALL tasks: vision, compression, reflection, drawing analysis
- Frame buffer wired into main camera loop at ~2fps

### Prompt System (Stable)
- System prompt: embodied first-person, no camera references, explicitly "not drawing right now"
- Mode selection: workspace > relational > observational > introspective (based on gaze/YOLO/activation)
- User prompt layers: situational line, mode context, drawing state, felt-state, desire, baseline context, thought thread
- Memory mode fires every 4 minutes ("I remember...") — generates interesting confabulated memories
- Token budget ~150 words with line-level trimming

### What's Broken
- **Super-frame mode** — `llama_video.LlamaServerClient.caption_video()` doesn't pass `chat_template_kwargs: {"enable_thinking": false}`, causing Qwen3.5 to dump full chain-of-thought. Needs either:
  - Patch `llama_video` client to support the flag
  - Or bypass `caption_video()` and build the API call ourselves from preprocessor output (recommended)
  - Set `VIDEO_MODE=superframe` to test once fixed
- **Memory system** — extensively audited, see `docs/memory-redesign-plan.md`

## Memory Redesign Status

### Completed (Phase 1: Reconnect Existing Wires)
- `context_compressor.get_current_desire()` now injected into caption prompt ("Preoccupied with: ...")
- `baseline_context` first sentence injected in observational/workspace modes
- Broken `get_session_greeting()` disabled (was injecting raw caption fragments as "I know this place — {garbage}")
- Error string sanitizer prevents `[WARNING]` text leaking into prompts
- Concept label validation tightened (rejects sentences with periods)
- Poisoned ChromaDB concepts, activation snapshot, and drawing memory cleaned

### Pending (Phases 2-5 — see docs/memory-redesign-plan.md)

**Phase 2: Core Memory Block** (1-2 hours)
- Extend `machine_identity.json` with `core_facts` section (place, people, drawings, self)
- Replace disabled `get_session_greeting()` with curated stable facts
- Updated by reflection worker when concepts pass observation threshold

**Phase 3: Fix Concept Storage Quality** (2-3 hours)
- Replace `_extract_canonical_name()` regex with LLM extraction
- Run once per compression cycle (every 8 captions), not per caption
- Extract from compression output, not raw monologue

**Phase 4: Delete Dead Code** (30 min)
- ~400 lines of orphaned functions identified across prompts.py, activation_memory.py, semantic_memory.py
- Full list in `docs/memory-redesign-plan.md` under "Dead Functions"

**Phase 5: Episodic Consolidation** (1 hour)
- Periodic promotion of repeated observations to core_facts
- Wire drawing events from episodic log into caption prompt

### Discovery Capture (Proposed, Not Started)
- Add a question to the existing introspection cycle (every ~12 captions):
  "From these recent thoughts, was there anything that felt like a personal memory or self-discovery?"
- Saves to existing `discoveries` list in `machine_identity.json`
- Model curates what's worth keeping — avoids saturation from memory mode outputs

## Key Architecture Decisions Made This Session

1. **Single backend**: llama-server only, no Ollama. One model (Qwen3.5-9B) for everything.
2. **Multi-image over super-frame**: Multi-image works reliably. Super-frame needs client fix.
3. **Memory systems are disconnected, not missing**: desires, beliefs, baseline_context, reflections all exist but weren't wired to the prompt. Phase 1 reconnected desires + baseline.
4. **Concept labels are unreliable**: `_extract_canonical_name()` regex extracts monologue fragments, not object concepts. Root cause of all "I know this place — {garbage}" issues.

## Disconnected Memory Systems (from audit)

Data that EXISTS but still doesn't reach the caption prompt:

| System | Data | Status |
|--------|------|--------|
| context_compression | `current_belief` | Generated, persisted. Only in introspective system prompt. |
| context_compression | `discoveries` | Generated, persisted. Only fed back to introspection itself. |
| semantic_memory | reflections | Stored in ChromaDB. `get_current_thread()` never called. |
| drawing_memory | artistic arc | Only in introspective mode + drawing pipeline. |
| episodic_log | drawing events | Recorded but never queried for prompt. |
| activation_memory | attention state | `generate_state_summary()` never called. |

## Files Changed This Session

| File | Change |
|------|--------|
| `config/config.py` | INFERENCE_BACKEND=llama_server, VIDEO_MODE_ENABLED=true, VIDEO_MODE=multi |
| `utils/inference.py` | NEW — unified query_model() router + VRAM lifecycle |
| `utils/llama_server.py` | Model paths hardcoded, Frame type fix, caption_video fix attempt |
| `captioner/frame_buffer.py` | Already existed, now wired into machine.py |
| `machine.py` | frame_buffer.push(), llama-server auto-start with Ollama fallback |
| `captioner/captioner.py` | Video routing in _process_frame(), all query_ollama -> query_model |
| `captioner/prompts.py` | System prompt rewrite (embodied POV), desire/baseline injection, error sanitizer, dead session_greeting disabled |
| `captioner/model_wrapper.py` | query_ollama -> query_model |
| `captioner/context_compression.py` | query_ollama -> query_model |
| `captioner/semantic_memory.py` | query_ollama -> query_model, concept validation tightened |
| `drawing/drawing.py` | query_ollama -> query_model, VRAM handoff via unified wrapper |
| `drawing/drawing_memory.py` | query_ollama -> query_model, poisoned entries cleaned |
| `grbl/grbl_utils.py` | query_ollama -> query_model |
| `mood/mood.py` | query_ollama -> query_model |
| `docs/memory-redesign-plan.md` | NEW — full architecture audit + phased redesign plan |

## Test Commands

```bash
# llama-server auto-starts with machine.py, but to run manually:
~/llama.cpp/build/bin/llama-server \
  -m ~/models/qwen3.5-9b/Qwen3.5-9B-Q5_K_M.gguf \
  --mmproj ~/models/qwen3.5-9b/mmproj-F16.gguf \
  --jinja --ctx-size 65536 --port 8080 -ngl 99

# Run the system (llama-server starts automatically)
python machine.py

# To fall back to Ollama:
INFERENCE_BACKEND=ollama python machine.py

# To test super-frame (once fixed):
VIDEO_MODE=superframe python machine.py
```
