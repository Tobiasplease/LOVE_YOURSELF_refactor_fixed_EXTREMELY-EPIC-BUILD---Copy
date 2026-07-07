Prompt & Caption Pipeline Overview

This document maps the full prompt/caption flow in the repo, highlights where prompts are built and passed, and lists unused or legacy functions that can be simplified or removed. It focuses on these modules:

- captioner/captioner.py
- captioner/prompt_interface.py
- captioner/prompts.py
- captioner/model_wrapper.py
- utils/ollama.py
- Related helpers: captioner/memory.py, captioner/context_compression.py, utils/drawing_state.py, config/model_settings.py


High-Level Flow

- Frame intake: `Captioner.update()` queues frames with mood/reactivity.
- Caption loop: `Captioner._caption_worker()` pulls a frame and calls `_process_frame()`.
- Awakening phases:
  - Phase 1 (internal): `Captioner.generate_internal_awakening()` → `utils.ollama.query_ollama()` with `prompts.INTERNAL_AWAKENING_TEMPLATE` and formatted `prompts.SYSTEM_PROMPT` (no image).
  - Phase 2 (environmental): `MultimodalModel.caption_image(..., first_time=True)` → `PromptInterface.build_caption_prompt_with_options()` → `prompts.build_environmental_caption_prompt()` with the first frame sent as image.
- Ongoing captions: `MultimodalModel.caption_image(..., flowing=True)` → `PromptInterface.build_caption_prompt_with_options()` → `prompts.build_ongoing_caption_prompt()`.
- Reflections (periodic): `MultimodalModel.reason_about_caption()` → `PromptInterface.build_reflection_prompt_with_options()` → `prompts.build_reflection_prompt()`.
- Drawing prompts (interval): `MultimodalModel.generate_drawing_prompt()` → `PromptInterface.build_drawing_prompt_with_options()` → `prompts.build_drawing_prompt()` → `drawing.DrawingController.handle_drawing_flow()`.
- Logging: All LLM calls flow through `utils.ollama.query_ollama()` which logs prompt, system prompt, options, and response via `event_logging`.


Prompt Building Responsibilities

- `captioner/prompt_interface.py` (single entry point)
  - Selects template based on context flags:
    - first_time=True → `build_environmental_caption_prompt`.
    - else → `build_ongoing_caption_prompt`.
  - Assembles model options via `config.model_settings.get_model_options(model)` and adds mild randomness (seed/temperature/top_p/…)
  - Chooses system prompt:
    - If `memory_ref.get_dynamic_system_context()` available → formats `prompts.SYSTEM_PROMPT` placeholders `{emotional_state}`, `{temporal_context}`, `{accumulated_understanding}`.
    - Else → `prompts.STATIC_SYSTEM_PROMPT`.
  - Reflection and drawing use analogous helpers to build prompt + options + system prompt.

- `captioner/prompts.py` (templates and builders)
  - System-level templates:
    - `SYSTEM_PROMPT` and `STATIC_SYSTEM_PROMPT` (captioning context).
    - `DRAWING_SYSTEM_PROMPT` (drawing intent output).
    - `SELF_CRITIQUE_SYSTEM_PROMPT` (used for post-drawing critique in `grbl/grbl_utils.py`).
  - Caption builders:
    - `build_environmental_caption_prompt(agent, mood, boredom, novelty, last_session_gap)` — first observation grounding.
    - `build_ongoing_caption_prompt(agent, last_caption)` — continuous, identity- and memory-aware observation.
  - Reflection builder:
    - `build_reflection_prompt(caption, extra, agent)` — short introspections using session time and memory.
  - Drawing builder:
    - `build_drawing_prompt(memory_ref, extra)` — composes a directive drawing intent using recent caption, memory snippets, last reflection, emotional description, and top motifs.
  - Utilities used in builders:
    - `beliefs_to_sentence`, `get_caption_emotion_context`, `extract_motifs_spacy`, `_is_significant_motif`.

- `captioner/model_wrapper.py` (pure API facade)
  - Delegates all prompt composition to `PromptInterface`, then calls `utils.ollama.query_ollama()`.
  - Cleans response with `_clean_response()` to strip unwanted boilerplate.

- `utils/ollama.py` (Ollama transport)
  - Accepts `prompt`, optional `system`, optional `image`, and `options`; supports streaming progress.
  - Centralized logging in `log_ollama_call()` with `prompt_type` labels (caption, reflection, drawing, awakening, …).
  - Blocks only during ComfyUI generation (not during CNC execution) so captions continue while drawing.


Context and Memory Inputs

- `captioner/memory.py` (via `Captioner`’s `MemoryMixin`):
  - Supplies temporal lines, top motifs, emotion context, baseline self-understanding, identity summary, and memory snippets used across builders.
  - Formats dynamic system prompt context via `get_dynamic_system_context()` (emotion, temporal context, accumulated understanding) consumed by `PromptInterface`.

- `captioner/context_compression.py` (referenced in `PromptInterface`):
  - Provides `get_baseline_context()` and `get_current_sentiment_context()` to reduce long-term context into short injections for prompts.

- `utils/drawing_state.py`:
  - Global drawing state for determining if the system is actively drawing. Builders use `DrawingState.get_drawing_info()` to switch to drawing-aware language inside `build_ongoing_caption_prompt`.


Where Prompts Are Sent

- Captions: `MultimodalModel.caption_image(...)` → `query_ollama(..., image=path, system=SYSTEM_PROMPT, options=...)` with `prompt_type="caption"`.
- Reflections: `MultimodalModel.reason_about_caption(...)` → `query_ollama(..., prompt_type="reflection")`.
- Drawing prompts: `MultimodalModel.generate_drawing_prompt(...)` → `query_ollama(..., prompt_type="drawing")`.
- Awakening (internal): `Captioner.generate_internal_awakening()` → `query_ollama(..., prompt_type="awakening")` (no image).


Unused or Legacy Functions (as of this repo state)

The following functions appear unused within the repository (no external calls found). They can likely be removed or inlined after a quick double-check for external integrations:

- captioner/prompts.py
  - `get_session_feeling(...)` — superseded by richer temporal context; only referenced in a commented line.
  - `build_caption_prompt(...)` — legacy wrapper that forwards to `build_ongoing_caption_prompt`; not called anywhere.
  - `build_simple_contextual_prompt(...)` — fallback prompt builder; not referenced.

- utils/drawing_state.py
  - `DrawingState.get_drawing_context_for_caption()` — string formatter for drawing context; builders consume `get_drawing_info()` directly instead.

- utils/prompt_compression.py
  - Entire module currently unused (no references to `compress_drawing_prompt`). Consider removing or wiring into drawing prompt UX if desired.
  - Minor: the internal import of `NUMBER_GENERATOR_SYSTEM_PROMPT` is unused — safe to delete if keeping the module.

- config/model_settings.py
  - `get_model_prompt_style(...)`, `get_emotion_aware_system_prompt(...)`, `is_qwen_model(...)`, `is_llava_model(...)` — not referenced.
  - Large `MODEL_SYSTEM_PROMPTS` entries beyond generation options are effectively unused because `PromptInterface` sources system prompts from `captioner/prompts.py`.

- perception/spatial_memory.py
  - The whole spatial memory module (`SpatialMemory`, `spatial_memory`) is not referenced anywhere. If spatial context in captions is still a goal, integrate its `get_spatial_context()` output into `build_ongoing_caption_prompt()`; otherwise consider removing to reduce surface area.


Improvement Opportunities

- Consolidate system prompts: keep `SYSTEM_PROMPT`, `STATIC_SYSTEM_PROMPT`, `DRAWING_SYSTEM_PROMPT` in one place (current approach) and remove dead model-specific prompt variants in `config/model_settings.py`.
- Remove legacy builders: drop `build_caption_prompt` and `build_simple_contextual_prompt` to avoid confusion with the centralized `PromptInterface` path.
- Integrate or delete spatial memory: if helpful, pipe `perception/spatial_memory.spatial_memory.get_spatial_context()` into `build_ongoing_caption_prompt` under a short, optional block; otherwise remove the module.
- Consider using `compress_drawing_prompt` to add a short, user-readable label for `state_manager.current_drawing_prompt` and logging, or delete the module.
- Keep system prompt formatting robust: `PromptInterface` already falls back to `STATIC_SYSTEM_PROMPT` on formatting errors; retain this guard.


Call Chain Cheat Sheet

- Caption (ongoing):
  `Captioner._process_frame()` → `MultimodalModel.caption_image()` → `PromptInterface.build_caption_prompt_with_options()` → `prompts.build_ongoing_caption_prompt()` → `utils.ollama.query_ollama()`

- Caption (first observation):
  `Captioner._process_frame()` (awaiting env phase) → `MultimodalModel.caption_image(..., first_time=True)` → `prompts.build_environmental_caption_prompt()` → `utils.ollama.query_ollama()`

- Reflection:
  `Captioner._process_frame()` (interval) → `MultimodalModel.reason_about_caption()` → `prompts.build_reflection_prompt()` → `utils.ollama.query_ollama()`

- Drawing Intent:
  `Captioner._process_frame()` (interval) → `MultimodalModel.generate_drawing_prompt()` → `prompts.build_drawing_prompt()` → `utils.ollama.query_ollama()` → `drawing.DrawingController`


Notes

- All LLM I/O is centralized in `utils/ollama.query_ollama()`, which sets `prompt_type` for logging and applies model `options` from `config.model_settings.get_model_options()`.
- Dynamic system prompt formatting pulls small, structured context from `MemoryMixin.get_dynamic_system_context()`; be careful to keep keys stable: `{emotional_state}`, `{temporal_context}`, `{accumulated_understanding}`.

