# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

This is a Python project. Common development commands:

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the main application (defaults = Qwen3.6-27B, hybrid stream shape)
python machine.py

# Explicit pins / A-B arms
./run_27b.sh          # force the 27B hybrid stack
./run_9b.sh           # the 9B arm, with the sampling it was tuned for

# Run with configuration overrides
python machine.py --config_override config/debug_config.json
python machine.py --config_override config/qwen_experiment.json

# Debug tools
python debug/log_viewer.py
python debug/force_memory_reset.py
python debug/test_comfy.py drawing/example_workflow.json
python debug/centerline_settings_explorer.py

# Code formatting (configured in pyproject.toml)
black . --line-length 150
isort . --profile black --line-length 150

# Linting
pylint . --max-line-length=150
flake8 . --max-line-length=150
```

## Architecture Overview

An AI-powered interactive mirror system. A camera observes a space, a vision-language model (Qwen3.5 via a patched llama-server — the sole inference backend) generates captions, and these drive mood analysis, physical outputs (servo, CNC arm), and periodic drawing generation (ComfyUI). The system is structured as a threaded Python application.

**`docs/runtime-map.md` is the maintained source of truth for what is actually live at runtime.** Update it whenever wiring changes. Historical/superseded plan docs live in `docs/archive/` — do not trust them against current code.

### Configuration Override System

Any variable in `config/config.py` can be overridden at runtime via a JSON file:

```bash
python machine.py --config_override config/debug_config.json    # Fast intervals for development
python machine.py --config_override config/qwen_experiment.json  # Qwen2.5-VL model settings
```

Platform-specific overrides in `config/gpu-peon/`, `config/impostor-bot-win/`, `config/jbe-osx/`.

### Core Components

- **machine.py**: Main entry point. Starts all threads, registers hooks, owns the main loop.
- **config/**: Centralized config system (`config.py`, `loader.py`, `model_settings.py`)
- **perception/**: Computer vision — face detection (OpenCV DNN), object detection (YOLO), spatial memory
- **captioner/**: AI captioning pipeline — the core of the system:
  - `captioner.py`: Main captioner class (MemoryMixin base in `memory.py`), runs caption/awakening/environmental cycles; salience assessment gates prompt interiority
  - `reflection.py`: Reflection loop — long-form thought on rotating subjects every ~20 quiet minutes, stored in ChromaDB
  - `semantic_memory.py`: ChromaDB-backed concept ledger (observations, reflections, familiarity)
  - `memory.py`: MemoryMixin — session memory, identity tracking, activation network integration
  - `activation_memory.py`: Activation-spreading memory network for concept recall and boredom scoring
  - `context_compression.py`: Compresses recent captions into evolving baseline context (every N captions via background thread)
  - `frame_buffer.py`: Rolling frame + detection-snapshot buffer feeding the caption loop
  - `model_wrapper.py`: Vision-model API wrapper (llama-server)
  - `prompt_interface.py`: Builds prompts + model options for caption and drawing calls
  - `prompts.py`: All prompt templates and builder functions
- **mood/**: Mood analysis engine, emotional state tracking
- **vision/**: Gaze tracking, frame diff, visual processing
- **breathing/**: Breathing simulation for servo life-like behavior
- **drawing/**: ComfyUI integration with workflow templates
- **servo_control/**: Arduino servo motor control with PWM lightbulb
- **hand_control/**: legacy remnant — only `hand_expression.py` survives (servo calibration tooling); runtime hand movement is `motor_panel/kinetic_bus.py`
- **grbl/**: GRBL CNC machine integration for precise drawing
- **bcnc/**: bCNC G-code processing
- **safety/**: ArUco marker detection and paper presence detection for CNC safety
- **event_logging/**: JSON event logging with run management
- **utils/**: Utility modules:
  - `llama_server.py`: llama-server process management + query paths (the inference backend)
  - `inference.py`: Model query interface (query_model / query_model_video)
  - `llm_log.py`: Per-call LLM logging (prompt/response/stream observability)
  - `state_manager.py`: Shared runtime state (drawing status, paper detection, etc.)
  - `hooks.py`: Hook registration and dispatch
  - `continuity.py`: Time-gap description helpers
  - `drawing_state.py`: Drawing state helpers
  - `caption_display.py`: Display formatting for captions
  - `nlp.py`: spaCy singleton (consumed by perception/vocab_promotion)
  - `view_orientation.py`: Camera orientation helpers
  - `progress_bar.py`, `error_tracking.py`: Dev utilities
- **labs/warp-fix-lab/**: Experimental drawing warp correction scripts (not integrated)

### Key Architecture Patterns

1. **Threaded Processing**: Camera, caption, and compression each run in separate threads
2. **Activation Memory System**: Concepts extracted from captions feed an activation network; novelty drives prompt-mode selection, boredom drives caption sampling (temperature/length)
3. **Context Compression**: Every N captions, a background thread compresses recent observations into a rolling `baseline_context` string, injected into system prompts
4. **Prompt Mode Routing**: `build_simple_caption_prompt` selects a mode (relational/observational/workspace/introspective/awakening) based on activation state; mode determines prompt content
5. **Multi-Step Drawing Analysis**: `context_rich_multi_step_drawing_analysis` runs a 5-step pipeline to generate drawing prompts from current memory state
6. **Configuration-Driven**: All tuning parameters in `config/config.py` with JSON override support
7. **CNC Safety Gate**: ArUco marker + paper detection (safety/) gate all physical drawing; no paper = no draw

### External Dependencies

- **Inference backend** (single, since July 2026): patched llama.cpp llama-server at http://localhost:8080 with Qwen3.5 video support (Conv3D super-frames) and assistant prefill; managed by `utils/llama_server.py`. ALL calls (captions, reflections, compression, drawing analysis) run on the one loaded model (`MODEL_NAME` label, weights from `LLAMA_MODEL_PATH`). Ollama + mistral-nemo were retired.
- **ComfyUI (Optional)**: AI image generation at http://localhost:8188
- **OpenCV DNN Models**: `models/deploy.prototxt` + `models/res10_300x300_ssd_iter_140000.caffemodel`
- **YOLO Models**: `models/yolov8n.pt` default (`yolov8m.pt` at repo root also available)
- **spaCy**: `en_core_web_sm` model for noun-chunk extraction in vocab promotion (utils/nlp.py singleton)
- **Physical Hardware (Optional)**: Arduino (servo/hand), GRBL CNC controller, uArm Swift Pro
- The system must run fully offline during exhibitions — all models and dependencies local

### Data Flow

1. Camera frames → face/object detection → spatial memory updates
2. Every caption cycle: frame(s) + memory context → vision model (llama-server) → caption text
3. Caption → activation memory update → concept activation/boredom scores
4. Every N captions: background compression → `baseline_context` updated
5. Caption + context → mood analysis → emotional state
6. Emotional state → servo/hand control decisions
7. Periodically: memory state → 5-step drawing analysis → ComfyUI → CNC execution
8. All events → JSON event log

### Environment Variables

- `LLAMA_SERVER_URL`: llama-server endpoint (default `http://localhost:8080`)
- `LLAMA_MODEL_PATH` / `LLAMA_MMPROJ_PATH`: model weights for llama-server
- `VIDEO_MODE_ENABLED` / `VIDEO_MODE`: multi-frame perception (`multi` default since July 2026 — plain stills on standard llama.cpp mtmd; `superframe` = patched Conv3D temporal encoding, needs steady frames)
- `MOTION_THRESHOLD`: frame-diff threshold below which a single still is sent
- `MOOD_SNAPSHOT_FOLDER`: Override default event log storage location

### Testing

The `debug/` folder has standalone component tests and calibration tools. Notable scripts:

- `log_viewer.py`: Interactive event log viewer
- `force_memory_reset.py`: Reset memory/state files
- `test_reflection_loop.py`: End-to-end check of the reflection loop
- `test_llama_server.py`, `test_live_video.py`: llama-server backend and video perception
- `test_comfy.py`: Test ComfyUI workflow execution
- `centerline_settings_explorer.py`: SVG centerline processing configuration
- `test_caption_flow.py`: Caption pipeline inspection
- `test_drawing_introspection.py`, `test_multi_step_drawing.py`: Drawing pipeline tests
- `servo_calibration_tool.py`, `test_left_arm_servos.py`: Servo/arm calibration
- `reset_cnc_state.py`: Reset CNC state after a stalled job

No formal test framework — all tests are standalone scripts.

Put all test and evaluation scripts in the `debug/` folder and all plan files in the `docs/` folder.

### Code Style

- Line length: 150 characters (configured in pyproject.toml)
- Black formatter + isort for import sorting
- Pylint and flake8 for linting
- Use comments very sparingly — only where logic is not self-evident
