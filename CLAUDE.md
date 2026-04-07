# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

This is a Python project. Common development commands:

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the main application
python machine.py

# Run with configuration overrides
python machine.py --config_override config/debug_config.json
python machine.py --config_override config/production_config.json
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

An AI-powered interactive mirror system. A camera observes a space, a vision-language model (Ollama/LLaVA) generates captions, and these drive mood analysis, physical outputs (servo, CNC arm), and periodic drawing generation (ComfyUI). The system is structured as a threaded Python application.

### Configuration Override System

Any variable in `config/config.py` can be overridden at runtime via a JSON file:

```bash
python machine.py --config_override config/debug_config.json    # Fast intervals for development
python machine.py --config_override config/production_config.json
python machine.py --config_override config/qwen_experiment.json  # Qwen2.5-VL model settings
```

Platform-specific overrides in `config/gpu-peon/`, `config/impostor-bot-win/`, `config/jbe-osx/`.

### Core Components

- **machine.py**: Main entry point. Starts all threads, registers hooks, owns the main loop.
- **config/**: Centralized config system (`config.py`, `loader.py`, `model_settings.py`)
- **perception/**: Computer vision — face detection (OpenCV DNN), object detection (YOLO), spatial memory
- **captioner/**: AI captioning pipeline — the core of the system:
  - `captioner.py`: Main captioner class, runs caption/awakening/environmental cycles
  - `memory.py`: Agent memory — observations, drawing history, mood tracking, temporal lines
  - `activation_memory.py`: Activation-spreading memory network for concept recall and boredom scoring
  - `context_compression.py`: Compresses recent captions into evolving baseline context (every N captions via background thread)
  - `model_wrapper.py`: Ollama API wrapper for the vision model
  - `prompt_interface.py`: Builds prompts + model options for caption, reflection, drawing calls
  - `prompts.py`: All prompt templates and builder functions
  - `subconscious.py`: Psychological synthesis layer (used by debug scripts; not called in main loop)
- **mood/**: Mood analysis engine, emotional state tracking
- **vision/**: Gaze tracking, frame diff, visual processing
- **breathing/**: Breathing simulation for servo life-like behavior
- **drawing/**: ComfyUI integration with workflow templates
- **servo_control/**: Arduino servo motor control with PWM lightbulb
- **hand_control/**: Emotional hand movement mapping and Arduino integration
- **grbl/**: GRBL CNC machine integration for precise drawing
- **bcnc/**: bCNC G-code processing
- **safety/**: ArUco marker detection and paper presence detection for CNC safety
- **event_logging/**: JSON event logging with run management
- **utils/**: Utility modules:
  - `ollama.py`: Ollama HTTP API wrapper
  - `state_manager.py`: Shared runtime state (drawing status, paper detection, etc.)
  - `hooks.py`: Hook registration and dispatch
  - `continuity.py`: Time-gap description helpers
  - `temporal_awareness.py`: Temporal context for prompts
  - `drawing_state.py`: Drawing state helpers
  - `caption_display.py`: Display formatting for captions
  - `pattern_recognition.py`: NLP pattern extraction (spaCy-based, used in memory pipeline)
  - `view_orientation.py`: Camera orientation helpers
  - `progress_bar.py`, `error_tracking.py`: Dev utilities
- **labs/warp-fix-lab/**: Experimental drawing warp correction scripts (not integrated)

### Key Architecture Patterns

1. **Threaded Processing**: Camera, caption, and compression each run in separate threads
2. **Activation Memory System**: Concepts extracted from captions feed an activation network; activation levels drive boredom detection and prompt mode selection
3. **Context Compression**: Every N captions, a background thread calls Ollama to compress recent observations into a rolling `baseline_context` string, injected into system prompts
4. **Prompt Mode Routing**: `build_simple_caption_prompt` selects a mode (relational/observational/restless/workspace/introspective/awakening) based on activation state; mode determines prompt content
5. **Multi-Step Drawing Analysis**: `context_rich_multi_step_drawing_analysis` runs a 5-step pipeline to generate drawing prompts from current memory state
6. **Configuration-Driven**: All tuning parameters in `config/config.py` with JSON override support
7. **CNC Safety Gate**: ArUco marker + paper detection (safety/) gate all physical drawing; no paper = no draw

### External Dependencies

- **Ollama API**: Must be running at http://localhost:11434 with a vision model loaded
  - Default: `llava:7b-v1.6-mistral-q5_1` — set `OLLAMA_MODEL` to override
  - Qwen experiment: `qwen2.5vl:7b`
- **ComfyUI (Optional)**: AI image generation at http://localhost:8188
- **OpenCV DNN Models**: `models/deploy.prototxt` + `models/res10_300x300_ssd_iter_140000.caffemodel`
- **YOLO Models**: `yolov8m.pt` and `yolov8n.pt` (included)
- **spaCy**: `en_core_web_sm` model for NLP in activation memory
- **Physical Hardware (Optional)**: Arduino (servo/hand), GRBL CNC controller
- **Whisper (Optional)**: Local speech recognition

### Data Flow

1. Camera frames → face/object detection → spatial memory updates
2. Every caption cycle: frame + memory context → Ollama vision model → caption text
3. Caption → activation memory update → concept activation/boredom scores
4. Every N captions: background compression → `baseline_context` updated
5. Caption + context → mood analysis → emotional state
6. Emotional state → servo/hand control decisions
7. Periodically: memory state → 5-step drawing analysis → ComfyUI → CNC execution
8. All events → JSON event log

### Environment Variables

- `MOOD_SNAPSHOT_FOLDER`: Override default event log storage location
- `OLLAMA_MODEL`: Specify Ollama model (default: `llava:7b-v1.6-mistral-q5_1`)

### Testing

The `debug/` folder has standalone component tests and calibration tools. Notable scripts:

- `log_viewer.py`: Interactive event log viewer
- `force_memory_reset.py`: Reset memory/state files
- `test_comfy.py`: Test ComfyUI workflow execution
- `centerline_settings_explorer.py`: SVG centerline processing configuration
- `test_caption_flow.py`, `test_prompt_flow.py`: Caption/prompt pipeline inspection
- `test_drawing_introspection.py`, `test_multi_step_drawing.py`: Drawing pipeline tests
- `servo_calibration_tool.py`, `test_left_arm_servos.py`: Servo/arm calibration
- `capture_paper_references.py`: Capture ArUco/paper reference images
- `reset_cnc_state.py`: Reset CNC state after a stalled job

No formal test framework — all tests are standalone scripts.

Put all test and evaluation scripts in the `debug/` folder and all plan files in the `docs/` folder.

### Code Style

- Line length: 150 characters (configured in pyproject.toml)
- Black formatter + isort for import sorting
- Pylint and flake8 for linting
- Use comments very sparingly — only where logic is not self-evident
