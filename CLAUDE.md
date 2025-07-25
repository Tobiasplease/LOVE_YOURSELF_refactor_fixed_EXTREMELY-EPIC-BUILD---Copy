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

# Test individual components
python debug/test_ollama_caption.py
python debug/test_comfy.py drawing/example_workflow.json
python debug/test_impostor_flow.py

# Code formatting (configured in pyproject.toml)
black . --line-length 150
isort . --profile black --line-length 150

# Linting
pylint . --max-line-length=150
flake8 . --max-line-length=150
```

## Architecture Overview

This is an AI-powered interactive mirror system that combines computer vision, mood analysis, and servo control. The system is structured as a modular Python application with threaded processing for real-time performance.

### Configuration Override System

The system supports runtime configuration overrides via JSON files passed to machine.py:

```bash
# Debug mode (faster intervals for development)
python machine.py --config_override config/debug_config.json

# Production mode (optimized intervals)
python machine.py --config_override config/production_config.json
```

**Available configurations:**
- `config/debug_config.json`: Fast intervals for development (REASON_INTERVAL: 30s, DRAWING_INTERVAL: 60s)
- `config/production_config.json`: Standard intervals for production use

**Creating custom overrides:** Any config variable in `config/config.py` can be overridden by creating a JSON file with the desired values. The system handles type conversion automatically.

### Core Components

- **machine.py**: Main application entry point and coordination loop with config override support
- **config/config.py**: Centralized configuration including camera, servo, and AI model settings
- **config/loader.py**: Configuration override system for runtime customization
- **perception/**: Computer vision modules (face detection, object detection with YOLO)
- **captioner/**: AI captioning system with memory management using Ollama API
- **mood/**: Mood analysis and emotional processing engine
- **vision/**: Gaze tracking and visual processing
- **breathing/**: Breathing simulation for life-like behavior
- **drawing/**: ComfyUI integration for AI image generation based on mood
- **servo_control/**: Arduino servo motor control for physical interaction
- **event_logging/**: JSON-based event logging and run management
- **utils/**: Utility modules including Ollama API wrapper and continuity helpers

### Key Architecture Patterns

1. **Threaded Processing**: Uses threading for real-time camera processing and AI analysis
2. **Modular Design**: Each major function is isolated in its own module
3. **Configuration-Driven**: All settings centralized in config/config.py
4. **External API Integration**: Uses Ollama for AI processing (llava model for vision) and ComfyUI for image generation
5. **Event-Driven Logging**: Comprehensive JSON logging of all system events
6. **Drawing Controller**: Intelligent decision-making for when to generate images based on mood, novelty, and boredom metrics

### External Dependencies

- **Ollama API**: Must be running locally at http://localhost:11434 with llava model (llava:7b-v1.6-mistral-q5_1)
- **ComfyUI (Optional)**: For AI image generation at http://localhost:8188/prompt
- **OpenCV DNN Models**: Requires face detection models in models/ directory:
  - deploy.prototxt
  - res10_300x300_ssd_iter_140000.caffemodel
- **YOLO Models**: yolov8m.pt and yolov8n.pt (included)
- **Arduino (Optional)**: For servo control if USE_SERVO=True

### Data Flow

1. Camera captures frames → face/object detection (OpenCV + YOLO)
2. Detected frames → mood analysis via Ollama
3. Mood data → drawing decisions (ComfyUI integration)
4. Mood data → servo positioning (if enabled)
5. All events → JSON logging with timestamps and run IDs
6. Images saved to mood_snapshots/ directory with event logs
7. Generated artwork stored with associated metadata

### Environment Variables

- `MOOD_SNAPSHOT_FOLDER`: Override default mood snapshot storage location
- `OLLAMA_MODEL`: Specify Ollama model (default: llava:7b-v1.6-mistral-q5_1)

### Testing

The debug/ folder contains individual component tests:
- `test_ollama_caption.py`: Tests Ollama API integration and captioning
- `test_comfy.py`: Tests ComfyUI workflow execution with JSON templates
- `test_impostor_flow.py`: Tests the complete impostor image generation pipeline

No formal test framework is configured - tests are standalone scripts.

### Code Style

- Line length: 150 characters (configured in pyproject.toml)
- Uses Black formatter with isort for import sorting
- Pylint and flake8 for linting
