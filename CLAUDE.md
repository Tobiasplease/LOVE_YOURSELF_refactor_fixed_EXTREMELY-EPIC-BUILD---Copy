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

# Additional debug tools
python debug/log_viewer.py
python debug/force_memory_reset.py
python debug/test_pipeline.py

# Code formatting (configured in pyproject.toml)
black . --line-length 150
isort . --profile black --line-length 150

# Linting
pylint . --max-line-length=150
flake8 . --max-line-length=150
```

## Architecture Overview

This is an AI-powered interactive mirror system that combines computer vision, mood analysis, physical control systems, and advanced processing capabilities. The system is structured as a modular Python application with threaded processing for real-time performance, supporting multiple output modalities including servo control, hand gestures, CNC drawing, and speech recognition.

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
- `config/gpu-peon/`: GPU-optimized configurations for high-performance systems
- `config/impostor-bot-win/`: Windows-specific production settings
- `config/jbe-osx/`: macOS-specific configurations

**Creating custom overrides:** Any config variable in `config/config.py` can be overridden by creating a JSON file with the desired values. The system handles type conversion automatically.

### Core Components

- **machine.py**: Main application entry point with debug mode and config override support
- **config/**: Centralized configuration system
  - **config.py**: Main configuration including camera, servo, and AI model settings
  - **loader.py**: Configuration override system for runtime customization
  - **Platform-specific configs**: GPU, Windows, macOS optimized settings
- **perception/**: Computer vision modules
  - Face detection, object detection with YOLO
  - Spatial memory and detection memory systems
- **captioner/**: AI captioning system with advanced memory management
  - Context compression, memory management, model wrapper
  - Prompt interface and template systems
- **mood/**: Mood analysis and emotional processing engine
- **vision/**: Gaze tracking and visual processing
- **breathing/**: Breathing simulation for life-like behavior
- **drawing/**: ComfyUI integration with multiple workflow templates
- **Physical Control Systems**:
  - **servo_control/**: Arduino servo motor control with PWM lightbulb control
  - **hand_control/**: Emotional hand movement mapping and Arduino integration
  - **grbl/**: GRBL CNC machine integration for precise drawing
  - **bcnc/**: bCNC G-code processing and workflow management
- **event_logging/**: JSON-based event logging, run management, and log types
- **utils/**: Extended utility modules
  - Ollama API wrapper, continuity helpers
  - Pattern recognition, motif scoring, thematic analysis
  - Progress tracking, error tracking, temporal awareness
- **labs/**: Experimental features
  - **warp-fix-lab/**: Drawing distortion correction experiments

### Key Architecture Patterns

1. **Threaded Processing**: Uses threading for real-time camera processing and AI analysis
2. **Modular Design**: Each major function is isolated in its own module with clear interfaces
3. **Configuration-Driven**: All settings centralized with platform-specific overrides
4. **Multi-Modal Output**: Supports multiple physical output systems (servo, CNC, hand control)
5. **External API Integration**:
   - Ollama for AI processing (llava model for vision)
   - ComfyUI for image generation with multiple workflow templates
6. **Event-Driven Architecture**:
   - Comprehensive JSON logging of all system events with run management
   - Lifetime state tracking and system state persistence
7. **Advanced Memory Systems**:
   - Context compression and memory management
   - Spatial memory for object tracking
   - Pattern recognition and thematic analysis
8. **Drawing Controller**: Intelligent decision-making for image generation based on mood, novelty, and boredom metrics
9. **Physical Integration**: CNC machine control with G-code processing and SVG conversion

### External Dependencies

- **Ollama API**: Must be running locally at http://localhost:11434 with llava model (llava:7b-v1.6-mistral-q5_1)
- **ComfyUI (Optional)**: For AI image generation at http://localhost:8188/prompt
  - Multiple workflow templates available for different deployment scenarios
- **OpenCV DNN Models**: Face detection models required in models/ directory:
  - deploy.prototxt
  - res10_300x300_ssd_iter_140000.caffemodel
- **YOLO Models**: yolov8m.pt and yolov8n.pt (included)
- **Physical Hardware (Optional)**:
  - **Arduino**: For servo control if USE_SERVO=True
  - **GRBL CNC Controller**: For precise drawing/engraving operations
  - **Hand Control Arduino**: For emotional gesture expression
- **Additional Models**:
  - **spaCy**: en_core_web_sm model for NLP processing
  - **Whisper (Optional)**: For local speech recognition

### Data Flow

1. **Input Processing**:
   - Camera captures frames → face/object detection (OpenCV + YOLO)
2. **AI Analysis**:
   - Detected frames → mood analysis via Ollama with context compression
   - Pattern recognition and thematic analysis on processed data
   - Memory system updates with spatial and temporal awareness
3. **Decision Making**:
   - Mood data → drawing decisions (ComfyUI integration with multiple templates)
   - Emotional state mapping → physical control decisions
4. **Physical Outputs** (if enabled):
   - Servo positioning for basic movement
   - Hand controller for emotional gestures
   - GRBL CNC control for precise drawing/engraving
5. **Data Persistence**:
   - All events → JSON logging with timestamps and run IDs
   - Images saved to event_log/ directory with associated metadata
   - Movement recordings stored for playback
   - Generated artwork stored with processing metadata
   - Lifetime state and system state persistence

### Environment Variables

- `MOOD_SNAPSHOT_FOLDER`: Override default event log storage location
- `OLLAMA_MODEL`: Specify Ollama model (default: llava:7b-v1.6-mistral-q5_1)
- Various hardware-specific environment variables for platform optimization

### Testing

The debug/ folder contains individual component tests and debugging tools:

- `test_ollama_caption.py`: Tests Ollama API integration and captioning
- `test_comfy.py`: Tests ComfyUI workflow execution with JSON templates
- `log_viewer.py`: Interactive event log viewing and analysis
- `force_memory_reset.py`: Memory system reset utility
- `centerline_settings_explorer.py`: SVG centerline processing configuration tool

No formal test framework is configured - tests are standalone scripts.

Put all test and evaluation scripts in the debug folder and all plan files in the docs folder.

### Code Style

- Line length: 150 characters (configured in pyproject.toml)
- Uses Black formatter with isort for import sorting
- Pylint and flake8 for linting with 150 character line length
- Dont use so many comments! Very sparingly.

### Additional Components

**Physical Integration:**

- Multiple Arduino firmware options for different control scenarios
- GRBL setup scripts and configuration utilities
- Movement recording and playback system
- SVG to G-code conversion with centerline processing

**Advanced Features:**

- Drawing warp correction laboratory
- Pattern recognition and motif extraction
- Thematic analysis and continuity tracking
- Comprehensive error tracking and progress monitoring
