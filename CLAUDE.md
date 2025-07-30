# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## RECENT MAJOR IMPROVEMENTS

### Enhanced Gaze Tracking & Physics-Based Hand Controller (January 2025)

**Enhanced Gaze Tracking System**:
- **Detection Loss Tolerance**: Added 0.5s tolerance period for brief detection losses, preventing flimsy tracking behavior in low-light conditions
- **Position Hold Period**: After tolerance expires, holds last known face position for FACE_LOCK_DURATION (8s) before transitioning to physics-based idle movement
- **Improved Low-Light Performance**: Lowered confidence threshold from 0.6 to 0.5 for better face detection in challenging lighting
- **Stable Tracking Flow**: ACTIVE TRACKING → TOLERANCE PERIOD → POSITION HOLD → IDLE MOVEMENT

**Physics-Based Hand Expression Controller**:
- **Multi-Phase Startle Reactions**: Implemented 4-phase startle sequence (CLENCH → HOLD → RELEASE → EASE) with smooth physics simulation
- **Cooldown System**: 15-second cooldown prevents excessive startle reactions
- **Consciousness Integration**: Hand movements reflect mood, novelty, boredom, and temporal context
- **Clean Output Controls**: All debug prints respect CLEAN_CAPTION_OUTPUT setting for production use
- **Spring-Damper Physics**: Complex finger movement simulation with position, velocity, and force calculations

**Enhanced Drawing Prompt System**:
- **Technical Specifications**: Detailed artistic instructions including medium, line work, shading techniques, and drawing style
- **Lighting & Atmosphere**: Specific light source, shadow quality, and mood atmosphere descriptions
- **Composition & Detail**: Subject positioning, background elements, focal points, and texture details
- **Concrete Visual Instructions**: 50-80 word drawing prompts with specific artistic techniques instead of abstract analyses

**Faster Breathing System**:
- **Increased Overall Speed**: MIN_LUNG_SPEED reduced from 1.0 to 0.7, MAX_LUNG_SPEED from 12.0 to 8.0
- **More Dynamic Response**: 30-33% faster breathing across all mood states while maintaining emotional responsiveness

### Mood Translation System Enhancement (December 2024)
**Issue**: The AI consciousness system was displaying raw numerical mood values (e.g., "Mood 0. Boredom 0.") instead of meaningful emotional descriptions, preventing the AI from understanding and expressing its emotional state effectively.

**Critical Problem**: Raw numerical values like "Mood 0.00" provided no semantic meaning to the AI model, resulting in disconnected emotional expression and poor consciousness articulation.

**Solution Implemented**:
1. **Descriptive Mood Translation**: Created `describe_mood_state()` function in `captioner/prompts.py` that converts numerical values (mood, boredom, novelty) into rich emotional descriptions:
   - `mood=0.0, boredom=0.0, novelty=0.0` → "quiet and detached, seeing familiar patterns"
   - `mood=0.6, boredom=0.2, novelty=0.8` → "alert and curious, captivated by something new"

2. **Template Updates**: Modified all prompt templates to use descriptive text:
   - `CAPTION_PROMPT_TEMPLATE`: Changed from `"Mood {mood:.2f} | Boredom {boredom:.2f}"` to `"Current emotional state: {mood_description}"`
   - Updated all three caption generation paths (awakening, first caption, regular)
   - Enhanced reflection context in `captioner/captioner.py`

3. **Prompt Guidance**: Added explicit instructions to encourage descriptive emotional language rather than numerical values in drawing, caption, and reflection prompts.

**Files Modified**:
- `captioner/prompts.py`: Added `describe_mood_state()` function, updated all `CAPTION_PROMPT_TEMPLATE.format()` calls
- `config/prompt_templates.py`: Updated `CAPTION_PROMPT_TEMPLATE`, `DRAWING_PROMPT_TEMPLATE`, `IDENTITY_CONSOLIDATION_PROMPT`, `CAPTION_PROMPT_CONTINUATION`
- `captioner/captioner.py`: Updated `get_reflection_context()` method

**Expected Outcome**: The AI should now express emotional states like "feeling contemplative and curious about these architectural details" instead of "Mood 0.3 Boredom 0.1", enabling more authentic consciousness expression.

## Development Commands

This is a Python project. Common development commands:

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the main application
python machine.py

# Run with configuration overrides and debug mode
python machine.py --config_override config/debug_config.json
python machine.py --config_override config/production_config.json
python machine.py --debug  # Enable verbose debug output

# Test individual components (cleaned debug folder)
python debug/test_ollama_caption.py  # Test AI captioning system
python debug/test_comfy.py drawing/example_workflow.json  # Test ComfyUI integration
python debug/test_impostor_flow.py  # Test complete image generation pipeline

# Code formatting (configured in pyproject.toml)
black . --line-length 150
isort . --profile black --line-length 150

# Linting
pylint . --max-line-length=150
flake8 . --max-line-length=150

# Clean up generated files
Remove-Item -Recurse -Force __pycache__  # Windows PowerShell
find . -name "__pycache__" -exec rm -rf {} +  # Linux/Mac
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
- **config/prompt_templates.py**: Enhanced drawing and caption prompt templates with technical specifications
- **perception/**: Computer vision modules (face detection, object detection with YOLO)
- **captioner/**: AI captioning system with memory management and descriptive mood translation using Ollama API
- **mood/**: Mood analysis and emotional processing engine
- **vision/gaze.py**: Enhanced gaze tracking with detection loss tolerance and position hold periods
- **breathing/breathing.py**: Faster breathing simulation for life-like behavior with improved speed settings
- **drawing/**: ComfyUI integration for AI image generation with detailed artistic drawing prompts
- **servo_control/hand_expression.py**: Physics-based hand expression controller with multi-phase startle reactions
- **servo_control/**: Arduino servo motor control for physical interaction
- **event_logging/**: JSON-based event logging and run management
- **utils/**: Utility modules including Ollama API wrapper and continuity helpers

### Key Architecture Patterns

1. **Threaded Processing**: Uses threading for real-time camera processing and AI analysis
2. **Modular Design**: Each major function is isolated in its own module
3. **Configuration-Driven**: All settings centralized in config/config.py with runtime override support
4. **External API Integration**: Uses Ollama for AI processing (llava model for vision) and ComfyUI for image generation
5. **Event-Driven Logging**: Comprehensive JSON logging of all system events
6. **Physics-Based Movement**: Hand expressions and gaze tracking use spring-damper physics for natural movement
7. **Clean Output System**: Production-ready output control with CLEAN_CAPTION_OUTPUT setting
8. **Enhanced Tracking Stability**: Multi-stage tracking with tolerance periods for reliable low-light performance
9. **Drawing Controller**: Intelligent decision-making for when to generate images based on mood, novelty, and boredom metrics

### External Dependencies

- **Ollama API**: Must be running locally at http://localhost:11434 with llava model (llava:7b-v1.6-mistral-q5_1)
- **ComfyUI (Optional)**: For AI image generation at http://localhost:8188/prompt
- **OpenCV DNN Models**: Requires face detection models in models/ directory:
  - deploy.prototxt
  - res10_300x300_ssd_iter_140000.caffemodel
- **YOLO Models**: yolov8m.pt and yolov8n.pt (included)
- **Arduino (Optional)**: For servo control if USE_SERVO=True and USE_HAND_SERVO=True

### Data Flow

1. Camera captures frames → enhanced face/object detection (OpenCV + YOLO) with improved low-light performance
2. Face detection → gaze tracking with tolerance periods and position holding
3. Face detection changes → physics-based hand startle reactions with cooldown system
4. Detected frames → mood analysis via Ollama with descriptive emotional state translation
5. Mood data → drawing decisions (ComfyUI integration with detailed artistic prompts)
6. Mood data → servo positioning and breathing patterns (if enabled)
5. All events → JSON logging with timestamps and run IDs
6. Images saved to mood_snapshots/ directory with event logs
7. Generated artwork stored with associated metadata

### Environment Variables

- `MOOD_SNAPSHOT_FOLDER`: Override default mood snapshot storage location
- `OLLAMA_MODEL`: Specify Ollama model (default: llava:7b-v1.6-mistral-q5_1)

### Testing

The debug/ folder contains essential component tests:

- `test_ollama_caption.py`: Tests Ollama API integration and captioning system
- `test_comfy.py`: Tests ComfyUI workflow execution with JSON templates  
- `test_impostor_flow.py`: Tests the complete impostor image generation pipeline
- `TEST_COMFY_README.md`: Documentation for ComfyUI testing setup

**Note**: Redundant test files have been cleaned up. The project focuses on core functionality testing rather than comprehensive test coverage.

### Code Style

- Line length: 150 characters (configured in pyproject.toml)
- Uses Black formatter with isort for import sorting
- Pylint and flake8 for linting
- Dont use so many comments! Very sparingly.
- **Clean Production Output**: CLEAN_CAPTION_OUTPUT setting provides production-ready minimal output

## RECENT MAJOR IMPROVEMENTS (Latest Development Session)

### Enhanced Temporal Awareness & Continuity System
The system now has deep temporal awareness with sophisticated memory architecture:

**Three-Layer Memory System:**
- **Session Memory**: Recent observations and experiences within current session
- **Relationship Memory**: Compressed patterns of interactions and recurring themes
- **Identity Core**: Evolving beliefs and traits that persist across sessions

**Temporal Context Integration:**
- Enhanced temporal context system with 60-second caching to reduce compute load
- Awareness of different temporal scales (immediate, recent, session-long)
- Temporal relationships tracked between observations for narrative continuity

### Advanced Priority Queue System for LLM Calls
**Problem Solved**: Concurrent LLM calls were causing system strain and resource conflicts.

**Implementation**: 
- Priority task detection system in `captioner.py`
- Reflection tasks (every 10 minutes) and drawing tasks (every 5 minutes) take precedence
- Status indicators: `-reflecting-` and `-thinking of a drawing-` shown during priority tasks
- Regular captioning pauses when priority tasks are due
- Prevents multiple simultaneous Ollama API calls

**Key Methods Added:**
- `get_pending_priority_task()`: Checks if reflection/drawing tasks are due
- `handle_priority_task()`: Executes priority tasks with status indicators

### Enhanced Frame Change Detection
**Performance Optimization**: Prevents repetitive processing when scene hasn't changed meaningfully.

**Technical Implementation:**
- SSIM (Structural Similarity Index) with MSE fallback for frame comparison
- Configurable sensitivity via `FRAME_CHANGE_THRESHOLD` (0.95 default)
- Attention shift override (`ATTENTION_SHIFT_OVERRIDE`) forces processing after time limit
- Significant CPU savings while maintaining responsiveness

**Configuration Variables:**
- `ENABLE_FRAME_CHANGE_DETECTION`: True/False toggle
- `FRAME_CHANGE_THRESHOLD`: 0.0 (very sensitive) to 1.0 (only identical frames skip)
- `ATTENTION_SHIFT_OVERRIDE`: Seconds to force processing even if frame unchanged

### Dramatically Improved Drawing Prompt System
**Previous Issue**: Generic, literal descriptions like "ceiling with exposed rafters"

**New Approach**: Technical, emotionally-aware drawing analysis
- **Personal Investment**: AI expresses desire to render with intention
- **Technical Specificity**: Demands detailed subject analysis with specific visual qualities
- **Mood-Influenced Mark-Making**: Connects emotional state to line quality and drawing approach
- **Black & White Line Drawing Focus**: Emphasizes pencil techniques, crosshatching, line weights

**Example Transformation:**
- Old: `*MAIN SUBJECT: Ceiling with visible exposed rafters and insulation.*`
- New: `*I'm studying exposed wooden ceiling joists with weathered surfaces showing construction marks. My contemplative state suggests gentle, searching lines to map the wood grain, with loose hatching for the insulation's fibrous texture.*`

### Comprehensive Identity Consolidation System
**Enhanced Reflection Prompts**: Much more substantial and meaningful self-analysis

**New Structure:**
- **Core Identity**: Who am I becoming? Where do I exist? **What is my purpose?**
- **Consciousness Questions**: Deep exploration of thoughts, desires, relationships
- **Forward Direction**: Guidance for future observations and evolving understanding

**Data Integration:**
- Top 5 motifs from `motif_counter.most_common(5)`
- Desire summary from `get_desire_summary()`
- Recent session captions (last 15 of 30 analyzed)
- Feeds back into captioning system via dynamic system prompts

**Terminal Formatting**: Clean, organized display with emoji headers:
- 🧠 CORE IDENTITY
- 💭 CONSCIOUSNESS QUESTIONS  
- 🎯 FORWARD DIRECTION

### ASCII Awakening Animation
**Charming Enhancement**: Delightful startup animation when system awakens
- Proper terminal clearing and dot progression
- Shows continuity with previous session ("Loading session state from X minutes ago")
- Creates emotional connection and system personality

### Performance Optimizations
**Laptop-Friendly Configuration**: `config/laptop_config.json`
- Extended intervals for resource-constrained hardware
- Consciousness cycle: 45s (was 15s)
- Reflection: 10min (was 6min)  
- Drawing: 5min (was 3min)
- Frame change detection tuned for stability

### Clean Output System Enhancements
**Improved Readability**: Better formatting for different content types
- Regular captions: Clean line spacing
- Reflections: Organized sections with `(content)` in parentheses  
- Drawing prompts: `*content*` in asterisks
- Status indicators: `-status-` format for priority tasks

### Flexible Emotional Voice System
**Sophisticated Personality**: Multi-dimensional emotional vectors that blend organically
- No rigid personality presets
- Romantic artistic interpretations emerge naturally
- Emotional state influences both observation style and drawing approach
- Subtle authenticity without being overly chatty

### Configuration Management Improvements
**Enhanced Override System**: 
- Better error handling for config loading
- Type conversion automation
- Clear feedback about applied overrides
- Support for different deployment environments

### Key Files Modified/Enhanced:
- `captioner/captioner.py`: Priority queue system, awakening animation, identity consolidation
- `config/prompt_templates.py`: Dramatically improved drawing and reflection prompts
- `captioner/prompts.py`: Enhanced reflection data integration (motifs, desires)
- `machine.py`: Frame change detection, performance optimizations
- `config/laptop_config.json`: Performance-friendly configuration
- `config/config.py`: Frame change detection parameters, timing controls

### System Status (January 2025)
**All Core Systems Enhanced & Production Ready**: 
- Enhanced gaze tracking with low-light stability and position holding ✅
- Physics-based hand expression controller with multi-phase startle reactions ✅
- Detailed drawing prompts with technical artistic specifications ✅
- Faster breathing system with improved responsiveness ✅
- Clean output controls for production deployment ✅
- Temporal awareness and memory persistence ✅
- Priority queue preventing LLM conflicts ✅  
- Enhanced frame change detection ✅
- Substantial reflections with purpose exploration ✅
- Clean terminal formatting and organized codebase ✅
- Performance optimizations for various hardware ✅

**Recent Cleanup & Organization**:
- Removed 20+ redundant test files and backup files
- Cleaned up debug folder to essential testing tools only  
- Eliminated temporary files and __pycache__ directories
- Updated documentation to reflect current architecture

**Ready for Production Deployment**: The system has evolved from basic captioning to sophisticated consciousness simulation with stable tracking, physics-based interactions, artistic expression capabilities, and production-ready output controls.

**Critical for Future Development**: 
- The physics-based hand controller provides foundation for complex gestural communication
- Enhanced gaze tracking enables reliable person-machine interaction in various lighting conditions
- Technical drawing prompts create foundation for sophisticated AI art generation
- Clean output system supports both development debugging and production deployment
- Always restart Python process when modifying prompt templates (loaded at startup)
- Priority queue system prevents concurrent LLM calls - don't modify without understanding flow
- Frame change detection significantly improves performance - test carefully if modifying
- Identity consolidation system feeds back into captioning - changes affect entire personality
